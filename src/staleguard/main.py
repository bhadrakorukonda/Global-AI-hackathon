"""StaleGuard FastAPI app (API layer: HTTP concerns only).

- POST /v1/chat/completions: transparent proxy to Bifrost. Verification is wired in at step 5, so every
  response carries x-staleguard-status: skipped.
- POST /verify: run the verification pipeline on arbitrary text (annotate mode).
- GET /health
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from tavily import AsyncTavilyClient

from staleguard import __version__
from staleguard.bifrost_client import (
    check_bifrost,
    forward_chat,
    forwardable_response_headers,
    make_http_client,
    make_llm_client,
    stream_chat,
)
from staleguard.config import Settings, get_settings
from staleguard.llm import LLMOutputError, OpenAIChatBackend
from staleguard.pipeline import Pipeline, PipelineDeps
from staleguard.schemas import HealthResponse, Report, StaleGuardStatus, VerifyRequest
from staleguard.verdict_cache import VerdictCache

__all__ = ["app"]

log = logging.getLogger("staleguard")

STATUS_HEADER = "x-staleguard-status"


def build_pipeline(settings: Settings) -> Pipeline | None:
    """Wire real backends. None when Tavily isn't configured (/verify then answers 503)."""
    if settings.tavily_api_key is None or not settings.tavily_api_key.get_secret_value():
        log.warning("TAVILY_API_KEY not set: /verify disabled")
        return None
    return Pipeline(
        PipelineDeps(
            llm=OpenAIChatBackend(make_llm_client(settings)),
            tavily=AsyncTavilyClient(settings.tavily_api_key.get_secret_value()),
            small_model=settings.small_model,
            judge_model=settings.judge_model,
            max_claims=settings.max_claims_per_answer,
            verdict_cache=VerdictCache(settings.verdict_cache_path) if settings.verdict_cache_enabled else None,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.bifrost_http = make_http_client(settings)
    app.state.pipeline = build_pipeline(settings)
    try:
        yield
    finally:
        await app.state.bifrost_http.aclose()
        if app.state.pipeline is not None and app.state.pipeline.deps.verdict_cache is not None:
            app.state.pipeline.deps.verdict_cache.close()


app = FastAPI(title="StaleGuard", version=__version__, lifespan=lifespan)


def get_bifrost_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.bifrost_http


def get_pipeline(request: Request) -> Pipeline:
    pipeline: Pipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Verification unavailable: TAVILY_API_KEY is not configured.")
    return pipeline


def _openai_error(status: int, message: str, err_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
        headers={STATUS_HEADER: "skipped"},
    )


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    bifrost = await check_bifrost(settings)
    return HealthResponse(status="ok" if bifrost.ok else "degraded", version=__version__, bifrost=bifrost)


@app.post("/verify", response_model=Report)
async def verify(req: VerifyRequest, pipeline: Pipeline = Depends(get_pipeline)) -> Report:
    """Verify arbitrary text (e.g. a RAG output) without an upstream chat call. Annotate mode only."""
    try:
        return await pipeline.run(req)
    except LLMOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # openai.APIError etc. — the LLM leg goes through Bifrost
        log.exception("verification failed")
        raise HTTPException(status_code=502, detail=f"Verification failed: {type(exc).__name__}: {exc}") from exc


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, http: httpx.AsyncClient = Depends(get_bifrost_http)) -> Response:
    try:
        body: Any = await request.json()
    except ValueError:
        return _openai_error(400, "Request body must be JSON.", "invalid_request_error")
    if not isinstance(body, dict):
        return _openai_error(400, "Request body must be a JSON object.", "invalid_request_error")

    try:
        if body.get("stream"):
            # v1: streaming is passed through unverified.
            upstream, chunks = await stream_chat(http, body, request.headers)
            headers = forwardable_response_headers(upstream.headers) | {STATUS_HEADER: "skipped"}
            return StreamingResponse(
                chunks,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                headers=headers,
            )

        result = await forward_chat(http, body, request.headers)
    except httpx.HTTPError as exc:
        log.warning("Bifrost unreachable: %s", exc)
        return _openai_error(502, f"Bifrost gateway unreachable: {type(exc).__name__}", "bifrost_unreachable")

    if result.cache_debug is not None:
        log.info(
            "bifrost cache: hit=%s type=%s id=%s",
            result.cache_debug.cache_hit, result.cache_debug.hit_type, result.cache_debug.cache_id,
        )
    status: StaleGuardStatus = "skipped"  # step 5 replaces this with the pipeline verdict
    return JSONResponse(status_code=result.status_code, content=result.body, headers=result.headers | {STATUS_HEADER: status})
