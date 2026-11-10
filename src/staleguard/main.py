"""StaleGuard FastAPI app (API layer: HTTP concerns only).

- POST /v1/chat/completions: OpenAI-compatible proxy to Bifrost that verifies the answer (staleguard.guard).
  Headers: x-sg-mode (annotate|rewrite|block), x-sg-include-domains (comma-separated); x-bf-* go to Bifrost.
  Streams pass through unverified (x-staleguard-status: skipped).
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

from staleguard import __version__
from staleguard.bifrost_client import check_bifrost, forwardable_response_headers, make_http_client, stream_chat
from staleguard.config import Settings, get_settings
from staleguard.factory import build_pipeline
from staleguard.guard import DOMAINS_HEADER, MODE_HEADER, STATUS_HEADER, guard_chat, parse_domains, parse_mode
from staleguard.llm import LLMOutputError
from staleguard.pipeline import Pipeline
from staleguard.schemas import HealthResponse, Report, VerifyRequest

__all__ = ["app"]

log = logging.getLogger("staleguard")


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


def get_optional_pipeline(request: Request) -> Pipeline | None:
    """None when Tavily isn't configured: the proxy then passes answers through unverified."""
    return getattr(request.app.state, "pipeline", None)


def get_pipeline(pipeline: Pipeline | None = Depends(get_optional_pipeline)) -> Pipeline:
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
async def chat_completions(
    request: Request,
    http: httpx.AsyncClient = Depends(get_bifrost_http),
    pipeline: Pipeline | None = Depends(get_optional_pipeline),
    settings: Settings = Depends(get_settings),
) -> Response:
    try:
        body: Any = await request.json()
    except ValueError:
        return _openai_error(400, "Request body must be JSON.", "invalid_request_error")
    if not isinstance(body, dict):
        return _openai_error(400, "Request body must be a JSON object.", "invalid_request_error")
    try:
        mode = parse_mode(request.headers.get(MODE_HEADER), settings.default_mode)
    except ValueError as exc:
        return _openai_error(400, str(exc), "invalid_request_error")

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

        result = await guard_chat(
            http=http,
            pipeline=pipeline,
            settings=settings,
            body=body,
            request_headers=request.headers,
            mode=mode,
            include_domains=parse_domains(request.headers.get(DOMAINS_HEADER)),
        )
    except httpx.HTTPError as exc:
        log.warning("Bifrost unreachable: %s", exc)
        return _openai_error(502, f"Bifrost gateway unreachable: {type(exc).__name__}", "bifrost_unreachable")

    return JSONResponse(status_code=result.status_code, content=result.body, headers=result.headers)
