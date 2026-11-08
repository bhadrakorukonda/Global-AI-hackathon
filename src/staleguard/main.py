"""StaleGuard FastAPI app.

Step 2: transparent OpenAI-compatible proxy to Bifrost. Verification is not wired in yet (step 5),
so every response carries x-staleguard-status: skipped.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from staleguard import __version__
from staleguard.bifrost_client import (
    check_bifrost,
    forward_chat,
    forwardable_response_headers,
    make_http_client,
    stream_chat,
)
from staleguard.config import Settings, get_settings
from staleguard.schemas import HealthResponse, StaleGuardStatus

log = logging.getLogger("staleguard")

STATUS_HEADER = "x-staleguard-status"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.bifrost_http = make_http_client(get_settings())
    try:
        yield
    finally:
        await app.state.bifrost_http.aclose()


app = FastAPI(title="StaleGuard", version=__version__, lifespan=lifespan)


def get_bifrost_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.bifrost_http


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
