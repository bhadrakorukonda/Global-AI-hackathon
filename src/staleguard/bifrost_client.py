"""Client for the Bifrost gateway: health probe, OpenAI SDK factory, chat forwarding.

Verified Bifrost behaviour this relies on is recorded in docs/bifrost-notes.md.
"""

from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from openai import AsyncOpenAI

from staleguard.config import Settings
from staleguard.schemas import CacheDebug, DependencyHealth, ForwardResult

# Bifrost does not need a client credential when enforce_auth_on_inference=false;
# the OpenAI SDK insists on *some* api_key, and Bifrost ignores this bearer value.
_PLACEHOLDER_API_KEY = "staleguard-local"

CHAT_PATH = "/v1/chat/completions"

# Request headers forwarded to Bifrost: all Bifrost control headers (x-bf-cache-key,
# x-bf-cache-ttl, x-bf-cache-threshold, x-bf-cache-no-store, x-bf-cache-type, x-bf-vk, ...).
_FORWARD_REQUEST_PREFIXES = ("x-bf-",)
# Response headers passed back to the client: Bifrost's own diagnostics (x-bifrost-cache-*, ...).
_FORWARD_RESPONSE_PREFIXES = ("x-bifrost-",)


def make_llm_client(settings: Settings) -> AsyncOpenAI:
    """All LLM calls go through Bifrost's OpenAI-compatible endpoint, never to providers directly."""
    return AsyncOpenAI(base_url=settings.bifrost_openai_base_url, api_key=_PLACEHOLDER_API_KEY)


def make_http_client(settings: Settings) -> httpx.AsyncClient:
    # Generous read timeout: a cache miss is a full LLM round trip.
    return httpx.AsyncClient(base_url=settings.bifrost_url.rstrip("/"), timeout=httpx.Timeout(120.0, connect=5.0))


def forwardable_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith(_FORWARD_REQUEST_PREFIXES)}


def forwardable_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith(_FORWARD_RESPONSE_PREFIXES)}


def parse_cache_debug(body: Any) -> CacheDebug | None:
    """Bifrost puts cache metadata at body.extra_fields.cache_debug (absent when no cache key was sent)."""
    if not isinstance(body, dict):
        return None
    raw = (body.get("extra_fields") or {}).get("cache_debug")
    return CacheDebug.model_validate(raw) if isinstance(raw, dict) else None


async def check_bifrost(settings: Settings, timeout_s: float = 3.0) -> DependencyHealth:
    url = f"{settings.bifrost_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        return DependencyHealth(ok=ok, url=url, detail=None if ok else f"HTTP {resp.status_code}: {resp.text[:200]}")
    except (httpx.HTTPError, ValueError) as exc:
        return DependencyHealth(ok=False, url=url, detail=f"{type(exc).__name__}: {exc}")


async def forward_chat(http: httpx.AsyncClient, body: dict[str, Any], headers: Mapping[str, str]) -> ForwardResult:
    """Forward a non-streaming chat completion to Bifrost unchanged. Raises httpx.HTTPError if unreachable."""
    resp = await http.post(CHAT_PATH, json=body, headers=forwardable_request_headers(headers))
    try:
        payload: Any = resp.json()
    except ValueError:
        payload = {"error": {"message": resp.text[:500], "type": "bifrost_invalid_json"}}
    return ForwardResult(
        status_code=resp.status_code,
        body=payload,
        headers=forwardable_response_headers(resp.headers),
        cache_debug=parse_cache_debug(payload),
    )


async def stream_chat(
    http: httpx.AsyncClient, body: dict[str, Any], headers: Mapping[str, str]
) -> tuple[httpx.Response, AsyncIterator[bytes]]:
    """Open a streaming chat completion. Caller must iterate the returned bytes (which closes the response)."""
    req = http.build_request("POST", CHAT_PATH, json=body, headers=forwardable_request_headers(headers))
    resp = await http.send(req, stream=True)

    async def _iter() -> AsyncIterator[bytes]:
        try:
            # Decoded bytes: we don't forward content-encoding, so raw (possibly gzipped) bytes would corrupt.
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return resp, _iter()
