"""Thin client for the Bifrost gateway. Step 1: health probe + OpenAI SDK client factory."""

import httpx
from openai import AsyncOpenAI

from staleguard.config import Settings
from staleguard.schemas import DependencyHealth

# Bifrost does not need a client credential when enforce_auth_on_inference=false;
# the OpenAI SDK insists on *some* api_key, and Bifrost ignores this bearer value.
_PLACEHOLDER_API_KEY = "staleguard-local"


def make_llm_client(settings: Settings) -> AsyncOpenAI:
    """All LLM calls go through Bifrost's OpenAI-compatible endpoint, never to providers directly."""
    return AsyncOpenAI(base_url=settings.bifrost_openai_base_url, api_key=_PLACEHOLDER_API_KEY)


async def check_bifrost(settings: Settings, timeout_s: float = 3.0) -> DependencyHealth:
    url = f"{settings.bifrost_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        return DependencyHealth(ok=ok, url=url, detail=None if ok else f"HTTP {resp.status_code}: {resp.text[:200]}")
    except (httpx.HTTPError, ValueError) as exc:
        return DependencyHealth(ok=False, url=url, detail=f"{type(exc).__name__}: {exc}")
