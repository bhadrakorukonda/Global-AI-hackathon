"""HTTP calls and view model for the Streamlit demo (demo/app.py).

No Streamlit imports here, so the parsing is unit-testable (tests/test_demo_view.py). The demo is a plain
client of the running StaleGuard proxy: it exercises exactly the path a real app would.
"""

import time
from typing import Any

import httpx
from pydantic import BaseModel

from staleguard.guard import (
    DOMAINS_HEADER,
    ERROR_HEADER,
    MODE_HEADER,
    STATUS_HEADER,
)
from staleguard.llm import message_text
from staleguard.schemas import CacheDebug, Mode, Report

__all__ = [
    "VERDICT_STYLE",
    "Exchange",
    "ask_guarded",
    "chat_body",
    "cost_rows",
    "parse_exchange",
    "seed_cache",
]

CACHE_KEY_HEADER = "x-bf-cache-key"

# verdict -> (Streamlit color name, short label)
VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "STALE": ("red", "Stale"),
    "CURRENT": ("green", "Current"),
    "CONFLICTING": ("orange", "Conflicting"),
    "UNVERIFIABLE": ("gray", "Unverifiable"),
}


class Exchange(BaseModel):
    """One question through the proxy, ready to render."""

    question: str
    mode: Mode
    http_status: int
    status: str  # x-staleguard-status: clean | stale | unverified | skipped
    model_answer: str  # what the model (or Bifrost's cache) said
    final_answer: str  # what StaleGuard returned to the client
    report: Report | None = None
    cache: CacheDebug | None = None
    error: str | None = None
    wall_ms: int = 0

    @property
    def upstream_ms(self) -> int | None:
        """Time spent in Bifrost/the model, i.e. everything StaleGuard didn't add."""
        return self.wall_ms - self.report.latency_ms if self.report else None


def chat_body(model: str, question: str) -> dict[str, Any]:
    """A single-turn chat request; low reasoning keeps the upstream answer fast for the demo."""
    return {"model": model, "messages": [{"role": "user", "content": question}], "reasoning_effort": "low"}


def _answer(body: Any) -> str:
    try:
        return message_text(body["choices"][0]["message"]["content"]) or ""
    except (KeyError, IndexError, TypeError):
        return ""


def parse_exchange(question: str, mode: Mode, status_code: int, headers: httpx.Headers | dict[str, str], body: Any, wall_ms: int) -> Exchange:
    """Turn the proxy's HTTP response into an Exchange. Never raises on odd bodies; errors are shown, not thrown."""
    headers = httpx.Headers(headers)
    if not isinstance(body, dict):
        return Exchange(question=question, mode=mode, http_status=status_code, status="skipped", model_answer="",
                        final_answer="", error=f"Unexpected response: {str(body)[:300]}", wall_ms=wall_ms)
    if status_code != 200:
        detail = body.get("error") or body.get("detail") or body
        return Exchange(question=question, mode=mode, http_status=status_code, status="skipped", model_answer="",
                        final_answer="", error=f"HTTP {status_code}: {str(detail)[:300]}", wall_ms=wall_ms)

    report = Report.model_validate(body["staleguard"]) if body.get("staleguard") else None
    raw_cache = (body.get("extra_fields") or {}).get("cache_debug")
    final = _answer(body)
    return Exchange(
        question=question,
        mode=mode,
        http_status=status_code,
        status=headers.get(STATUS_HEADER, report.verdict_summary if report else "skipped"),
        model_answer=(report.original_answer if report and report.original_answer else final),
        final_answer=final,
        report=report,
        cache=CacheDebug.model_validate(raw_cache) if raw_cache else None,
        error=headers.get(ERROR_HEADER),
        wall_ms=wall_ms,
    )


async def ask_guarded(
    http: httpx.AsyncClient,
    proxy_url: str,
    *,
    model: str,
    question: str,
    mode: Mode,
    cache_key: str | None = None,
    include_domains: list[str] | None = None,
) -> Exchange:
    """Send the question through StaleGuard (/v1/chat/completions) with the demo's per-request policy headers."""
    headers = {MODE_HEADER: mode}
    if cache_key:
        headers[CACHE_KEY_HEADER] = cache_key
    if include_domains:
        headers[DOMAINS_HEADER] = ",".join(include_domains)
    started = time.perf_counter()
    try:
        resp = await http.post(f"{proxy_url.rstrip('/')}/v1/chat/completions", json=chat_body(model, question), headers=headers)
    except httpx.HTTPError as exc:
        return Exchange(question=question, mode=mode, http_status=0, status="skipped", model_answer="", final_answer="",
                        error=f"Can't reach StaleGuard at {proxy_url}: {type(exc).__name__}", wall_ms=0)
    wall_ms = round((time.perf_counter() - started) * 1000)
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    return parse_exchange(question, mode, resp.status_code, resp.headers, body, wall_ms)


async def seed_cache(http: httpx.AsyncClient, bifrost_url: str, *, model: str, question: str, cache_key: str) -> tuple[str, str | None]:
    """Ask Bifrost directly (bypassing StaleGuard) so the answer lands in its semantic cache unverified.

    Stands in for "an answer cached last month". Returns (answer, cache_id).
    """
    resp = await http.post(f"{bifrost_url.rstrip('/')}/v1/chat/completions", json=chat_body(model, question),
                           headers={CACHE_KEY_HEADER: cache_key})
    resp.raise_for_status()
    body = resp.json()
    cd = (body.get("extra_fields") or {}).get("cache_debug") or {}
    return _answer(body), cd.get("cache_id")


def cost_rows(report: Report) -> list[tuple[str, str]]:
    """(label, value) pairs for the "cost of this check" footer."""
    tokens = report.tokens_by_stage
    per_stage = ", ".join(f"{stage} {n:,}" for stage, n in tokens.items()) or "none"
    models = ", ".join(f"{stage}: {m}" for stage, m in report.models_by_stage.items()) or "n/a"
    return [
        ("Added latency", f"{report.latency_ms:,} ms"),
        ("Claims", f"{report.claims_checked} checked of {report.claims_extracted} extracted"),
        ("Tavily searches", str(report.tavily_calls)),
        ("Verdict-cache hits", str(report.verdict_cache_hits)),
        ("Tokens", f"{sum(tokens.values()):,} ({per_stage})"),
        ("Models", models),
    ]
