"""Verified chat completions (service layer): forward to Bifrost -> verify the answer -> act -> evict.

Policy decisions live here; HTTP parsing lives in main.py.

- Verification needs a successful, non-streaming response with assistant text. Anything else (upstream
  error, tool-call-only reply, Tavily not configured) passes through with status `skipped`.
- Fail open: if verification itself fails, the client still gets Bifrost's answer, marked `skipped`, with
  the reason in `x-staleguard-error`. A fact-checker outage must not take the chat API down with it.
- Eviction: if any claim is STALE and Bifrost returned a cache_id, that entry is evicted. This covers a
  cache *hit* that served a stale answer (SPEC §3), and also a *miss*: Bifrost has just cached the new
  stale answer under that id (the cache_id on a miss is the entry it wrote; docs/bifrost-notes.md), and
  evicting it stops it being replayed. Eviction ignores the confidence threshold: a wrong eviction only
  costs one regeneration.
"""

import copy
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from staleguard.bifrost_client import evict_cache_entry, forward_chat
from staleguard.config import Settings
from staleguard.llm import UsageTracker, message_text
from staleguard.pipeline import Pipeline
from staleguard.pipeline.act import actionable_stale, block_message, rewrite_answer
from staleguard.schemas import ForwardResult, Mode, Report, StaleGuardStatus, Verdict, VerifyRequest

__all__ = [
    "DOMAINS_HEADER",
    "MODE_HEADER",
    "GuardResult",
    "STATUS_HEADER",
    "guard_chat",
    "parse_domains",
    "parse_mode",
]

log = logging.getLogger(__name__)

MODE_HEADER = "x-sg-mode"
DOMAINS_HEADER = "x-sg-include-domains"
STATUS_HEADER = "x-staleguard-status"
CLAIMS_HEADER = "x-staleguard-claims-checked"
ACTION_HEADER = "x-staleguard-action"
EVICTED_HEADER = "x-staleguard-cache-evicted"
ERROR_HEADER = "x-staleguard-error"

_MODES: tuple[Mode, ...] = ("annotate", "rewrite", "block")


@dataclass
class GuardResult:
    status_code: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


def parse_mode(value: str | None, default: Mode) -> Mode:
    """x-sg-mode header -> Mode. Raises ValueError on an unknown mode."""
    if value is None or not value.strip():
        return default
    mode = value.strip().lower()
    if mode not in _MODES:
        raise ValueError(f"{MODE_HEADER} must be one of {', '.join(_MODES)}; got {value!r}")
    return cast(Mode, mode)


def parse_domains(value: str | None) -> list[str] | None:
    """x-sg-include-domains header (comma-separated) -> trusted-domain policy for Tavily."""
    domains = [d.strip() for d in (value or "").split(",") if d.strip()]
    return domains or None


def _last_user_text(body: dict[str, Any]) -> str | None:
    for msg in reversed(body.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return message_text(msg.get("content"))
    return None


def _answer_text(body: Any) -> str | None:
    try:
        return message_text(body["choices"][0]["message"].get("content"))
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def _skipped(result: ForwardResult, error: str | None = None) -> GuardResult:
    headers = result.headers | {STATUS_HEADER: "skipped"}
    if error:
        headers[ERROR_HEADER] = error[:200]
    return GuardResult(result.status_code, result.body, headers)


def _with_answer(body: dict[str, Any], text: str) -> dict[str, Any]:
    new = copy.deepcopy(body)
    new["choices"][0]["message"]["content"] = text  # plain string: valid OpenAI shape for every client
    return new


async def guard_chat(
    *,
    http: httpx.AsyncClient,
    pipeline: Pipeline | None,
    settings: Settings,
    body: dict[str, Any],
    request_headers: Mapping[str, str],
    mode: Mode,
    include_domains: list[str] | None = None,
) -> GuardResult:
    """Forward a non-streaming chat completion and verify the answer. Raises httpx.HTTPError if Bifrost is down."""
    result = await forward_chat(http, body, request_headers)
    cd = result.cache_debug
    if cd is not None:
        log.info("bifrost cache: hit=%s type=%s id=%s", cd.cache_hit, cd.hit_type, cd.cache_id)

    answer = _answer_text(result.body) if result.status_code == 200 else None
    if pipeline is None or not answer or not answer.strip():
        return _skipped(result)

    question = _last_user_text(body)
    started = time.perf_counter()
    try:
        report = await pipeline.run(VerifyRequest(text=answer, question=question, include_domains=include_domains))
    except Exception as exc:  # fail open (module docstring)
        log.exception("verification failed; returning the unverified answer")
        return _skipped(result, f"verification failed: {type(exc).__name__}")

    out_body: dict[str, Any] = result.body
    update: dict[str, Any] = {"mode": mode, "bifrost_cache_hit": cd.cache_hit if cd else None}
    headers = dict(result.headers)

    stale = actionable_stale(report, settings.stale_confidence_threshold)
    if stale and mode == "block":
        out_body = _with_answer(out_body, block_message(stale))
        update |= {"action": "blocked", "original_answer": answer}
    elif stale and mode == "rewrite":
        usage = UsageTracker()
        try:
            new_answer = await rewrite_answer(
                pipeline.deps.llm, model=pipeline.deps.judge_model, question=question,
                answer=answer, stale=stale, today=pipeline.deps.today(), usage=usage,
            )
        except Exception as exc:  # keep the annotated original rather than failing the request
            log.exception("rewrite failed; returning the annotated original")
            headers[ERROR_HEADER] = f"rewrite failed: {type(exc).__name__}"
        else:
            out_body = _with_answer(out_body, new_answer)
            update |= {"action": "rewritten", "original_answer": answer}
        update |= {
            "tokens_by_stage": report.tokens_by_stage | usage.snapshot(),
            "models_by_stage": report.models_by_stage | usage.models(),
        }

    if cd is not None and cd.cache_id and any(v.verdict is Verdict.STALE for v in report.claims):
        evicted = await evict_cache_entry(http, cd.cache_id, settings.bifrost_setup_token)
        update |= {"cache_evicted": evicted, "evicted_cache_id": cd.cache_id if evicted else None}

    update["latency_ms"] = round((time.perf_counter() - started) * 1000)  # all added latency, incl. rewrite + eviction
    final: Report = report.model_copy(update=update)
    if out_body is result.body:
        out_body = copy.copy(result.body)
    out_body["staleguard"] = final.model_dump(mode="json")

    status: StaleGuardStatus = final.verdict_summary
    headers |= {
        STATUS_HEADER: status,
        CLAIMS_HEADER: str(final.claims_checked),
        ACTION_HEADER: final.action,
        EVICTED_HEADER: str(final.cache_evicted).lower(),
    }
    return GuardResult(result.status_code, out_body, headers)
