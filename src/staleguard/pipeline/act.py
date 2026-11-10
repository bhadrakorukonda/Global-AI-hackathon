"""Stage 5: act on a Report (SPEC §3 Act). Decides what the proxy does to the answer.

- annotate: answer unchanged; the report is attached (the proxy does that, nothing to do here).
- rewrite:  regenerate the answer with the corrected claims and citations (judge model, via Bifrost).
- block:    replace the answer with a deterministic refusal listing the stale claims (no LLM call).

rewrite and block only act on STALE claims with a correction at or above the confidence threshold, so a
low-confidence STALE is reported but never changes what the user sees.
"""

import json
from datetime import date

from staleguard.llm import ChatBackend, UsageTracker, structured_call
from staleguard.schemas import ClaimVerdict, Report, RewriteLLMOutput, Verdict

__all__ = ["actionable_stale", "block_message", "rewrite_answer"]


def actionable_stale(report: Report, threshold: float) -> list[ClaimVerdict]:
    """STALE claims confident enough to change the answer (and that carry a correction)."""
    return [
        v for v in report.claims
        if v.verdict is Verdict.STALE and v.corrected_claim and v.confidence >= threshold
    ]


def block_message(stale: list[ClaimVerdict]) -> str:
    """User-facing refusal for block mode. Deterministic so it can't itself contain a stale claim."""
    lines = ["StaleGuard withheld this answer because it contains out-of-date information:", ""]
    for v in stale:
        lines.append(f"- \"{v.claim.text}\" is out of date. Current: {v.corrected_claim}")
        lines.extend(f"  Source: {e.title} — {e.url}" + (f" ({e.published_date})" if e.published_date else "") for e in v.evidence)
    lines += ["", "Ask again to get an updated answer."]
    return "\n".join(lines)


def _corrections_input(stale: list[ClaimVerdict]) -> str:
    """Corrections with globally numbered sources, so the rewrite's [n] markers are unambiguous."""
    sources: dict[str, int] = {}
    payload = []
    for v in stale:
        cited = []
        for e in v.evidence:
            n = sources.setdefault(e.url, len(sources) + 1)
            cited.append({"n": n, "title": e.title, "url": e.url, "published": e.published_date})
        payload.append({"stale_claim": v.claim.text, "corrected": v.corrected_claim, "sources": cited})
    return json.dumps(payload, ensure_ascii=False, indent=1)


async def rewrite_answer(
    backend: ChatBackend,
    *,
    model: str,
    question: str | None,
    answer: str,
    stale: list[ClaimVerdict],
    today: date,
    usage: UsageTracker,
) -> str:
    """Rewrite `answer` with the stale claims corrected and cited. Raises LLMOutputError on bad output."""
    out = await structured_call(
        backend,
        model=model,
        prompt="rewrite",
        variables={
            "today": today.isoformat(),
            "question": question or "(not given)",
            "answer": answer,
            "corrections": _corrections_input(stale),
        },
        output_model=RewriteLLMOutput,
        stage="rewrite",
        usage=usage,
        reasoning_effort="low",
    )
    return out.answer
