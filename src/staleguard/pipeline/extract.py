"""Stage 1 — split an answer into atomic, self-contained claims (small model).

The same call also writes a *neutral* search query per claim (the topic, without the claimed value), at
no extra LLM cost: searching with the claim text itself pulls in pages that repeat the stale value
(measured live, docs/bifrost-notes.md §3).

It also proposes each claim's volatility class (`class_hint`), so the classify stage needs no LLM call
of its own in the common case: one small-model round trip instead of two on every answer.
"""

import re

from staleguard.llm import ChatBackend, UsageTracker, structured_call
from staleguard.schemas import Claim, ExtractLLMOutput

__all__ = ["extract_claims", "strip_code"]

# Fenced blocks and inline code are out of scope (SPEC §10: no verifying claims inside code).
_FENCED = re.compile(r"```.*?(```|\Z)", re.DOTALL)
_INLINE = re.compile(r"`[^`\n]*`")


def strip_code(text: str) -> str:
    """Remove fenced code blocks entirely; keep inline code text (it is often a version or name) unquoted."""
    return _INLINE.sub(lambda m: m.group(0).strip("`"), _FENCED.sub(" ", text)).strip()


async def extract_claims(
    backend: ChatBackend, *, model: str, text: str, question: str | None, usage: UsageTracker
) -> list[Claim]:
    """Return claims numbered from 1. Empty if the text has no prose or no factual claims."""
    prose = strip_code(text)
    if not prose:
        return []
    out = await structured_call(
        backend,
        model=model,
        prompt="extract",
        variables={"question": question or "", "text": prose},
        output_model=ExtractLLMOutput,
        stage="extract",
        usage=usage,
        reasoning_effort="low",
    )
    kept = [c for c in out.claims if c.claim.strip()]
    return [
        Claim(id=i, text=c.claim.strip(), search_query=(c.query or "").strip() or None, class_hint=c.claim_class)
        for i, c in enumerate(kept, start=1)
    ]
