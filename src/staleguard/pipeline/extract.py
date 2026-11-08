"""Stage 1 — split an answer into atomic, self-contained claims (small model)."""

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
    texts = [c.strip() for c in out.claims if c and c.strip()]
    return [Claim(id=i, text=t) for i, t in enumerate(texts, start=1)]
