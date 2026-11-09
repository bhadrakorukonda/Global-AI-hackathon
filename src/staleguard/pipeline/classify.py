"""Stage 2 — classify claims by volatility: deterministic rules first, small model for the rest.

Rules only ever mark a claim *volatile*; they never declare one static. Claims no rule matches go to
the LLM in one batched call, which may answer `static`.
"""

import json
import re
from dataclasses import dataclass
from datetime import date

from staleguard.llm import ChatBackend, UsageTracker, structured_call
from staleguard.schemas import CLASS_TTL, Claim, ClaimClass, ClassifyLLMOutput

__all__ = ["classify_claims", "classify_by_rules", "select_for_verification"]

@dataclass(frozen=True)
class _Rule:
    name: str
    claim_class: ClaimClass
    pattern: re.Pattern[str]


def _rx(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# Order = priority when several classes match: the most specific class wins, relative_time is the fallback.
_RULES: tuple[_Rule, ...] = (
    _Rule("currency_amount", ClaimClass.PRICE, _rx(r"[$€£₹¥]\s?\d|\b\d+(?:[.,]\d+)?\s?(?:usd|eur|inr|dollars?|euros?|rupees?)\b")),
    _Rule("price_word", ClaimClass.PRICE, _rx(r"\b(?:costs?|priced?|pricing|per (?:month|year|seat|user|token|million tokens?)|subscription fee|fee)\b")),
    _Rule("version_number", ClaimClass.VERSION, _rx(r"\b(?:version|release|v)\s?\d+(?:\.\d+)+\b|\b\d+\.\d+(?:\.\d+)?\b(?=[^.]*\b(?:release|released|version|stable|lts)\b)")),
    _Rule("version_word", ClaimClass.VERSION, _rx(r"\b(?:latest|newest|current|stable|lts)\s+(?:\w+\s+){0,2}(?:version|release)\b")),
    _Rule("role_title", ClaimClass.ROLE, _rx(r"\b(?:ceo|cto|cfo|coo|chief \w+ officer|president|vice president|prime minister|chancellor|chair(?:man|woman|person)?|head of|founder and ceo|governor|mayor|minister|secretary of|director of)\b")),
    _Rule("event_state", ClaimClass.EVENT_STATUS, _rx(r"\b(?:pending|ongoing|underway|upcoming|in talks|in negotiations|expected to|scheduled (?:to|for)|has (?:been )?announced|is (?:being )?(?:acquired|reviewed|delayed|investigated)|merger|acquisition|lawsuit|on track to|in beta|in preview)\b")),
    _Rule("policy_term", ClaimClass.POLICY, _rx(r"\b(?:regulations?|regulators?|guidelines?|directive|mandat(?:e|es|ory)|requires? by law|legal(?:ly)? required|compliance|banned|bans?|statute|ordinance)\b|\b[A-Z][\w-]*(?: [A-Z][\w-]*)* (?:Act|Law|Regulation)\b", re.UNICODE)),
    _Rule("statistic_term", ClaimClass.STATISTIC, _rx(r"\b(?:population|gdp|inflation|unemployment|market share|revenue|valuation|ranked|ranking|most popular)\b|\b\d+(?:[.,]\d+)?\s?(?:%|percent|million|billion|trillion|crore|lakh)\b|\b\d[\d,.]*\+?\s(?:monthly |daily |active )*(?:users|subscribers|downloads|customers)\b")),
    _Rule("relative_word", ClaimClass.RELATIVE_TIME, _rx(r"\b(?:latest|newest|current(?:ly)?|now|today|nowadays|recent(?:ly)?|as of|so far|to date|up[- ]to[- ]date|at the moment|this (?:year|month|week|quarter))\b")),
)

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def classify_by_rules(text: str, today: date) -> tuple[ClaimClass | None, list[str]]:
    """Return (highest-priority matching class, names of all rules that fired). (None, []) if nothing matched.

    A year >= last year also counts as relative_time: "in 2026" is a claim about the present.
    """
    fired = [r for r in _RULES if r.pattern.search(text)]
    signals = [r.name for r in fired]
    if any(int(y) >= today.year - 1 for y in _YEAR.findall(text)):
        signals.append("recent_year")
        if not fired:
            return ClaimClass.RELATIVE_TIME, signals
    return (fired[0].claim_class if fired else None), signals


async def classify_claims(
    backend: ChatBackend, *, model: str, claims: list[Claim], today: date, usage: UsageTracker
) -> list[Claim]:
    """Classify every claim. Rule hits skip the LLM; the remainder go in one batched LLM call."""
    classified: list[Claim] = []
    unresolved: list[Claim] = []
    for claim in claims:
        cls, signals = classify_by_rules(claim.text, today)
        if cls is not None:
            classified.append(claim.model_copy(update={"claim_class": cls, "classified_by": "rules", "signals": signals}))
        else:
            unresolved.append(claim)

    if unresolved:
        out = await structured_call(
            backend,
            model=model,
            prompt="classify",
            variables={"claims": json.dumps([{"id": c.id, "text": c.text} for c in unresolved], ensure_ascii=False)},
            output_model=ClassifyLLMOutput,
            stage="classify",
            usage=usage,
            reasoning_effort="low",
        )
        by_id = {item.id: item.claim_class for item in out.items}
        for claim in unresolved:
            # A claim the model skipped is treated as volatile-by-context rather than silently dropped.
            cls = by_id.get(claim.id, ClaimClass.RELATIVE_TIME)
            classified.append(claim.model_copy(update={"claim_class": cls, "classified_by": "llm"}))

    return sorted(classified, key=lambda c: c.id)


def select_for_verification(claims: list[Claim], max_claims: int) -> list[Claim]:
    """Drop static claims; keep the `max_claims` most volatile (shortest TTL first, then answer order)."""
    volatile = [c for c in claims if c.claim_class not in (None, ClaimClass.STATIC)]
    ranked = sorted(volatile, key=lambda c: (CLASS_TTL[c.claim_class], c.id))  # type: ignore[index]
    return sorted(ranked[:max_claims], key=lambda c: c.id)
