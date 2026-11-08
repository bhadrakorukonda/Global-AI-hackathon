"""Stage 4 — judge each claim against its evidence (judge model, one batched call).

Batched rather than one call per claim to stay inside Gemini free-tier request limits; the output is
still one verdict per claim. Evidence rules are enforced in code, not trusted to the model:
no cited evidence -> UNVERIFIABLE, never STALE.
"""

import json
from datetime import date

from staleguard.llm import ChatBackend, UsageTracker, structured_call
from staleguard.schemas import Claim, ClaimVerdict, Evidence, JudgeLLMItem, JudgeLLMOutput, SearchResult, Verdict

__all__ = ["apply_evidence_rules", "judge_claims"]


def _judge_input(claims: list[Claim], searches: dict[int, SearchResult]) -> str:
    payload = []
    for c in claims:
        s = searches.get(c.id)
        evidence = s.evidence if s else []
        payload.append(
            {
                "id": c.id,
                "claim": c.text,
                "claim_class": c.claim_class,
                "search_summary": s.answer if s else None,
                "evidence": [
                    {"n": n, "title": e.title, "url": e.url, "published": e.published_date, "snippet": e.snippet}
                    for n, e in enumerate(evidence, start=1)
                ],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=1)


def apply_evidence_rules(claim: Claim, item: JudgeLLMItem | None, evidence: list[Evidence]) -> ClaimVerdict:
    """Turn the model's raw verdict into a ClaimVerdict, enforcing SPEC §3's evidence requirements."""
    if item is None:
        return ClaimVerdict(claim=claim, verdict=Verdict.UNVERIFIABLE, confidence=0.0, rationale="Judge returned no verdict.")

    cited = [evidence[i - 1] for i in dict.fromkeys(item.evidence_ids) if 1 <= i <= len(evidence)]
    verdict, confidence, rationale = item.verdict, item.confidence, item.rationale
    if verdict is not Verdict.UNVERIFIABLE and not cited:
        verdict, confidence = Verdict.UNVERIFIABLE, 0.0
        rationale = f"Downgraded: judge said {item.verdict.value} without citing evidence."
    corrected = item.corrected_claim if verdict is Verdict.STALE else None
    return ClaimVerdict(
        claim=claim, verdict=verdict, corrected_claim=corrected, confidence=confidence, rationale=rationale, evidence=cited
    )


async def judge_claims(
    backend: ChatBackend,
    *,
    model: str,
    claims: list[Claim],
    searches: list[SearchResult],
    today: date,
    usage: UsageTracker,
) -> list[ClaimVerdict]:
    """Judge all claims in one call. Claims with no evidence at all skip the model and are UNVERIFIABLE."""
    by_id = {s.claim_id: s for s in searches}
    with_evidence = [c for c in claims if by_id.get(c.id) and by_id[c.id].evidence]

    items: dict[int, JudgeLLMItem] = {}
    if with_evidence:
        out = await structured_call(
            backend,
            model=model,
            prompt="judge",
            variables={"today": today.isoformat(), "claims": _judge_input(with_evidence, by_id)},
            output_model=JudgeLLMOutput,
            stage="judge",
            usage=usage,
        )
        items = {v.id: v for v in out.verdicts}

    verdicts = []
    for c in claims:
        search = by_id.get(c.id)
        evidence = search.evidence if search else []
        if not evidence:
            reason = f"No evidence found ({search.error})." if search and search.error else "No evidence found."
            v = ClaimVerdict(claim=c, verdict=Verdict.UNVERIFIABLE, confidence=0.0, rationale=reason)
        else:
            v = apply_evidence_rules(c, items.get(c.id), evidence)
        verdicts.append(v.model_copy(update={"search_query": search.query if search else None}))
    return verdicts
