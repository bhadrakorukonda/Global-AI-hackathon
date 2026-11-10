"""LangGraph wiring plus Report assembly.

    extract -> classify -> cache_lookup -> verify -> judge -> cache_store -> END
       |           |            |
       +-----------+------------+--> END   (fast paths: no claims / nothing volatile / all cached)
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from staleguard.llm import ChatBackend, UsageTracker
from staleguard.pipeline.classify import classify_claims, select_for_verification
from staleguard.pipeline.extract import extract_claims
from staleguard.pipeline.judge import judge_claims
from staleguard.pipeline.verify import SearchBackend, verify_claims
from staleguard.schemas import Claim, ClaimVerdict, Report, SearchResult, Verdict, VerifyRequest
from staleguard.verdict_cache import VerdictCache

__all__ = ["Pipeline", "PipelineDeps", "summarize"]


@dataclass(frozen=True)
class PipelineDeps:
    llm: ChatBackend
    tavily: SearchBackend
    small_model: str
    judge_model: str
    max_claims: int = 5
    verdict_cache: VerdictCache | None = None  # None = verify every claim every time
    today: Callable[[], date] = field(default=date.today)


class _State(TypedDict, total=False):
    request: VerifyRequest
    today: date
    usage: UsageTracker
    claims: list[Claim]
    selected: list[Claim]
    cached: list[ClaimVerdict]
    to_verify: list[Claim]
    searches: list[SearchResult]
    verdicts: list[ClaimVerdict]


def summarize(verdicts: list[ClaimVerdict]) -> str:
    """stale if any claim is STALE; unverified if any is UNVERIFIABLE/CONFLICTING; else clean (incl. nothing to check)."""
    kinds = {v.verdict for v in verdicts}
    if Verdict.STALE in kinds:
        return "stale"
    if kinds & {Verdict.UNVERIFIABLE, Verdict.CONFLICTING}:
        return "unverified"
    return "clean"


class Pipeline:
    """Compiled verification graph. Build once per app; `run` is safe to call concurrently."""

    def __init__(self, deps: PipelineDeps) -> None:
        self.deps = deps
        self._graph = self._build()

    def _build(self) -> Any:
        d = self.deps

        async def extract(s: _State) -> _State:
            r = s["request"]
            return {"claims": await extract_claims(d.llm, model=d.small_model, text=r.text, question=r.question, usage=s["usage"])}

        async def classify(s: _State) -> _State:
            claims = await classify_claims(d.llm, model=d.small_model, claims=s["claims"], today=s["today"], usage=s["usage"])
            return {"claims": claims, "selected": select_for_verification(claims, d.max_claims)}

        def cache_lookup(s: _State) -> _State:
            domains = s["request"].include_domains
            cached: list[ClaimVerdict] = []
            to_verify: list[Claim] = []
            for claim in s["selected"]:
                hit = d.verdict_cache.get(claim.text, domains) if d.verdict_cache else None
                if hit is None:
                    to_verify.append(claim)
                else:
                    # Keep this answer's claim (id, wording, class); reuse the stored judgement.
                    cached.append(hit.model_copy(update={"claim": claim, "from_cache": True}))
            return {"cached": cached, "to_verify": to_verify}

        async def verify(s: _State) -> _State:
            return {"searches": await verify_claims(d.tavily, s["to_verify"], s["today"], s["request"].include_domains)}

        async def judge(s: _State) -> _State:
            verdicts = await judge_claims(
                d.llm, model=d.judge_model, claims=s["to_verify"], searches=s["searches"], today=s["today"], usage=s["usage"]
            )
            return {"verdicts": verdicts}

        def cache_store(s: _State) -> _State:
            if d.verdict_cache is not None:
                failed = {r.claim_id for r in s["searches"] if r.error}  # don't remember outages as verdicts
                for v in s["verdicts"]:
                    if v.claim.id not in failed:
                        d.verdict_cache.put(v, s["request"].include_domains)
            return {}

        g = StateGraph(_State)
        for name, fn in [("extract", extract), ("classify", classify), ("cache_lookup", cache_lookup),
                         ("verify", verify), ("judge", judge), ("cache_store", cache_store)]:
            g.add_node(name, fn)
        g.add_edge(START, "extract")
        g.add_conditional_edges("extract", lambda s: "classify" if s["claims"] else END, ["classify", END])
        g.add_conditional_edges("classify", lambda s: "cache_lookup" if s["selected"] else END, ["cache_lookup", END])
        g.add_conditional_edges("cache_lookup", lambda s: "verify" if s["to_verify"] else END, ["verify", END])
        g.add_edge("verify", "judge")
        g.add_edge("judge", "cache_store")
        g.add_edge("cache_store", END)
        return g.compile()

    async def run(self, request: VerifyRequest) -> Report:
        started = time.perf_counter()
        usage = UsageTracker()
        final: _State = await self._graph.ainvoke({"request": request, "today": self.deps.today(), "usage": usage})
        cached = final.get("cached", [])
        verdicts = sorted(cached + final.get("verdicts", []), key=lambda v: v.claim.id)
        return Report(
            verdict_summary=summarize(verdicts),  # type: ignore[arg-type]
            claims=verdicts,
            claims_extracted=len(final.get("claims", [])),
            claims_checked=len(final.get("selected", [])),
            latency_ms=round((time.perf_counter() - started) * 1000),
            tokens_by_stage=usage.snapshot(),
            models_by_stage=usage.models(),
            tavily_calls=len(final.get("searches", [])),
            verdict_cache_hits=len(cached),
        )
