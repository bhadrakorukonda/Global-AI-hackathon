"""LangGraph wiring: extract -> classify -> (fast path: END) | verify -> judge -> END, plus Report assembly."""

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

__all__ = ["Pipeline", "PipelineDeps", "summarize"]


@dataclass(frozen=True)
class PipelineDeps:
    llm: ChatBackend
    tavily: SearchBackend
    small_model: str
    judge_model: str
    max_claims: int = 5
    today: Callable[[], date] = field(default=date.today)


class _State(TypedDict, total=False):
    request: VerifyRequest
    today: date
    usage: UsageTracker
    claims: list[Claim]
    selected: list[Claim]
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

        async def verify(s: _State) -> _State:
            return {"searches": await verify_claims(d.tavily, s["selected"], s["request"].include_domains)}

        async def judge(s: _State) -> _State:
            verdicts = await judge_claims(
                d.llm, model=d.judge_model, claims=s["selected"], searches=s["searches"], today=s["today"], usage=s["usage"]
            )
            return {"verdicts": verdicts}

        g = StateGraph(_State)
        g.add_node("extract", extract)
        g.add_node("classify", classify)
        g.add_node("verify", verify)
        g.add_node("judge", judge)
        g.add_edge(START, "extract")
        g.add_conditional_edges("extract", lambda s: "classify" if s["claims"] else END, ["classify", END])
        # Fast path: no volatile claims -> no Tavily, no judge.
        g.add_conditional_edges("classify", lambda s: "verify" if s["selected"] else END, ["verify", END])
        g.add_edge("verify", "judge")
        g.add_edge("judge", END)
        return g.compile()

    async def run(self, request: VerifyRequest) -> Report:
        started = time.perf_counter()
        usage = UsageTracker()
        final: _State = await self._graph.ainvoke({"request": request, "today": self.deps.today(), "usage": usage})
        verdicts = final.get("verdicts", [])
        return Report(
            verdict_summary=summarize(verdicts),  # type: ignore[arg-type]
            claims=verdicts,
            claims_extracted=len(final.get("claims", [])),
            claims_checked=len(final.get("selected", [])),
            latency_ms=round((time.perf_counter() - started) * 1000),
            tokens_by_stage=usage.snapshot(),
            tavily_calls=len(final.get("searches", [])),
        )
