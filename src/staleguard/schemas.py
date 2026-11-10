"""Pydantic models for every StaleGuard input/output: HTTP, Bifrost passthrough, and the verification pipeline."""

from datetime import timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CLASS_TTL",
    "Action",
    "CacheDebug",
    "Claim",
    "ClaimClass",
    "ClaimVerdict",
    "ClassifyLLMItem",
    "ClassifyLLMOutput",
    "DependencyHealth",
    "Evidence",
    "ExtractLLMOutput",
    "ExtractedClaim",
    "ForwardResult",
    "HealthResponse",
    "JudgeLLMItem",
    "JudgeLLMOutput",
    "Mode",
    "Report",
    "RewriteLLMOutput",
    "SearchResult",
    "StaleGuardStatus",
    "Verdict",
    "VerifyRequest",
]

# ---------------------------------------------------------------- health / proxy


class DependencyHealth(BaseModel):
    ok: bool
    url: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    bifrost: DependencyHealth


class CacheDebug(BaseModel):
    """Bifrost's extra_fields.cache_debug (Go: BifrostCacheMetadata). Shapes verified live, see docs/bifrost-notes.md."""

    model_config = ConfigDict(extra="allow")

    cache_hit: bool
    # Present on hits AND misses (on a miss it is the id of the entry just written).
    cache_id: str | None = None
    hit_type: Literal["direct", "semantic"] | None = None
    similarity: float | None = None
    threshold: float | None = None
    requested_provider: str | None = None
    requested_model: str | None = None
    provider_used: str | None = None
    model_used: str | None = None
    input_tokens: int | None = None
    cache_hit_latency: int | None = None


class ForwardResult(BaseModel):
    """Result of forwarding a chat completion to Bifrost."""

    status_code: int
    body: Any
    headers: dict[str, str]
    cache_debug: CacheDebug | None = None


# Values of the x-staleguard-status response header / Report.verdict_summary.
StaleGuardStatus = Literal["clean", "stale", "unverified", "skipped"]
# Proxy action policy, chosen per request with the x-sg-mode header (SPEC §3 Act).
Mode = Literal["annotate", "rewrite", "block"]
# What the proxy did to the answer.
Action = Literal["none", "rewritten", "blocked"]

# ---------------------------------------------------------------- pipeline domain


class ClaimClass(StrEnum):
    """Volatility classes from SPEC §2. Everything except STATIC gets verified."""

    VERSION = "version"
    ROLE = "role"
    PRICE = "price"
    STATISTIC = "statistic"
    POLICY = "policy"
    EVENT_STATUS = "event_status"
    RELATIVE_TIME = "relative_time"
    STATIC = "static"


# Re-verify TTL per class (SPEC §2). None = never verified. Drives claim ranking and the verdict cache.
CLASS_TTL: dict[ClaimClass, timedelta | None] = {
    ClaimClass.PRICE: timedelta(days=1),
    ClaimClass.EVENT_STATUS: timedelta(days=1),
    ClaimClass.RELATIVE_TIME: timedelta(days=1),
    ClaimClass.VERSION: timedelta(days=7),
    ClaimClass.POLICY: timedelta(days=14),
    ClaimClass.ROLE: timedelta(days=30),
    ClaimClass.STATISTIC: timedelta(days=30),
    ClaimClass.STATIC: None,
}


class Verdict(StrEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNVERIFIABLE = "UNVERIFIABLE"
    CONFLICTING = "CONFLICTING"


class Claim(BaseModel):
    """An atomic, self-contained claim extracted from an answer."""

    id: int
    text: str
    claim_class: ClaimClass | None = None  # None until classified
    classified_by: Literal["rules", "llm"] | None = None
    signals: list[str] = Field(default_factory=list, description="Names of the rules that fired.")
    search_query: str | None = Field(default=None, description="Neutral web query for the claim's topic (from extract).")


class Evidence(BaseModel):
    url: str
    title: str
    snippet: str
    published_date: str | None = Field(default=None, description="ISO date (YYYY-MM-DD) when the source provides one.")
    score: float | None = None


class SearchResult(BaseModel):
    """Tavily evidence gathered for one claim."""

    claim_id: int
    query: str
    evidence: list[Evidence] = Field(default_factory=list)
    answer: str | None = Field(default=None, description="Tavily's own summary answer, if requested.")
    error: str | None = None


class ClaimVerdict(BaseModel):
    claim: Claim
    verdict: Verdict
    corrected_claim: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list, description="Only the evidence the judge cited.")
    search_query: str | None = None
    from_cache: bool = Field(default=False, description="Served from the verdict cache (no Tavily call, no judge call).")


class Report(BaseModel):
    """The `staleguard` object: /verify's response body, and attached to every verified proxy response."""

    verdict_summary: Literal["clean", "stale", "unverified"]
    claims: list[ClaimVerdict]
    claims_extracted: int
    claims_checked: int
    latency_ms: int
    tokens_by_stage: dict[str, int]
    models_by_stage: dict[str, str] = Field(default_factory=dict, description="Model that actually answered each stage (shows Bifrost failover).")
    tavily_calls: int
    verdict_cache_hits: int = 0
    # Proxy-only fields (defaults on /verify).
    mode: Mode = "annotate"
    action: Action = "none"
    original_answer: str | None = Field(default=None, description="The upstream answer, when rewrite/block replaced it.")
    bifrost_cache_hit: bool | None = Field(default=None, description="None when the request sent no x-bf-cache-key.")
    cache_evicted: bool = False
    evicted_cache_id: str | None = None


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, description="Answer text to verify.")
    question: str | None = Field(default=None, description="Optional question the text answers; helps make claims self-contained.")
    include_domains: list[str] | None = Field(default=None, description="Trusted domains to prefer in Tavily search.")


# ---------------------------------------------------------------- LLM structured outputs


class ExtractedClaim(BaseModel):
    claim: str
    query: str | None = None


class ExtractLLMOutput(BaseModel):
    claims: list[ExtractedClaim]

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_list(cls, v: Any) -> Any:
        """Flash-Lite sometimes returns the claims array without the {"claims": ...} wrapper (seen live)."""
        return {"claims": v} if isinstance(v, list) else v

    @field_validator("claims", mode="before")
    @classmethod
    def _accept_bare_strings(cls, v: Any) -> Any:
        """Tolerate a model that returns plain claim strings (no query): the claim text becomes the query."""
        return [{"claim": c} if isinstance(c, str) else c for c in v] if isinstance(v, list) else v


class ClassifyLLMItem(BaseModel):
    id: int
    claim_class: ClaimClass


class ClassifyLLMOutput(BaseModel):
    items: list[ClassifyLLMItem]


class JudgeLLMItem(BaseModel):
    id: int
    verdict: Verdict
    corrected_claim: str | None = None
    evidence_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class JudgeLLMOutput(BaseModel):
    verdicts: list[JudgeLLMItem]


class RewriteLLMOutput(BaseModel):
    answer: str = Field(min_length=1)
