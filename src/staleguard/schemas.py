"""Pydantic models for every StaleGuard input/output: HTTP, Bifrost passthrough, and the verification pipeline."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CacheDebug",
    "Claim",
    "ClaimClass",
    "ClaimVerdict",
    "ClassifyLLMItem",
    "ClassifyLLMOutput",
    "DependencyHealth",
    "Evidence",
    "ExtractLLMOutput",
    "ForwardResult",
    "HealthResponse",
    "JudgeLLMItem",
    "JudgeLLMOutput",
    "Report",
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


class Report(BaseModel):
    """The `staleguard` object returned by /verify (and, from step 5, attached to proxied responses)."""

    verdict_summary: Literal["clean", "stale", "unverified"]
    claims: list[ClaimVerdict]
    claims_extracted: int
    claims_checked: int
    latency_ms: int
    tokens_by_stage: dict[str, int]
    tavily_calls: int
    cache_evicted: bool = False


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, description="Answer text to verify.")
    question: str | None = Field(default=None, description="Optional question the text answers; helps make claims self-contained.")
    include_domains: list[str] | None = Field(default=None, description="Trusted domains to prefer in Tavily search.")


# ---------------------------------------------------------------- LLM structured outputs


class ExtractLLMOutput(BaseModel):
    claims: list[str]


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
