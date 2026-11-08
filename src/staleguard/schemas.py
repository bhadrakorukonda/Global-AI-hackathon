"""Pydantic models for StaleGuard inputs/outputs. Pipeline models (Claim, Evidence, Verdict, Report) land in step 3."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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


# Values of the x-staleguard-status response header.
StaleGuardStatus = Literal["clean", "stale", "unverified", "skipped"]
