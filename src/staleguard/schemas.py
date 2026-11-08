"""Pydantic models for StaleGuard inputs/outputs. Pipeline models (Claim, Evidence, Verdict, Report) land in step 3."""

from typing import Literal

from pydantic import BaseModel


class DependencyHealth(BaseModel):
    ok: bool
    url: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    bifrost: DependencyHealth
