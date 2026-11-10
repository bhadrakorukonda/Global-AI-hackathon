"""Wire the verification pipeline from Settings (infrastructure). Shared by the API and the eval runner,
so both run exactly the same models, fallbacks and Tavily client."""

import dataclasses
import logging
from typing import Any

from tavily import AsyncTavilyClient

from staleguard.bifrost_client import make_llm_client
from staleguard.config import Settings
from staleguard.llm import OpenAIChatBackend
from staleguard.pipeline import Pipeline, PipelineDeps
from staleguard.verdict_cache import VerdictCache

__all__ = ["build_pipeline"]

log = logging.getLogger(__name__)


def build_pipeline(settings: Settings, **overrides: Any) -> Pipeline | None:
    """Real backends via Bifrost + Tavily. None when Tavily isn't configured.

    `overrides` replace PipelineDeps fields (the eval uses verdict_cache, volatility_filter, max_claims).
    """
    if settings.tavily_api_key is None or not settings.tavily_api_key.get_secret_value():
        log.warning("TAVILY_API_KEY not set: verification disabled")
        return None
    if "verdict_cache" not in overrides:  # don't open the default SQLite file when the caller brings its own
        overrides["verdict_cache"] = VerdictCache(settings.verdict_cache_path) if settings.verdict_cache_enabled else None
    deps = PipelineDeps(
        llm=OpenAIChatBackend(
            make_llm_client(settings),
            fallbacks={settings.small_model: settings.small_model_fallbacks, settings.judge_model: settings.judge_model_fallbacks},
        ),
        tavily=AsyncTavilyClient(settings.tavily_api_key.get_secret_value()),
        small_model=settings.small_model,
        judge_model=settings.judge_model,
        max_claims=settings.max_claims_per_answer,
    )
    return Pipeline(dataclasses.replace(deps, **overrides))
