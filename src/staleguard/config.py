"""Settings loaded from environment / .env. Secrets live only in .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bifrost_url: str = "http://localhost:8080"
    # Sent as X-Bifrost-Setup-Token on Bifrost /api/* calls (e.g. cache eviction).
    bifrost_setup_token: SecretStr | None = None

    small_model: str = "gemini/gemini-3.5-flash-lite"
    judge_model: str = "groq/openai/gpt-oss-120b"  # Groq free tier: 30 RPM / 1K RPD

    tavily_api_key: SecretStr | None = None
    # Cap on claims verified per answer (most volatile first) — bounds latency and Tavily credits.
    max_claims_per_answer: int = 5
    # SQLite verdict cache. Disable for the eval ablation ("verify every claim").
    verdict_cache_enabled: bool = True
    verdict_cache_path: str = "data/verdicts.sqlite3"

    @property
    def bifrost_openai_base_url(self) -> str:
        return f"{self.bifrost_url.rstrip('/')}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
