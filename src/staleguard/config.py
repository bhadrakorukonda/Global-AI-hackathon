"""Settings loaded from environment / .env. Secrets live only in .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bifrost_url: str = "http://localhost:8080"
    # Sent as X-Bifrost-Setup-Token on Bifrost /api/* calls (e.g. cache eviction).
    bifrost_setup_token: SecretStr | None = None

    small_model: str = "gemini/gemini-3.6-flash"
    judge_model: str = "gemini/gemini-3.1-pro-preview"

    tavily_api_key: SecretStr | None = None

    @property
    def bifrost_openai_base_url(self) -> str:
        return f"{self.bifrost_url.rstrip('/')}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
