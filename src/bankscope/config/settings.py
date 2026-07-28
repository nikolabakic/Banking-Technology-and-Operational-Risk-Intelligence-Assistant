from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    sec_user_agent: str
    sec_requests_per_second: float = Field(default=4, gt=0, le=10)
    sec_max_concurrency: int = Field(default=3, ge=1)
    sec_timeout_seconds: float = Field(default=30, gt=0)

    bank_registry_path: Path = Path("config/banks.yaml")
    raw_data_dir: Path = Path("data/raw/sec")
    processed_data_dir: Path = Path("data/processed")
    manifest_dir: Path = Path("artifacts/manifests")

    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @field_validator("sec_user_agent")
    @classmethod
    def validate_sec_user_agent(cls, value: str) -> str:
        value = value.strip()

        if "@" not in value:
            raise ValueError("SEC_USER_AGENT mora sadržati kontakt email adresu.")

        return value


@lru_cache
def get_settings() -> ApplicationSettings:
    return ApplicationSettings()
