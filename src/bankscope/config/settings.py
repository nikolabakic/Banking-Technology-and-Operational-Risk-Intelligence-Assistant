from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    sec_user_agent: str
    sec_requests_per_second: float = Field(default=4, gt=0, le=10)
    sec_timeout_seconds: float = Field(default=30, gt=0)

    bank_registry_path: Path = PROJECT_ROOT / "config/banks.yaml"
    raw_data_dir: Path = PROJECT_ROOT / "data/raw/sec"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"

    @field_validator("sec_user_agent")
    @classmethod
    def validate_sec_user_agent(cls, value: str) -> str:
        value = value.strip()

        if "@" not in value:
            raise ValueError("SEC_USER_AGENT must contain a contact email address.")

        return value


@lru_cache
def get_settings() -> ApplicationSettings:
    return ApplicationSettings()
