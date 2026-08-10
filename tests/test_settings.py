from pathlib import Path

import pytest
from pydantic import ValidationError

from bankscope.config.settings import ApplicationSettings, get_settings


@pytest.fixture
def valid_settings_data(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    for field_name in ApplicationSettings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    return {
        "_env_file": None,
        "sec_user_agent": "BankScopeRAG test@example.com",
    }


def test_settings_load_valid_values(
    valid_settings_data: dict[str, object],
) -> None:
    settings = ApplicationSettings(
        **valid_settings_data,
        sec_requests_per_second=5,
    )

    assert settings.sec_user_agent == "BankScopeRAG test@example.com"
    assert settings.sec_requests_per_second == 5
    assert settings.sec_timeout_seconds == 30
    assert settings.openai_model == "AZURE_GPT_4o_2024_1120"


def test_settings_reject_user_agent_without_email(
    valid_settings_data: dict[str, object],
) -> None:
    invalid_data = valid_settings_data | {"sec_user_agent": "BankScopeRAG"}

    with pytest.raises(ValidationError, match="SEC_USER_AGENT"):
        ApplicationSettings(**invalid_data)


@pytest.mark.parametrize("requests_per_second", [0, 10.01])
def test_settings_reject_invalid_sec_rate(
    valid_settings_data: dict[str, object],
    requests_per_second: float,
) -> None:
    with pytest.raises(ValidationError):
        ApplicationSettings(
            **valid_settings_data,
            sec_requests_per_second=requests_per_second,
        )


def test_settings_hide_openai_api_key(
    valid_settings_data: dict[str, object],
) -> None:
    api_key = "sk-test-secret-value"
    settings = ApplicationSettings(
        **valid_settings_data,
        openai_api_key=api_key,
    )

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == api_key
    assert api_key not in repr(settings)


def test_get_settings_returns_cached_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEC_USER_AGENT", "BankScopeRAG cache@example.com")
    get_settings.cache_clear()

    try:
        first_settings = get_settings()
        second_settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert first_settings is second_settings
