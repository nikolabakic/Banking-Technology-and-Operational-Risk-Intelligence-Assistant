import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bankscope.config.settings import ApplicationSettings, get_settings
from bankscope.observability.logging import JsonFormatter, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TEST_PATH = PROJECT_ROOT / "scripts" / "run_smoke_test.py"


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


@pytest.fixture
def preserve_root_logger() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    yield

    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)


def test_settings_load_valid_values(valid_settings_data: dict[str, object]) -> None:
    settings = ApplicationSettings(
        **valid_settings_data,
        sec_requests_per_second=5,
    )

    assert settings.sec_user_agent == "BankScopeRAG test@example.com"
    assert settings.sec_requests_per_second == 5
    assert settings.sec_max_concurrency == 3
    assert settings.sec_timeout_seconds == 30
    assert settings.log_level == "INFO"
    assert settings.openai_api_key is None


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
    assert "**********" in repr(settings)


def test_get_settings_returns_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "BankScopeRAG cache@example.com")
    get_settings.cache_clear()

    try:
        first_settings = get_settings()
        second_settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert first_settings is second_settings


def test_json_formatter_returns_required_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="bankscope.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=100,
        msg="Test message %s",
        args=("value",),
        exc_info=None,
    )

    log_data = json.loads(formatter.format(record))

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "bankscope.test"
    assert log_data["message"] == "Test message value"
    assert log_data["module"] == Path(__file__).stem
    assert log_data["line"] == 100
    assert "function" in log_data
    assert datetime.fromisoformat(log_data["timestamp"]).utcoffset() is not None


def test_json_formatter_includes_exception() -> None:
    formatter = JsonFormatter()

    try:
        raise ValueError("Expected test error")
    except ValueError:
        exception_info = sys.exc_info()

    record = logging.LogRecord(
        name="bankscope.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=120,
        msg="Operation failed",
        args=(),
        exc_info=exception_info,
    )

    log_data = json.loads(formatter.format(record))

    assert "ValueError: Expected test error" in log_data["exception"]


def test_configure_logging_outputs_one_json_log(
    capsys: pytest.CaptureFixture[str],
    preserve_root_logger: None,
) -> None:
    configure_logging("INFO")
    configure_logging("INFO")

    root_logger = logging.getLogger()
    logging.getLogger("bankscope.test").info("Application started")
    output_lines = capsys.readouterr().out.strip().splitlines()

    assert len(root_logger.handlers) == 1
    assert len(output_lines) == 1
    assert json.loads(output_lines[0])["message"] == "Application started"


def test_smoke_test_runs_without_network(tmp_path: Path) -> None:
    assert SMOKE_TEST_PATH.is_file(), f"Nedostaje smoke test: {SMOKE_TEST_PATH}"

    raw_data_dir = tmp_path / "raw"
    processed_data_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifests"
    source_dir = PROJECT_ROOT / "src"

    environment = os.environ.copy()
    environment.update(
        {
            "APP_ENV": "test",
            "LOG_LEVEL": "INFO",
            "SEC_USER_AGENT": "BankScopeRAG smoke@example.com",
            "SEC_REQUESTS_PER_SECOND": "4",
            "SEC_MAX_CONCURRENCY": "3",
            "SEC_TIMEOUT_SECONDS": "30",
            "BANK_REGISTRY_PATH": str(tmp_path / "banks.yaml"),
            "RAW_DATA_DIR": str(raw_data_dir),
            "PROCESSED_DATA_DIR": str(processed_data_dir),
            "MANIFEST_DIR": str(manifest_dir),
            "OPENAI_API_KEY": "",
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (str(source_dir), environment.get("PYTHONPATH")),
                )
            ),
        }
    )

    result = subprocess.run(
        [sys.executable, str(SMOKE_TEST_PATH)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    log_entries = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    messages = [entry["message"] for entry in log_entries]

    assert messages == [
        "Application settings loaded successfully",
        "Verified 4 project directories",
        "BankScope smoke test passed",
    ]
    assert raw_data_dir.is_dir()
    assert processed_data_dir.is_dir()
    assert manifest_dir.is_dir()
    assert (tmp_path / "artifacts" / "logs").is_dir()
