import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest

from bankscope.observability.logging import JsonFormatter, configure_logging


@pytest.fixture
def preserve_root_logger() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    yield

    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)


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
