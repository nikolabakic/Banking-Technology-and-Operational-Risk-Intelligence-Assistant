from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _environment_value(primary: str, fallback: str) -> str | None:
    return os.getenv(primary) or os.getenv(fallback)


def access_model() -> Any:
    """Create the authenticated Unique OpenAI-compatible client."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install the optional LLM dependencies with 'pip install -e .[llm]'."
        ) from error

    api_key = _environment_value("api_key", "OPENAI_API_KEY")
    base_url = _environment_value("base_url", "OPENAI_API_BASE_URL")
    headers = {
        "x-app-id": _environment_value("x-app-id", "OPENAI_X_APP_ID"),
        "x-user-id": _environment_value("x-user-id", "OPENAI_X_USER_ID"),
        "x-company-id": _environment_value("x-company-id", "OPENAI_X_COMPANY_ID"),
        "x-api-version": _environment_value("x-api-version", "OPENAI_X_API_VERSION"),
    }
    missing = [
        name
        for name, value in {"api_key": api_key, "base_url": base_url, **headers}.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing model access environment values: {', '.join(missing)}.")
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
