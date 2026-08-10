from __future__ import annotations

from typing import Any

from bankscope.config.settings import ApplicationSettings
from bankscope.llm.model_access import access_model


def create_openai_client(settings: ApplicationSettings) -> Any:
    """Obtain the authenticated corporate model client through model_access."""
    del settings  # Model selection remains separate from client authentication.
    return access_model()
