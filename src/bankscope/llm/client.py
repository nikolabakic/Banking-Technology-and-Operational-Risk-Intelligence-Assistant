from __future__ import annotations

from typing import Any

from bankscope.config.settings import ApplicationSettings
from bankscope.generation.answer_generator import GPT51_MODEL_MARKERS
from bankscope.llm.model_access import access_model, model_access_parameters


def create_openai_client(settings: ApplicationSettings) -> Any:
    """Obtain the authenticated corporate model client through model_access."""
    del settings  # Model selection remains separate from client authentication.
    return access_model()


def create_langchain_chat_model(settings: ApplicationSettings, *, model: str | None = None) -> Any:
    """Create the LangChain chat-model adapter with the same corporate access contract."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install the optional LLM dependencies with 'pip install -e .[llm]'."
        ) from error

    model_name = (model or settings.openai_model).strip()
    options: dict[str, Any] = {
        **model_access_parameters(),
        "model": model_name,
        "timeout": settings.llm_request_timeout_seconds,
        "max_retries": 1,
    }
    if any(marker in model_name.upper() for marker in GPT51_MODEL_MARKERS):
        options["max_completion_tokens"] = settings.llm_conversation_max_output_tokens
    else:
        options.update(
            {
                "max_tokens": settings.llm_conversation_max_output_tokens,
                "temperature": settings.llm_temperature,
            }
        )
    return ChatOpenAI(**options)
