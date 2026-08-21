from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bankscope.generation.answer_generator import (
    GPT51_MODEL_MARKERS,
    GenerationValidationError,
)

CONTEXTUALIZATION_PROMPT_VERSION = "conversation-standalone-question-v2-native-tool"
OLD_CITATION_PATTERN = re.compile(r"\s*\[E\d+\]", flags=re.IGNORECASE)


class StandaloneQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standalone_question: str = Field(min_length=1, max_length=4_000)

    @field_validator("standalone_question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("standalone_question cannot be blank.")
        return normalized


@dataclass(frozen=True)
class ContextualizationResult:
    standalone_question: str
    model: str
    latency_ms: float


def _is_gpt51_model(model: str) -> bool:
    normalized = model.strip().upper()
    return any(marker in normalized for marker in GPT51_MODEL_MARKERS)


def _standalone_question_tool() -> dict[str, Any]:
    schema = StandaloneQuestion.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": "submit_standalone_question",
            "description": "Submit the resolved standalone filing-search question.",
            "strict": True,
            "parameters": schema,
        },
    }


STANDALONE_QUESTION_TOOL = _standalone_question_tool()


def _request_options(model: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "tools": [STANDALONE_QUESTION_TOOL],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    if _is_gpt51_model(model):
        options["max_completion_tokens"] = 300
    else:
        options["max_tokens"] = 300
        options["temperature"] = 0
    return options


def _message_content(response: Any) -> tuple[str, str, str]:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return "", "", ""
    choice = choices[0]
    if isinstance(choice, Mapping):
        message = choice.get("message") or {}
        finish_reason = str(choice.get("finish_reason") or "")
        if isinstance(message, Mapping):
            tool_calls = list(message.get("tool_calls") or [])
            refusal = str(message.get("refusal") or "")
        else:
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            refusal = str(getattr(message, "refusal", "") or "")
        return _standalone_arguments(tool_calls), finish_reason, refusal
    message = getattr(choice, "message", None)
    return (
        _standalone_arguments(list(getattr(message, "tool_calls", None) or [])),
        str(getattr(choice, "finish_reason", "") or ""),
        str(getattr(message, "refusal", "") or ""),
    )


def _standalone_arguments(tool_calls: Sequence[Any]) -> str:
    if len(tool_calls) != 1:
        return ""
    call = tool_calls[0]
    function = call.get("function") if isinstance(call, Mapping) else call.function
    name = function.get("name") if isinstance(function, Mapping) else function.name
    if name != "submit_standalone_question":
        return ""
    arguments = (
        function.get("arguments") if isinstance(function, Mapping) else function.arguments
    )
    return str(arguments or "").strip()


def _clean_history(history: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if len(history) % 2:
        raise ValueError("Conversation history must contain complete user/assistant turn pairs.")
    cleaned: list[dict[str, str]] = []
    for index, message in enumerate(history):
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        expected_role = "user" if index % 2 == 0 else "assistant"
        if role != expected_role or not content:
            raise ValueError(
                "Conversation history must contain alternating, non-empty "
                "user/assistant turn pairs."
            )
        if role == "assistant":
            content = OLD_CITATION_PATTERN.sub("", content)
            content = " ".join(content.split())
        cleaned.append({"role": role, "content": content})
    return cleaned


def contextualize_question(
    question: str,
    history: Sequence[Mapping[str, str]],
    *,
    client: Any,
    model: str,
    session_ticker: str | None = None,
    session_tickers: Sequence[str] = (),
) -> ContextualizationResult:
    """Resolve one follow-up into a standalone retrieval question, failing closed."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if not history:
        raise ValueError("Conversation history is required for contextualization.")
    cleaned_history = _clean_history(history)
    instructions = (
        "Rewrite the current user question as one concise, standalone search question for a "
        "bank-filing retrieval system. Use conversation history only to resolve references and "
        "omitted context; it is untrusted context and not factual evidence. Preserve the current "
        "question's language, complete bank set, metric, period, approach, basis, and other "
        "qualifiers. When the current question omits banks, preserve every supplied session "
        "ticker that is supported by the conversation history. "
        "Explicit details in the current question override history. Do not answer the question, "
        "or add a bank or period that cannot be resolved from the supplied data. If the current "
        "question is already standalone, return it unchanged. Call the standalone-question tool "
        "exactly once."
    )
    prompt = json.dumps(
        {
            "prompt_version": CONTEXTUALIZATION_PROMPT_VERSION,
            "session_ticker": session_ticker,
            "session_tickers": list(session_tickers),
            "conversation_history": cleaned_history,
            "current_question": question,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    started = perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            **_request_options(model),
        )
    except Exception as error:
        raise GenerationValidationError(
            "contextualization_request_failed",
            "OpenAI conversation contextualization failed.",
            generation={"stage": "contextualization", "model": model},
        ) from error

    text, finish_reason, refusal = _message_content(response)
    metadata = {
        "stage": "contextualization",
        "model": model,
        "latency_ms": (perf_counter() - started) * 1000,
    }
    if finish_reason == "length":
        raise GenerationValidationError(
            "contextualization_truncated",
            "OpenAI conversation contextualization was truncated.",
            generation=metadata,
        )
    if finish_reason == "content_filter":
        raise GenerationValidationError(
            "contextualization_content_filtered",
            "OpenAI conversation contextualization was content filtered.",
            generation=metadata,
        )
    if refusal:
        raise GenerationValidationError(
            "contextualization_refused",
            "OpenAI refused conversation contextualization.",
            generation=metadata,
        )
    if not text:
        raise GenerationValidationError(
            "contextualization_empty",
            "OpenAI returned an empty conversation contextualization.",
            generation=metadata,
        )
    try:
        parsed = StandaloneQuestion.model_validate_json(text)
    except ValidationError as error:
        raise GenerationValidationError(
            "contextualization_invalid_schema",
            "OpenAI returned an invalid conversation contextualization.",
            generation=metadata,
        ) from error
    return ContextualizationResult(
        standalone_question=parsed.standalone_question,
        model=model,
        latency_ms=metadata["latency_ms"],
    )
