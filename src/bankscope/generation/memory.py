"""Model-first bounded conversation compaction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bankscope.generation.answer_generator import GPT51_MODEL_MARKERS, GenerationValidationError

CONVERSATION_SUMMARY_PROMPT_VERSION = "conversation-summary-tool-v1"
CONVERSATION_SUMMARY_TIMEOUT_SECONDS = 30.0


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000)


def _summary_tool() -> dict[str, Any]:
    schema = ConversationSummary.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": "save_conversation_summary",
            "description": "Save the updated bounded summary of this BankScope conversation.",
            "strict": True,
            "parameters": schema,
        },
    }


CONVERSATION_SUMMARY_TOOL = _summary_tool()


def _request_options(model: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "tools": [CONVERSATION_SUMMARY_TOOL],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "timeout": CONVERSATION_SUMMARY_TIMEOUT_SECONDS,
    }
    if any(marker in model.strip().upper() for marker in GPT51_MODEL_MARKERS):
        options["max_completion_tokens"] = 1_500
    else:
        options.update({"max_tokens": 1_500, "temperature": 0})
    return options


def _tool_arguments(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        raise ValueError("conversation summary response has no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else choice.message
    finish_reason = (
        str(choice.get("finish_reason") or "")
        if isinstance(choice, Mapping)
        else str(getattr(choice, "finish_reason", "") or "")
    )
    refusal = (
        str(message.get("refusal") or "")
        if isinstance(message, Mapping)
        else str(getattr(message, "refusal", "") or "")
    )
    tool_calls = (
        list(message.get("tool_calls") or [])
        if isinstance(message, Mapping)
        else list(getattr(message, "tool_calls", None) or [])
    )
    if finish_reason in {"length", "content_filter"} or refusal or len(tool_calls) != 1:
        raise ValueError("conversation summary response is unusable")
    call = tool_calls[0]
    function = call.get("function") if isinstance(call, Mapping) else call.function
    name = (
        str(function.get("name") or "")
        if isinstance(function, Mapping)
        else str(getattr(function, "name", "") or "")
    )
    if name != "save_conversation_summary":
        raise ValueError("conversation summary used an unknown tool")
    return (
        str(function.get("arguments") or "")
        if isinstance(function, Mapping)
        else str(getattr(function, "arguments", "") or "")
    )


def summarize_conversation(
    existing_summary: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    client: Any,
    model: str,
) -> str:
    """Merge older raw turns into one model-authored conversation summary."""

    if not messages:
        raise ValueError("messages are required for conversation compaction.")
    system = (
        "Summarize older BankScope conversation turns for future conversational continuity. "
        "Preserve user instructions and style preferences, current banks/topics, unresolved "
        "questions, and referents needed to understand later messages. Distinguish user requests "
        "from assistant statements. Do not treat assistant claims as filing evidence, do not copy "
        "citation labels, and do not add facts. Merge with the existing summary and call the "
        "summary tool exactly once."
    )
    payload = json.dumps(
        {
            "prompt_version": CONVERSATION_SUMMARY_PROMPT_VERSION,
            "existing_summary": existing_summary,
            "older_messages": [
                {
                    "role": str(message.get("role") or ""),
                    "content": str(message.get("content") or ""),
                }
                for message in messages
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
            **_request_options(model),
        )
        return ConversationSummary.model_validate_json(_tool_arguments(response)).summary.strip()
    except (TypeError, ValueError, ValidationError) as error:
        raise GenerationValidationError(
            "conversation_summary_invalid",
            "The model returned an invalid conversation summary.",
            generation={"stage": "compacting_conversation", "model": model},
        ) from error
    except Exception as error:
        raise GenerationValidationError(
            "conversation_summary_request_failed",
            "Conversation compaction failed.",
            generation={"stage": "compacting_conversation", "model": model},
        ) from error
