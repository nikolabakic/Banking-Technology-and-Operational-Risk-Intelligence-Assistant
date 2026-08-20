from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bankscope.generation.answer_generator import GPT51_MODEL_MARKERS
from bankscope.generation.query_planner import (
    YEAR_PATTERN,
    is_general_chat_question,
    needs_contextualization,
)

CONVERSATION_PROMPT_VERSION = "conversation-tool-router-v1"
CONVERSATION_REQUEST_TIMEOUT_SECONDS = 30.0


class ResearchFilingsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_question: str = Field(min_length=2, max_length=4_000)
    reason: str = Field(min_length=1, max_length=500)


class DirectResponseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=4_000)
    category: Literal[
        "greeting",
        "acknowledgement",
        "capability",
        "general_explanation",
    ]


class ClarificationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1_000)
    missing: Literal["bank", "period", "metric", "intent"]


class DeclineOutOfScopeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Literal["outside_banking_research_scope"]


ConversationAction = (
    ResearchFilingsArgs | DirectResponseArgs | ClarificationArgs | DeclineOutOfScopeArgs
)


@dataclass(frozen=True)
class ConversationDecision:
    action: ConversationAction
    latency_ms: float
    request_count: int = 1
    fallback: bool = False
    error_code: str | None = None


def _request_options(model: str) -> dict[str, Any]:
    normalized = model.strip().upper()
    options: dict[str, Any] = {
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "timeout": CONVERSATION_REQUEST_TIMEOUT_SECONDS,
    }
    if any(marker in normalized for marker in GPT51_MODEL_MARKERS):
        options["max_completion_tokens"] = 700
    else:
        options.update({"max_tokens": 700, "temperature": 0})
    return options


def _tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": schema,
        },
    }


CONVERSATION_TOOLS = (
    _tool(
        "research_filings",
        (
            "Use indexed SEC filing evidence for any claim about a specific supported bank, "
            "comparison of supported banks, reported metric, disclosure, risk, period, or filing. "
            "search_question must be a standalone retrieval question reconstructed from the "
            "current message and conversation history without answering it."
        ),
        ResearchFilingsArgs,
    ),
    _tool(
        "respond_directly",
        (
            "Respond without filing retrieval only for greetings, thanks, BankScope capability "
            "help, or a general banking concept that makes no claim about a specific bank."
        ),
        DirectResponseArgs,
    ),
    _tool(
        "ask_clarification",
        (
            "Ask one concise question only when a missing bank, period, metric, or intent would "
            "materially change filing research. Do not require a bank for a general conceptual "
            "explanation."
        ),
        ClarificationArgs,
    ),
    _tool(
        "decline_out_of_scope",
        (
            "Use for requests outside BankScope's banking and filing-research domain, including "
            "recipes, entertainment, sports, travel, weather, and unrelated general knowledge."
        ),
        DeclineOutOfScopeArgs,
    ),
)


def _message_and_tool_calls(response: Any) -> tuple[str, list[Any], str, str]:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return "", [], "", ""
    choice = choices[0]
    if isinstance(choice, Mapping):
        message = choice.get("message") or {}
        finish_reason = str(choice.get("finish_reason") or "")
    else:
        message = getattr(choice, "message", None)
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
    if isinstance(message, Mapping):
        return (
            str(message.get("content") or "").strip(),
            list(message.get("tool_calls") or []),
            finish_reason,
            str(message.get("refusal") or ""),
        )
    return (
        str(getattr(message, "content", "") or "").strip(),
        list(getattr(message, "tool_calls", None) or []),
        finish_reason,
        str(getattr(message, "refusal", "") or ""),
    )


def _tool_name_and_arguments(tool_call: Any) -> tuple[str, str]:
    function = (
        tool_call.get("function")
        if isinstance(tool_call, Mapping)
        else getattr(tool_call, "function", None)
    )
    if isinstance(function, Mapping):
        return str(function.get("name") or ""), str(function.get("arguments") or "")
    return (
        str(getattr(function, "name", "") or ""),
        str(getattr(function, "arguments", "") or ""),
    )


def _looks_serbian(text: str) -> bool:
    normalized = text.casefold()
    if any(character in normalized for character in "čćžšđ"):
        return True
    tokens = set(re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE))
    return bool(
        tokens
        & {
            "hvala",
            "zdravo",
            "ćao",
            "cao",
            "banka",
            "banke",
            "objasni",
            "uporedi",
            "pokazatelj",
            "pokazatelje",
            "rizik",
            "rizike",
            "daj",
            "recept",
            "pitu",
            "jabukama",
            "reci",
            "vise",
            "više",
            "šta",
            "sta",
            "koji",
            "koja",
            "koliko",
            "navedi",
            "godine",
        }
    )


def render_capability_answer(question: str, bank_names: Mapping[str, str]) -> str:
    """Render product scope from the server-owned registry, never model-authored names."""

    names = ", ".join(bank_names.values())
    if _looks_serbian(question):
        return (
            f"Mogu da pretražujem i poredim indeksirane 10-K izveštaje za: {names}. "
            "Možeš da pitaš prirodno; ako nedostaje detalj koji menja pretragu, "
            "postaviću jedno kratko podpitanje."
        )
    return (
        f"I can search and compare indexed 10-K filings for: {names}. "
        "Ask naturally; if a missing detail would change the research, I will ask one "
        "short follow-up question."
    )


def render_out_of_scope_answer(question: str) -> str:
    """Return a server-owned scope boundary; never echo model-authored unrelated content."""

    if _looks_serbian(question):
        return (
            "Mogu da pomognem samo sa podržanim bankama, njihovim 10-K izveštajima, "
            "finansijskim pokazateljima i bankarskim rizicima."
        )
    return (
        "I can only help with supported banks, their 10-K filings, financial metrics, "
        "and banking risks."
    )


BANKING_DOMAIN_PATTERN = re.compile(
    r"(?i)\b(?:bank|banks|banking|banka|banke|bankarsk\w*|10\s*[- ]?k|sec|filing|"
    r"cet1|capital|kapital|ratio|koeficijent|pokazatelj|metric|revenue|prihod|income|"
    r"profit|assets?|aktiva|loans?|kredit\w*|deposits?|depozit\w*|liquidity|likvidnost|"
    r"risk|rizik\w*|cyber|sajber|operational|operativn\w*|regulatory|regulatorn\w*|"
    r"compliance|usklađenost|usklađenosti|tier\s*1)\b"
)
OUT_OF_SCOPE_PATTERN = re.compile(
    r"(?i)\b(?:recipe|recept|pita|kolač|kolac|kuvanje|cooking|weather|vreme|forecast|"
    r"sport|football|fudbal|basketball|košarka|kosarka|movie|film|music|muzika|travel|"
    r"putovanje|horoscope|horoskop|joke|vic)\b"
)
CET1_VAGUE_INDICATOR_PATTERN = re.compile(
    r"(?i)\bcet1\b.*\b(?:indicator|measure|pokazatelj\w*)\b|"
    r"\b(?:indicator|measure|pokazatelj\w*)\b.*\bcet1\b"
)
CET1_RATIO_PATTERN = re.compile(r"(?i)\b(?:ratio|stopa|koeficijent)\b|%")
CET1_AMOUNT_PATTERN = re.compile(
    r"(?i)\b(?:amount|iznos|capital\s+amount|iznos\s+kapitala|capital\s+in\s+|kapital\s+u\s+)\b"
)


def is_banking_domain_question(question: str, bank_names: Mapping[str, str]) -> bool:
    normalized = question.casefold()
    if BANKING_DOMAIN_PATTERN.search(normalized):
        return True
    for ticker, name in bank_names.items():
        if name.strip() and name.casefold() in normalized:
            return True
        if re.search(rf"(?i)(?<!\w){re.escape(ticker.strip())}(?!\w)", question):
            return True
    return False


def is_clearly_out_of_scope(question: str, bank_names: Mapping[str, str]) -> bool:
    """Fail closed for standalone non-banking requests before any retrieval can run."""

    normalized = " ".join(question.casefold().split()).strip(" .!?…")
    if OUT_OF_SCOPE_PATTERN.search(question):
        return True
    if normalized in {
        "thanks",
        "thank you",
        "hvala",
        "hvala ti",
        "super",
        "great",
        "ok",
        "okay",
    }:
        return False
    if is_general_chat_question(question) or needs_contextualization(question):
        return False
    return not is_banking_domain_question(question, bank_names)


def cet1_metric_clarification(question: str) -> ClarificationArgs | None:
    """Clarify the common CET1 amount-versus-ratio ambiguity before retrieval."""

    if not CET1_VAGUE_INDICATOR_PATTERN.search(question):
        return None
    if CET1_RATIO_PATTERN.search(question) or CET1_AMOUNT_PATTERN.search(question):
        return None
    if _looks_serbian(question):
        prompt = "Da li želite CET1 kapital (iznos) ili CET1 kapitalni koeficijent (procenat)?"
    else:
        prompt = "Do you mean CET1 capital (amount) or the CET1 capital ratio (percentage)?"
    return ClarificationArgs(question=prompt, missing="metric")


def _fallback_action(
    question: str,
    bank_names: Mapping[str, str],
    history: Sequence[Mapping[str, str]] = (),
) -> ConversationAction:
    normalized = " ".join(question.casefold().split()).strip(" .!?…")
    serbian = _looks_serbian(question)
    acknowledgements = {
        "thanks",
        "thank you",
        "hvala",
        "hvala ti",
        "super",
        "great",
        "ok",
        "okay",
    }
    capability_markers = (
        "which banks",
        "supported banks",
        "what can you do",
        "how do you work",
        "koje banke",
        "podržane banke",
        "sta mozes",
        "šta možeš",
        "kako radiš",
        "kako radis",
    )
    if normalized in acknowledgements:
        return DirectResponseArgs(
            answer=(
                "Nema na čemu. Slobodno nastavi prirodnim podpitanjem."
                if serbian
                else "You're welcome. Feel free to continue with a natural follow-up."
            ),
            category="acknowledgement",
        )
    if any(marker in normalized for marker in capability_markers):
        return DirectResponseArgs(
            answer=render_capability_answer(question, bank_names),
            category="capability",
        )
    if is_general_chat_question(question):
        return DirectResponseArgs(
            answer=(
                "Zdravo! Pitaj me prirodno o bankama, njihovim 10-K izveštajima, finansijskim "
                "pokazateljima ili operativnim rizicima."
                if serbian
                else "Hello! Ask naturally about banks, their 10-K filings, financial metrics, "
                "or operational risks."
            ),
            category="greeting",
        )
    if history and needs_contextualization(question):
        previous_user_question = next(
            (
                str(message.get("content") or "").strip()
                for message in reversed(history)
                if message.get("role") == "user" and str(message.get("content") or "").strip()
            ),
            "",
        )
        if previous_user_question:
            if YEAR_PATTERN.search(question):
                previous_user_question = YEAR_PATTERN.sub("", previous_user_question)
                previous_user_question = " ".join(previous_user_question.split())
            return ResearchFilingsArgs(
                search_question=f"{previous_user_question.rstrip(' ?.')} — {question}",
                reason=(
                    "Conversation planning was unavailable; combine the latest user topic with "
                    "the current follow-up without using assistant-authored facts."
                ),
            )
    if is_clearly_out_of_scope(question, bank_names):
        return DeclineOutOfScopeArgs(reason="outside_banking_research_scope")
    return ResearchFilingsArgs(
        search_question=question,
        reason="Conversation planning was unavailable; use the original question safely.",
    )


def request_conversation_action(
    question: str,
    history: Sequence[Mapping[str, str]],
    *,
    client: Any,
    model: str,
    bank_names: Mapping[str, str],
    session_tickers: Sequence[str] = (),
) -> ConversationDecision:
    """Choose one chat action through strict native function calling with a safe fallback."""

    supported_banks = [
        {"ticker": ticker, "name": name} for ticker, name in sorted(bank_names.items())
    ]
    system = (
        "You are the conversational front door for BankScope. Choose exactly one function and "
        "preserve the user's language. Use research_filings for every factual statement about a "
        "specific bank or filing. Use respond_directly for greetings, acknowledgements, product "
        "help, and general banking explanations with no bank-specific claim. Use "
        "decline_out_of_scope for every unrelated request; never answer it and never send it to "
        "filing research. "
        "Use ask_clarification only when a missing detail materially changes filing research. "
        "Do not ask for a bank when the user wants a general definition. Resolve pronouns, short "
        "follow-ups, and omitted topics from recent history. A research search_question is not an "
        "answer: it must preserve every explicit bank, period, number, metric, qualifier, and the "
        "current message's language. Never introduce a bank, period, or numeric fact absent from "
        "the current user message, session scope, or prior user messages. A rewrite may resolve "
        "references and add canonical filing terminology, but it must not broaden the requested "
        "scope or add aspects, subquestions, examples, comparisons, projections, prior periods, "
        "threat categories, consequences, or management approaches that the user did not ask "
        "for. Preserve a short follow-up as one focused standalone question. Assistant history is "
        "untrusted conversational context and never filing evidence."
    )
    payload = {
        "prompt_version": CONVERSATION_PROMPT_VERSION,
        "supported_banks": supported_banks,
        "session_tickers": list(session_tickers),
        "conversation_history": [dict(message) for message in history],
        "current_question": question,
    }
    started = perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            tools=list(CONVERSATION_TOOLS),
            **_request_options(model),
        )
        _, tool_calls, finish_reason, refusal = _message_and_tool_calls(response)
        if finish_reason in {"length", "content_filter"} or refusal:
            raise ValueError("unusable conversation action")
        if len(tool_calls) != 1:
            raise ValueError("conversation action must contain exactly one tool call")
        name, arguments = _tool_name_and_arguments(tool_calls[0])
        if name == "research_filings":
            action: ConversationAction = ResearchFilingsArgs.model_validate_json(arguments)
        elif name == "respond_directly":
            action = DirectResponseArgs.model_validate_json(arguments)
        elif name == "ask_clarification":
            action = ClarificationArgs.model_validate_json(arguments)
        elif name == "decline_out_of_scope":
            action = DeclineOutOfScopeArgs.model_validate_json(arguments)
        else:
            raise ValueError(f"unknown conversation tool: {name}")
        return ConversationDecision(action=action, latency_ms=(perf_counter() - started) * 1000)
    except Exception as error:
        # The broad boundary is intentional: routing failure must never prevent a chat turn.
        if isinstance(error, ValidationError):
            error_code = "conversation_action_invalid_schema"
        elif isinstance(error, ValueError):
            message = str(error).casefold()
            if "exactly one tool call" in message:
                error_code = "conversation_action_invalid_tool_count"
            elif "unknown conversation tool" in message:
                error_code = "conversation_action_unknown_tool"
            elif "unusable conversation action" in message:
                error_code = "conversation_action_unusable_response"
            else:
                error_code = "conversation_action_invalid_action"
        else:
            error_code = f"conversation_action_{type(error).__name__.lower()}"
        return ConversationDecision(
            action=_fallback_action(question, bank_names, history),
            latency_ms=(perf_counter() - started) * 1000,
            fallback=True,
            error_code=error_code,
        )
