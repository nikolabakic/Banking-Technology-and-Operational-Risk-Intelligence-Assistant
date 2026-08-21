from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bankscope.generation.answer_generator import (
    CITATION_PATTERN,
    GPT51_MODEL_MARKERS,
    NUMBER_TOKEN_PATTERN,
)
from bankscope.generation.query_planner import (
    YEAR_PATTERN,
    is_general_chat_question,
    needs_contextualization,
)
from bankscope.sec.bank_resolver import BankResolution, resolve_bank
from bankscope.sec.company_registry import bank_identifier_variants, normalize_bank_text

CONVERSATION_PROMPT_VERSION = "conversation-langgraph-router-v4-model-context"
CONVERSATION_REQUEST_TIMEOUT_SECONDS = 30.0
OUT_OF_SCOPE_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.5
QUALIFIER_PATTERN = re.compile(
    r"\b(?:always|never|only|approximately|about|at\s+least|at\s+most|"
    r"materially|primarily|isključivo|uvek|nikad|približno|najmanje|najviše)\b",
    re.IGNORECASE,
)

RouteAction = Literal[
    "filing_research",
    "direct_response",
    "clarification",
    "out_of_scope",
    "web_research",
]
DirectCategory = Literal[
    "greeting",
    "acknowledgement",
    "capability",
    "general_explanation",
    "contextual_transform",
]
MissingDetail = Literal["bank", "period", "metric", "intent"]


class RouteDecision(BaseModel):
    """One strict semantic route produced by the conversation model."""

    model_config = ConfigDict(extra="forbid")

    action: RouteAction
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    search_question: str | None = Field(..., min_length=2, max_length=4_000)
    response_text: str | None = Field(..., min_length=1, max_length=4_000)
    category: DirectCategory | None = Field(...)
    missing: MissingDetail | None = Field(...)
    citation_ids: list[str] = Field(..., max_length=20)
    presentation_guidance: str | None = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Style guidance for filing_research or web_research only; null for every other action."
        ),
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> RouteDecision:
        if self.action in {"filing_research", "web_research"}:
            if not self.search_question:
                raise ValueError(f"{self.action} requires search_question")
            if (
                self.response_text is not None
                or self.category is not None
                or self.missing is not None
                or self.citation_ids
            ):
                raise ValueError(f"{self.action} permits only search_question")
        elif self.action == "direct_response":
            if not self.response_text or not self.category:
                raise ValueError("direct_response requires response_text and category")
            if (
                self.search_question is not None
                or self.missing is not None
                or self.presentation_guidance is not None
            ):
                raise ValueError("direct_response permits only response_text and category")
            if self.category != "contextual_transform" and self.citation_ids:
                raise ValueError("Only a contextual transform may reuse citations")
        elif self.action == "clarification":
            if not self.response_text or not self.missing:
                raise ValueError("clarification requires response_text and missing")
            if (
                self.search_question is not None
                or self.category is not None
                or self.citation_ids
                or self.presentation_guidance is not None
            ):
                raise ValueError("clarification permits only response_text and missing")
        elif any(
            value is not None
            for value in (
                self.search_question,
                self.response_text,
                self.category,
                self.missing,
                self.presentation_guidance,
            )
        ) or self.citation_ids:
            raise ValueError("out_of_scope requires all action-specific fields to be null")
        return self


class ResearchFilingsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_question: str = Field(min_length=2, max_length=4_000)
    reason: str = Field(min_length=1, max_length=500)
    presentation_guidance: str | None = Field(default=None, max_length=300)


class WebResearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_question: str = Field(min_length=2, max_length=4_000)
    reason: str = Field(min_length=1, max_length=500)
    presentation_guidance: str | None = Field(default=None, max_length=300)


class DirectResponseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=4_000)
    category: DirectCategory
    citation_ids: list[str] = Field(default_factory=list, max_length=20)


class ClarificationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1_000)
    missing: MissingDetail


class DeclineOutOfScopeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Literal["outside_banking_research_scope"]


ConversationAction = (
    ResearchFilingsArgs
    | WebResearchArgs
    | DirectResponseArgs
    | ClarificationArgs
    | DeclineOutOfScopeArgs
)


@dataclass(frozen=True)
class ConversationDecision:
    action: ConversationAction
    latency_ms: float
    request_count: int = 1
    fallback: bool = False
    error_code: str | None = None
    route_action: RouteAction = "clarification"
    confidence: float = 0
    reason: str = ""
    router_backend: str = "langgraph"
    graph_nodes: tuple[str, ...] = ()
    source_policy: str = "filings_first_web_for_current_external"


class _ConversationGraphState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    conversation_summary: str
    previous_answer: dict[str, Any] | None
    session_tickers: list[str]
    bank_names: dict[str, str]
    bank_aliases: dict[str, tuple[str, ...]]
    resolution: BankResolution
    banking_domain: bool
    filing_signal: bool
    explicit_filing_source: bool
    web_signal: bool
    route: RouteDecision
    latency_ms: float
    fallback: bool
    error_code: str | None
    graph_nodes: list[str]


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


def _route_tool() -> dict[str, Any]:
    schema = RouteDecision.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": "route_conversation",
            "description": (
                "Choose one BankScope conversation route. This is a routing directive, not a "
                "factual answer about a specific bank."
            ),
            "strict": True,
            "parameters": schema,
        },
    }


CONVERSATION_TOOLS = (_route_tool(),)


def _message_and_tool_calls(response: Any) -> tuple[list[Any], str, str]:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return [], "", ""
    choice = choices[0]
    if isinstance(choice, Mapping):
        message = choice.get("message") or {}
        finish_reason = str(choice.get("finish_reason") or "")
    else:
        message = getattr(choice, "message", None)
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
    if isinstance(message, Mapping):
        return (
            list(message.get("tool_calls") or []),
            finish_reason,
            str(message.get("refusal") or ""),
        )
    return (
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
            "rizik",
            "daj",
            "recept",
            "reci",
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
    if _looks_serbian(question):
        return (
            "Mogu da pomognem sa podržanim bankama, njihovim 10-K izveštajima, "
            "finansijskim pokazateljima i opštim bankarskim temama."
        )
    return (
        "I can help with supported banks, their 10-K filings, financial metrics, "
        "and general banking topics."
    )


def render_web_unavailable_answer(question: str) -> str:
    if _looks_serbian(question):
        return (
            "Za ovo pitanje su potrebni aktuelni spoljni podaci, ali web pretraga još nije "
            "omogućena. Mogu i dalje da istražim indeksirane 10-K izveštaje."
        )
    return (
        "This question needs current external data, but web search is not enabled yet. "
        "I can still research the indexed 10-K filings."
    )


BANKING_DOMAIN_PATTERN = re.compile(
    r"(?i)\b(?:bank\w*|banka|banke|bankarsk\w*|10\s*[- ]?k|sec|filing\w*|"
    r"cet1|capital|kapital|ratio|koeficijent|pokazatelj\w*|metric\w*|revenue|prihod|"
    r"income|profit|assets?|aktiva|loans?|kredit\w*|deposits?|depozit\w*|liquidity|"
    r"likvidnost|risk\w*|rizik\w*|cyber\w*|sajber\w*|operational\w*|operativn\w*|"
    r"regulatory\w*|regulatorn\w*|compliance|usklađen\w*|tier\s*1)\b"
)
FILING_RESEARCH_PATTERN = re.compile(
    r"(?i)\b(?:10\s*[- ]?k|sec|filing\w*|reported|disclos\w*|material|framework|"
    r"cet1|ratio|metric\w*|revenue|income|assets?|loans?|deposits?|liquidity|"
    r"risk\w*|rizik\w*|cyber\w*|sajber\w*|operational\w*|operativn\w*|"
    r"regulatory\w*|compliance|capital|20\d{2})\b"
)
EXPLICIT_FILING_SOURCE_PATTERN = re.compile(
    r"(?i)\b(?:10\s*[- ]?k|sec\s+filing|filing\w*|reported|disclos\w*|"
    r"fiscal\s+year|annual\s+report|izveštaj\w*|izvestaj\w*)\b"
)
WEB_RESEARCH_PATTERN = re.compile(
    r"(?i)\b(?:today|today's|current|currently|latest|live|yesterday|this\s+morning|"
    r"share\s+price|stock\s+price|market\s+price|trading|traded|news|danas|današnj\w*|"
    r"trenutn\w*|najnovij\w*|juče|juc[e]?|vesti|cena\s+akcij\w*)\b"
)
MULTI_BANK_NO_TOPIC_TOKENS = {
    "a",
    "and",
    "are",
    "bank",
    "banka",
    "bankama",
    "banke",
    "banaka",
    "best",
    "better",
    "between",
    "bolja",
    "bolje",
    "bolji",
    "compare",
    "compared",
    "compares",
    "comparing",
    "comparison",
    "did",
    "difference",
    "differences",
    "differ",
    "do",
    "does",
    "each",
    "for",
    "how",
    "i",
    "in",
    "is",
    "između",
    "izmedju",
    "kako",
    "koja",
    "koje",
    "koji",
    "me",
    "molim",
    "najbolja",
    "najbolje",
    "najbolji",
    "of",
    "or",
    "other",
    "please",
    "poređenje",
    "poredjenje",
    "poredi",
    "razlika",
    "razlike",
    "razlikuju",
    "sa",
    "se",
    "tell",
    "than",
    "the",
    "them",
    "these",
    "those",
    "to",
    "u",
    "uporedi",
    "versus",
    "vs",
    "was",
    "were",
    "what",
    "which",
    "with",
    "za",
}
EXPLICIT_OUT_OF_SCOPE_PATTERN = re.compile(
    r"(?i)\b(?:recipe|recept|kuvanje|cooking|weather\s+forecast|vremenska\s+prognoza|"
    r"football|fudbal|basketball|košarka|kosarka|horoscope|horoskop)\b"
)
CET1_VAGUE_INDICATOR_PATTERN = re.compile(
    r"(?i)\bcet1\b.*\b(?:indicator|measure|pokazatelj\w*)\b|"
    r"\b(?:indicator|measure|pokazatelj\w*)\b.*\bcet1\b"
)
CET1_RATIO_PATTERN = re.compile(r"(?i)\b(?:ratio|stopa|koeficijent)\b|%")
CET1_AMOUNT_PATTERN = re.compile(
    r"(?i)\b(?:amount|iznos|capital\s+amount|iznos\s+kapitala|capital\s+in\s+|kapital\s+u\s+)\b"
)


def is_banking_domain_question(
    question: str,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    normalized = question.casefold()
    if BANKING_DOMAIN_PATTERN.search(normalized):
        return True
    aliases = bank_aliases or {}
    for ticker, name in bank_names.items():
        identifiers = (name, *aliases.get(ticker, ()))
        if any(identifier.strip().casefold() in normalized for identifier in identifiers):
            return True
        if re.search(rf"(?i)(?<!\w){re.escape(ticker.strip())}(?!\w)", question):
            return True
    return False


def has_filing_research_signal(question: str) -> bool:
    return bool(FILING_RESEARCH_PATTERN.search(question))


def has_web_research_signal(question: str) -> bool:
    return bool(WEB_RESEARCH_PATTERN.search(question))


def _multi_bank_has_topic(
    question: str,
    resolution: BankResolution,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]],
) -> bool:
    """Return whether a multi-bank request names something substantive to compare."""

    normalized = f" {normalize_bank_text(question)} "
    identifiers: set[str] = set()
    for ticker in resolution.tickers:
        values = [bank_names.get(ticker, ticker), *bank_aliases.get(ticker, ())]
        if ticker != "C":
            values.append(ticker)
        for value in values:
            identifiers.update(bank_identifier_variants(value))
    for identifier in sorted(
        identifiers, key=lambda value: (len(value.split()), len(value)), reverse=True
    ):
        normalized = normalized.replace(f" {identifier} ", " ")

    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    substantive = [
        token
        for token in tokens
        if token not in MULTI_BANK_NO_TOPIC_TOKENS and not re.fullmatch(r"(?:19|20)\d{2}", token)
    ]
    return bool(substantive)


def is_clearly_out_of_scope(
    question: str,
    bank_names: Mapping[str, str] | None = None,
) -> bool:
    """Return only a high-precision fallback signal; never use as a normal pre-router veto."""

    del bank_names
    return bool(EXPLICIT_OUT_OF_SCOPE_PATTERN.search(question))


def cet1_metric_clarification(question: str) -> ClarificationArgs | None:
    if not CET1_VAGUE_INDICATOR_PATTERN.search(question):
        return None
    if CET1_RATIO_PATTERN.search(question) or CET1_AMOUNT_PATTERN.search(question):
        return None
    if _looks_serbian(question):
        prompt = "Da li želite CET1 kapital (iznos) ili CET1 kapitalni koeficijent (procenat)?"
    else:
        prompt = "Do you mean CET1 capital (amount) or the CET1 capital ratio (percentage)?"
    return ClarificationArgs(question=prompt, missing="metric")


def _acknowledgement_or_capability_route(
    question: str, bank_names: Mapping[str, str]
) -> RouteDecision | None:
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
    if normalized in acknowledgements:
        return RouteDecision(
            action="direct_response",
            confidence=1,
            reason="deterministic_acknowledgement",
            search_question=None,
            response_text=(
                "Nema na čemu. Slobodno nastavi prirodnim podpitanjem."
                if serbian
                else "You're welcome. Feel free to continue with a natural follow-up."
            ),
            category="acknowledgement",
            missing=None,
            citation_ids=[],
            presentation_guidance=None,
        )
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
    if any(marker in normalized for marker in capability_markers):
        return RouteDecision(
            action="direct_response",
            confidence=1,
            reason="deterministic_capability",
            search_question=None,
            response_text=render_capability_answer(question, bank_names),
            category="capability",
            missing=None,
            citation_ids=[],
            presentation_guidance=None,
        )
    if is_general_chat_question(question):
        return RouteDecision(
            action="direct_response",
            confidence=1,
            reason="deterministic_greeting",
            search_question=None,
            response_text=(
                "Zdravo! Pitaj me prirodno o bankama, njihovim 10-K izveštajima, "
                "finansijskim pokazateljima ili operativnim rizicima."
                if serbian
                else "Hello! Ask naturally about banks, their 10-K filings, financial metrics, "
                "or operational risks."
            ),
            category="greeting",
            missing=None,
            citation_ids=[],
            presentation_guidance=None,
        )
    return None


def _fallback_route(
    question: str,
    bank_names: Mapping[str, str],
    history: Sequence[Mapping[str, str]],
    *,
    resolution: BankResolution,
    banking_domain: bool,
) -> RouteDecision:
    direct = _acknowledgement_or_capability_route(question, bank_names)
    if direct is not None:
        return direct
    if is_clearly_out_of_scope(question):
        return RouteDecision(
            action="out_of_scope",
            confidence=1,
            reason="explicit_unrelated_fallback",
            search_question=None,
            response_text=None,
            category=None,
            missing=None,
            citation_ids=[],
            presentation_guidance=None,
        )
    if resolution.source == "question" or banking_domain:
        return RouteDecision(
            action="filing_research",
            confidence=0.75,
            reason="supported_bank_or_banking_signal_fallback",
            search_question=question,
            response_text=None,
            category=None,
            missing=None,
            citation_ids=[],
            presentation_guidance=None,
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
            return RouteDecision(
                action="filing_research",
                confidence=0.7,
                reason="referential_history_fallback",
                search_question=f"{previous_user_question.rstrip(' ?.')} — {question}",
                response_text=None,
                category=None,
                missing=None,
                citation_ids=[],
                presentation_guidance=None,
            )
    return RouteDecision(
        action="clarification",
        confidence=0.5,
        reason="ambiguous_intent_fallback",
        search_question=None,
        response_text=(
            "Na koje bankarsko pitanje želite da se nadovežem?"
            if _looks_serbian(question)
            else "Which banking question would you like me to help with?"
        ),
        category=None,
        missing="intent",
        citation_ids=[],
        presentation_guidance=None,
    )


def _route_system_prompt() -> str:
    return (
        "Understand the BankScope conversation and return one strict route. Use filing_research "
        "for new factual claims about a supported bank or filing, and web_research for current or "
        "external facts. Use direct_response for greetings, acknowledgements, BankScope product "
        "help, banking or risk concepts that need no bank-specific facts, and transformations of "
        "the immediately previous grounded answer. Recipes, entertainment, creative writing, and "
        "other clearly non-banking requests are out_of_scope, not general_explanation. For a "
        "contextual transform, "
        "set category=contextual_transform, preserve only facts already present, and reuse only "
        "the supplied previous-answer citation labels. A request for a shorter answer must "
        "materially condense the previous answer by retaining its essential points. Use "
        "clarification only when a missing "
        "detail materially changes the task, and out_of_scope only when clearly unrelated. "
        "The supplied bank_resolution is authoritative. Preserve every explicit bank, period, "
        "number, metric, and qualifier in a search_question. For filing_research or web_research, "
        "infer style preferences and put only style/format instructions in presentation_guidance. "
        "First scan all prior user messages for standing instructions such as 'from now on', "
        "'always', or an explicit preference; these remain active until the user resets them. For "
        "every research action with an active standing style preference, presentation_guidance "
        "must be non-null. Preserve a numeric length limit only when the user explicitly supplied "
        "one; otherwise express concise preferences qualitatively and let the answer model choose "
        "the appropriate length for the question. "
        "For a direct contextual_transform, apply the requested style directly to response_text, "
        "set presentation_guidance to null, and list exactly its inline citation labels in "
        "citation_ids. Every nullable action field is required: set unused fields to null and "
        "citation_ids to an empty list when citations are not allowed. Assistant history supports "
        "conversational continuity and direct transformation, but it is never filing evidence for "
        "a new factual answer."
    )


def _route_payload(
    state: _ConversationGraphState,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]],
) -> str:
    resolution = state["resolution"]
    supported_banks = [
        {
            "ticker": ticker,
            "name": name,
            "aliases": list(bank_aliases.get(ticker, ())),
        }
        for ticker, name in sorted(bank_names.items())
    ]
    previous = state.get("previous_answer") or {}
    safe_previous = None
    if previous:
        safe_previous = {
            "question": previous.get("question"),
            "answer": previous.get("answer"),
            "ticker": previous.get("ticker"),
            "tickers": list(previous.get("tickers") or []),
            "citation_ids": [
                str(item.get("label") or "").strip().upper()
                for item in previous.get("citations") or []
                if str(item.get("label") or "").strip()
            ],
        }
    payload = {
        "prompt_version": CONVERSATION_PROMPT_VERSION,
        "supported_banks": supported_banks,
        "available_sources": {"filings": True, "web": False},
        "session_tickers": state["session_tickers"],
        "bank_resolution": resolution.as_dict(),
        "positive_signals": {
            "banking_domain": state["banking_domain"],
            "filing_research": state["filing_signal"],
            "explicit_filing_source": state["explicit_filing_source"],
            "web_research": state["web_signal"],
        },
        "conversation_summary": state.get("conversation_summary") or "",
        "conversation_history": state["history"],
        "previous_grounded_answer": safe_previous,
        "current_question": state["question"],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _native_route(client: Any, model: str, system: str, payload: str) -> RouteDecision:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
        tools=list(CONVERSATION_TOOLS),
        **_request_options(model),
    )
    tool_calls, finish_reason, refusal = _message_and_tool_calls(response)
    if finish_reason in {"length", "content_filter"} or refusal:
        raise ValueError("unusable conversation route")
    if len(tool_calls) != 1:
        raise ValueError("conversation route must contain exactly one tool call")
    name, arguments = _tool_name_and_arguments(tool_calls[0])
    if name != "route_conversation":
        raise ValueError(f"unknown conversation tool: {name}")
    return RouteDecision.model_validate_json(arguments)


def _route_error_code(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "conversation_route_invalid_schema"
    if isinstance(error, ValueError):
        message = str(error).casefold()
        if "exactly one tool call" in message:
            return "conversation_route_invalid_tool_count"
        if "unknown conversation tool" in message:
            return "conversation_route_unknown_tool"
        if "unusable conversation route" in message:
            return "conversation_route_unusable_response"
        return "conversation_route_invalid_action"
    return f"conversation_route_{type(error).__name__.lower()}"


def _apply_route_policy(route: RouteDecision, state: _ConversationGraphState) -> RouteDecision:
    resolution = state["resolution"]
    explicit_supported_bank = resolution.source == "question" and bool(resolution.tickers)
    multi_bank = resolution.status == "multiple" and 2 <= len(resolution.tickers) <= 4
    if (
        route.action == "direct_response"
        and route.category == "general_explanation"
        and is_clearly_out_of_scope(state["question"], state["bank_names"])
    ):
        return RouteDecision(
            action="out_of_scope",
            confidence=max(route.confidence, OUT_OF_SCOPE_CONFIDENCE_THRESHOLD),
            reason="explicit_non_banking_request_requires_scope_decline",
            search_question=None,
            response_text=None,
            category=None,
            missing=None,
            citation_ids=[],
            presentation_guidance=None,
        )
    if route.action == "direct_response" and route.category == "contextual_transform":
        previous = state.get("previous_answer")
        if not previous:
            return RouteDecision(
                action="clarification",
                confidence=route.confidence,
                reason="contextual_transform_requires_previous_answer",
                search_question=None,
                response_text="Which previous answer would you like me to transform?",
                category=None,
                missing="intent",
                citation_ids=[],
                presentation_guidance=None,
            )
        allowed_citations = {
            str(item.get("label") or "").strip().upper()
            for item in previous.get("citations") or []
        }
        inline_citations = set(CITATION_PATTERN.findall(str(route.response_text or "")))
        if (
            set(route.citation_ids) != inline_citations
            or set(route.citation_ids) - allowed_citations
        ):
            return RouteDecision(
                action="clarification",
                confidence=route.confidence,
                reason="contextual_transform_invalid_citations",
                search_question=None,
                response_text="I could not safely preserve the previous answer's citations.",
                category=None,
                missing="intent",
                citation_ids=[],
                presentation_guidance=None,
            )
        previous_numbers = set(NUMBER_TOKEN_PATTERN.findall(str(previous.get("answer") or "")))
        response_numbers = set(NUMBER_TOKEN_PATTERN.findall(str(route.response_text or "")))
        if response_numbers - previous_numbers:
            return RouteDecision(
                action="clarification",
                confidence=route.confidence,
                reason="contextual_transform_added_number",
                search_question=None,
                response_text="I could not safely transform the answer without changing a number.",
                category=None,
                missing="intent",
                citation_ids=[],
                presentation_guidance=None,
            )
        allowed_tickers = {
            str(value).strip().upper()
            for value in [
                previous.get("ticker"),
                *(previous.get("tickers") or []),
            ]
            if value
        }
        response_resolution = resolve_bank(
            str(route.response_text or ""),
            bank_names=state["bank_names"],
            bank_aliases=state["bank_aliases"],
        )
        if set(response_resolution.tickers) - allowed_tickers:
            if explicit_supported_bank:
                return RouteDecision(
                    action="filing_research",
                    confidence=max(route.confidence, 0.8),
                    reason="new_bank_fact_requires_filing_research",
                    search_question=state["question"],
                    response_text=None,
                    category=None,
                    missing=None,
                    citation_ids=[],
                    presentation_guidance=None,
                )
            return RouteDecision(
                action="clarification",
                confidence=route.confidence,
                reason="contextual_transform_changed_bank",
                search_question=None,
                response_text="I could not safely transform the answer without changing the bank.",
                category=None,
                missing="intent",
                citation_ids=[],
                presentation_guidance=None,
            )
        previous_qualifiers = {
            value.casefold()
            for value in QUALIFIER_PATTERN.findall(str(previous.get("answer") or ""))
        }
        response_qualifiers = {
            value.casefold()
            for value in QUALIFIER_PATTERN.findall(str(route.response_text or ""))
        }
        if response_qualifiers - previous_qualifiers:
            return RouteDecision(
                action="clarification",
                confidence=route.confidence,
                reason="contextual_transform_added_qualifier",
                search_question=None,
                response_text=(
                    "I could not safely transform the answer without changing a qualifier."
                ),
                category=None,
                missing="intent",
                citation_ids=[],
                presentation_guidance=None,
            )
    if multi_bank and not _multi_bank_has_topic(
        state["question"],
        resolution,
        state["bank_names"],
        state["bank_aliases"],
    ):
        return RouteDecision(
            action="clarification",
            confidence=max(route.confidence, 0.8),
            reason="multi_bank_comparison_requires_topic_or_metric",
            search_question=None,
            response_text=(
                "Koju temu ili pokazatelj želite da uporedim za navedene banke?"
                if _looks_serbian(state["question"])
                else "Which topic or metric would you like me to compare for those banks?"
            ),
            category=None,
            missing="metric",
            citation_ids=[],
            presentation_guidance=None,
        )
    requires_web = (
        explicit_supported_bank and state["web_signal"] and not state["explicit_filing_source"]
    )
    if requires_web and route.action not in {"web_research", "clarification"}:
        return RouteDecision(
            action="web_research",
            confidence=max(route.confidence, 0.8),
            reason="current_external_bank_claim_requires_web_research",
            search_question=state["question"],
            response_text=None,
            category=None,
            missing=None,
            citation_ids=[],
            presentation_guidance=route.presentation_guidance,
        )
    requires_filing = (
        explicit_supported_bank
        and (state["filing_signal"] or multi_bank)
        and (not state["web_signal"] or state["explicit_filing_source"])
    )
    if requires_filing and route.action not in {"filing_research", "clarification"}:
        return RouteDecision(
            action="filing_research",
            confidence=max(route.confidence, 0.8),
            reason="bank_specific_filing_claim_requires_filing_research",
            search_question=state["question"],
            response_text=None,
            category=None,
            missing=None,
            citation_ids=[],
            presentation_guidance=route.presentation_guidance,
        )
    if route.action == "out_of_scope" and route.confidence < OUT_OF_SCOPE_CONFIDENCE_THRESHOLD:
        return RouteDecision(
            action="clarification",
            confidence=route.confidence,
            reason="low_confidence_out_of_scope_requires_clarification",
            search_question=None,
            response_text=(
                "Da li želite da ovo povežem sa bankom, 10-K izveštajem ili bankarskim rizikom?"
                if _looks_serbian(state["question"])
                else "Would you like to connect this to a bank, 10-K filing, or banking risk?"
            ),
            category=None,
            missing="intent",
            citation_ids=[],
            presentation_guidance=None,
        )
    if route.confidence < LOW_CONFIDENCE_THRESHOLD and route.action not in {
        "direct_response",
        "clarification",
    }:
        return RouteDecision(
            action="clarification",
            confidence=route.confidence,
            reason="low_confidence_route_requires_clarification",
            search_question=None,
            response_text=(
                "Možete li malo precizirati koje bankarsko pitanje ili izvor želite?"
                if _looks_serbian(state["question"])
                else "Could you clarify which banking question or source you want me to use?"
            ),
            category=None,
            missing="intent",
            citation_ids=[],
            presentation_guidance=None,
        )
    return route


def _route_to_action(route: RouteDecision) -> ConversationAction:
    if route.action == "filing_research":
        return ResearchFilingsArgs(
            search_question=str(route.search_question),
            reason=route.reason,
            presentation_guidance=route.presentation_guidance,
        )
    if route.action == "web_research":
        return WebResearchArgs(
            search_question=str(route.search_question),
            reason=route.reason,
            presentation_guidance=route.presentation_guidance,
        )
    if route.action == "direct_response":
        return DirectResponseArgs(
            answer=str(route.response_text),
            category=route.category or "general_explanation",
            citation_ids=route.citation_ids,
        )
    if route.action == "clarification":
        return ClarificationArgs(
            question=str(route.response_text), missing=route.missing or "intent"
        )
    return DeclineOutOfScopeArgs(reason="outside_banking_research_scope")


class ConversationGraph:
    """Compile and run the bounded conversation-routing workflow once per pipeline."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        bank_names: Mapping[str, str],
        bank_aliases: Mapping[str, Sequence[str]] | None = None,
        chat_model: Any | None = None,
        backend: Literal["langgraph", "legacy"] = "langgraph",
    ) -> None:
        self.client = client
        self.model = model
        self.bank_names = dict(bank_names)
        self.bank_aliases = {
            ticker: tuple(aliases) for ticker, aliases in (bank_aliases or {}).items()
        }
        self.chat_model = chat_model
        self.backend = backend
        self._structured_model = None
        if chat_model is not None:
            self._structured_model = chat_model.with_structured_output(
                RouteDecision, method="function_calling", strict=True
            )
        self._graph = self._compile_graph() if backend == "langgraph" else None

    def _compile_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise RuntimeError(
                "Install the optional LLM dependencies with 'pip install -e .[llm]'."
            ) from error

        builder = StateGraph(_ConversationGraphState)
        builder.add_node("prepare", self._prepare_node)
        builder.add_node("route", self._route_node)
        builder.add_node("validate_route", self._validate_route_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "route")
        builder.add_edge("route", "validate_route")
        builder.add_edge("validate_route", END)
        return builder.compile()

    def _prepare_node(self, state: _ConversationGraphState) -> dict[str, Any]:
        question = state["question"]
        resolution = resolve_bank(
            question,
            bank_names=self.bank_names,
            bank_aliases=self.bank_aliases,
            session_tickers=state["session_tickers"],
        )
        return {
            "resolution": resolution,
            "bank_names": self.bank_names,
            "bank_aliases": self.bank_aliases,
            "banking_domain": is_banking_domain_question(
                question, self.bank_names, self.bank_aliases
            ),
            "filing_signal": has_filing_research_signal(question),
            "explicit_filing_source": bool(EXPLICIT_FILING_SOURCE_PATTERN.search(question)),
            "web_signal": has_web_research_signal(question),
            "graph_nodes": ["prepare"],
        }

    def _route_node(self, state: _ConversationGraphState) -> dict[str, Any]:
        started = perf_counter()
        try:
            system = _route_system_prompt()
            payload = _route_payload(state, self.bank_names, self.bank_aliases)
            if self._structured_model is not None:
                result = self._structured_model.invoke([("system", system), ("human", payload)])
                route = (
                    result
                    if isinstance(result, RouteDecision)
                    else RouteDecision.model_validate(result)
                )
            else:
                route = _native_route(self.client, self.model, system, payload)
            return {
                "route": route,
                "latency_ms": (perf_counter() - started) * 1000,
                "fallback": False,
                "error_code": None,
                "graph_nodes": [*state.get("graph_nodes", []), "route"],
            }
        except Exception as error:
            route = _fallback_route(
                state["question"],
                self.bank_names,
                state["history"],
                resolution=state["resolution"],
                banking_domain=state["banking_domain"],
            )
            return {
                "route": route,
                "latency_ms": (perf_counter() - started) * 1000,
                "fallback": True,
                "error_code": _route_error_code(error),
                "graph_nodes": [*state.get("graph_nodes", []), "route"],
            }

    @staticmethod
    def _validate_route_node(state: _ConversationGraphState) -> dict[str, Any]:
        route = _apply_route_policy(state["route"], state)
        return {
            "route": route,
            "graph_nodes": [*state.get("graph_nodes", []), "validate_route"],
        }

    def route(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
        *,
        session_tickers: Sequence[str] = (),
        conversation_summary: str = "",
        previous_answer: Mapping[str, Any] | None = None,
    ) -> ConversationDecision:
        initial: _ConversationGraphState = {
            "question": question,
            "history": [dict(message) for message in history],
            "session_tickers": list(session_tickers),
            "conversation_summary": conversation_summary,
            "previous_answer": dict(previous_answer) if previous_answer else None,
        }
        if self._graph is not None:
            state = self._graph.invoke(initial)
        else:
            state = {**initial, **self._prepare_node(initial)}
            state.update(self._route_node(state))
            state.update(self._validate_route_node(state))
            state["graph_nodes"] = ["legacy_route", "validate_route"]
        route = state["route"]
        return ConversationDecision(
            action=_route_to_action(route),
            latency_ms=float(state.get("latency_ms") or 0),
            request_count=1,
            fallback=bool(state.get("fallback")),
            error_code=state.get("error_code"),
            route_action=route.action,
            confidence=route.confidence,
            reason=route.reason,
            router_backend=self.backend,
            graph_nodes=tuple(state.get("graph_nodes") or ()),
        )


def request_conversation_action(
    question: str,
    history: Sequence[Mapping[str, str]],
    *,
    client: Any,
    model: str,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]] | None = None,
    session_tickers: Sequence[str] = (),
    conversation_summary: str = "",
    previous_answer: Mapping[str, Any] | None = None,
    chat_model: Any | None = None,
    backend: Literal["langgraph", "legacy"] = "legacy",
) -> ConversationDecision:
    """Compatibility entrypoint; production compiles ConversationGraph once in the pipeline."""

    graph = ConversationGraph(
        client=client,
        model=model,
        bank_names=bank_names,
        bank_aliases=bank_aliases,
        chat_model=chat_model,
        backend=backend,
    )
    return graph.route(
        question,
        history,
        session_tickers=session_tickers,
        conversation_summary=conversation_summary,
        previous_answer=previous_answer,
    )
