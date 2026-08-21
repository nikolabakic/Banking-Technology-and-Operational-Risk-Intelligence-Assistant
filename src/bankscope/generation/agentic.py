from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from bankscope.generation.answer_generator import GPT51_MODEL_MARKERS, GenerationValidationError

AGENT_STEP_PROMPT_VERSION = "agentic-rag-loop-v3-native-tools"
VERIFIER_PROMPT_VERSION = "agentic-rag-verifier-v2-native-tools"
AGENTIC_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_AGENT_MODEL_REQUESTS = 3
MAX_AGENT_TOOL_ACTIONS = 1
MAX_VERIFIER_REQUESTS = 1
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?%?")
TIER_ONE_PATTERN = re.compile(r"(?i)\btier\s+1\b")


def _numeric_facts(text: str) -> set[str]:
    """Extract numeric facts while allowing the canonical term ``Tier 1``."""
    without_tier_one = TIER_ONE_PATTERN.sub("tier one", text)
    return {
        value.rstrip("%").replace(",", ".") for value in NUMBER_PATTERN.findall(without_tier_one)
    }


class AgenticPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["generate", "rewrite_search", "expand_context", "abstain"]
    reason_code: str = Field(min_length=1, max_length=80)
    explanation: str = Field(min_length=1, max_length=500)
    rewritten_query: str | None = Field(default=None, max_length=4_000)
    anchor_target_chunk_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_action_arguments(self) -> AgenticPlan:
        if self.action == "rewrite_search":
            if not self.rewritten_query or not self.rewritten_query.strip():
                raise ValueError("rewrite_search requires rewritten_query.")
            if self.anchor_target_chunk_id is not None:
                raise ValueError("rewrite_search cannot include an anchor.")
        elif self.action == "expand_context":
            if not self.anchor_target_chunk_id or not self.anchor_target_chunk_id.strip():
                raise ValueError("expand_context requires anchor_target_chunk_id.")
            if self.rewritten_query is not None:
                raise ValueError("expand_context cannot include a rewritten query.")
        elif self.rewritten_query is not None or self.anchor_target_chunk_id is not None:
            raise ValueError("generate and abstain cannot include action arguments.")
        return self


class SearchHybridStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search_hybrid"]
    query: str = Field(min_length=2, max_length=4_000)
    reason: str = Field(min_length=1, max_length=500)


class SearchExactStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search_exact"]
    terms: list[str] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_terms(self) -> SearchExactStep:
        if any(not 2 <= len(term.strip()) <= 120 for term in self.terms):
            raise ValueError("Exact terms must contain 2 to 120 characters.")
        return self


class ReadContextStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["read_context"]
    anchor_target_chunk_id: str = Field(min_length=1, max_length=256)
    before: int = Field(default=2, ge=0, le=3)
    after: int = Field(default=2, ge=0, le=3)
    reason: str = Field(min_length=1, max_length=500)


class FinishStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["finish"]
    status: Literal["sufficient", "unsupported"]
    reason: str = Field(min_length=1, max_length=500)
    supporting_target_chunk_ids: list[str] = Field(default_factory=list, max_length=10)


AgentStep = Annotated[
    SearchHybridStep | SearchExactStep | ReadContextStep | FinishStep,
    Field(discriminator="action"),
]
AGENT_STEP_ADAPTER = TypeAdapter(AgentStep)


class EvidenceVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "missing", "unsupported"]
    explanation: str = Field(min_length=1, max_length=700)
    missing_aspects: list[str] = Field(default_factory=list, max_length=8)
    supporting_target_chunk_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_status_fields(self) -> EvidenceVerdict:
        if self.status == "missing" and not self.missing_aspects:
            raise ValueError("A missing verdict requires missing_aspects.")
        if self.status != "missing" and self.missing_aspects:
            raise ValueError("Only a missing verdict may include missing_aspects.")
        return self


class SearchHybridArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=4_000)
    reason: str = Field(min_length=1, max_length=500)


class SearchExactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[str] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_terms(self) -> SearchExactArgs:
        if any(not 2 <= len(term.strip()) <= 120 for term in self.terms):
            raise ValueError("Exact terms must contain 2 to 120 characters.")
        return self


class ReadContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_target_chunk_id: str = Field(min_length=1, max_length=256)
    before: int = Field(ge=0, le=3)
    after: int = Field(ge=0, le=3)
    reason: str = Field(min_length=1, max_length=500)


class FinishArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "unsupported"]
    reason: str = Field(min_length=1, max_length=500)
    supporting_target_chunk_ids: list[str] = Field(max_length=10)


class EvidenceVerdictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "missing", "unsupported"]
    explanation: str = Field(min_length=1, max_length=700)
    missing_aspects: list[str] = Field(max_length=8)
    supporting_target_chunk_ids: list[str] = Field(max_length=10)

    @model_validator(mode="after")
    def validate_status_fields(self) -> EvidenceVerdictArgs:
        if self.status == "missing" and not self.missing_aspects:
            raise ValueError("A missing verdict requires missing_aspects.")
        if self.status != "missing" and self.missing_aspects:
            raise ValueError("Only a missing verdict may include missing_aspects.")
        return self


@dataclass
class AgentState:
    ticker: str
    question: str
    evidence: list[dict[str, Any]]
    model_requests: int = 0
    tool_actions: int = 0
    verifier_requests: int = 0
    consecutive_schema_failures: int = 0
    executed_searches: set[str] = field(default_factory=set)
    read_windows: set[tuple[str, int, int]] = field(default_factory=set)
    verifier_feedback: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_model_requests(self) -> int:
        return max(0, MAX_AGENT_MODEL_REQUESTS - self.model_requests)

    @property
    def remaining_tool_actions(self) -> int:
        return max(0, MAX_AGENT_TOOL_ACTIONS - self.tool_actions)


@dataclass(frozen=True)
class ModelDecision:
    value: AgentStep | EvidenceVerdict
    latency_ms: float
    request_count: int = 1
    fallback: bool = False
    error_code: str | None = None


def _tool_request_options(model: str) -> dict[str, Any]:
    normalized = model.strip().upper()
    options: dict[str, Any] = {
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    if any(marker in normalized for marker in GPT51_MODEL_MARKERS):
        options["max_completion_tokens"] = 500
    else:
        options.update({"max_tokens": 500, "temperature": 0})
    return options


def _native_tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
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


AGENT_STEP_TOOLS = (
    _native_tool(
        "search_hybrid",
        "Run one semantic and lexical search within the current bank's filing corpus.",
        SearchHybridArgs,
    ),
    _native_tool(
        "search_exact",
        "Search one to eight literal filing phrases within the current bank only.",
        SearchExactArgs,
    ),
    _native_tool(
        "read_context",
        "Read a bounded canonical window around one supplied narrative target chunk ID.",
        ReadContextArgs,
    ),
    _native_tool(
        "finish",
        "Finish retrieval from current evidence as sufficient or unsupported.",
        FinishArgs,
    ),
)

EVIDENCE_VERDICT_TOOL = _native_tool(
    "submit_evidence_verdict",
    "Submit the independent groundedness verdict for the supplied evidence.",
    EvidenceVerdictArgs,
)


def _message_tool_calls(response: Any) -> tuple[list[Any], str, str]:
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


def _call_tool_model(
    *,
    client: Any,
    model: str,
    system: str,
    payload: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], float, str, str]:
    started = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        tools=[dict(tool) for tool in tools],
        **_tool_request_options(model),
        timeout=AGENTIC_REQUEST_TIMEOUT_SECONDS,
    )
    tool_calls, finish_reason, refusal = _message_tool_calls(response)
    return tool_calls, (perf_counter() - started) * 1000, finish_reason, refusal


def evidence_previews(
    evidence: Sequence[Mapping[str, Any]], *, limit: int = 10
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for item in evidence[:limit]:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        body = str(item.get("evidence") or item.get("document") or "")
        previews.append(
            {
                "target_chunk_id": item.get("target_chunk_id"),
                "ticker": item.get("ticker") or metadata.get("ticker"),
                "record_type": item.get("record_type") or metadata.get("record_type"),
                "report_date": metadata.get("report_date"),
                "section_title": metadata.get("section_title"),
                "preview": body[:1_600],
            }
        )
    return previews


def request_agent_step(state: AgentState, *, client: Any, model: str) -> ModelDecision:
    """Request one bounded search/read/finish decision from the model."""
    system = (
        "You control retrieval for exactly one bank filing. Choose one action. search_hybrid may "
        "translate or augment a non-English question with canonical English filing terminology, "
        "but must retain every explicit year and must never add a numeric fact. search_exact uses "
        "one to eight literal phrases (never regex) for exact names, acronyms, and user-provided "
        "numbers. read_context may use only a supplied narrative target ID. Assess only the part "
        "of a comparison that belongs to current_ticker; missing peer-bank evidence is irrelevant. "
        "Do not finish unsupported merely because initial top results miss: try search first "
        "unless the requested period is explicitly beyond the filing corpus. Finish sufficient "
        "only when all parts for current_ticker are supported. Call exactly one supplied function."
    )
    try:
        tool_calls, latency_ms, finish_reason, refusal = _call_tool_model(
            client=client,
            model=model,
            system=system,
            payload={
                "prompt_version": AGENT_STEP_PROMPT_VERSION,
                "question": state.question,
                "current_ticker": state.ticker,
                "remaining_model_requests": state.remaining_model_requests,
                "remaining_tool_actions": state.remaining_tool_actions,
                "evidence": evidence_previews(state.evidence),
                "executed_searches": sorted(state.executed_searches),
                "read_windows": [list(window) for window in sorted(state.read_windows)],
                "verifier_feedback": state.verifier_feedback[-2:],
                "recent_trace": state.trace[-4:],
            },
            tools=AGENT_STEP_TOOLS,
        )
    except Exception as error:
        raise GenerationValidationError(
            "agentic_step_request_failed",
            "OpenAI agent-step request failed.",
            generation={"stage": "agentic_retrieval", "request_count": 1},
        ) from error
    metadata = {"stage": "agentic_retrieval", "latency_ms": latency_ms, "request_count": 1}
    if finish_reason == "length":
        code = "agentic_step_truncated"
    elif finish_reason == "content_filter" or refusal:
        code = "agentic_step_filtered"
    elif len(tool_calls) != 1:
        code = "agentic_step_empty"
    else:
        code = ""
    if code:
        raise GenerationValidationError(
            code, "OpenAI returned an unusable agent step.", generation=metadata
        )
    try:
        name, arguments = _tool_name_and_arguments(tool_calls[0])
        if name == "search_hybrid":
            args = SearchHybridArgs.model_validate_json(arguments)
            step: AgentStep = SearchHybridStep(action="search_hybrid", **args.model_dump())
        elif name == "search_exact":
            exact_args = SearchExactArgs.model_validate_json(arguments)
            step = SearchExactStep(action="search_exact", **exact_args.model_dump())
        elif name == "read_context":
            context_args = ReadContextArgs.model_validate_json(arguments)
            step = ReadContextStep(action="read_context", **context_args.model_dump())
        elif name == "finish":
            finish_args = FinishArgs.model_validate_json(arguments)
            step = FinishStep(action="finish", **finish_args.model_dump())
        else:
            raise ValueError(f"Unknown agent action: {name}")
    except (TypeError, ValueError, ValidationError) as error:
        raise GenerationValidationError(
            "agentic_step_invalid_schema",
            "OpenAI returned an invalid agent function call.",
            generation=metadata,
        ) from error
    return ModelDecision(step, latency_ms)


def verify_evidence(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    ticker: str,
    client: Any,
    model: str,
) -> ModelDecision:
    system = (
        "Independently verify whether the supplied evidence answers every part of the question for "
        "current_ticker only. Ignore missing peer-bank evidence. Use sufficient only for direct "
        "complete support, missing when another targeted search/read could fill named gaps, and "
        "unsupported only when the requested fact is outside the filing corpus. Supporting IDs "
        "must come from supplied evidence. Call submit_evidence_verdict exactly once."
    )
    try:
        tool_calls, latency_ms, finish_reason, refusal = _call_tool_model(
            client=client,
            model=model,
            system=system,
            payload={
                "prompt_version": VERIFIER_PROMPT_VERSION,
                "question": question,
                "current_ticker": ticker,
                "evidence": evidence_previews(evidence),
            },
            tools=(EVIDENCE_VERDICT_TOOL,),
        )
    except Exception as error:
        raise GenerationValidationError(
            "agentic_verdict_request_failed",
            "OpenAI evidence-verifier request failed.",
            generation={"stage": "verifying_evidence", "request_count": 1},
        ) from error
    metadata = {"stage": "verifying_evidence", "latency_ms": latency_ms, "request_count": 1}
    if finish_reason in {"length", "content_filter"} or refusal or len(tool_calls) != 1:
        raise GenerationValidationError(
            "agentic_verdict_unusable",
            "OpenAI returned an unusable evidence verdict.",
            generation=metadata,
        )
    try:
        name, arguments = _tool_name_and_arguments(tool_calls[0])
        if name != "submit_evidence_verdict":
            raise ValueError(f"Unknown verifier action: {name}")
        verdict_args = EvidenceVerdictArgs.model_validate_json(arguments)
        verdict = EvidenceVerdict(**verdict_args.model_dump())
    except (TypeError, ValueError, ValidationError) as error:
        raise GenerationValidationError(
            "agentic_verdict_invalid_schema",
            "OpenAI returned an invalid evidence-verdict function call.",
            generation=metadata,
        ) from error
    known_ids = {str(item.get("target_chunk_id") or "") for item in evidence}
    if set(verdict.supporting_target_chunk_ids) - known_ids:
        raise GenerationValidationError(
            "agentic_verdict_unknown_evidence",
            "The evidence verdict referenced an unknown target ID.",
            generation=metadata,
        )
    return ModelDecision(verdict, latency_ms)


def validate_agent_step(step: AgentStep, state: AgentState) -> None:
    """Enforce query preservation and bank-scoped action arguments at runtime."""
    if isinstance(step, SearchHybridStep):
        missing_years = set(YEAR_PATTERN.findall(state.question)) - set(
            YEAR_PATTERN.findall(step.query)
        )
        if missing_years:
            raise GenerationValidationError(
                "agentic_search_lost_period",
                "The search query did not preserve every explicit period.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )
        original_numbers = _numeric_facts(state.question)
        search_numbers = _numeric_facts(step.query)
        if search_numbers - original_numbers:
            raise GenerationValidationError(
                "agentic_search_added_numeric_fact",
                "The search query introduced a numeric fact not present in the question.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )
    elif isinstance(step, SearchExactStep):
        original_numbers = _numeric_facts(state.question)
        term_numbers = set().union(*(_numeric_facts(term) for term in step.terms))
        if term_numbers - original_numbers:
            raise GenerationValidationError(
                "agentic_exact_added_numeric_fact",
                "Exact search introduced a numeric fact not present in the question.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )
    elif isinstance(step, ReadContextStep):
        anchor = next(
            (
                item
                for item in state.evidence
                if item.get("target_chunk_id") == step.anchor_target_chunk_id
            ),
            None,
        )
        if anchor is None:
            raise GenerationValidationError(
                "agentic_anchor_not_in_results",
                "Context reading requires an anchor returned by an earlier search.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )
        metadata = anchor.get("metadata") if isinstance(anchor.get("metadata"), Mapping) else {}
        anchor_ticker = str(anchor.get("ticker") or metadata.get("ticker") or "").upper()
        record_type = str(anchor.get("record_type") or metadata.get("record_type") or "").lower()
        if anchor_ticker != state.ticker.upper() or record_type != "text":
            raise GenerationValidationError(
                "agentic_anchor_outside_scope",
                "Context reading requires narrative evidence from the current bank.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )
    elif isinstance(step, FinishStep):
        known_ids = {str(item.get("target_chunk_id") or "") for item in state.evidence}
        if set(step.supporting_target_chunk_ids) - known_ids:
            raise GenerationValidationError(
                "agentic_finish_unknown_evidence",
                "Finish referenced evidence that was not retrieved.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )
        if (
            step.status == "unsupported"
            and state.tool_actions == 0
            and not _is_future_question(state.question, state.evidence)
        ):
            raise GenerationValidationError(
                "agentic_premature_unsupported",
                "At least one targeted search is required before unsupported.",
                generation={"stage": "agentic_retrieval", "request_count": 1},
            )


def _is_future_question(question: str, evidence: Sequence[Mapping[str, Any]]) -> bool:
    requested = [int(year) for year in YEAR_PATTERN.findall(question)]
    available: list[int] = []
    for item in evidence:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        report_date = str(metadata.get("report_date") or "")
        available.extend(int(year) for year in YEAR_PATTERN.findall(report_date))
    return bool(requested and available and max(requested) > max(available))


def validate_plan_scope(
    plan: AgenticPlan,
    original_question: str,
    evidence: Sequence[Mapping[str, Any]],
    ticker: str,
) -> None:
    if plan.action == "rewrite_search":
        missing_years = set(YEAR_PATTERN.findall(original_question)) - set(
            YEAR_PATTERN.findall(plan.rewritten_query or "")
        )
        if missing_years:
            raise GenerationValidationError(
                "agentic_rewrite_lost_period",
                "The rewritten query did not preserve every explicit period.",
                generation={"stage": "assessing_evidence"},
            )
        original_numbers = _numeric_facts(original_question)
        rewritten_numbers = _numeric_facts(plan.rewritten_query or "")
        if rewritten_numbers - original_numbers:
            raise GenerationValidationError(
                "agentic_rewrite_added_numeric_fact",
                "The rewritten query introduced a numeric fact not present in the question.",
                generation={"stage": "assessing_evidence"},
            )
    if plan.action != "expand_context":
        return
    anchor = next(
        (item for item in evidence if item.get("target_chunk_id") == plan.anchor_target_chunk_id),
        None,
    )
    if anchor is None:
        raise GenerationValidationError(
            "agentic_anchor_not_in_initial_results",
            "The context anchor was not in the initial retrieval results.",
            generation={"stage": "assessing_evidence"},
        )
    metadata = anchor.get("metadata") if isinstance(anchor.get("metadata"), Mapping) else {}
    anchor_ticker = str(anchor.get("ticker") or metadata.get("ticker") or "").upper()
    record_type = str(anchor.get("record_type") or metadata.get("record_type") or "")
    if anchor_ticker != ticker.upper():
        code = "agentic_anchor_crossed_bank"
    elif record_type != "text":
        code = "agentic_anchor_not_narrative"
    else:
        return
    raise GenerationValidationError(
        code,
        "The context anchor is outside the allowed narrative bank scope.",
        generation={"stage": "assessing_evidence"},
    )


def deduplicate_evidence(
    preferred: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*preferred, *existing]:
        target_id = str(item.get("target_chunk_id") or "")
        if not target_id or target_id in seen:
            continue
        seen.add(target_id)
        merged.append(dict(item))
        if len(merged) == limit:
            break
    return merged


class CanonicalContextExpander:
    """Bounded read of canonical neighbours; never accepts paths or arbitrary selectors."""

    def __init__(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        self._chunks = [dict(chunk) for chunk in chunks]
        self._index = {
            str(chunk.get("target_chunk_id")): index
            for index, chunk in enumerate(self._chunks)
            if chunk.get("target_chunk_id")
        }

    def expand(
        self,
        anchor_target_chunk_id: str,
        *,
        ticker: str,
        before: int = 1,
        after: int = 1,
    ) -> list[dict[str, Any]]:
        if not 0 <= before <= 3 or not 0 <= after <= 3:
            raise ValueError("Context windows must be between zero and three chunks.")
        index = self._index.get(anchor_target_chunk_id)
        if index is None:
            raise GenerationValidationError(
                "agentic_anchor_unknown",
                "The context anchor is not in the canonical corpus.",
                generation={"stage": "expanding_context"},
            )
        anchor = self._chunks[index]
        anchor_meta = anchor.get("metadata") if isinstance(anchor.get("metadata"), Mapping) else {}
        if str(anchor_meta.get("ticker") or "").upper() != ticker.upper():
            raise GenerationValidationError(
                "agentic_expansion_crossed_bank",
                "Context expansion attempted to cross a bank boundary.",
                generation={"stage": "expanding_context"},
            )
        if str(anchor.get("record_type") or anchor_meta.get("record_type") or "") != "text":
            return [self._as_evidence(anchor)]
        accession = str(anchor_meta.get("accession_number") or "")
        expanded = []
        for candidate_index in range(
            max(0, index - before), min(len(self._chunks), index + after + 1)
        ):
            candidate = self._chunks[candidate_index]
            metadata = (
                candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
            )
            if (
                str(metadata.get("ticker") or "").upper() == ticker.upper()
                and str(metadata.get("accession_number") or "") == accession
                and str(candidate.get("record_type") or metadata.get("record_type") or "") == "text"
            ):
                expanded.append(self._as_evidence(candidate))
        return expanded

    @staticmethod
    def _as_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(record)
        output["evidence"] = str(record.get("evidence") or record.get("document") or "")
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        output.setdefault("ticker", metadata.get("ticker"))
        output.setdefault("record_type", metadata.get("record_type"))
        output["retrieval_source"] = "canonical_context_expansion"
        return output
