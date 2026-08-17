from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from bankscope.generation.answer_generator import GPT51_MODEL_MARKERS, GenerationValidationError

ROUTER_PROMPT_VERSION = "agentic-rag-router-v1"
PLANNER_PROMPT_VERSION = "agentic-rag-evidence-planner-v1"
AGENT_STEP_PROMPT_VERSION = "agentic-rag-loop-v2"
VERIFIER_PROMPT_VERSION = "agentic-rag-verifier-v1"
AGENTIC_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_AGENT_MODEL_REQUESTS = 6
MAX_AGENT_TOOL_ACTIONS = 4
MAX_VERIFIER_REQUESTS = 2
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?%?")
TIER_ONE_PATTERN = re.compile(r"(?i)\btier\s+1\b")


def _numeric_facts(text: str) -> set[str]:
    """Extract numeric facts while allowing the canonical term ``Tier 1``."""
    without_tier_one = TIER_ONE_PATTERN.sub("tier one", text)
    return {
        value.rstrip("%").replace(",", ".")
        for value in NUMBER_PATTERN.findall(without_tier_one)
    }


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["general_chat", "domain_rag"]
    reason_code: str = Field(min_length=1, max_length=80)
    explanation: str = Field(min_length=1, max_length=500)


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
    value: RouteDecision | AgenticPlan | AgentStep | EvidenceVerdict
    latency_ms: float
    request_count: int = 1
    fallback: bool = False
    error_code: str | None = None


def _request_options(model: str) -> dict[str, Any]:
    options: dict[str, Any] = {"response_format": {"type": "json_object"}}
    normalized = model.strip().upper()
    if any(marker in normalized for marker in GPT51_MODEL_MARKERS):
        options["max_completion_tokens"] = 500
    else:
        options.update({"max_tokens": 500, "temperature": 0})
    return options


def _message_content(response: Any) -> tuple[str, str, str]:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return "", "", ""
    choice = choices[0]
    message = (
        choice.get("message")
        if isinstance(choice, Mapping)
        else getattr(choice, "message", None)
    )
    finish_reason = (
        str(choice.get("finish_reason") or "")
        if isinstance(choice, Mapping)
        else str(getattr(choice, "finish_reason", "") or "")
    )
    if isinstance(message, Mapping):
        return (
            str(message.get("content") or "").strip(),
            finish_reason,
            str(message.get("refusal") or ""),
        )
    return (
        str(getattr(message, "content", "") or "").strip(),
        finish_reason,
        str(getattr(message, "refusal", "") or ""),
    )


def _call_json_model(
    *, client: Any, model: str, system: str, payload: Mapping[str, Any]
) -> tuple[str, float, str, str]:
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
        **_request_options(model),
        timeout=AGENTIC_REQUEST_TIMEOUT_SECONDS,
    )
    text, finish_reason, refusal = _message_content(response)
    return text, (perf_counter() - started) * 1000, finish_reason, refusal


def route_question(question: str, *, client: Any, model: str) -> ModelDecision:
    """Route conservatively: every invalid or unavailable decision falls back to RAG."""
    schema = json.dumps(RouteDecision.model_json_schema(), separators=(",", ":"))
    system = (
        "Classify a BankScope chat request. general_chat is allowed only for greetings, "
        "farewells, thanks, or help about BankScope features. Every banking, finance, filing, "
        "company, regulation, or risk question must be domain_rag. Ambiguity must be domain_rag. "
        "Return only JSON matching this schema: " + schema
    )
    started = perf_counter()
    try:
        text, latency_ms, finish_reason, refusal = _call_json_model(
            client=client,
            model=model,
            system=system,
            payload={"prompt_version": ROUTER_PROMPT_VERSION, "question": question},
        )
        if finish_reason in {"length", "content_filter"} or refusal or not text:
            raise ValueError("unusable routing response")
        decision = RouteDecision.model_validate_json(text)
        return ModelDecision(decision, latency_ms)
    except Exception:
        return ModelDecision(
            RouteDecision(
                route="domain_rag",
                reason_code="routing_fallback_domain",
                explanation="Routing was unavailable or invalid; the safe domain route was used.",
            ),
            (perf_counter() - started) * 1000,
            fallback=True,
        )


def plan_evidence(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    ticker: str,
    client: Any,
    model: str,
) -> ModelDecision:
    schema = json.dumps(AgenticPlan.model_json_schema(), separators=(",", ":"))
    previews = []
    for item in evidence[:5]:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        text = str(item.get("evidence") or item.get("document") or "")
        previews.append(
            {
                "target_chunk_id": item.get("target_chunk_id"),
                "ticker": item.get("ticker") or metadata.get("ticker"),
                "record_type": item.get("record_type") or metadata.get("record_type"),
                "report_date": metadata.get("report_date"),
                "section_title": metadata.get("section_title"),
                "preview": text[:1_200],
            }
        )
    system = (
        "Assess evidence for one bank-filing question and choose exactly one action. generate when "
        "the supplied evidence is sufficient; rewrite_search only when a terminology-preserving "
        "rewrite is likely to retrieve the missing evidence; expand_context only when the answer "
        "likely sits immediately beside one supplied narrative chunk; abstain when this filing "
        "corpus cannot support the answer. A rewrite must preserve the question language, ticker, "
        "period, metric, variant, and qualifiers. It must not insert a numeric value or factual "
        "claim learned from the evidence. An anchor must be one of the supplied IDs. "
        "Return only JSON matching this schema: " + schema
    )
    try:
        text, latency_ms, finish_reason, refusal = _call_json_model(
            client=client,
            model=model,
            system=system,
            payload={
                "prompt_version": PLANNER_PROMPT_VERSION,
                "question": question,
                "ticker": ticker,
                "evidence": previews,
            },
        )
    except Exception as error:
        raise GenerationValidationError(
            "agentic_plan_request_failed",
            "OpenAI evidence assessment failed.",
            generation={"stage": "assessing_evidence", "model": model},
        ) from error
    metadata = {"stage": "assessing_evidence", "model": model, "latency_ms": latency_ms}
    if finish_reason == "length":
        code = "agentic_plan_truncated"
    elif finish_reason == "content_filter" or refusal:
        code = "agentic_plan_filtered"
    elif not text:
        code = "agentic_plan_empty"
    else:
        code = ""
    if code:
        raise GenerationValidationError(
            code,
            "OpenAI returned an unusable evidence plan.",
            generation=metadata,
        )
    try:
        plan = AgenticPlan.model_validate_json(text)
    except ValidationError as error:
        raise GenerationValidationError(
            "agentic_plan_invalid_schema",
            "OpenAI returned an invalid evidence plan.",
            generation=metadata,
        ) from error
    validate_plan_scope(plan, question, evidence, ticker)
    return ModelDecision(plan, latency_ms)


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
    schema = json.dumps(AGENT_STEP_ADAPTER.json_schema(), separators=(",", ":"))
    system = (
        "You control retrieval for exactly one bank filing. Choose one action. search_hybrid may "
        "translate or augment a non-English question with canonical English filing terminology, "
        "but must retain every explicit year and must never add a numeric fact. search_exact uses "
        "one to eight literal phrases (never regex) for exact names, acronyms, and user-provided "
        "numbers. read_context may use only a supplied narrative target ID. Assess only the part "
        "of a comparison that belongs to current_ticker; missing peer-bank evidence is irrelevant. "
        "Do not finish unsupported merely because initial top results miss: try search first "
        "unless the requested period is explicitly beyond the filing corpus. Finish sufficient "
        "only when all parts for current_ticker are supported. Return only JSON matching this "
        "schema: " + schema
    )
    try:
        text, latency_ms, finish_reason, refusal = _call_json_model(
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
    elif not text:
        code = "agentic_step_empty"
    else:
        code = ""
    if code:
        raise GenerationValidationError(
            code, "OpenAI returned an unusable agent step.", generation=metadata
        )
    try:
        step = AGENT_STEP_ADAPTER.validate_json(text)
    except ValidationError as error:
        raise GenerationValidationError(
            "agentic_step_invalid_schema",
            "OpenAI returned an invalid agent step.",
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
    schema = json.dumps(EvidenceVerdict.model_json_schema(), separators=(",", ":"))
    system = (
        "Independently verify whether the supplied evidence answers every part of the question for "
        "current_ticker only. Ignore missing peer-bank evidence. Use sufficient only for direct "
        "complete support, missing when another targeted search/read could fill named gaps, and "
        "unsupported only when the requested fact is outside the filing corpus. Supporting IDs "
        "must come from supplied evidence. Return only JSON matching this schema: " + schema
    )
    try:
        text, latency_ms, finish_reason, refusal = _call_json_model(
            client=client,
            model=model,
            system=system,
            payload={
                "prompt_version": VERIFIER_PROMPT_VERSION,
                "question": question,
                "current_ticker": ticker,
                "evidence": evidence_previews(evidence),
            },
        )
    except Exception as error:
        raise GenerationValidationError(
            "agentic_verdict_request_failed",
            "OpenAI evidence-verifier request failed.",
            generation={"stage": "verifying_evidence", "request_count": 1},
        ) from error
    metadata = {"stage": "verifying_evidence", "latency_ms": latency_ms, "request_count": 1}
    if finish_reason in {"length", "content_filter"} or refusal or not text:
        raise GenerationValidationError(
            "agentic_verdict_unusable",
            "OpenAI returned an unusable evidence verdict.",
            generation=metadata,
        )
    try:
        verdict = EvidenceVerdict.model_validate_json(text)
    except ValidationError as error:
        raise GenerationValidationError(
            "agentic_verdict_invalid_schema",
            "OpenAI returned an invalid evidence verdict.",
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
        if step.status == "unsupported" and state.tool_actions == 0 and not _is_future_question(
            state.question, state.evidence
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
                candidate.get("metadata")
                if isinstance(candidate.get("metadata"), Mapping)
                else {}
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
