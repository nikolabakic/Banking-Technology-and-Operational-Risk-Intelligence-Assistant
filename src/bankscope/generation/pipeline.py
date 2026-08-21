from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import numpy as np

from bankscope.generation.agentic import (
    MAX_AGENT_MODEL_REQUESTS,
    MAX_AGENT_TOOL_ACTIONS,
    MAX_VERIFIER_REQUESTS,
    AgentState,
    CanonicalContextExpander,
    EvidenceVerdict,
    FinishStep,
    ReadContextStep,
    SearchExactStep,
    SearchHybridStep,
    deduplicate_evidence,
    request_agent_step,
    validate_agent_step,
    verify_evidence,
)
from bankscope.generation.answer_generator import (
    CITATION_PATTERN,
    GenerationValidationError,
    generate_answer,
    render_unsupported_answer,
)
from bankscope.generation.comparison_generator import synthesize_comparison
from bankscope.generation.conversation import (
    ClarificationArgs,
    ConversationGraph,
    DeclineOutOfScopeArgs,
    DirectResponseArgs,
    ResearchFilingsArgs,
    WebResearchArgs,
    render_capability_answer,
    render_out_of_scope_answer,
    render_web_unavailable_answer,
)
from bankscope.generation.memory import summarize_conversation
from bankscope.generation.query_planner import (
    build_bank_subquestion,
    build_retrieval_queries,
    remove_untrusted_numeric_facts,
    round_robin_evidence,
    validate_contextualized_rewrite,
)
from bankscope.io import read_jsonl, sha256_file
from bankscope.retrieval.glossary_locators import validate_glossary_locators
from bankscope.retrieval.hybrid_retriever import HybridRetriever
from bankscope.retrieval.mixed_retriever import MixedRetriever
from bankscope.retrieval.qdrant_retriever import (
    DEFAULT_COLLECTION_NAME,
    QdrantRetriever,
    load_qdrant_manifest,
)
from bankscope.sec.bank_resolver import BankResolution, resolve_bank
from bankscope.sec.company_registry import load_bank_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = PROJECT_ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = PROJECT_ROOT / "data/processed/tables.jsonl"
DEFAULT_GLOSSARY_LOCATORS = PROJECT_ROOT / "data/processed/lexical_glossary_locators_v1.jsonl"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data/processed/qdrant"
DEFAULT_QDRANT_MANIFEST = PROJECT_ROOT / "data/processed/qdrant_manifest.json"
DEFAULT_BANK_REGISTRY = PROJECT_ROOT / "config/banks.yaml"


class QueryEncoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class SentenceTransformerQueryEncoder:
    """Load one SentenceTransformer and reuse it for every question in a run."""

    def __init__(self, model_name: str, model_revision: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_options: dict[str, Any] = {"local_files_only": True}
        if model_revision.strip().lower() != "unknown":
            model_options["revision"] = model_revision
        self.model = SentenceTransformer(model_name, **model_options)

    def encode(self, text: str) -> np.ndarray:
        vector = self.model.encode_query(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32)


@dataclass(frozen=True)
class AnswerRun:
    output: dict[str, Any]
    evidence: list[dict[str, Any]]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    diagnostics: dict[str, Any] | None = None
    stage_trace: tuple[dict[str, Any], ...] = ()
    agentic_plans: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RetrievalRun:
    evidence: list[dict[str, Any]]
    ticker: str
    status: str
    initial_evidence_count: int
    embedding_latency_ms: float
    retrieval_latency_ms: float
    orchestration_latency_ms: float
    model_request_count: int
    diagnostics: dict[str, Any]
    stage_trace: tuple[dict[str, Any], ...] = ()
    agentic_plans: tuple[dict[str, Any], ...] = ()


class BankAnswerPipeline:
    """Reusable grounded-answer pipeline for single-bank and comparison questions."""

    def __init__(
        self,
        *,
        retriever: Any,
        query_encoder: QueryEncoder,
        client: Any,
        generation_model: str,
        temperature: float = 0,
        close_callback: Any | None = None,
        dense_model: dict[str, str] | None = None,
        bank_names: dict[str, str] | None = None,
        bank_aliases: dict[str, tuple[str, ...]] | None = None,
        agentic_rag_enabled: bool = False,
        context_expander: CanonicalContextExpander | None = None,
        conversation_model: Any | None = None,
        conversation_router_backend: Literal["langgraph", "legacy"] = "langgraph",
    ) -> None:
        if not generation_model.strip():
            raise ValueError("generation_model cannot be empty.")
        self.retriever = retriever
        self.query_encoder = query_encoder
        self.client = client
        self.generation_model = generation_model
        self.temperature = temperature
        self._close_callback = close_callback
        self._closed = False
        self.dense_model = dict(dense_model or {})
        self.bank_names = {
            ticker.strip().upper(): name.strip()
            for ticker, name in (bank_names or {}).items()
            if ticker.strip() and name.strip()
        }
        self.bank_aliases = {
            ticker.strip().upper(): tuple(alias.strip() for alias in aliases if alias.strip())
            for ticker, aliases in (bank_aliases or {}).items()
            if ticker.strip()
        }
        self.agentic_rag_enabled = agentic_rag_enabled
        self.context_expander = context_expander
        self.conversation_graph = ConversationGraph(
            client=client,
            model=generation_model,
            bank_names=self.bank_names,
            bank_aliases=self.bank_aliases,
            chat_model=conversation_model,
            backend=conversation_router_backend,
        )
        self._research_handlers: dict[str, Callable[..., AnswerRun | ResearchFilingsArgs]] = {
            "filing_research": self._filing_research_handler,
            "web_research": self._web_unavailable_run,
        }

    @classmethod
    def from_paths(
        cls,
        *,
        client: Any,
        generation_model: str,
        temperature: float = 0,
        chunks_path: str | Path = DEFAULT_CHUNKS,
        tables_path: str | Path = DEFAULT_TABLES,
        glossary_locators_path: str | Path = DEFAULT_GLOSSARY_LOCATORS,
        qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
        qdrant_manifest_path: str | Path = DEFAULT_QDRANT_MANIFEST,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        query_encoder: QueryEncoder | None = None,
        bank_registry_path: str | Path = DEFAULT_BANK_REGISTRY,
        agentic_rag_enabled: bool = False,
        conversation_model: Any | None = None,
        conversation_router_backend: Literal["langgraph", "legacy"] = "langgraph",
    ) -> BankAnswerPipeline:
        chunks_path = Path(chunks_path)
        tables_path = Path(tables_path)
        glossary_locators_path = Path(glossary_locators_path)
        manifest_path = Path(qdrant_manifest_path)
        manifest = load_qdrant_manifest(manifest_path)
        dense_model = manifest.get("dense_model")
        if not isinstance(dense_model, dict):
            raise ValueError("Qdrant manifest has no valid dense_model.")
        model_name = str(dense_model.get("name") or "")
        model_revision = str(dense_model.get("revision") or "")
        if not model_name or not model_revision:
            raise ValueError("Qdrant manifest has incomplete dense model metadata.")

        expected_chunks_hash = str(
            manifest.get("sources", {}).get("chunks", {}).get("sha256") or ""
        )
        if expected_chunks_hash != sha256_file(chunks_path):
            raise ValueError("chunks.jsonl does not match the Qdrant manifest.")

        tables = read_jsonl(tables_path)
        records = read_jsonl(chunks_path)
        glossary_locators = read_jsonl(glossary_locators_path)
        validate_glossary_locators(glossary_locators, records, tables)
        registry = load_bank_registry(bank_registry_path)
        enabled_banks = [bank for bank in registry.banks if bank.enabled]
        bank_names = {bank.ticker: bank.legal_name for bank in enabled_banks}
        bank_aliases = {bank.ticker: bank.aliases for bank in enabled_banks}
        qdrant = QdrantRetriever(
            qdrant_path,
            tables,
            manifest_path=manifest_path,
            collection_name=collection_name,
            tables_path=tables_path,
        )
        try:
            # Open the exclusive local store before loading the comparatively slow
            # embedding model, so a second BankScope process fails immediately.
            encoder = query_encoder or SentenceTransformerQueryEncoder(model_name, model_revision)
            retriever = MixedRetriever(
                qdrant,
                HybridRetriever(records, tables=tables, lexical_records=glossary_locators),
            )
            return cls(
                retriever=retriever,
                query_encoder=encoder,
                client=client,
                generation_model=generation_model,
                temperature=temperature,
                close_callback=qdrant.close,
                dense_model={"name": model_name, "revision": model_revision},
                bank_names=bank_names,
                bank_aliases=bank_aliases,
                agentic_rag_enabled=agentic_rag_enabled,
                context_expander=CanonicalContextExpander(records),
                conversation_model=conversation_model,
                conversation_router_backend=conversation_router_backend,
            )
        except Exception:
            qdrant.close()
            raise

    def close(self) -> None:
        if not self._closed and self._close_callback is not None:
            self._close_callback()
        self._closed = True

    def __enter__(self) -> BankAnswerPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def compact_conversation(
        self,
        existing_summary: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> str:
        """Compact older thread turns with the configured generation model."""

        return summarize_conversation(
            existing_summary,
            messages,
            client=self.client,
            model=self.generation_model,
        )

    def _diagnostics(
        self,
        *,
        route: str,
        outcome: str,
        stages: Sequence[Mapping[str, Any]],
        initial_evidence_count: int,
        final_evidence_count: int,
        model_request_count: int,
        bank_plans: Sequence[Mapping[str, Any]] = (),
        output: Mapping[str, Any] | None = None,
        failed_stage: str | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        action_budget_ok = all(
            int(plan.get("tool_action_count") or plan.get("additional_action_count") or 0)
            <= MAX_AGENT_TOOL_ACTIONS
            for plan in bank_plans
        )
        orchestration_budget = MAX_AGENT_MODEL_REQUESTS * max(1, len(bank_plans))
        # Routing/contextualization and answer generation are outside the per-bank retrieval loop.
        request_budget = orchestration_budget + 2 + 2 * max(1, len(bank_plans))
        if not self.agentic_rag_enabled:
            request_budget = 6
        bank_isolation_ok = all(bool(plan.get("bank_isolation_ok", True)) for plan in bank_plans)
        citations_ok = True
        if output is not None and route == "domain_rag":
            if output.get("status") == "supported":
                citations_ok = bool(output.get("citations"))
            for result in output.get("bank_results") or []:
                if result.get("status") == "supported" and not result.get("citations"):
                    citations_ok = False
        checks = {
            "pipeline_completed": failed_stage is None,
            "plan_schema": all(bool(plan.get("schema_valid", True)) for plan in bank_plans),
            "query_preservation": all(
                bool(plan.get("query_preservation_ok", True)) for plan in bank_plans
            ),
            "citation_contract": citations_ok,
            "bank_isolation": bank_isolation_ok,
            "action_budget": action_budget_ok,
            "request_budget": 0 <= model_request_count <= request_budget,
        }
        routing_stage = next(
            (dict(stage) for stage in stages if str(stage.get("stage")) == "routing"),
            {},
        )
        presentation_guidance = (
            (output or {}).get("contextualization", {}).get("presentation_guidance")
            if isinstance((output or {}).get("contextualization"), Mapping)
            else None
        )
        citation_source = "none"
        if output and output.get("citations"):
            citation_source = (
                "previous_grounded_answer"
                if output.get("dialog_act") == "contextual_transform"
                else "retrieved_evidence"
            )
        return {
            "route": route,
            "router_backend": routing_stage.get("router_backend"),
            "route_action": routing_stage.get("route_action"),
            "route_confidence": routing_stage.get("route_confidence"),
            "route_reason": routing_stage.get("route_reason"),
            "graph_nodes": list(routing_stage.get("graph_nodes") or []),
            "source_policy": routing_stage.get("source_policy"),
            "context_message_count": routing_stage.get("context_message_count", 0),
            "context_estimated_tokens": routing_stage.get("context_estimated_tokens", 0),
            "summary_used": bool(routing_stage.get("summary_used")),
            "summary_updated": bool(routing_stage.get("summary_updated")),
            "presentation_guidance": presentation_guidance,
            "citation_source": citation_source,
            "agentic_rag_enabled": self.agentic_rag_enabled,
            "outcome": outcome,
            "failed_stage": failed_stage,
            "error_code": error_code,
            "stages": [dict(stage) for stage in stages],
            "initial_evidence_count": initial_evidence_count,
            "final_evidence_count": final_evidence_count,
            "model_request_count": model_request_count,
            "bank_plans": [dict(plan) for plan in bank_plans],
            "quality_gate": {"passed": all(checks.values()), "checks": checks},
        }

    def _run_agentic_loop(
        self,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        *,
        ticker: str,
        record_type: str | None,
        candidate_k: int,
        rrf_k: int,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float, float, int, str]:
        baseline_evidence = [dict(item) for item in evidence]
        state = AgentState(
            ticker=ticker,
            question=question,
            evidence=[dict(item) for item in evidence],
        )
        embedding_ms = 0.0
        retrieval_ms = 0.0
        orchestration_ms = 0.0
        final_status = "sufficient"
        schema_valid = True

        while state.model_requests < MAX_AGENT_MODEL_REQUESTS:
            request_recorded = False
            if on_progress is not None:
                on_progress(
                    "assessing_evidence",
                    {
                        "message": f"Assessing evidence for {ticker}...",
                        "ticker": ticker,
                        "step": state.model_requests + 1,
                    },
                )
            try:
                decision = request_agent_step(
                    state, client=self.client, model=self.generation_model
                )
                state.model_requests += decision.request_count
                orchestration_ms += decision.latency_ms
                request_recorded = True
                step = decision.value
                if not isinstance(
                    step, (SearchHybridStep, SearchExactStep, ReadContextStep, FinishStep)
                ):
                    raise RuntimeError("Agent returned an unexpected step type.")
                validate_agent_step(step, state)
                state.consecutive_schema_failures = 0
                if state.tool_actions >= MAX_AGENT_TOOL_ACTIONS and not isinstance(
                    step, FinishStep
                ):
                    step = FinishStep(
                        action="finish",
                        status="sufficient" if state.evidence else "unsupported",
                        reason="The bounded retrieval tool budget is exhausted.",
                        supporting_target_chunk_ids=[
                            str(item.get("target_chunk_id") or "")
                            for item in state.evidence[:10]
                            if item.get("target_chunk_id")
                        ],
                    )
            except GenerationValidationError as error:
                if not request_recorded:
                    state.model_requests += int(error.generation.get("request_count") or 1)
                    orchestration_ms += float(error.generation.get("latency_ms") or 0.0)
                state.consecutive_schema_failures += 1
                schema_valid = False
                state.trace.append(
                    {
                        "action": "invalid_step",
                        "error_code": error.code,
                        "result": "Retry with the strict action schema.",
                    }
                )
                if state.consecutive_schema_failures >= 2:
                    final_status = "sufficient" if state.evidence else "unsupported"
                    break
                continue

            if isinstance(step, SearchHybridStep):
                key = "hybrid:" + " ".join(step.query.casefold().split())
                if key in state.executed_searches:
                    state.trace.append(
                        {"action": step.action, "query": step.query, "result": "No new evidence"}
                    )
                    state.tool_actions += 1
                    continue
                state.executed_searches.add(key)
                state.tool_actions += 1
                try:
                    started = perf_counter()
                    vector = self.query_encoder.encode(step.query)
                    embedding_ms += (perf_counter() - started) * 1000
                    started = perf_counter()
                    revised = self.retriever.search_hybrid(
                        step.query,
                        vector,
                        limit=5,
                        candidate_k=candidate_k,
                        rrf_k=rrf_k,
                        ticker=ticker,
                        record_type=record_type,
                    )
                except Exception as error:
                    state.trace.append(
                        {
                            "action": step.action,
                            "query": step.query,
                            "result": f"Error: {type(error).__name__}",
                        }
                    )
                    continue
                elapsed = (perf_counter() - started) * 1000
                retrieval_ms += elapsed
                before_ids = {item.get("target_chunk_id") for item in state.evidence}
                state.evidence = deduplicate_evidence(revised, state.evidence, limit=10)
                new_ids = [
                    item.get("target_chunk_id")
                    for item in state.evidence
                    if item.get("target_chunk_id") not in before_ids
                ]
                state.trace.append(
                    {
                        "action": step.action,
                        "query": step.query,
                        "reason": step.reason,
                        "new_target_chunk_ids": new_ids,
                        "latency_ms": elapsed,
                    }
                )
            elif isinstance(step, SearchExactStep):
                key = "exact:" + "|".join(sorted(term.casefold().strip() for term in step.terms))
                if key in state.executed_searches:
                    state.trace.append(
                        {"action": step.action, "terms": step.terms, "result": "No new evidence"}
                    )
                    state.tool_actions += 1
                    continue
                state.executed_searches.add(key)
                state.tool_actions += 1
                try:
                    started = perf_counter()
                    exact = self.retriever.search_exact(
                        step.terms,
                        limit=20,
                        ticker=ticker,
                        record_type=record_type,
                    )
                except Exception as error:
                    state.trace.append(
                        {
                            "action": step.action,
                            "terms": step.terms,
                            "result": f"Error: {type(error).__name__}",
                        }
                    )
                    continue
                elapsed = (perf_counter() - started) * 1000
                retrieval_ms += elapsed
                before_ids = {item.get("target_chunk_id") for item in state.evidence}
                state.evidence = deduplicate_evidence(exact, state.evidence, limit=10)
                new_ids = [
                    item.get("target_chunk_id")
                    for item in state.evidence
                    if item.get("target_chunk_id") not in before_ids
                ]
                state.trace.append(
                    {
                        "action": step.action,
                        "terms": step.terms,
                        "reason": step.reason,
                        "new_target_chunk_ids": new_ids,
                        "latency_ms": elapsed,
                    }
                )
            elif isinstance(step, ReadContextStep):
                window = (step.anchor_target_chunk_id, step.before, step.after)
                state.tool_actions += 1
                if window in state.read_windows:
                    state.trace.append(
                        {
                            "action": step.action,
                            "anchor_target_chunk_id": step.anchor_target_chunk_id,
                            "result": "No new evidence",
                        }
                    )
                    continue
                state.read_windows.add(window)
                if self.context_expander is None:
                    state.trace.append(
                        {"action": step.action, "result": "Error: context reader unavailable"}
                    )
                    continue
                try:
                    started = perf_counter()
                    expanded = self.context_expander.expand(
                        step.anchor_target_chunk_id,
                        ticker=ticker,
                        before=step.before,
                        after=step.after,
                    )
                except Exception as error:
                    state.trace.append(
                        {
                            "action": step.action,
                            "anchor_target_chunk_id": step.anchor_target_chunk_id,
                            "result": f"Error: {type(error).__name__}",
                        }
                    )
                    continue
                elapsed = (perf_counter() - started) * 1000
                retrieval_ms += elapsed
                before_ids = {item.get("target_chunk_id") for item in state.evidence}
                state.evidence = deduplicate_evidence(expanded, state.evidence, limit=10)
                new_ids = [
                    item.get("target_chunk_id")
                    for item in state.evidence
                    if item.get("target_chunk_id") not in before_ids
                ]
                state.trace.append(
                    {
                        "action": step.action,
                        "anchor_target_chunk_id": step.anchor_target_chunk_id,
                        "before": step.before,
                        "after": step.after,
                        "reason": step.reason,
                        "new_target_chunk_ids": new_ids,
                        "latency_ms": elapsed,
                    }
                )
            else:
                if (
                    state.verifier_requests >= MAX_VERIFIER_REQUESTS
                    or state.remaining_model_requests == 0
                ):
                    final_status = step.status
                    state.trace.append(step.model_dump())
                    break
                try:
                    verdict_decision = verify_evidence(
                        question,
                        state.evidence,
                        ticker=ticker,
                        client=self.client,
                        model=self.generation_model,
                    )
                    state.model_requests += verdict_decision.request_count
                    state.verifier_requests += 1
                    orchestration_ms += verdict_decision.latency_ms
                    verdict = verdict_decision.value
                    if not isinstance(verdict, EvidenceVerdict):
                        raise RuntimeError("Verifier returned an unexpected decision type.")
                    state.trace.append({"action": "verify_evidence", **verdict.model_dump()})
                    if verdict.status == "missing" and state.remaining_tool_actions > 0:
                        state.verifier_feedback.extend(verdict.missing_aspects)
                        continue
                    final_status = "sufficient" if verdict.status == "sufficient" else "unsupported"
                    break
                except GenerationValidationError as error:
                    state.model_requests += int(error.generation.get("request_count") or 1)
                    state.verifier_requests += 1
                    orchestration_ms += float(error.generation.get("latency_ms") or 0.0)
                    schema_valid = False
                    state.trace.append({"action": "invalid_verdict", "error_code": error.code})
                    final_status = "sufficient" if state.evidence else "unsupported"
                    break

            if state.tool_actions >= MAX_AGENT_TOOL_ACTIONS:
                # The next model call must finish; make that constraint visible to the agent.
                state.verifier_feedback.append(
                    "Tool budget exhausted; finish from current evidence."
                )

        payload = {
            "ticker": ticker,
            "action": "agentic_loop",
            "final_status": final_status,
            "steps": state.trace,
            "effective_queries": sorted(state.executed_searches),
            "read_windows": [list(window) for window in sorted(state.read_windows)],
            "model_request_count": state.model_requests,
            "tool_action_count": state.tool_actions,
            "verifier_request_count": state.verifier_requests,
            "additional_action_count": state.tool_actions,
            "assessment_latency_ms": orchestration_ms,
            "schema_valid": schema_valid,
            "query_preservation_ok": not any(
                trace.get("error_code")
                in {
                    "agentic_search_lost_period",
                    "agentic_search_added_numeric_fact",
                    "agentic_exact_added_numeric_fact",
                }
                for trace in state.trace
            ),
            "bank_isolation_ok": True,
            "fallback": not schema_valid,
        }
        # Agentic retrieval is corrective and additive. A model verdict must never erase the
        # validated baseline path; generation remains the final groundedness/abstention gate.
        final_evidence = deduplicate_evidence(baseline_evidence, state.evidence, limit=10)
        return (
            final_evidence,
            payload,
            embedding_ms,
            retrieval_ms + orchestration_ms,
            state.model_requests,
            final_status,
        )

    def retrieve_evidence(
        self,
        question: str,
        *,
        ticker: str,
        original_question: str | None = None,
        record_type: str | None = None,
        limit: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> RetrievalRun:
        """Retrieve bank-scoped evidence without invoking answer generation."""
        if self._closed:
            raise RuntimeError("The answer pipeline is closed.")
        question = question.strip()
        normalized_ticker = ticker.strip().upper()
        if not question or not normalized_ticker:
            raise ValueError("Question and ticker cannot be empty.")
        if limit <= 0 or candidate_k < limit or rrf_k <= 0:
            raise ValueError("Invalid retrieval limits.")
        stages: list[dict[str, Any]] = []
        bank_name = self.bank_names.get(normalized_ticker, normalized_ticker)
        retrieval_queries = build_retrieval_queries(
            question,
            ticker=normalized_ticker,
            bank_name=bank_name,
            original_question=original_question,
        )
        if on_progress is not None:
            on_progress(
                "embedding",
                {
                    "message": (
                        "Encoding section-diverse summary searches..."
                        if len(retrieval_queries) > 1
                        else "Encoding the question..."
                    ),
                    "ticker": normalized_ticker,
                    "query_count": len(retrieval_queries),
                },
            )
        embedding_ms = 0.0
        vectors: list[np.ndarray] = []
        for retrieval_query in retrieval_queries:
            started = perf_counter()
            vectors.append(self.query_encoder.encode(retrieval_query))
            embedding_ms += (perf_counter() - started) * 1000
        stages.append({"stage": "embedding", "status": "completed", "latency_ms": embedding_ms})
        if on_progress is not None:
            on_progress(
                "retrieving",
                {
                    "message": (
                        "Searching key sections of the indexed filing..."
                        if len(retrieval_queries) > 1
                        else "Searching indexed filings..."
                    ),
                    "ticker": normalized_ticker,
                    "query_count": len(retrieval_queries),
                },
            )
        retrieval_ms = 0.0
        result_groups: list[list[dict[str, Any]]] = []
        per_query_limit = limit if len(retrieval_queries) == 1 else 2
        for retrieval_query, vector in zip(retrieval_queries, vectors, strict=True):
            started = perf_counter()
            results = self.retriever.search_hybrid(
                retrieval_query,
                vector,
                limit=per_query_limit,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                ticker=normalized_ticker,
                record_type=record_type,
            )
            retrieval_ms += (perf_counter() - started) * 1000
            result_groups.append([dict(item) for item in results])
        evidence_limit = limit if len(retrieval_queries) == 1 else max(limit, 10)
        evidence = [
            dict(item) for item in round_robin_evidence(result_groups, limit=evidence_limit)
        ]
        initial_count = len(evidence)
        stages.append({"stage": "retrieving", "status": "completed", "latency_ms": retrieval_ms})
        plans: list[dict[str, Any]] = []
        requests = 0
        orchestration_ms = 0.0
        status = "sufficient" if evidence else "unsupported"
        if self.agentic_rag_enabled:
            evidence, plan, extra_embedding, extra_work, requests, status = self._run_agentic_loop(
                question,
                evidence,
                ticker=normalized_ticker,
                record_type=record_type,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                on_progress=on_progress,
            )
            plans.append(plan)
            embedding_ms += extra_embedding
            orchestration_ms = float(plan.get("assessment_latency_ms") or 0.0)
            retrieval_ms += max(0.0, extra_work - orchestration_ms)
            stages.append(
                {
                    "stage": "agentic_retrieval",
                    "status": "completed",
                    "latency_ms": extra_work,
                    "ticker": normalized_ticker,
                }
            )
        diagnostics = {
            "ticker": normalized_ticker,
            "status": status,
            "queries": list(retrieval_queries),
            "initial_evidence_count": initial_count,
            "final_evidence_count": len(evidence),
            "model_request_count": requests,
            "bank_plans": plans,
            "quality_gate": {
                "passed": all(
                    bool(plan.get(key, True))
                    for plan in plans
                    for key in ("query_preservation_ok", "bank_isolation_ok")
                )
                and all(
                    int(plan.get("tool_action_count") or 0) <= MAX_AGENT_TOOL_ACTIONS
                    for plan in plans
                )
                and all(bool(plan.get("schema_valid", True)) for plan in plans)
                and requests <= MAX_AGENT_MODEL_REQUESTS,
            },
        }
        return RetrievalRun(
            evidence=evidence,
            ticker=normalized_ticker,
            status=status,
            initial_evidence_count=initial_count,
            embedding_latency_ms=embedding_ms,
            retrieval_latency_ms=retrieval_ms,
            orchestration_latency_ms=orchestration_ms,
            model_request_count=requests,
            diagnostics=diagnostics,
            stage_trace=tuple(stages),
            agentic_plans=tuple(plans),
        )

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        tickers: Sequence[str] = (),
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        conversation_summary: str = "",
        previous_answer: Mapping[str, Any] | None = None,
        conversation_metadata: Mapping[str, Any] | None = None,
        record_type: str | None = None,
        limit: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> AnswerRun:
        if self._closed:
            raise RuntimeError("The answer pipeline is closed.")
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if candidate_k < limit:
            raise ValueError("candidate_k must be at least limit.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        stages: list[dict[str, Any]] = []
        route = "domain_rag"
        orchestration_request_count = 0
        history = list(conversation_history or ())
        chat_mode = conversation_history is not None
        explicit_resolution = resolve_bank(
            question,
            bank_names=self.bank_names,
            bank_aliases=self.bank_aliases,
        )

        available_history_turns = len(history) // 2
        selected_history: list[Mapping[str, str]] = history
        presentation_guidance: str | None = None

        standalone_question = question
        contextualization_model: str | None = None
        contextualization_latency_ms = 0.0
        contextualization_fallback = False
        contextualization_error_code: str | None = None

        if chat_mode:
            if on_progress is not None:
                on_progress("routing", {"message": "Routing the request..."})
            session_scope = list(
                dict.fromkeys(
                    value.strip().upper()
                    for value in (tickers or ((ticker,) if ticker else ()))
                    if value and value.strip()
                )
            )
            decision = self.conversation_graph.route(
                question,
                selected_history,
                session_tickers=session_scope,
                conversation_summary=conversation_summary,
                previous_answer=previous_answer,
            )
            orchestration_request_count += decision.request_count
            action = decision.action
            stages.append(
                {
                    "stage": "routing",
                    "status": "completed",
                    "latency_ms": decision.latency_ms,
                    "source": "conversation_graph",
                    "action": type(action).__name__,
                    "router_backend": decision.router_backend,
                    "route_action": decision.route_action,
                    "route_confidence": decision.confidence,
                    "route_reason": decision.reason,
                    "graph_nodes": list(decision.graph_nodes),
                    "source_policy": decision.source_policy,
                    "fallback": decision.fallback,
                    "error_code": decision.error_code,
                    "context_message_count": len(selected_history),
                    "context_estimated_tokens": int(
                        (conversation_metadata or {}).get("estimated_tokens") or 0
                    ),
                    "summary_used": bool(conversation_summary),
                    "summary_updated": bool(
                        (conversation_metadata or {}).get("summary_updated")
                    ),
                }
            )
            if isinstance(action, (ResearchFilingsArgs, WebResearchArgs)):
                handled = self._research_handlers[decision.route_action](
                    question,
                    action,
                    stages=stages,
                    model_request_count=orchestration_request_count,
                    history_turns=len(selected_history) // 2,
                    available_history_turns=available_history_turns,
                    fallback=decision.fallback,
                    error_code=decision.error_code,
                    history=selected_history,
                    session_tickers=session_scope,
                )
                if isinstance(handled, AnswerRun):
                    return handled
                action = handled
            if isinstance(action, DirectResponseArgs):
                if action.category == "capability":
                    action = DirectResponseArgs(
                        answer=render_capability_answer(question, self.bank_names),
                        category="capability",
                    )
                elif action.category == "general_explanation" and explicit_resolution.tickers:
                    action = ResearchFilingsArgs(
                        search_question=question,
                        reason="Bank-specific context requires grounded filing research.",
                    )
                if isinstance(action, DirectResponseArgs):
                    return self._direct_conversation_run(
                        question,
                        action,
                        stages=stages,
                        model_request_count=orchestration_request_count,
                        history_turns=len(selected_history) // 2,
                        available_history_turns=available_history_turns,
                        fallback=decision.fallback,
                        error_code=decision.error_code,
                        source_answer=previous_answer,
                    )
            if isinstance(action, ClarificationArgs):
                return self._clarification_run(
                    question,
                    action,
                    stages=stages,
                    model_request_count=orchestration_request_count,
                    history_turns=len(selected_history) // 2,
                    available_history_turns=available_history_turns,
                    fallback=decision.fallback,
                    error_code=decision.error_code,
                )
            if isinstance(action, DeclineOutOfScopeArgs):
                return self._out_of_scope_run(
                    question,
                    stages=stages,
                    model_request_count=orchestration_request_count,
                    history_turns=len(selected_history) // 2,
                    available_history_turns=available_history_turns,
                    fallback=decision.fallback,
                    error_code=decision.error_code,
                )
            if not isinstance(action, ResearchFilingsArgs):
                raise RuntimeError("Conversation routing returned an unexpected action.")
            standalone_question = action.search_question
            presentation_guidance = action.presentation_guidance
            contextualization_model = self.generation_model
            contextualization_latency_ms = decision.latency_ms
            contextualization_fallback = decision.fallback
            contextualization_error_code = decision.error_code
            if on_progress is not None:
                on_progress(
                    "contextualizing",
                    {"message": "Preparing the filing search..."},
                )
            user_history = [
                str(message.get("content") or "")
                for message in selected_history
                if message.get("role") == "user"
            ]
            standalone_question, removed_assistant_number = remove_untrusted_numeric_facts(
                standalone_question,
                current_question=question,
                allowed_user_context=user_history,
            )
            if removed_assistant_number:
                contextualization_fallback = True
                contextualization_error_code = "contextualization_removed_assistant_numeric_fact"
            try:
                validate_contextualized_rewrite(
                    question,
                    standalone_question,
                    allowed_user_context=user_history,
                )
                planned_resolution = resolve_bank(
                    standalone_question,
                    bank_names=self.bank_names,
                    bank_aliases=self.bank_aliases,
                    session_ticker=ticker,
                    session_tickers=tickers,
                )
                explicit_scope = set(explicit_resolution.tickers)
                planned_scope = set(planned_resolution.tickers)
                history_scope = {
                    history_ticker
                    for message in selected_history
                    if message.get("role") == "user"
                    for history_ticker in resolve_bank(
                        str(message.get("content") or ""),
                        bank_names=self.bank_names,
                        bank_aliases=self.bank_aliases,
                    ).tickers
                }
                allowed_scope = explicit_scope | set(session_scope) | history_scope
                if explicit_scope - planned_scope:
                    raise GenerationValidationError(
                        "contextualization_lost_bank_scope",
                        "The filing search dropped a bank named in the current question.",
                        generation={"stage": "contextualizing"},
                    )
                if planned_scope - allowed_scope:
                    raise GenerationValidationError(
                        "contextualization_added_bank_scope",
                        "The filing search introduced a bank outside the conversation scope.",
                        generation={"stage": "contextualizing"},
                    )
            except GenerationValidationError as error:
                # Search-query planning is fail-open. The original message remains authoritative;
                # grounded answer generation and citation validation remain fail-closed.
                standalone_question = question
                contextualization_fallback = True
                contextualization_error_code = error.code
            stages.append(
                {
                    "stage": "contextualizing",
                    "status": "completed",
                    "latency_ms": contextualization_latency_ms,
                    "fallback": contextualization_fallback,
                    "error_code": contextualization_error_code,
                }
            )
        contextualization_applied = standalone_question != question
        contextualization_payload = {
            "applied": contextualization_applied,
            "history_turns": len(selected_history) // 2,
            "available_history_turns": available_history_turns,
            "standalone_question": standalone_question,
            "model": contextualization_model,
            "latency_ms": contextualization_latency_ms,
            "source": "conversation_graph" if chat_mode else "not_requested",
            "fallback": contextualization_fallback,
            "error_code": contextualization_error_code,
            "conversation_summary_used": bool(conversation_summary),
            "presentation_guidance": presentation_guidance,
            "skip_reason": (
                None
                if contextualization_applied
                else "no_history"
                if not history
                else "original_question_fallback"
                if contextualization_fallback
                else "current_question_is_standalone"
            ),
        }

        if on_progress is not None:
            on_progress("resolving_bank", {"message": "Identifying the bank..."})
        resolution = resolve_bank(
            standalone_question,
            bank_names=self.bank_names,
            bank_aliases=self.bank_aliases,
            session_ticker=ticker,
            session_tickers=tickers,
        )
        if contextualization_applied:
            explicit_scope = set(explicit_resolution.tickers)
            final_scope = set(resolution.tickers)
            session_scope = {
                value.strip().upper()
                for value in (tickers or ((ticker,) if ticker else ()))
                if value and value.strip()
            }
            history_scope = {
                history_ticker
                for message in selected_history
                if message.get("role") == "user"
                for history_ticker in resolve_bank(
                    str(message.get("content") or ""),
                    bank_names=self.bank_names,
                    bank_aliases=self.bank_aliases,
                ).tickers
            }
            if explicit_scope - final_scope:
                raise GenerationValidationError(
                    "contextualization_lost_bank_scope",
                    "The standalone question dropped a bank named in the current question.",
                    generation={"stage": "contextualizing"},
                )
            allowed_scope = explicit_scope | session_scope | history_scope
            if final_scope - allowed_scope:
                raise GenerationValidationError(
                    "contextualization_added_bank_scope",
                    "The standalone question introduced a bank outside the current thread scope.",
                    generation={"stage": "contextualizing"},
                )
        stages.append({"stage": "resolving_bank", "status": "completed", "latency_ms": 0.0})
        if resolution.status in {"missing", "too_many"}:
            return self._ambiguous_bank_run(
                question,
                resolution,
                contextualization_payload,
                stages=stages,
                model_request_count=orchestration_request_count,
            )
        if resolution.status == "multiple":
            return self._comparison_run(
                question,
                standalone_question,
                resolution,
                contextualization_payload,
                record_type=record_type,
                limit=limit,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                on_progress=on_progress,
                stages=stages,
                orchestration_request_count=orchestration_request_count,
                presentation_guidance=presentation_guidance,
            )
        resolved_ticker = str(resolution.ticker)

        retrieval_run = self.retrieve_evidence(
            standalone_question,
            ticker=resolved_ticker,
            original_question=question,
            record_type=record_type,
            limit=limit,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            on_progress=on_progress,
        )
        evidence = retrieval_run.evidence
        embedding_latency_ms = retrieval_run.embedding_latency_ms
        retrieval_latency_ms = retrieval_run.retrieval_latency_ms
        initial_evidence_count = retrieval_run.initial_evidence_count
        bank_plans = [dict(plan) for plan in retrieval_run.agentic_plans]
        orchestration_request_count += retrieval_run.model_request_count
        stages.extend(dict(stage) for stage in retrieval_run.stage_trace)

        if on_progress is not None:
            on_progress(
                "generating",
                {"message": "Generating a grounded answer...", "evidence_count": len(evidence)},
            )
        started = perf_counter()
        answer = generate_answer(
            question,
            evidence,
            client=self.client,
            model=self.generation_model,
            expected_ticker=resolved_ticker,
            expected_bank_name=self.bank_names.get(resolved_ticker, resolved_ticker),
            expected_record_type=record_type,
            temperature=self.temperature,
            resolved_question=standalone_question,
            presentation_guidance=presentation_guidance,
        )
        generation_latency_ms = (perf_counter() - started) * 1000
        stages.append(
            {"stage": "generating", "status": "completed", "latency_ms": generation_latency_ms}
        )
        if on_progress is not None:
            on_progress("validating", {"message": "Validating the grounded answer..."})
        stages.append({"stage": "validating", "status": "completed", "latency_ms": 0.0})
        output = {
            "question": question,
            "dialog_act": "answer",
            "ticker": resolved_ticker,
            "contextualization": contextualization_payload,
            "bank_resolution": resolution.as_dict(),
            "retrieval": {
                "backend": "mixed",
                "mode": "hybrid",
                "evidence_count": len(evidence),
                "queries": list(retrieval_run.diagnostics.get("queries") or []),
            },
            **answer,
        }
        generation_requests = int((answer.get("generation") or {}).get("request_count") or 0)
        diagnostics = self._diagnostics(
            route=route,
            outcome=str(answer.get("status") or "unknown"),
            stages=stages,
            initial_evidence_count=initial_evidence_count,
            final_evidence_count=len(evidence),
            model_request_count=orchestration_request_count + generation_requests,
            bank_plans=bank_plans,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=evidence,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            diagnostics=diagnostics,
            stage_trace=tuple(stages),
            agentic_plans=tuple(bank_plans),
        )

    @staticmethod
    def _relabel_bank_result(
        result: Mapping[str, Any], next_index: int
    ) -> tuple[dict[str, Any], int]:
        output = dict(result)
        mapping: dict[str, str] = {}
        citations: list[dict[str, Any]] = []
        for raw in result.get("citations") or []:
            citation = dict(raw)
            old_label = str(citation.get("label") or "").strip().upper()
            new_label = f"E{next_index}"
            next_index += 1
            mapping[old_label] = new_label
            citation["label"] = new_label
            citations.append(citation)
        output["citations"] = citations
        answer = str(output.get("answer") or "")
        output["answer"] = CITATION_PATTERN.sub(
            lambda match: f"[{mapping.get(match.group(1), match.group(1))}]", answer
        )
        return output, next_index

    def _comparison_run(
        self,
        question: str,
        standalone_question: str,
        resolution: BankResolution,
        contextualization: Mapping[str, Any],
        *,
        record_type: str | None,
        limit: int,
        candidate_k: int,
        rrf_k: int,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None,
        stages: list[dict[str, Any]],
        orchestration_request_count: int,
        presentation_guidance: str | None,
    ) -> AnswerRun:
        selected_tickers = resolution.tickers
        embedding_latency_ms = 0.0
        retrieval_latency_ms = 0.0
        all_evidence: list[dict[str, Any]] = []
        bank_results: list[dict[str, Any]] = []
        per_bank_retrieval: list[dict[str, Any]] = []
        bank_generation_latency_ms = 0.0
        initial_evidence_count = 0
        bank_plans: list[dict[str, Any]] = []
        next_citation_index = 1
        for ticker in selected_tickers:
            bank_name = self.bank_names.get(ticker, ticker)
            bank_question = build_bank_subquestion(
                standalone_question,
                ticker=ticker,
                selected_tickers=selected_tickers,
                bank_names=self.bank_names,
                bank_aliases=self.bank_aliases,
            )
            original_bank_question = build_bank_subquestion(
                question,
                ticker=ticker,
                selected_tickers=selected_tickers,
                bank_names=self.bank_names,
                bank_aliases=self.bank_aliases,
            )
            retrieval_run = self.retrieve_evidence(
                bank_question,
                ticker=ticker,
                original_question=original_bank_question,
                record_type=record_type,
                limit=limit,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                on_progress=on_progress,
            )
            evidence = retrieval_run.evidence
            initial_evidence_count += retrieval_run.initial_evidence_count
            embedding_latency_ms += retrieval_run.embedding_latency_ms
            retrieval_latency_ms += retrieval_run.retrieval_latency_ms
            orchestration_request_count += retrieval_run.model_request_count
            bank_plans.extend(dict(plan) for plan in retrieval_run.agentic_plans)
            stages.extend({**dict(stage), "ticker": ticker} for stage in retrieval_run.stage_trace)
            all_evidence.extend(evidence)
            per_bank_retrieval.append(
                {
                    "ticker": ticker,
                    "query": bank_question,
                    "queries": list(retrieval_run.diagnostics.get("queries") or []),
                    "evidence_count": len(evidence),
                    "latency_ms": retrieval_run.retrieval_latency_ms,
                }
            )

            if on_progress is not None:
                on_progress(
                    "generating",
                    {
                        "message": f"Generating the {ticker} answer...",
                        "ticker": ticker,
                        "evidence_count": len(evidence),
                    },
                )
            started = perf_counter()
            try:
                answer = generate_answer(
                    original_bank_question,
                    evidence,
                    client=self.client,
                    model=self.generation_model,
                    expected_ticker=ticker,
                    expected_bank_name=bank_name,
                    expected_record_type=record_type,
                    temperature=self.temperature,
                    resolved_question=bank_question,
                    comparison_scope=True,
                    presentation_guidance=presentation_guidance,
                )
            except GenerationValidationError as error:
                generation = dict(error.generation)
                answer = {
                    "status": "unsupported",
                    "answer_type": "narrative",
                    "answer": render_unsupported_answer(question),
                    "facts": None,
                    "reason": (
                        f"The {ticker} answer failed output validation ({error.code}); "
                        "the other banks were still evaluated."
                    ),
                    "citations": [],
                    "generation": {
                        **generation,
                        "model": self.generation_model,
                        "request_count": int(generation.get("request_count") or 0),
                        "final_status": "unsupported",
                        "error_code": error.code,
                    },
                }
                per_bank_retrieval[-1]["generation_error_code"] = error.code
            bank_generation_latency_ms += (perf_counter() - started) * 1000
            relabeled, next_citation_index = self._relabel_bank_result(answer, next_citation_index)
            bank_results.append({"ticker": ticker, "bank_name": bank_name, **relabeled})

        supported_count = sum(result["status"] == "supported" for result in bank_results)
        if supported_count == len(bank_results):
            status = "supported"
        elif supported_count:
            status = "partial"
        else:
            status = "unsupported"

        if on_progress is not None and supported_count:
            on_progress(
                "synthesizing",
                {"message": "Synthesizing the bank comparison...", "tickers": selected_tickers},
            )
        started = perf_counter()
        synthesis = synthesize_comparison(
            question,
            bank_results,
            client=self.client,
            model=self.generation_model,
            resolved_question=standalone_question,
            presentation_guidance=presentation_guidance,
        )
        synthesis_latency_ms = (perf_counter() - started) * 1000
        generation_latency_ms = bank_generation_latency_ms + synthesis_latency_ms
        stages.append(
            {"stage": "generating", "status": "completed", "latency_ms": bank_generation_latency_ms}
        )
        if synthesis_latency_ms:
            stages.append(
                {"stage": "synthesizing", "status": "completed", "latency_ms": synthesis_latency_ms}
            )
        if on_progress is not None:
            on_progress("validating", {"message": "Validating the bank comparison..."})
        stages.append({"stage": "validating", "status": "completed", "latency_ms": 0.0})

        citations = [
            dict(citation) for result in bank_results for citation in result.get("citations") or []
        ]
        generation_request_count = sum(
            int((result.get("generation") or {}).get("request_count") or 0)
            for result in bank_results
        ) + int(synthesis["generation"].get("request_count") or 0)
        output = {
            "question": question,
            "dialog_act": "answer",
            "mode": "comparison",
            "ticker": None,
            "tickers": list(selected_tickers),
            "contextualization": dict(contextualization),
            "bank_resolution": resolution.as_dict(),
            "retrieval": {
                "backend": "mixed",
                "mode": "hybrid",
                "evidence_count": len(all_evidence),
                "per_bank": per_bank_retrieval,
            },
            "status": status,
            "answer_type": "narrative",
            "answer": synthesis["answer"],
            "facts": None,
            "reason": (
                "All selected banks have grounded answers."
                if status == "supported"
                else "Only some selected banks have grounded answers."
                if status == "partial"
                else "No selected bank has sufficient evidence."
            ),
            "citations": citations,
            "bank_results": bank_results,
            "generation": {
                **synthesis["generation"],
                "request_count": generation_request_count,
                "bank_request_count": generation_request_count
                - int(synthesis["generation"].get("request_count") or 0),
                "synthesis_request_count": int(synthesis["generation"].get("request_count") or 0),
                "final_status": status,
            },
        }
        diagnostics = self._diagnostics(
            route="domain_rag",
            outcome=status,
            stages=stages,
            initial_evidence_count=initial_evidence_count,
            final_evidence_count=len(all_evidence),
            model_request_count=orchestration_request_count + generation_request_count,
            bank_plans=bank_plans,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=all_evidence,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            diagnostics=diagnostics,
            stage_trace=tuple(stages),
            agentic_plans=tuple(bank_plans),
        )

    def _direct_conversation_run(
        self,
        question: str,
        action: DirectResponseArgs,
        *,
        stages: Sequence[Mapping[str, Any]],
        model_request_count: int,
        history_turns: int,
        available_history_turns: int,
        fallback: bool,
        error_code: str | None,
        source_answer: Mapping[str, Any] | None,
    ) -> AnswerRun:
        source_citations = {
            str(item.get("label") or "").strip().upper(): dict(item)
            for item in (source_answer or {}).get("citations") or []
        }
        citations = [
            source_citations[label]
            for label in action.citation_ids
            if label in source_citations
        ]
        contextual_transform = action.category == "contextual_transform"
        output: dict[str, Any] = {
            "question": question,
            "dialog_act": action.category,
            "ticker": (source_answer or {}).get("ticker") if contextual_transform else None,
            "tickers": (
                list((source_answer or {}).get("tickers") or [])
                if contextual_transform
                else []
            ),
            "contextualization": {
                "applied": False,
                "history_turns": history_turns,
                "available_history_turns": available_history_turns,
                "standalone_question": question,
                "model": self.generation_model,
                "latency_ms": 0.0,
                "source": "conversation_graph",
                "fallback": fallback,
                "error_code": error_code,
                "presentation_guidance": None,
                "skip_reason": "retrieval_not_required",
            },
            "bank_resolution": {
                "status": "not_required",
                "source": "conversation_graph",
                "ticker": None,
                "detected_tickers": [],
            },
            "retrieval": {"backend": "none", "mode": "none", "evidence_count": 0},
            "status": "supported",
            "answer_type": "narrative",
            "answer": action.answer,
            "facts": None,
            "reason": (
                "The model transformed the immediately previous grounded answer."
                if contextual_transform
                else "The conversation action does not require filing evidence."
            ),
            "reason_code": f"conversation_{action.category}",
            "citations": citations,
            "generation": {
                "model": self.generation_model,
                "final_status": "supported",
                "request_count": 0,
            },
        }
        diagnostics = self._diagnostics(
            route="general_chat",
            outcome="supported",
            stages=stages,
            initial_evidence_count=0,
            final_evidence_count=0,
            model_request_count=model_request_count,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            diagnostics=diagnostics,
            stage_trace=tuple(dict(stage) for stage in stages),
        )

    def _out_of_scope_run(
        self,
        question: str,
        *,
        stages: Sequence[Mapping[str, Any]],
        model_request_count: int,
        history_turns: int,
        available_history_turns: int,
        fallback: bool,
        error_code: str | None,
    ) -> AnswerRun:
        output: dict[str, Any] = {
            "question": question,
            "dialog_act": "out_of_scope",
            "ticker": None,
            "tickers": [],
            "contextualization": {
                "applied": False,
                "history_turns": history_turns,
                "available_history_turns": available_history_turns,
                "standalone_question": question,
                "model": self.generation_model,
                "latency_ms": 0.0,
                "source": "conversation_graph",
                "fallback": fallback,
                "error_code": error_code,
                "skip_reason": "outside_banking_research_scope",
            },
            "bank_resolution": {
                "status": "not_required",
                "source": "conversation_graph",
                "ticker": None,
                "detected_tickers": [],
            },
            "retrieval": {"backend": "none", "mode": "none", "evidence_count": 0},
            "status": "unsupported",
            "answer_type": "narrative",
            "answer": render_out_of_scope_answer(question),
            "facts": None,
            "reason": "The request is outside BankScope's supported research domain.",
            "reason_code": "outside_banking_research_scope",
            "citations": [],
            "generation": {
                "model": self.generation_model,
                "final_status": "unsupported",
                "request_count": 0,
            },
        }
        diagnostics = self._diagnostics(
            route="scope_guard",
            outcome="unsupported",
            stages=stages,
            initial_evidence_count=0,
            final_evidence_count=0,
            model_request_count=model_request_count,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            diagnostics=diagnostics,
            stage_trace=tuple(dict(stage) for stage in stages),
        )

    def _web_unavailable_run(
        self,
        question: str,
        action: WebResearchArgs,
        *,
        stages: Sequence[Mapping[str, Any]],
        model_request_count: int,
        history_turns: int,
        available_history_turns: int,
        fallback: bool,
        error_code: str | None,
        history: Sequence[Mapping[str, str]],
        session_tickers: Sequence[str],
    ) -> AnswerRun:
        standalone_question = action.search_question
        try:
            user_history = [
                str(message.get("content") or "")
                for message in history
                if message.get("role") == "user"
            ]
            validate_contextualized_rewrite(
                question,
                standalone_question,
                allowed_user_context=user_history,
            )
            explicit_scope = set(
                resolve_bank(
                    question,
                    bank_names=self.bank_names,
                    bank_aliases=self.bank_aliases,
                ).tickers
            )
            planned_scope = set(
                resolve_bank(
                    standalone_question,
                    bank_names=self.bank_names,
                    bank_aliases=self.bank_aliases,
                    session_tickers=session_tickers,
                ).tickers
            )
            history_scope = {
                ticker
                for message in history
                if message.get("role") == "user"
                for ticker in resolve_bank(
                    str(message.get("content") or ""),
                    bank_names=self.bank_names,
                    bank_aliases=self.bank_aliases,
                ).tickers
            }
            allowed_scope = explicit_scope | set(session_tickers) | history_scope
            if explicit_scope - planned_scope:
                raise GenerationValidationError(
                    "contextualization_lost_bank_scope",
                    "The web search dropped a bank named in the current question.",
                )
            if planned_scope - allowed_scope:
                raise GenerationValidationError(
                    "contextualization_added_bank_scope",
                    "The web search introduced a bank outside the conversation scope.",
                )
        except GenerationValidationError as error:
            standalone_question = question
            fallback = True
            error_code = error.code
        output: dict[str, Any] = {
            "question": question,
            "dialog_act": "web_research_unavailable",
            "ticker": None,
            "tickers": [],
            "contextualization": {
                "applied": standalone_question != question,
                "history_turns": history_turns,
                "available_history_turns": available_history_turns,
                "standalone_question": standalone_question,
                "model": self.generation_model,
                "latency_ms": 0.0,
                "source": "conversation_graph",
                "fallback": fallback,
                "error_code": error_code,
                "skip_reason": "web_search_unavailable",
            },
            "bank_resolution": {
                "status": "not_required",
                "source": "conversation_graph",
                "ticker": None,
                "detected_tickers": [],
            },
            "retrieval": {"backend": "none", "mode": "none", "evidence_count": 0},
            "status": "unsupported",
            "answer_type": "narrative",
            "answer": render_web_unavailable_answer(question),
            "facts": None,
            "reason": action.reason,
            "reason_code": "web_search_unavailable",
            "citations": [],
            "generation": {
                "model": self.generation_model,
                "final_status": "unsupported",
                "request_count": 0,
            },
        }
        diagnostics = self._diagnostics(
            route="web_research",
            outcome="unsupported",
            stages=stages,
            initial_evidence_count=0,
            final_evidence_count=0,
            model_request_count=model_request_count,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            diagnostics=diagnostics,
            stage_trace=tuple(dict(stage) for stage in stages),
        )

    @staticmethod
    def _filing_research_handler(
        question: str,
        action: ResearchFilingsArgs,
        **routing_context: Any,
    ) -> ResearchFilingsArgs:
        """Keep filing retrieval behind the same registry used by future source handlers."""

        del question, routing_context
        return action

    def _clarification_run(
        self,
        question: str,
        action: ClarificationArgs,
        *,
        stages: Sequence[Mapping[str, Any]],
        model_request_count: int,
        history_turns: int,
        available_history_turns: int,
        fallback: bool,
        error_code: str | None,
    ) -> AnswerRun:
        output: dict[str, Any] = {
            "question": question,
            "dialog_act": "clarification",
            "ticker": None,
            "contextualization": {
                "applied": False,
                "history_turns": history_turns,
                "available_history_turns": available_history_turns,
                "standalone_question": question,
                "model": self.generation_model,
                "latency_ms": 0.0,
                "source": "conversation_graph",
                "fallback": fallback,
                "error_code": error_code,
                "skip_reason": f"missing_{action.missing}",
            },
            "bank_resolution": {
                "status": "not_required",
                "source": "conversation_graph",
                "ticker": None,
                "detected_tickers": [],
            },
            "retrieval": {"backend": "none", "mode": "none", "evidence_count": 0},
            "status": "ambiguous",
            "answer_type": "narrative",
            "answer": action.question,
            "facts": None,
            "reason": f"Filing research needs the missing {action.missing}.",
            "reason_code": f"missing_{action.missing}",
            "citations": [],
            "generation": {
                "model": self.generation_model,
                "final_status": "ambiguous",
                "request_count": 0,
            },
        }
        diagnostics = self._diagnostics(
            route="domain_rag",
            outcome="ambiguous",
            stages=stages,
            initial_evidence_count=0,
            final_evidence_count=0,
            model_request_count=model_request_count,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            diagnostics=diagnostics,
            stage_trace=tuple(dict(stage) for stage in stages),
        )

    def _ambiguous_bank_run(
        self,
        question: str,
        resolution: BankResolution,
        contextualization: Mapping[str, Any],
        *,
        stages: Sequence[Mapping[str, Any]] = (),
        model_request_count: int = 0,
    ) -> AnswerRun:
        if resolution.status == "too_many":
            answer = "Možete porediti najviše četiri podržane banke u jednom pitanju."
            reason_code = "too_many_banks_identified"
            reason = "Pitanje pominje više od četiri podržane banke."
        elif resolution.status == "multiple":
            names = [self.bank_names.get(ticker, ticker) for ticker in resolution.detected_tickers]
            answer = (
                "Trenutno mogu da odgovorim za jednu banku po pitanju. "
                f"Pronašao sam {', '.join(names)}; navedite jednu banku."
            )
            reason_code = "multiple_banks_identified"
            reason = "Pitanje pominje više podržanih banaka."
        else:
            answer = (
                "Nisam mogao da odredim na koju banku se pitanje odnosi. "
                "Navedite naziv ili ticker, na primer 'JPMorgan Chase' ili 'JPM'."
            )
            reason_code = "bank_not_identified"
            reason = "Pitanje ne sadrži prepoznat naziv ili ticker podržane banke."
        output = {
            "question": question,
            "dialog_act": "clarification",
            "ticker": None,
            "contextualization": dict(contextualization),
            "bank_resolution": resolution.as_dict(),
            "retrieval": {
                "backend": "mixed",
                "mode": "hybrid",
                "evidence_count": 0,
            },
            "status": "ambiguous",
            "answer_type": "narrative",
            "answer": answer,
            "facts": None,
            "reason": reason,
            "reason_code": reason_code,
            "citations": [],
            "generation": {
                "model": self.generation_model,
                "final_status": "ambiguous",
                "request_count": 0,
            },
        }
        diagnostics = self._diagnostics(
            route="domain_rag",
            outcome="ambiguous",
            stages=stages,
            initial_evidence_count=0,
            final_evidence_count=0,
            model_request_count=model_request_count,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            diagnostics=diagnostics,
            stage_trace=tuple(dict(stage) for stage in stages),
        )

    def _general_chat_run(
        self,
        question: str,
        *,
        stages: Sequence[Mapping[str, Any]],
        model_request_count: int,
    ) -> AnswerRun:
        answer = (
            "Zdravo! Mogu da pomognem sa pitanjima o podržanim bankama, njihovim SEC "
            "izveštajima, finansijskim pokazateljima i operativnim rizicima."
        )
        output: dict[str, Any] = {
            "question": question,
            "dialog_act": "capability",
            "ticker": None,
            "contextualization": {
                "applied": False,
                "history_turns": 0,
                "standalone_question": question,
                "model": None,
                "latency_ms": 0.0,
            },
            "bank_resolution": {
                "status": "not_required",
                "source": "router",
                "ticker": None,
                "detected_tickers": [],
            },
            "retrieval": {"backend": "mixed", "mode": "hybrid", "evidence_count": 0},
            "status": "supported",
            "answer_type": "narrative",
            "answer": answer,
            "facts": None,
            "reason": "General BankScope greeting or product-help request.",
            "reason_code": "general_chat",
            "citations": [],
            "generation": {
                "model": self.generation_model,
                "final_status": "supported",
                "request_count": 0,
            },
        }
        diagnostics = self._diagnostics(
            route="general_chat",
            outcome="supported",
            stages=stages,
            initial_evidence_count=0,
            final_evidence_count=0,
            model_request_count=model_request_count,
            output=output,
        )
        output["diagnostics"] = diagnostics
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            diagnostics=diagnostics,
            stage_trace=tuple(dict(stage) for stage in stages),
        )


# Backward-compatible import for existing scripts and third-party callers.
SingleBankAnswerPipeline = BankAnswerPipeline
