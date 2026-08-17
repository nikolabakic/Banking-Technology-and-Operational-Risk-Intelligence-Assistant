from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from bankscope.generation.agentic import (
    AGENT_STEP_ADAPTER,
    AgenticPlan,
    AgentState,
    CanonicalContextExpander,
    FinishStep,
    ReadContextStep,
    SearchExactStep,
    SearchHybridStep,
    deduplicate_evidence,
    route_question,
    validate_agent_step,
    validate_plan_scope,
)
from bankscope.generation.answer_generator import GenerationValidationError
from bankscope.generation.pipeline import BankAnswerPipeline
from bankscope.io import read_jsonl
from scripts.evaluate_agentic_rag import run_mode, validate_challenge


def evidence(target: str, ticker: str = "JPM", accession: str = "filing-1") -> dict:
    return {
        "target_chunk_id": target,
        "record_type": "text",
        "document": target,
        "metadata": {
            "ticker": ticker,
            "record_type": "text",
            "accession_number": accession,
        },
    }


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        ("generate", {}),
        ("rewrite_search", {"rewritten_query": "JPM CET1 ratio 2025 standardized"}),
        ("expand_context", {"anchor_target_chunk_id": "chunk-1"}),
        ("abstain", {}),
    ],
)
def test_all_four_plans_have_strict_action_arguments(action, arguments) -> None:
    plan = AgenticPlan(
        action=action,
        reason_code="test_reason",
        explanation="Deterministic test decision.",
        **arguments,
    )
    assert plan.action == action

    with pytest.raises(ValidationError):
        AgenticPlan(
            action="generate",
            reason_code="bad",
            explanation="Invalid extra action argument.",
            rewritten_query="not allowed",
        )


def test_rewrite_must_preserve_explicit_period() -> None:
    plan = AgenticPlan(
        action="rewrite_search",
        reason_code="retrieval_term_mismatch",
        explanation="Use filing terminology.",
        rewritten_query="JPM CET1 ratio standardized approach",
    )
    with pytest.raises(GenerationValidationError) as caught:
        validate_plan_scope(plan, "What was JPM CET1 in 2025?", [evidence("chunk-1")], "JPM")
    assert caught.value.code == "agentic_rewrite_lost_period"


def test_rewrite_cannot_add_numeric_facts_from_evidence() -> None:
    plan = AgenticPlan(
        action="rewrite_search",
        reason_code="retrieval_term_mismatch",
        explanation="Use filing terminology.",
        rewritten_query="Compare JPM and BAC CET1 for 2025; BAC reported 11.4%.",
    )
    with pytest.raises(GenerationValidationError) as caught:
        validate_plan_scope(
            plan,
            "Compare JPM and BAC CET1 for 2025.",
            [evidence("chunk-1")],
            "JPM",
        )
    assert caught.value.code == "agentic_rewrite_added_numeric_fact"


def test_context_expansion_is_radius_one_and_accession_scoped() -> None:
    chunks = [
        evidence("previous"),
        evidence("anchor"),
        evidence("next"),
        evidence("other-filing", accession="filing-2"),
    ]
    expanded = CanonicalContextExpander(chunks).expand("anchor", ticker="JPM")
    assert [item["target_chunk_id"] for item in expanded] == ["previous", "anchor", "next"]
    assert all(item["retrieval_source"] == "canonical_context_expansion" for item in expanded)


def test_context_expansion_accepts_bounded_asymmetric_window() -> None:
    chunks = [evidence(f"chunk-{index}") for index in range(7)]
    expanded = CanonicalContextExpander(chunks).expand(
        "chunk-3", ticker="JPM", before=3, after=2
    )
    assert [item["target_chunk_id"] for item in expanded] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
        "chunk-5",
    ]
    with pytest.raises(ValueError, match="zero and three"):
        CanonicalContextExpander(chunks).expand("chunk-3", ticker="JPM", before=4)


def test_context_expansion_rejects_cross_bank_anchor() -> None:
    expander = CanonicalContextExpander([evidence("bac", ticker="BAC")])
    with pytest.raises(GenerationValidationError) as caught:
        expander.expand("bac", ticker="JPM")
    assert caught.value.code == "agentic_expansion_crossed_bank"


def test_preferred_evidence_is_first_deduplicated_and_bounded() -> None:
    original = [evidence(f"old-{index}") for index in range(5)]
    preferred = [evidence("new"), evidence("old-1")]
    merged = deduplicate_evidence(preferred, original, limit=5)
    assert [item["target_chunk_id"] for item in merged] == [
        "new",
        "old-1",
        "old-0",
        "old-2",
        "old-3",
    ]


def test_invalid_router_output_falls_back_to_domain_rag() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"), finish_reason="stop")]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    result = route_question(
        "What is JPM operational risk?",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="test-model",
    )
    assert result.value.route == "domain_rag"
    assert result.fallback is True


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "search_hybrid", "query": "JPM 2025 CET1", "reason": "Canonical terms"},
        {"action": "search_exact", "terms": ["Common Equity Tier 1"], "reason": "Exact"},
        {
            "action": "read_context",
            "anchor_target_chunk_id": "chunk-1",
            "before": 2,
            "after": 3,
            "reason": "Complete section",
        },
        {
            "action": "finish",
            "status": "sufficient",
            "reason": "Direct support",
            "supporting_target_chunk_ids": ["chunk-1"],
        },
    ],
)
def test_agent_step_discriminated_schema(payload: dict) -> None:
    step = AGENT_STEP_ADAPTER.validate_python(payload)
    assert step.action == payload["action"]
    with pytest.raises(ValidationError):
        AGENT_STEP_ADAPTER.validate_python({**payload, "unexpected": True})


def test_agent_step_preserves_period_and_numeric_facts() -> None:
    state = AgentState(
        ticker="JPM",
        question="Koji je JPM CET1 procenat za 2025?",
        evidence=[evidence("chunk-1")],
    )
    validate_agent_step(
        SearchHybridStep(
            action="search_hybrid",
            query="JPM 2025 Standardized Common Equity Tier 1 capital ratio",
            reason="Use canonical English filing terms.",
        ),
        state,
    )
    with pytest.raises(GenerationValidationError) as caught:
        validate_agent_step(
            SearchHybridStep(
                action="search_hybrid",
                query="JPM Standardized CET1 ratio 14.6%",
                reason="Invalid evidence-derived value.",
            ),
            state,
        )
    assert caught.value.code == "agentic_search_lost_period"


def test_agent_step_blocks_premature_unsupported_and_unknown_anchor() -> None:
    state = AgentState(
        ticker="JPM",
        question="What is JPM operational risk in 2025?",
        evidence=[evidence("chunk-1")],
    )
    with pytest.raises(GenerationValidationError) as caught:
        validate_agent_step(
            FinishStep(action="finish", status="unsupported", reason="Not in top results."),
            state,
        )
    assert caught.value.code == "agentic_premature_unsupported"
    with pytest.raises(GenerationValidationError) as caught:
        validate_agent_step(
            ReadContextStep(
                action="read_context",
                anchor_target_chunk_id="missing",
                reason="Read around a missing result.",
            ),
            state,
        )
    assert caught.value.code == "agentic_anchor_not_in_results"


def test_exact_step_rejects_evidence_derived_numbers() -> None:
    state = AgentState(
        ticker="JPM",
        question="What was JPM CET1 in 2025?",
        evidence=[evidence("chunk-1")],
    )
    with pytest.raises(GenerationValidationError) as caught:
        validate_agent_step(
            SearchExactStep(
                action="search_exact",
                terms=["14.6%"],
                reason="Do not leak an answer into search.",
            ),
            state,
        )
    assert caught.value.code == "agentic_exact_added_numeric_fact"


class LoopEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class LoopRetriever:
    def __init__(self) -> None:
        self.exact_calls: list[tuple[list[str], dict]] = []

    def search_hybrid(self, query, vector, **kwargs):
        return [evidence("initial", ticker=kwargs["ticker"])]

    def search_exact(self, terms, **kwargs):
        self.exact_calls.append((list(terms), kwargs))
        return [evidence("target", ticker=kwargs["ticker"])]


class SequenceCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        message = SimpleNamespace(content=content, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def test_retrieve_evidence_runs_exact_search_then_independent_verifier() -> None:
    completions = SequenceCompletions(
        [
            '{"action":"search_exact","terms":["Common Equity Tier 1 capital ratio"],'
            '"reason":"Use the filing phrase."}',
            '{"action":"finish","status":"sufficient","reason":"Target found",'
            '"supporting_target_chunk_ids":["target"]}',
            '{"status":"sufficient","explanation":"Direct evidence",'
            '"missing_aspects":[],"supporting_target_chunk_ids":["target"]}',
        ]
    )
    retriever = LoopRetriever()
    pipeline = BankAnswerPipeline(
        retriever=retriever,
        query_encoder=LoopEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="test-model",
        agentic_rag_enabled=True,
    )

    run = pipeline.retrieve_evidence("What was JPM CET1 in 2025?", ticker="JPM")

    assert run.status == "sufficient"
    assert [item["target_chunk_id"] for item in run.evidence[:2]] == ["target", "initial"]
    assert run.model_request_count == 3
    assert retriever.exact_calls[0][1]["ticker"] == "JPM"
    plan = run.agentic_plans[0]
    assert plan["tool_action_count"] == 1
    assert [step["action"] for step in plan["steps"]] == [
        "search_exact",
        "verify_evidence",
    ]


def test_retrieve_evidence_survives_two_schema_failures_with_safe_fallback() -> None:
    completions = SequenceCompletions(["not-json", "still-not-json"])
    pipeline = BankAnswerPipeline(
        retriever=LoopRetriever(),
        query_encoder=LoopEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="test-model",
        agentic_rag_enabled=True,
    )

    run = pipeline.retrieve_evidence("What was JPM CET1 in 2025?", ticker="JPM")

    assert run.status == "sufficient"
    assert [item["target_chunk_id"] for item in run.evidence] == ["initial"]
    assert run.model_request_count == 2
    assert run.agentic_plans[0]["fallback"] is True
    assert run.diagnostics["quality_gate"]["passed"] is False


def test_agentic_evaluator_keeps_retrieval_metrics_when_generation_fails() -> None:
    retrieval = SimpleNamespace(
        evidence=[evidence("target")],
        agentic_plans=(
            {
                "steps": [],
                "model_request_count": 1,
                "tool_action_count": 0,
                "bank_isolation_ok": True,
                "query_preservation_ok": True,
            },
        ),
        diagnostics={"quality_gate": {"passed": True}},
        status="sufficient",
        embedding_latency_ms=1.0,
        retrieval_latency_ms=2.0,
        orchestration_latency_ms=3.0,
    )

    class FailingGenerationPipeline:
        agentic_rag_enabled = False

        def retrieve_evidence(self, question, *, ticker):
            return retrieval

        def answer(self, *args, **kwargs):
            raise GenerationValidationError("invalid_schema", "Generation failed.")

    result = run_mode(
        FailingGenerationPipeline(),  # type: ignore[arg-type]
        {
            "query_id": "q1",
            "query": "What was JPM CET1?",
            "tickers": ["JPM"],
            "source_query_ids": ["source"],
        },
        {"source": {"relevant_target_chunk_ids": ["target"]}},
        enabled=True,
    )

    assert result["executed"] is True
    assert result["hit_at_5"] is True
    assert result["end_to_end"]["executed"] is False
    assert result["end_to_end"]["error_code"] == "invalid_schema"


def test_frozen_agentic_challenge_has_required_distribution() -> None:
    challenge = read_jsonl("data/evaluation/agentic_rag_challenge_v1.jsonl")
    qrels = {row["query_id"]: row for row in read_jsonl("data/evaluation/queries.jsonl")}
    validate_challenge(challenge, qrels)
