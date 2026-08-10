from bankscope.evaluation.answer_metrics import (
    evaluate_answer,
    expected_answer_status,
    summarize_answer_metrics,
)


def query(**changes):
    value = {
        "status": "answerable",
        "relevant_target_chunk_ids": ["chunk-1", "chunk-2"],
        "expected_value": 14.6,
        "expected_unit": "percent",
        "expected_period": "2025-12-31",
        "expected_entity": "JPMorgan Chase & Co.",
        "expected_variant": "Standardized",
    }
    value.update(changes)
    return value


def answer(**changes):
    value = {
        "status": "supported",
        "answer": "JPMorgan Chase's Standardized ratio was 14.6% in 2025 [E1].",
        "citations": [{"target_chunk_id": "chunk-1"}],
    }
    value.update(changes)
    return value


def test_expected_status_mapping() -> None:
    assert expected_answer_status("answerable") == "supported"
    assert expected_answer_status("ambiguous") == "ambiguous"
    assert expected_answer_status("unsupported") == "unsupported"


def test_structured_answer_and_relevant_citation_match() -> None:
    metrics = evaluate_answer(query(), answer())

    assert metrics["status_correct"] == 1
    assert metrics["structured"] == {
        "value_match": 1,
        "unit_match": 1,
        "period_match": 1,
        "entity_match": 1,
        "variant_match": 1,
    }
    assert metrics["citations"]["citation_precision"] == 1.0
    assert metrics["citations"]["citation_complete"] == 1


def test_wrong_structured_fields_and_partial_evidence_groups_fail() -> None:
    metrics = evaluate_answer(
        query(
            required_evidence_groups=[
                {"target_chunk_ids": ["chunk-1"]},
                {"target_chunk_ids": ["chunk-2"]},
            ]
        ),
        answer(
            answer="Citigroup's Advanced ratio was 13.1 dollars in 2024 [E1] [E2].",
            citations=[
                {"target_chunk_id": "chunk-1"},
                {"target_chunk_id": "unrelated"},
            ],
        ),
    )

    assert set(metrics["structured"].values()) == {0}
    assert metrics["citations"]["citation_precision"] == 0.5
    assert metrics["citations"]["required_group_coverage"] == 0.5
    assert metrics["citations"]["citation_complete"] == 0


def test_unsupported_answer_is_correct_and_clean_without_citations() -> None:
    metrics = evaluate_answer(
        query(status="unsupported"),
        answer(status="unsupported", answer="Not supported.", citations=[]),
    )

    assert metrics["status_correct"] == 1
    assert metrics["structured"] == {}
    assert metrics["citations"]["citation_complete"] == 1


def test_summary_keeps_semantic_judge_separate() -> None:
    rows = [
        {
            "metrics": evaluate_answer(query(), answer()),
            "judge": {"correctness": True, "completeness": False, "groundedness": True},
        },
        {"error": {"message": "failed"}},
    ]

    summary = summarize_answer_metrics(rows)

    assert summary["query_count"] == 2
    assert summary["evaluated_count"] == 1
    assert summary["error_count"] == 1
    assert summary["status_accuracy"] == 1.0
    assert summary["semantic_correctness_rate"] == 1.0
    assert summary["semantic_completeness_rate"] == 0.0


def test_unsupported_only_summary_has_no_supported_citation_rate() -> None:
    metrics = evaluate_answer(
        query(status="unsupported"),
        answer(status="unsupported", answer="Not supported.", citations=[]),
    )

    summary = summarize_answer_metrics([{"metrics": metrics}])

    assert summary["citation_relevant_hit_rate"] is None
    assert summary["mean_citation_precision"] is None
    assert summary["citation_complete_rate"] == 1.0
