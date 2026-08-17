import pytest

from bankscope.generation import GPT51_CANDIDATE_MODEL, GenerationValidationError
from bankscope.io import read_jsonl
from scripts.evaluate_answers import (
    DEFAULT_OUTPUT,
    _cited_evidence,
    _error_payload,
    assess_generation_quality_gate,
    select_generation_queries,
    validate_citation_audit,
    validate_generation_queries,
    validate_output_model,
)


def test_frozen_queries_select_exact_single_bank_scope() -> None:
    queries = read_jsonl("data/evaluation/queries.jsonl")

    validate_generation_queries(queries)
    eligible, skipped = select_generation_queries(queries)

    assert len(eligible) == 30
    assert sum(query["status"] == "answerable" for query in eligible) == 29
    assert sum(query["status"] == "unsupported" for query in eligible) == 1
    assert len(skipped) == 4
    assert sum(item["reason"] == "cross_bank_out_of_single_bank_scope" for item in skipped) == 3
    assert sum(item["reason"] == "missing_ticker_out_of_single_bank_scope" for item in skipped) == 1


def test_query_id_filter_keeps_excluded_query_visible() -> None:
    queries = read_jsonl("data/evaluation/queries.jsonl")

    eligible, skipped = select_generation_queries(queries, ["dev_ambiguous_bank_cet1_2025"])

    assert eligible == []
    assert skipped[0]["query_id"] == "dev_ambiguous_bank_cet1_2025"


def test_versioned_citation_audit_matches_frozen_queries() -> None:
    queries = read_jsonl("data/evaluation/queries.jsonl")
    records = read_jsonl("data/evaluation/generation_citation_audit_v2.jsonl")

    audit = validate_citation_audit(records, queries)

    accepted = audit["dev_bac_bana_standardized_cet1_2025_metadata"]
    assert accepted["accepted_additional_target_chunk_ids"] == [
        "648648bf7a1b5a9c476ffa4072265f994a31bd2e042851f6475b6ecc5f214460"
    ]
    assert (
        audit["dev_c_standardized_cet1_ratio_2025"]["reviewed_targets"][0]["classification"]
        == "relevant_but_insufficient_precision"
    )
    assert (
        audit["dev_bac_bana_expansion_2025"]["reviewed_targets"][0]["classification"]
        == "does_not_support_claim"
    )
    accepted_tfc = audit["dev_tfc_cet1_ratio_corporation_2025"]
    assert len(accepted_tfc["accepted_additional_target_chunk_ids"]) == 2
    assert audit["dev_jpm_standardized_cet1_requirement_2025"][
        "accepted_additional_target_chunk_ids"
    ] == ["a9a562bc58279de88399041989871810cec4bb84ae9ee0de59aee219c7d8785b"]
    assert (
        audit["dev_pnc_total_deposits_2025"]["reviewed_targets"][0]["classification"]
        == "relevant_but_insufficient_precision"
    )


def test_citation_audit_requires_accepted_ids_to_have_accepted_reviews() -> None:
    queries = [{"query_id": "q1"}]
    records = [
        {
            "audit_version": 2,
            "query_id": "q1",
            "accepted_additional_target_chunk_ids": ["chunk-1"],
            "reviewed_targets": [
                {
                    "target_chunk_id": "chunk-1",
                    "classification": "relevant_but_insufficient_precision",
                    "note": "Not exact enough.",
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="accepted IDs do not match"):
        validate_citation_audit(records, queries)


def test_semantic_judge_evidence_is_limited_to_actual_citations() -> None:
    answer = {"citations": [{"target_chunk_id": "chunk-2"}]}
    evidence = [
        {"target_chunk_id": "chunk-1", "evidence": "Uncited."},
        {"target_chunk_id": "chunk-2", "evidence": "Cited."},
        {"target_chunk_id": "chunk-3", "evidence": "Also uncited."},
    ]

    assert _cited_evidence(answer, evidence) == [evidence[1]]


def test_generation_validation_error_preserves_stable_code() -> None:
    error = GenerationValidationError("invalid_schema", "Invalid response.")

    assert _error_payload(error) == {
        "type": "GenerationValidationError",
        "message": "Invalid response.",
        "code": "invalid_schema",
    }


def test_default_candidate_artifact_requires_gpt51_model() -> None:
    validate_output_model(DEFAULT_OUTPUT, GPT51_CANDIDATE_MODEL)

    with pytest.raises(ValueError, match="reserved"):
        validate_output_model(DEFAULT_OUTPUT, "AZURE_GPT_4o_2024_1120")
    with pytest.raises(ValueError, match="filtered run"):
        validate_output_model(DEFAULT_OUTPUT, GPT51_CANDIDATE_MODEL, ["q1"])

    validate_output_model(DEFAULT_OUTPUT.with_name("custom.json"), "another-model")


def test_generation_quality_gate_encodes_frozen_acceptance_contract() -> None:
    rows = []
    for _ in range(17):
        rows.append(
            {
                "expected_numeric": True,
                "metrics": {
                    "expected_status": "supported",
                    "citations": {"citation_support_precision": 1.0},
                },
                "answer": {
                    "answer_type": "numeric",
                    "generation": {"validation_checks": ["numeric_value_in_cited_evidence"]},
                },
            }
        )
    for _ in range(12):
        rows.append(
            {
                "expected_numeric": False,
                "metrics": {
                    "expected_status": "supported",
                    "citations": {"citation_support_precision": 1.0},
                },
                "answer": {"answer_type": "narrative"},
                "judge": {"groundedness": True},
            }
        )
    rows.append(
        {
            "expected_numeric": False,
            "metrics": {
                "expected_status": "unsupported",
                "citations": {"citation_support_precision": None},
            },
        }
    )
    summary = {
        "evaluated_count": 30,
        "error_count": 0,
        "status_accuracy": 1.0,
        "value_match_count": 17,
        "value_match_rate": 1.0,
        "unit_match_count": 17,
        "unit_match_rate": 1.0,
        "period_match_count": 17,
        "period_match_rate": 1.0,
        "entity_match_count": 17,
        "entity_match_rate": 1.0,
        "variant_match_count": 11,
        "variant_match_rate": 1.0,
        "generation_request_count": 29,
    }

    gate = assess_generation_quality_gate(rows, summary)

    assert gate["passed"] is True
    rows[17]["judge"]["groundedness"] = False
    assert assess_generation_quality_gate(rows, summary)["passed"] is False
