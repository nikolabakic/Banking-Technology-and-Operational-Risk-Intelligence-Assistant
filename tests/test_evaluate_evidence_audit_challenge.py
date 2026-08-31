from collections import Counter

import pytest

from bankscope.io import read_jsonl
from scripts.evaluate_evidence_audit_challenge import (
    DEFAULT_CHALLENGE,
    EXPECTED_CATEGORIES,
    ROOT,
    summarize_challenge,
    validate_challenge,
)


def test_challenge_contract_is_separate_and_references_local_corpus() -> None:
    queries = read_jsonl(DEFAULT_CHALLENGE)
    records = read_jsonl(ROOT / "data/processed/chunks.jsonl")

    validate_challenge(
        queries,
        {str(record.get("target_chunk_id") or "") for record in records},
    )

    assert len(read_jsonl(ROOT / "data/evaluation/queries.jsonl")) == 34
    assert Counter(str(query["challenge_category"]) for query in queries) == EXPECTED_CATEGORIES


def test_challenge_rejects_relevant_and_trap_overlap() -> None:
    queries = read_jsonl(DEFAULT_CHALLENGE)
    changed = [dict(query) for query in queries]
    changed[4]["trap_target_chunk_ids"] = [changed[4]["relevant_target_chunk_ids"][0]]
    corpus_ids = {
        str(target_id)
        for query in queries
        for field in ("relevant_target_chunk_ids", "trap_target_chunk_ids")
        for target_id in query.get(field, [])
    }

    with pytest.raises(ValueError, match="overlaps relevant and trap IDs"):
        validate_challenge(changed, corpus_ids)


def test_challenge_summary_keeps_audit_and_trap_results_descriptive() -> None:
    rows = [
        {
            "challenge_category": "citation_evidence_trap",
            "required_claims": ["claim"],
            "trap_avoidance": True,
            "metrics": {
                "status_correct": 1,
                "expected_status": "supported",
                "citations": {
                    "citation_relevant_hit": 1,
                    "citation_complete": 1,
                    "citation_precision": 1.0,
                    "citation_support_hit": 1,
                    "citation_support_complete": 1,
                    "citation_support_precision": 1.0,
                },
                "structured": {},
            },
            "answer": {
                "generation": {"request_count": 1},
                "evidence_audit": {"status": "review_recommended"},
            },
        }
    ]

    summary = summarize_challenge(rows)

    assert summary["trap_avoidance_count"] == 1
    assert summary["evidence_audit_status_counts"] == {"review_recommended": 1}
    assert summary["manual_claim_review_case_count"] == 1
