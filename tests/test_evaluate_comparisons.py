from scripts.evaluate_comparisons import evaluate_comparison_run, select_comparison_queries


def test_select_comparison_queries_requires_frozen_shape() -> None:
    rows = [
        {
            "query_id": f"q{index}",
            "question_type": "cross_bank_coverage",
            "required_evidence_groups": [{}, {}],
        }
        for index in range(3)
    ]
    assert len(select_comparison_queries(rows)) == 3


def test_comparison_checks_group_coverage_and_citation_ownership() -> None:
    query = {
        "required_evidence_groups": [
            {"entity": "Bank A", "target_chunk_ids": ["a"]},
            {"entity": "Bank B", "target_chunk_ids": ["b"]},
        ]
    }
    output = {
        "status": "supported",
        "tickers": ["A", "B"],
        "bank_results": [
            {
                "ticker": "A",
                "status": "supported",
                "citations": [{"label": "E1", "ticker": "A"}],
            },
            {
                "ticker": "B",
                "status": "supported",
                "citations": [{"label": "E2", "ticker": "B"}],
            },
        ],
    }
    checks = evaluate_comparison_run(
        query,
        output,
        [{"target_chunk_id": "a"}, {"target_chunk_id": "b"}],
        entity_tickers={"Bank A": "A", "Bank B": "B"},
    )

    assert checks["retrieval_complete"] is True
    assert checks["expected_tickers_supported"] is True
    assert checks["citation_ownership_violations"] == []
