from bankscope.io import read_jsonl
from scripts.evaluate_answers import select_generation_queries, validate_generation_queries


def test_frozen_queries_select_exact_single_bank_scope() -> None:
    queries = read_jsonl("data/evaluation/queries.jsonl")

    validate_generation_queries(queries)
    eligible, skipped = select_generation_queries(queries)

    assert len(eligible) == 26
    assert sum(query["status"] == "answerable" for query in eligible) == 25
    assert sum(query["status"] == "unsupported" for query in eligible) == 1
    assert len(skipped) == 4
    assert sum(item["reason"] == "cross_bank_out_of_single_bank_scope" for item in skipped) == 3
    assert sum(item["reason"] == "missing_ticker_out_of_single_bank_scope" for item in skipped) == 1


def test_query_id_filter_keeps_excluded_query_visible() -> None:
    queries = read_jsonl("data/evaluation/queries.jsonl")

    eligible, skipped = select_generation_queries(queries, ["dev_ambiguous_bank_cet1_2025"])

    assert eligible == []
    assert skipped[0]["query_id"] == "dev_ambiguous_bank_cet1_2025"
