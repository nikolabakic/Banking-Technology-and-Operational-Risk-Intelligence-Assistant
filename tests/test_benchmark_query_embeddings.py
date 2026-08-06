from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.benchmark_query_embeddings import DEFAULT_QUERIES, load_query_texts


def test_load_query_texts_and_limit() -> None:
    queries = load_query_texts(DEFAULT_QUERIES, limit=1)

    assert len(queries) == 1
    assert queries[0]


@pytest.mark.parametrize("limit", [0, -1])
def test_load_query_texts_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="--limit must be at least 1"):
        load_query_texts(DEFAULT_QUERIES, limit=limit)


def test_load_query_texts_rejects_empty_query() -> None:
    with (
        patch("scripts.benchmark_query_embeddings.read_jsonl", return_value=[{"query": ""}]),
        pytest.raises(ValueError, match="non-empty 'query'"),
    ):
        load_query_texts(DEFAULT_QUERIES)
