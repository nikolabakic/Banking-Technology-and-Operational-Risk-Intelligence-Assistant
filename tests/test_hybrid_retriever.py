from typing import Any

import numpy as np
import pytest

from bankscope.retrieval.hybrid_retriever import (
    HybridRetriever,
    normalize_lexical_text,
    reciprocal_rank_fusion,
)


def record(
    record_id: str,
    *,
    ticker: str = "JPM",
    record_type: str = "text",
    embedding_text: str | None = None,
    document: str | None = None,
    table_id: str | None = None,
) -> dict[str, Any]:
    metadata = {"ticker": ticker}
    if table_id:
        metadata["table_id"] = table_id
    return {
        "record_id": record_id,
        "target_chunk_id": table_id or record_id,
        "record_type": record_type,
        "embedding_text": embedding_text or f"retrieval text {record_id}",
        "document": document or f"evidence {record_id}",
        "metadata": metadata,
    }


def ranked(target_id: str, method: str, rank: int) -> dict[str, Any]:
    return {
        "record_index": rank - 1,
        "record_id": f"record::{target_id}",
        "target_chunk_id": target_id,
        "record_type": "text",
        "ticker": "JPM",
        "embedding_text": target_id,
        "retrieval_text": target_id,
        "document": target_id,
        "evidence": target_id,
        "metadata": {},
        "retrieval_method": method,
        "rank": rank,
        "score": 1.0 / rank,
    }


def test_bm25_indexes_embedding_text_and_hydrates_table_evidence() -> None:
    records = [
        record("narrative", embedding_text="operational risk controls"),
        record(
            "table-description",
            record_type="table",
            embedding_text="Liquidity coverage ratio was 115 percent in 2025",
            document="Short generated table description",
            table_id="table-7",
        ),
    ]
    tables = [
        {
            "table_id": "table-7",
            "document": "| Metric | 2025 |\n|---|---|\n| LCR | 115% |",
            "metadata": {"ticker": "JPM"},
        }
    ]
    retriever = HybridRetriever(records, tables=tables)

    [result] = retriever.search_bm25("2025 liquidity coverage 115%", limit=1, record_type="TABLE")

    assert result["target_chunk_id"] == "table-7"
    assert result["retrieval_text"].startswith("Liquidity coverage")
    assert result["embedding_text"] == result["retrieval_text"]
    assert result["document"].startswith("| Metric")
    assert result["evidence"] == result["document"]


def test_bm25_uses_glossary_locators_and_deduplicates_parent_tables() -> None:
    parent = record(
        "table-description",
        record_type="table",
        embedding_text="Acronym table",
        table_id="table-7",
    )
    locators = [
        record(
            "bana-locator",
            record_type="table",
            embedding_text="BANA stands for Bank of America National Association",
            table_id="table-7",
        ),
        record(
            "bana-duplicate-locator",
            record_type="table",
            embedding_text="BANA definition Bank of America National Association",
            table_id="table-7",
        ),
    ]
    tables = [{"table_id": "table-7", "document": "| BANA | Full definition |"}]
    retriever = HybridRetriever([parent], tables=tables, lexical_records=locators)

    results = retriever.search_bm25("What does BANA stand for?", limit=3)

    assert [result["target_chunk_id"] for result in results] == ["table-7"]
    assert results[0]["record_id"] == "bana-locator"
    assert results[0]["document"] == "| BANA | Full definition |"


def test_dense_search_applies_ticker_and_record_type_filters() -> None:
    records = [
        record("jpm-text", ticker="JPM"),
        record("wfc-text", ticker="WFC"),
        record("wfc-table", ticker="WFC", record_type="table", table_id="wfc-table"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.9, 0.1]], dtype=np.float32)
    tables = [{"table_id": "wfc-table", "document": "| WFC table |"}]
    retriever = HybridRetriever(records, embeddings, tables)

    results = retriever.search_dense(
        np.asarray([1.0, 0.0]), limit=3, ticker="wfc", record_type="text"
    )

    assert [result["record_id"] for result in results] == ["wfc-text"]
    assert results[0]["ticker"] == "WFC"


def test_dense_search_preserves_record_order_when_scores_tie() -> None:
    records = [record("first"), record("second")]
    embeddings = np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    retriever = HybridRetriever(records, embeddings)

    results = retriever.search_dense(np.asarray([1.0, 0.0]), limit=2)

    assert [result["record_id"] for result in results] == ["first", "second"]


def test_reciprocal_rank_fusion_is_deduplicated_and_deterministic() -> None:
    results = reciprocal_rank_fusion(
        [ranked("a", "dense", 1), ranked("b", "dense", 2)],
        [ranked("b", "bm25", 1), ranked("c", "bm25", 2)],
        limit=3,
        rrf_k=60,
    )

    assert [result["target_chunk_id"] for result in results] == ["b", "a", "c"]
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    assert results[0]["score"] == results[0]["rrf_score"]
    assert [result["rank"] for result in results] == [1, 2, 3]


def test_hybrid_fuses_candidate_window_before_applying_output_limit() -> None:
    records = [
        record("a", embedding_text="generic evidence"),
        record("b", embedding_text="special evidence"),
    ]
    retriever = HybridRetriever(records, np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))

    results = retriever.search_hybrid("special", np.asarray([1.0, 0.0]), limit=1, candidate_k=2)

    assert [result["target_chunk_id"] for result in results] == ["b"]


def test_retriever_rejects_shape_order_and_query_errors() -> None:
    records = [record("first"), record("second")]

    with pytest.raises(ValueError, match="counts do not match"):
        HybridRetriever(records, np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        HybridRetriever([records[0], records[0]], np.eye(2, dtype=np.float32))

    retriever = HybridRetriever(records, np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions do not match"):
        retriever.search_dense(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="zero norm"):
        retriever.search_dense(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="No records match"):
        retriever.search_bm25("risk", ticker="BAC")
    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search_hybrid("risk", np.ones(2), limit=2, candidate_k=1)


def test_bm25_does_not_require_embeddings_but_dense_does() -> None:
    retriever = HybridRetriever([record("risk", embedding_text="operational risk")])

    assert retriever.search_bm25("operational", limit=1)[0]["record_id"] == "risk"
    with pytest.raises(ValueError, match="requires document embeddings"):
        retriever.search_dense(np.ones(2))
    with pytest.raises(ValueError, match="requires document embeddings"):
        retriever.search_hybrid("risk", np.ones(2))


def test_table_records_require_a_complete_table_store() -> None:
    table_record = record("description", record_type="table", table_id="table-1")

    with pytest.raises(ValueError, match="table store is required"):
        HybridRetriever([table_record])
    with pytest.raises(ValueError, match="unknown table IDs"):
        HybridRetriever([table_record], tables=[])


def test_financial_lexical_normalization_removes_numeric_commas() -> None:
    assert normalize_lexical_text("Assets were 12,345 \u2014 unchanged") == (
        "Assets were 12345 - unchanged"
    )
