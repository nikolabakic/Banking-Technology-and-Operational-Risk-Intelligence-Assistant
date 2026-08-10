import json

import numpy as np
from qdrant_client import QdrantClient, models

from bankscope.io import sha256_file, write_jsonl
from bankscope.retrieval.qdrant_retriever import QdrantRetriever


def test_persistent_qdrant_search_filters_and_table_hydration(tmp_path) -> None:
    qdrant_path = tmp_path / "qdrant"
    tables_path = tmp_path / "tables.jsonl"
    manifest_path = tmp_path / "qdrant_manifest.json"
    collection = "fixture"
    write_jsonl(
        tables_path,
        [{"table_id": "table-1", "document": "| Metric | Value |\n| --- | --- |\n| Risk | 42 |"}],
    )

    payloads = [
        {
            "record_id": "jpm-risk",
            "target_chunk_id": "jpm-risk",
            "record_type": "text",
            "ticker": "JPM",
            "bank_name": "JPMorgan Chase & Co.",
            "embedding_text": "operational risk capital",
            "document": "Operational risk evidence.",
            "metadata": {"ticker": "JPM"},
        },
        {
            "record_id": "bac-liquidity",
            "target_chunk_id": "bac-liquidity",
            "record_type": "text",
            "ticker": "BAC",
            "bank_name": "Bank of America Corporation",
            "embedding_text": "liquidity coverage ratio",
            "document": "Liquidity evidence.",
            "metadata": {"ticker": "BAC"},
        },
        {
            "record_id": "jpm-table-description",
            "target_chunk_id": "table-1",
            "table_id": "table-1",
            "record_type": "table",
            "ticker": "JPM",
            "bank_name": "JPMorgan Chase & Co.",
            "embedding_text": "operational risk table",
            "document": "Description used for retrieval.",
            "metadata": {"ticker": "JPM", "table_id": "table-1"},
        },
    ]
    dense = ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0])
    client = QdrantClient(path=str(qdrant_path))
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upsert(
        collection,
        points=[
            models.PointStruct(
                id=index,
                vector={
                    "dense": dense[index - 1],
                    "sparse": models.Document(
                        text=payloads[index - 1]["embedding_text"], model="Qdrant/bm25"
                    ),
                },
                payload=payloads[index - 1],
            )
            for index in range(1, 4)
        ],
        wait=True,
    )
    client.close()

    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "collection_name": collection,
                "point_count": 3,
                "dense_dimension": 4,
                "dense_model": {"name": "fixture", "revision": "fixture"},
                "sparse_model": "Qdrant/bm25",
                "sources": {
                    "tables": {"path": str(tables_path), "sha256": sha256_file(tables_path)}
                },
            }
        ),
        encoding="utf-8",
    )

    retriever = QdrantRetriever(
        qdrant_path,
        [{"table_id": "table-1", "document": "| Metric | Value |\n| --- | --- |\n| Risk | 42 |"}],
        manifest_path=manifest_path,
        tables_path=tables_path,
    )
    try:
        [table_result] = retriever.search_dense(
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            limit=1,
            ticker="jpm",
            record_type="TABLE",
        )
        assert table_result["target_chunk_id"] == "table-1"
        assert table_result["document"].startswith("| Metric | Value |")

        [sparse_result] = retriever.search_bm25("liquidity coverage", limit=1, ticker="bac")
        assert sparse_result["record_id"] == "bac-liquidity"

        hybrid = retriever.search_hybrid(
            "operational risk",
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            limit=2,
            candidate_k=3,
            ticker="JPM",
        )
        assert {result["record_id"] for result in hybrid} == {
            "jpm-risk",
            "jpm-table-description",
        }
    finally:
        retriever.close()
