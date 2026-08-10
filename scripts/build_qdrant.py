"""Build the persistent local Qdrant retrieval collection from frozen artifacts."""

from __future__ import annotations

import argparse
import json
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from bankscope.io import load_embedding_archive, read_jsonl, sha256_file
from bankscope.retrieval.hybrid_retriever import get_field, get_retrieval_text
from bankscope.retrieval.qdrant_retriever import (
    DEFAULT_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_MODEL_NAME,
    SPARSE_VECTOR_NAME,
)
from bankscope.sec.company_registry import load_bank_registry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"
DEFAULT_CORPUS_MANIFEST = ROOT / "data/processed/manifest.json"
DEFAULT_QDRANT_PATH = ROOT / "data/processed/qdrant"
DEFAULT_QDRANT_MANIFEST = ROOT / "data/processed/qdrant_manifest.json"
DEFAULT_BANKS = ROOT / "config/banks.yaml"
EXPECTED_DENSE_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--banks", type=Path, default=DEFAULT_BANKS)
    parser.add_argument("--path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def _table_index(tables: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for table in tables:
        table_id = str(get_field(table, "table_id") or "").strip()
        if not table_id or not str(table.get("document") or "").strip():
            raise ValueError("Every table must have a table_id and document.")
        if table_id in indexed:
            raise ValueError(f"Duplicate table_id: {table_id}.")
        indexed[table_id] = table
    return indexed


def _validate_corpus_manifest(
    path: Path, *, record_count: int, table_point_count: int
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load corpus manifest: {path}") from error
    if int(manifest.get("chunk_count", -1)) != record_count:
        raise ValueError("Corpus manifest chunk_count does not match chunks.jsonl.")
    if int(manifest.get("table_chunk_count", -1)) != table_point_count:
        raise ValueError("Corpus manifest table_chunk_count does not match chunks.jsonl.")
    return manifest


def _payload(record: dict[str, Any], bank_names: dict[str, str]) -> dict[str, Any]:
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    ticker = str(get_field(record, "ticker") or "").upper()
    record_type = str(get_field(record, "record_type") or "").lower()
    target_chunk_id = str(record.get("target_chunk_id") or "").strip()
    if not ticker or ticker not in bank_names:
        raise ValueError(f"Record {record.get('record_id')} has an unknown ticker: {ticker}.")
    if record_type not in {"text", "table"} or not target_chunk_id:
        raise ValueError(f"Record {record.get('record_id')} has invalid canonical fields.")
    payload = {
        "record_id": str(record["record_id"]),
        "target_chunk_id": target_chunk_id,
        "record_type": record_type,
        "ticker": ticker,
        "bank_name": bank_names[ticker],
        "embedding_text": get_retrieval_text(record),
        "document": str(record.get("document") or get_retrieval_text(record)),
        "metadata": metadata,
    }
    if record_type == "table":
        payload["table_id"] = str(get_field(record, "table_id") or target_chunk_id)
    return payload


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    records = read_jsonl(args.chunks)
    tables = read_jsonl(args.tables)
    record_ids = [str(record.get("record_id") or "").strip() for record in records]
    if any(not record_id for record_id in record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("Retrieval record IDs must be non-empty and unique.")
    chunks_sha256 = sha256_file(args.chunks)
    archive = load_embedding_archive(args.embeddings, expected_record_ids=record_ids)
    if archive["input_sha256"] != chunks_sha256:
        raise ValueError("Embedding archive input hash does not match chunks.jsonl.")
    embeddings = archive["embeddings"]
    if embeddings.shape[1] != EXPECTED_DENSE_DIMENSION:
        raise ValueError(f"Expected {EXPECTED_DENSE_DIMENSION}-dimensional dense vectors.")

    tables_by_id = _table_index(tables)
    table_ids = {
        str(get_field(record, "table_id") or record.get("target_chunk_id") or "")
        for record in records
        if str(get_field(record, "record_type") or "").lower() == "table"
    }
    if missing_tables := table_ids - tables_by_id.keys():
        raise ValueError(f"Table records reference unknown table IDs: {sorted(missing_tables)}")
    corpus_manifest = _validate_corpus_manifest(
        args.corpus_manifest, record_count=len(records), table_point_count=len(table_ids)
    )
    registry = load_bank_registry(args.banks)
    bank_names = {bank.ticker: bank.legal_name for bank in registry.banks}
    payloads = [_payload(record, bank_names) for record in records]
    point_ids = [str(uuid5(NAMESPACE_URL, record_id)) for record_id in record_ids]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("Deterministic Qdrant point IDs are not unique.")

    args.path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(args.path))
    try:
        exists = client.collection_exists(args.collection)
        if exists and not args.recreate:
            raise ValueError(
                f"Collection '{args.collection}' already exists; pass --recreate to rebuild it."
            )
        if exists:
            client.delete_collection(args.collection)
        client.create_collection(
            collection_name=args.collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=EXPECTED_DENSE_DIMENSION, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Payload indexes have no effect in the local Qdrant.*"
            )
            for field in ("ticker", "record_type"):
                client.create_payload_index(
                    args.collection, field, field_schema=models.PayloadSchemaType.KEYWORD
                )

        for start in range(0, len(records), args.batch_size):
            stop = min(start + args.batch_size, len(records))
            points = [
                models.PointStruct(
                    id=point_ids[index],
                    vector={
                        DENSE_VECTOR_NAME: embeddings[index].tolist(),
                        SPARSE_VECTOR_NAME: models.Document(
                            text=payloads[index]["embedding_text"], model=SPARSE_MODEL_NAME
                        ),
                    },
                    payload=payloads[index],
                )
                for index in range(start, stop)
            ]
            client.upsert(args.collection, points=points, wait=True)
            print(f"Indexed {stop}/{len(records)}")

        info = client.get_collection(args.collection)
        if info.points_count != len(records):
            raise ValueError(f"Qdrant point count mismatch: {info.points_count} != {len(records)}")
        vectors = info.config.params.vectors
        dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if dense is None or dense.size != EXPECTED_DENSE_DIMENSION:
            raise ValueError("Qdrant dense vector schema validation failed.")
    finally:
        client.close()

    qdrant_manifest = {
        "format_version": 1,
        "collection_name": args.collection,
        "qdrant_client_version": version("qdrant-client"),
        "point_count": len(records),
        "text_point_count": len(records) - len(table_ids),
        "table_point_count": len(table_ids),
        "dense_dimension": EXPECTED_DENSE_DIMENSION,
        "dense_model": {
            "name": archive["model_name"],
            "revision": archive["model_revision"],
        },
        "sparse_model": SPARSE_MODEL_NAME,
        "candidate_k": 30,
        "rrf_k": 60,
        "corpus_version": corpus_manifest.get("corpus_version"),
        "sources": {
            "chunks": {"path": str(args.chunks.resolve()), "sha256": chunks_sha256},
            "tables": {
                "path": str(args.tables.resolve()),
                "sha256": sha256_file(args.tables),
            },
            "embeddings": {
                "path": str(args.embeddings.resolve()),
                "sha256": sha256_file(args.embeddings),
            },
        },
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(qdrant_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "collection": args.collection,
                "point_count": len(records),
                "table_point_count": len(table_ids),
                "dense_dimension": EXPECTED_DENSE_DIMENSION,
                "manifest": str(args.output_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
