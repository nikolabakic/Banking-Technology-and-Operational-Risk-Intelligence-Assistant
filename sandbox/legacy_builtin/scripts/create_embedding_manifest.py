from __future__ import annotations

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/processed/chunks/sec_10k_chunks.jsonl"
PROXIES_PATH = ROOT / "data/processed/table_proxies/sec_10k_table_proxies.jsonl"
RECORDS_PATH = ROOT / "data/processed/embedding_records/sec_10k_embedding_records.jsonl"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings/qwen3_embedding_0_6b_records.npz"
MANIFEST_PATH = ROOT / "data/processed/embeddings/qwen3_embedding_0_6b_manifest.json"

PACKAGE_NAMES = (
    "numpy",
    "sentence-transformers",
    "transformers",
    "torch",
    "huggingface-hub",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def artifact_info(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def get_field(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]

    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get(field)

    return None


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}

    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not_installed"

    return versions


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def main() -> None:
    required_paths = (
        CHUNKS_PATH,
        PROXIES_PATH,
        RECORDS_PATH,
        EMBEDDINGS_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    records = load_jsonl(RECORDS_PATH)
    proxies = load_jsonl(PROXIES_PATH)

    record_ids = [str(get_field(record, "record_id")) for record in records]
    target_chunk_ids = [str(get_field(record, "target_chunk_id")) for record in records]

    record_type_counts = Counter(str(get_field(record, "record_type")) for record in records)
    ticker_counts = Counter(str(get_field(record, "ticker")) for record in records)

    ticker_type_counts: dict[str, dict[str, int]] = {}

    for record in records:
        ticker = str(get_field(record, "ticker"))
        record_type = str(get_field(record, "record_type"))

        ticker_type_counts.setdefault(ticker, {})
        ticker_type_counts[ticker][record_type] = ticker_type_counts[ticker].get(record_type, 0) + 1

    proxy_version_counts = Counter(str(proxy.get("proxy_version", "missing")) for proxy in proxies)

    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as archive:
        embeddings = archive["embeddings"]
        npz_record_ids = archive["record_ids"].astype(str).tolist()
        model_name = str(archive["model_name"].item())

        norms = np.linalg.norm(embeddings, axis=1)

        if embeddings.shape[0] != len(records):
            raise ValueError("Embedding and record counts do not match.")

        if npz_record_ids != record_ids:
            raise ValueError("NPZ and JSONL record ID order does not match.")

        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Duplicate record IDs detected.")

        if len(set(target_chunk_ids)) != len(target_chunk_ids):
            raise ValueError("Duplicate target chunk IDs detected.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or Inf values.")

        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Embeddings are not L2-normalized.")

        if len(proxies) != record_type_counts["table"]:
            raise ValueError("Table record and proxy counts do not match.")

        manifest = {
            "schema_version": 1,
            "baseline_name": "qwen3_embedding_0_6b_records",
            "git_commit": get_git_commit(),
            "artifacts": {
                "chunks": artifact_info(CHUNKS_PATH),
                "table_proxies": artifact_info(PROXIES_PATH),
                "embedding_records": artifact_info(RECORDS_PATH),
                "embeddings": artifact_info(EMBEDDINGS_PATH),
            },
            "corpus": {
                "chunk_count": count_lines(CHUNKS_PATH),
                "embedding_record_count": len(records),
                "unique_record_id_count": len(set(record_ids)),
                "unique_target_chunk_id_count": len(set(target_chunk_ids)),
                "record_type_counts": dict(sorted(record_type_counts.items())),
                "ticker_counts": dict(sorted(ticker_counts.items())),
                "ticker_type_counts": {
                    ticker: dict(sorted(type_counts.items()))
                    for ticker, type_counts in sorted(ticker_type_counts.items())
                },
            },
            "table_proxies": {
                "count": len(proxies),
                "version_counts": dict(sorted(proxy_version_counts.items())),
            },
            "embedding_model": {
                "model_name": model_name,
                "model_revision": None,
                "revision_status": "not_captured_at_generation",
            },
            "embedding_matrix": {
                "array_key": "embeddings",
                "record_id_key": "record_ids",
                "count": int(embeddings.shape[0]),
                "dimension": int(embeddings.shape[1]),
                "dtype": str(embeddings.dtype),
                "l2_normalized": True,
                "minimum_norm": float(norms.min()),
                "maximum_norm": float(norms.max()),
                "npz_jsonl_id_order_match": True,
            },
            "environment": {
                "generation_environment": "not_captured",
                "verification_environment": get_package_versions(),
            },
            "known_limitations": [
                "Model revision was not captured during generation.",
                "Generation library versions were not captured.",
                "USB and WFC filings contain partial primary documents.",
            ],
        }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Manifest written: {MANIFEST_PATH}")
    print(f"Records: {manifest['corpus']['embedding_record_count']}")
    print(f"Embedding SHA-256: {manifest['artifacts']['embeddings']['sha256']}")
    print(f"Model revision status: {manifest['embedding_model']['revision_status']}")


if __name__ == "__main__":
    main()
