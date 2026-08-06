"""Small, shared helpers for BankScope data files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

JsonRecord = dict[str, Any]
EmbeddingArchive = dict[str, Any]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    """Read JSON objects from *path*, ignoring empty lines."""

    input_path = Path(path)
    records: list[JsonRecord] = []

    try:
        input_file = input_path.open(encoding="utf-8")
    except OSError as error:
        raise OSError(f"Cannot open JSONL file: {input_path}") from error

    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}: {error.msg}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {input_path}")

            records.append(value)

    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write mappings as UTF-8 JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)

            for record_number, record in enumerate(records, start=1):
                if not isinstance(record, Mapping):
                    raise TypeError(f"JSONL record {record_number} is not a mapping")

                try:
                    line = json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"JSONL record {record_number} is not JSON serializable"
                    ) from error

                output_file.write(line + "\n")

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    input_path = Path(path)
    digest = hashlib.sha256()

    try:
        input_file = input_path.open("rb")
    except OSError as error:
        raise OSError(f"Cannot open file for hashing: {input_path}") from error

    with input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _read_scalar_text(archive: Any, key: str, path: Path) -> str:
    value = np.asarray(archive[key])

    if value.ndim != 0:
        raise ValueError(f"Embedding archive field '{key}' must be a scalar: {path}")

    item = value.item()

    if isinstance(item, bytes):
        try:
            text = item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Embedding archive field '{key}' is not valid UTF-8: {path}"
            ) from error
    elif isinstance(item, str):
        text = item
    else:
        raise ValueError(f"Embedding archive field '{key}' must be text: {path}")

    if not text.strip():
        raise ValueError(f"Embedding archive field '{key}' cannot be empty: {path}")

    return text


def load_embedding_archive(
    path: str | Path,
    expected_record_ids: Sequence[str] | None = None,
) -> EmbeddingArchive:
    """Load an embedding NPZ and validate its vectors, IDs, and provenance."""

    archive_path = Path(path)
    required_keys = {
        "embeddings",
        "record_ids",
        "model_name",
        "model_revision",
        "input_sha256",
    }

    try:
        archive_context = np.load(archive_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot load embedding archive: {archive_path}") from error

    with archive_context as archive:
        missing_keys = sorted(required_keys - set(archive.files))

        if missing_keys:
            raise ValueError(
                f"Embedding archive is missing required fields {missing_keys}: {archive_path}"
            )

        embeddings = np.array(archive["embeddings"], copy=True)
        raw_record_ids = np.asarray(archive["record_ids"])
        model_name = _read_scalar_text(archive, "model_name", archive_path)
        model_revision = _read_scalar_text(archive, "model_revision", archive_path)
        input_sha256 = _read_scalar_text(archive, "input_sha256", archive_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding matrix must be two-dimensional, got {embeddings.shape}: {archive_path}"
        )

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Embedding matrix cannot be empty: {archive_path}")

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Embedding matrix must use float32, got {embeddings.dtype}: {archive_path}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(f"Embedding matrix contains NaN or infinite values: {archive_path}")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        maximum_deviation = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(
            "Embedding vectors must have unit norm; "
            f"maximum deviation is {maximum_deviation:.8f}: {archive_path}"
        )

    if raw_record_ids.ndim != 1:
        raise ValueError(f"record_ids must be a one-dimensional array: {archive_path}")

    if raw_record_ids.dtype.kind not in {"S", "U"}:
        raise ValueError(f"record_ids must contain strings: {archive_path}")

    record_ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_record_ids.tolist()
    ]

    if embeddings.shape[0] != len(record_ids):
        raise ValueError(
            "Embedding row count does not match record_ids: "
            f"{embeddings.shape[0]} != {len(record_ids)} ({archive_path})"
        )

    if any(not record_id.strip() for record_id in record_ids):
        raise ValueError(f"record_ids cannot contain empty values: {archive_path}")

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"record_ids must be unique: {archive_path}")

    if not _SHA256_PATTERN.fullmatch(input_sha256):
        raise ValueError(f"input_sha256 must be a lowercase SHA-256 digest: {archive_path}")

    if expected_record_ids is not None:
        if isinstance(expected_record_ids, (str, bytes)):
            raise TypeError("expected_record_ids must be a sequence of strings")

        expected = list(expected_record_ids)

        if any(not isinstance(record_id, str) for record_id in expected):
            raise TypeError("expected_record_ids must contain only strings")

        if record_ids != expected:
            raise ValueError(
                f"Embedding record_ids do not match the expected order: {archive_path}"
            )

    return {
        "embeddings": embeddings,
        "record_ids": record_ids,
        "model_name": model_name,
        "model_revision": model_revision,
        "input_sha256": input_sha256,
    }
