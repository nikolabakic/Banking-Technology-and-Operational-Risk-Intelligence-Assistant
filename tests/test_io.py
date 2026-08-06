import hashlib
from pathlib import Path

import numpy as np
import pytest

from bankscope.io import load_embedding_archive, read_jsonl, sha256_file, write_jsonl


def test_jsonl_round_trip_is_utf8_and_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    records = [{"record_id": "a", "text": "štednja"}, {"record_id": "b", "value": 2}]

    write_jsonl(path, records)
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert read_jsonl(path) == records


def test_read_jsonl_reports_line_for_invalid_record(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\n[1, 2]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"object on line 2"):
        read_jsonl(path)


def test_write_jsonl_does_not_replace_file_on_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"original":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="record 2"):
        write_jsonl(path, [{"valid": True}, {"invalid": {1, 2}}])

    assert path.read_text(encoding="utf-8") == '{"original":true}\n'


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    content = b"BankScope\x00data"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def _write_embedding_archive(path: Path, **changes: object) -> None:
    values: dict[str, object] = {
        "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "record_ids": np.asarray(["first", "second"]),
        "model_name": np.asarray("test/model"),
        "model_revision": np.asarray("abc123"),
        "input_sha256": np.asarray("a" * 64),
    }
    values.update(changes)
    np.savez(path, **values)


def test_load_embedding_archive_returns_validated_values(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    _write_embedding_archive(path)

    archive = load_embedding_archive(path, expected_record_ids=["first", "second"])

    assert archive["record_ids"] == ["first", "second"]
    assert archive["model_name"] == "test/model"
    assert archive["model_revision"] == "abc123"
    assert archive["input_sha256"] == "a" * 64
    np.testing.assert_array_equal(
        archive["embeddings"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"embeddings": np.ones((2, 2), dtype=np.float64)}, "float32"),
        ({"embeddings": np.ones((2, 2), dtype=np.float32)}, "unit norm"),
        ({"record_ids": np.asarray(["same", "same"])}, "unique"),
        ({"record_ids": np.asarray(["only-one"])}, "row count"),
        ({"input_sha256": np.asarray("not-a-hash")}, "SHA-256"),
    ],
)
def test_load_embedding_archive_rejects_invalid_data(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "invalid.npz"
    _write_embedding_archive(path, **changes)

    with pytest.raises(ValueError, match=message):
        load_embedding_archive(path)


def test_load_embedding_archive_checks_expected_order(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    _write_embedding_archive(path)

    with pytest.raises(ValueError, match="expected order"):
        load_embedding_archive(path, expected_record_ids=["second", "first"])


def test_load_embedding_archive_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(
        path,
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        record_ids=np.asarray(["first"]),
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_embedding_archive(path)
