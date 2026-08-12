"""Resolve persisted citations against the active canonical corpus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bankscope.io import read_jsonl, sha256_file


class StaleCitationError(RuntimeError):
    """Raised when a citation belongs to a different corpus revision."""


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, Mapping) else {}


class CitationSourceResolver:
    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        tables: Sequence[Mapping[str, Any]],
        *,
        corpus_hash: str,
    ) -> None:
        self.chunks = [dict(record) for record in chunks]
        self.tables = {
            str(record.get("table_id") or record.get("target_chunk_id")): dict(record)
            for record in tables
        }
        self.by_target = {
            str(record["target_chunk_id"]): (index, record)
            for index, record in enumerate(self.chunks)
        }
        self.corpus_hash = corpus_hash

    @classmethod
    def from_paths(cls, chunks_path: str | Path, tables_path: str | Path) -> CitationSourceResolver:
        chunks_path = Path(chunks_path)
        return cls(
            read_jsonl(chunks_path),
            read_jsonl(tables_path),
            corpus_hash=sha256_file(chunks_path),
        )

    @staticmethod
    def _canonical_document(
        record: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]]
    ) -> str:
        if str(record.get("record_type") or "").lower() == "table":
            target = str(record.get("target_chunk_id") or "")
            table = tables.get(target)
            if table is not None:
                return str(table.get("document") or "")
        return str(record.get("document") or "")

    def context(
        self,
        target_chunk_id: str,
        *,
        expected_corpus_hash: str,
        radius: int = 1,
    ) -> dict[str, Any]:
        if expected_corpus_hash != self.corpus_hash:
            raise StaleCitationError("The citation was created from a different corpus revision.")
        found = self.by_target.get(target_chunk_id)
        if found is None:
            raise KeyError(target_chunk_id)
        index, anchor = found
        metadata = dict(_metadata(anchor))
        record_type = str(anchor.get("record_type") or metadata.get("record_type") or "")
        context_records: list[dict[str, Any]] = []
        if record_type.lower() == "text":
            accession = str(metadata.get("accession_number") or "")
            start = max(0, index - radius)
            end = min(len(self.chunks), index + radius + 1)
            for candidate_index in range(start, end):
                candidate = self.chunks[candidate_index]
                candidate_metadata = dict(_metadata(candidate))
                if str(candidate_metadata.get("accession_number") or "") != accession:
                    continue
                context_records.append(
                    {
                        "target_chunk_id": str(candidate.get("target_chunk_id") or ""),
                        "role": (
                            "anchor"
                            if candidate_index == index
                            else "previous"
                            if candidate_index < index
                            else "next"
                        ),
                        "record_type": str(candidate.get("record_type") or ""),
                        "document": self._canonical_document(candidate, self.tables),
                        "metadata": candidate_metadata,
                    }
                )
        else:
            context_records.append(
                {
                    "target_chunk_id": target_chunk_id,
                    "role": "anchor",
                    "record_type": record_type,
                    "document": self._canonical_document(anchor, self.tables),
                    "metadata": metadata,
                }
            )
        return {
            "target_chunk_id": target_chunk_id,
            "record_type": record_type,
            "ticker": str(anchor.get("ticker") or metadata.get("ticker") or ""),
            "source_url": str(metadata.get("source_url") or ""),
            "corpus_hash": self.corpus_hash,
            "chunks": context_records,
        }
