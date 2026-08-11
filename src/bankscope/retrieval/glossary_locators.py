from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from bankscope.parsing.tables import extract_table_title, is_table_value, normalize_space

GLOSSARY_LOCATOR_VERSION = "bankscope-glossary-locator-v1"
GLOSSARY_TITLE_MARKERS = ("acronym", "glossary")
HEADER_KEYS = {"abbreviation", "acronym", "term"}
HEADER_VALUES = {"definition", "description", "meaning"}

Record = dict[str, Any]


def is_glossary_table(table: Mapping[str, Any]) -> bool:
    """Recognize glossary and acronym tables without depending on stored classification."""
    if str(table.get("table_type") or "").casefold() == "glossary":
        return True

    title = extract_table_title(str(table.get("document") or "")) or ""
    compact_title = re.sub(r"[^a-z]+", "", title.casefold())
    return any(marker in compact_title for marker in GLOSSARY_TITLE_MARKERS)


def _text_pairs(table: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for matrix in table.get("cell_matrices") or []:
        for row in matrix:
            cells = [normalize_space(cell) for cell in row]
            for index in range(0, len(cells) - 1, 2):
                key, value = cells[index], cells[index + 1]
                if not key or not value or is_table_value(key) or is_table_value(value):
                    continue
                if not any(character.isalpha() for character in key):
                    continue
                if not any(character.isalpha() for character in value):
                    continue
                if key.casefold() in HEADER_KEYS and value.casefold() in HEADER_VALUES:
                    continue
                pair = (key, value)
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _context_prefix(parent: Mapping[str, Any]) -> str:
    embedding_text = str(parent.get("embedding_text") or "")
    return embedding_text.split("\n\n", maxsplit=1)[0].strip()


def build_glossary_locators(
    records: Sequence[Mapping[str, Any]], tables: Sequence[Mapping[str, Any]]
) -> list[Record]:
    """Build small lexical-only entries that hydrate back to complete glossary tables."""
    parents = {
        str(record.get("target_chunk_id") or ""): record
        for record in records
        if str(record.get("record_type") or "").casefold() == "table"
    }
    locators: list[Record] = []
    for table in tables:
        if not is_glossary_table(table):
            continue
        table_id = str(table.get("table_id") or "").strip()
        parent = parents.get(table_id)
        if parent is None:
            continue
        metadata = parent.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        for key, value in _text_pairs(table):
            locator_text = f"{key} stands for {value}."
            identity = f"{table_id}\0{key}\0{value}"
            locator_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            locator_metadata = {
                **metadata,
                "locator_type": "glossary_pair",
                "locator_version": GLOSSARY_LOCATOR_VERSION,
                "locator_key": key,
                "locator_value": value,
                "source_record_id": str(parent.get("record_id") or ""),
            }
            context = _context_prefix(parent)
            content = "\n".join(
                [
                    "Glossary definition",
                    f"Abbreviation: {key}",
                    locator_text,
                    f"Definition: {key}: {value}",
                ]
            )
            locators.append(
                {
                    "record_id": f"glossary_locator::{table_id}::{locator_id}",
                    "target_chunk_id": table_id,
                    "record_type": "table",
                    "embedding_text": f"{context}\n\n{content}" if context else content,
                    "document": locator_text,
                    "metadata": locator_metadata,
                }
            )
    return locators


def validate_glossary_locators(
    locators: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    tables: Sequence[Mapping[str, Any]],
) -> None:
    record_ids = [str(locator.get("record_id") or "").strip() for locator in locators]
    if any(not record_id for record_id in record_ids):
        raise ValueError("Every glossary locator must have a record_id.")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Glossary locator record IDs must be unique.")

    parent_ids = {
        str(record.get("target_chunk_id") or "")
        for record in records
        if str(record.get("record_type") or "").casefold() == "table"
    }
    table_ids = {str(table.get("table_id") or "") for table in tables}
    for locator in locators:
        target_id = str(locator.get("target_chunk_id") or "").strip()
        metadata = locator.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if str(locator.get("record_type") or "").casefold() != "table":
            raise ValueError("Glossary locators must use record_type='table'.")
        if target_id not in parent_ids or target_id not in table_ids:
            raise ValueError(f"Glossary locator references unknown parent table: {target_id}.")
        if metadata.get("locator_version") != GLOSSARY_LOCATOR_VERSION:
            raise ValueError(f"Glossary locator {locator['record_id']} has an invalid version.")
        if not str(locator.get("embedding_text") or "").strip():
            raise ValueError(f"Glossary locator {locator['record_id']} has no retrieval text.")
