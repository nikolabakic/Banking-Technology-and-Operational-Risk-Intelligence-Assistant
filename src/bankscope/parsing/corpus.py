from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from bankscope.parsing.tables import (
    PARSER_NAME,
    PARSER_VERSION,
    build_table_record,
    deduplicate,
    describe_table,
    extract_sec2md_table_grids,
    extract_table_title,
    make_chunk_id,
    parse_markdown_table_matrices,
)

CORPUS_VERSION = "bankscope-corpus-v1"
TEXT_TARGET_TOKENS = 512
TEXT_MAX_TOKENS = 1024
TEXT_OVERLAP_TOKENS = 64
MAX_EMBEDDING_TOKENS = 2048

SEC_ITEM_PATTERN = re.compile(r"(?im)^(?:\*\*|__)?\s*item\s+(\d{1,2}[a-c]?)\s*[.:]")
GLOSSARY_ENTRY_PATTERN = re.compile(r"\*\*[^*\n]{1,180}\*\*")

Record = dict[str, Any]
TokenCounter = Callable[[str], int]


def as_dict(value: Any) -> Record:
    if isinstance(value, Mapping):
        return dict(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    raise TypeError(f"Expected a mapping or Pydantic model, got {type(value)!r}.")


def extract_explicit_sec_item(content: str) -> tuple[str | None, str, float]:
    match = SEC_ITEM_PATTERN.search(content[:500])
    if match is None:
        return None, "unassigned", 0.0
    return f"Item {match.group(1).upper()}", "explicit_heading_in_chunk", 1.0


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def looks_like_navigation(content: str) -> bool:
    normalized = normalize_space(content).casefold()
    if not normalized:
        return True
    return (
        normalized.startswith("**form 10-k index**")
        or normalized.startswith("form 10-k index")
        or normalized.startswith("**table of contents**")
        or normalized.startswith("table of contents")
    )


def looks_like_page_furniture(content: str) -> bool:
    normalized = normalize_space(content)
    if not normalized or re.fullmatch(r"\d{1,4}", normalized):
        return True
    return bool(
        re.fullmatch(
            r"(?i)JPMorgan Chase\s*&\s*Co\.?/?\s*2025 Form 10-K\s*\|?\s*\d{1,4}",
            normalized,
        )
    )


def extract_section_title(content: str) -> str | None:
    match = re.match(r"\s*(?:\*\*|__)([^*\n]{1,160})(?:\*\*|__)", content)
    if match is None:
        return None
    return normalize_space(match.group(1)).rstrip(".:") or None


def is_heading_start(content: str) -> bool:
    title = extract_section_title(content)
    if title is None:
        return False
    first_block = content.strip().split("\n\n", maxsplit=1)[0]
    return len(first_block) <= 180


def split_markdown_units(content: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*(])", paragraph)
            if sentence.strip()
        ]
        units.extend(sentences or [paragraph])
    return units


def split_to_token_limit(
    content: str,
    token_count: TokenCounter,
    *,
    target_tokens: int = TEXT_TARGET_TOKENS,
    max_tokens: int = TEXT_MAX_TOKENS,
    overlap_tokens: int = TEXT_OVERLAP_TOKENS,
) -> list[str]:
    """Legacy narrative splitter retained so existing target IDs stay stable."""
    if token_count(content) <= max_tokens:
        return [content.strip()]

    units = split_markdown_units(content)
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return

        chunks.append("\n\n".join(current).strip())
        overlap: list[str] = []
        for unit in reversed(current):
            candidate = [unit, *overlap]
            if token_count("\n\n".join(candidate)) > overlap_tokens:
                break
            overlap = candidate
        current = overlap

    for unit in units:
        if token_count(unit) > max_tokens:
            words = unit.split()
            word_buffer: list[str] = []
            for word in words:
                candidate = " ".join([*word_buffer, word])
                if word_buffer and token_count(candidate) > max_tokens:
                    if current:
                        flush()
                    chunks.append(" ".join(word_buffer))
                    word_buffer = [word]
                else:
                    word_buffer.append(word)
            if word_buffer:
                if current:
                    flush()
                chunks.append(" ".join(word_buffer))
            continue

        candidate = "\n\n".join([*current, unit])
        if current and (
            token_count(candidate) > max_tokens
            or token_count("\n\n".join(current)) >= target_tokens
        ):
            flush()
        current.append(unit)

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def split_glossary_entries(content: str) -> tuple[str | None, list[str]]:
    content = re.sub(
        r"^\s*\*\*Glossary of Terms and Acronyms\*\*\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    matches = list(GLOSSARY_ENTRY_PATTERN.finditer(content))
    if not matches:
        return content.strip() or None, []

    prefix = content[: matches[0].start()].strip() or None
    entries = [
        content[match.start() : matches[index + 1].start()].strip()
        if index + 1 < len(matches)
        else content[match.start() :].strip()
        for index, match in enumerate(matches)
    ]
    return prefix, [entry for entry in entries if entry]


def page_maps(pages: Sequence[Any]) -> tuple[dict[int, int | None], set[int]]:
    display_pages: dict[int, int | None] = {}
    glossary_pages: set[int] = set()
    for source_page in pages:
        page = as_dict(source_page)
        page_number = int(page["number"])
        content = str(page.get("content") or "")
        display_pages[page_number] = page.get("display_page")
        if content.lstrip().casefold().startswith("**glossary of terms and acronyms**"):
            glossary_pages.add(page_number)
    return display_pages, glossary_pages


def ordered_elements(pages: Sequence[Any]) -> list[Record]:
    elements: list[Record] = []
    seen_ids: set[str] = set()
    for source_page in pages:
        page = as_dict(source_page)
        page_number = int(page["number"])
        for source_element in page.get("elements") or []:
            element = as_dict(source_element)
            element_id = str(element["id"])
            if element_id in seen_ids:
                continue
            seen_ids.add(element_id)
            element.setdefault("page_start", page_number)
            element.setdefault("page_end", page_number)
            elements.append(element)
    return elements


def _build_embedding_text(
    filing: Mapping[str, Any],
    *,
    content: str,
    record_type: str,
    page_start: int,
    page_end: int,
    section_title: str | None,
) -> str:
    year = str(filing.get("report_date", ""))[:4]
    lines = [
        f"Bank: {filing.get('ticker', '')}",
        f"Entity: {filing.get('legal_name', filing.get('ticker', ''))}",
        f"Report: {year} 10-K",
        f"Evidence type: {record_type}",
        f"Internal pages: {page_start}-{page_end}",
    ]
    if section_title:
        lines.append(f"Section: {section_title}")
    return "\n".join([*lines, "", content])


def _base_metadata(
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
    content: str,
    record_type: str,
    page_start: int,
    page_end: int,
    start_display_page: int | None,
    end_display_page: int | None,
    element_ids: Sequence[str],
    xbrl_tags: Sequence[str],
    section_title: str | None = None,
) -> Record:
    sec_item, sec_item_source, sec_item_confidence = extract_explicit_sec_item(content)
    metadata: Record = {
        "ticker": str(filing.get("ticker", "")),
        "cik": str(filing.get("cik", "")),
        "accession_number": str(filing["accession_number"]),
        "filing_date": str(filing.get("filing_date", "")),
        "report_date": str(filing.get("report_date", "")),
        "source_url": str(filing.get("source_url", "")),
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "chunk_variant": "structure_aware",
        "chunker_version": CORPUS_VERSION,
        "raw_sha256": raw_sha256,
        "record_type": record_type,
        "page_start": page_start,
        "page_end": page_end,
        "element_ids": deduplicate(element_ids),
        "xbrl_tags": sorted(set(xbrl_tags)),
        "retrieval_eligible": True,
        "sec_item": sec_item,
        "sec_item_source": sec_item_source,
        "sec_item_confidence": sec_item_confidence,
        "section_title": section_title or extract_section_title(content),
    }
    if start_display_page is not None:
        metadata["start_display_page"] = start_display_page
    if end_display_page is not None:
        metadata["end_display_page"] = end_display_page
    return metadata


def _build_text_record(
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
    content: str,
    page_start: int,
    page_end: int,
    start_display_page: int | None,
    end_display_page: int | None,
    element_ids: Sequence[str],
    xbrl_tags: Sequence[str],
    child_key: str,
    text_kind: str = "narrative",
) -> Record:
    content = content.strip()
    element_ids = deduplicate(element_ids)
    chunk_id = make_chunk_id(
        accession_number=str(filing["accession_number"]),
        variant="structure_aware",
        content=content,
        element_ids=element_ids,
        child_key=child_key,
    )
    metadata = _base_metadata(
        filing,
        raw_sha256=raw_sha256,
        content=content,
        record_type="text",
        page_start=page_start,
        page_end=page_end,
        start_display_page=start_display_page,
        end_display_page=end_display_page,
        element_ids=element_ids,
        xbrl_tags=xbrl_tags,
    )
    metadata["text_kind"] = text_kind
    section_title = metadata.get("section_title")
    return {
        "record_id": f"structure_aware::{chunk_id}",
        "target_chunk_id": chunk_id,
        "record_type": "text",
        "embedding_text": _build_embedding_text(
            filing,
            content=content,
            record_type="text",
            page_start=page_start,
            page_end=page_end,
            section_title=str(section_title) if section_title else None,
        ),
        "document": content,
        "metadata": metadata,
    }


def _build_table_chunk(
    table: Record,
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
    description: str,
    provenance: Record,
) -> Record:
    table_id = str(table["table_id"])
    table_metadata = table["metadata"]
    page_start = int(table_metadata["page_start"])
    page_end = int(table_metadata["page_end"])
    section_title = table_metadata.get("section_title")
    metadata = _base_metadata(
        filing,
        raw_sha256=raw_sha256,
        content=description,
        record_type="table",
        page_start=page_start,
        page_end=page_end,
        start_display_page=table_metadata.get("start_display_page"),
        end_display_page=table_metadata.get("end_display_page"),
        element_ids=table_metadata["element_ids"],
        xbrl_tags=table_metadata["xbrl_tags"],
        section_title=str(section_title) if section_title else None,
    )
    metadata.update(
        {
            "table_id": table_id,
            "parent_id": table_id,
            "logical_table_id": table_id,
            "table_element_id": table_metadata["table_element_id"],
            "table_index": table["table_index"],
            "table_type": table["table_type"],
            "description_provenance": provenance,
        }
    )
    return {
        "record_id": f"structure_aware::table::{table_id}",
        "target_chunk_id": table_id,
        "record_type": "table",
        "embedding_text": _build_embedding_text(
            filing,
            content=description,
            record_type="table",
            page_start=page_start,
            page_end=page_end,
            section_title=str(section_title) if section_title else None,
        ),
        "document": description,
        "metadata": metadata,
    }


def validate_corpus(
    chunks: Sequence[Record],
    tables: Sequence[Record],
    *,
    token_count: TokenCounter | None = None,
) -> None:
    record_ids = [str(chunk["record_id"]) for chunk in chunks]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Chunk record IDs must be unique.")

    table_ids = [str(table["table_id"]) for table in tables]
    if len(table_ids) != len(set(table_ids)):
        raise ValueError("Table IDs must be unique.")

    table_chunk_counts = Counter(
        str(chunk["target_chunk_id"]) for chunk in chunks if chunk["record_type"] == "table"
    )
    for table in tables:
        expected = 1 if table["retrieval_eligible"] else 0
        actual = table_chunk_counts[str(table["table_id"])]
        if actual != expected:
            raise ValueError(
                f"Table {table['table_id']} has {actual} description chunks; expected {expected}."
            )

    unknown_targets = set(table_chunk_counts) - set(table_ids)
    if unknown_targets:
        raise ValueError(f"Table chunks reference missing tables: {sorted(unknown_targets)}.")

    for chunk in chunks:
        if chunk["record_type"] not in {"text", "table"}:
            raise ValueError(f"Unsupported record_type: {chunk['record_type']!r}.")
        if not str(chunk.get("document") or "").strip():
            raise ValueError(f"Empty chunk document: {chunk['record_id']}.")
        if token_count is not None:
            embedding_tokens = token_count(str(chunk.get("embedding_text") or "")) + 1
            if embedding_tokens > MAX_EMBEDDING_TOKENS:
                raise ValueError(
                    f"Embedding input exceeds {MAX_EMBEDDING_TOKENS} tokens: "
                    f"{chunk['record_id']} ({embedding_tokens})."
                )


def build_corpus(
    pages: Sequence[Any],
    filing: Mapping[str, Any],
    raw_sha256: str,
    token_count: TokenCounter,
    annotated_html: str | None = None,
    description_mode: str = "local",
    llm_client: Any = None,
    llm_model: str = "gpt-4o",
) -> tuple[list[Record], list[Record]]:
    """Build retrieval chunks plus a lossless, separately stored table corpus."""
    if description_mode not in {"local", "openai"}:
        raise ValueError("description_mode must be 'local' or 'openai'.")

    display_pages, glossary_pages = page_maps(pages)
    elements = ordered_elements(pages)
    table_grids_by_element = extract_sec2md_table_grids(annotated_html) if annotated_html else {}
    chunks: list[Record] = []
    tables: list[Record] = []
    narrative_buffer: list[Record] = []
    narrative_index = 0
    table_index = 0
    current_section_title: str | None = None
    active_llm_client = llm_client

    def display_page(page_number: int) -> int | None:
        value = display_pages.get(page_number)
        return int(value) if value is not None else None

    def emit_narrative_buffer() -> None:
        nonlocal narrative_buffer, narrative_index
        if not narrative_buffer:
            return

        combined = "\n\n".join(str(element["content"]).strip() for element in narrative_buffer)
        element_ids = deduplicate(element["id"] for element in narrative_buffer)
        tags = deduplicate(tag for element in narrative_buffer for tag in element.get("tags") or [])
        page_start = min(int(element["page_start"]) for element in narrative_buffer)
        page_end = max(int(element["page_end"]) for element in narrative_buffer)

        for part_index, part in enumerate(split_to_token_limit(combined, token_count)):
            if looks_like_navigation(part) or looks_like_page_furniture(part):
                continue
            chunks.append(
                _build_text_record(
                    filing,
                    raw_sha256=raw_sha256,
                    content=part,
                    page_start=page_start,
                    page_end=page_end,
                    start_display_page=display_page(page_start),
                    end_display_page=display_page(page_end),
                    element_ids=element_ids,
                    xbrl_tags=tags,
                    child_key=f"narrative:{narrative_index}:{part_index}",
                )
            )

        narrative_index += 1
        narrative_buffer = []

    for element_index, element in enumerate(elements):
        content = str(element.get("content") or "").strip()
        kind = str(element.get("kind") or "text").casefold()
        if not content and kind != "table":
            continue

        element_id = str(element["id"])
        page_start = int(element["page_start"])
        page_end = int(element["page_end"])
        tags = deduplicate(element.get("tags") or [])

        if page_start in glossary_pages and kind == "text":
            emit_narrative_buffer()
            prefix, entries = split_glossary_entries(content)
            if prefix:
                narrative_buffer = [{**element, "content": prefix}]
                emit_narrative_buffer()

            for entry_index, entry in enumerate(entries):
                for part_index, part in enumerate(split_to_token_limit(entry, token_count)):
                    chunks.append(
                        _build_text_record(
                            filing,
                            raw_sha256=raw_sha256,
                            content=part,
                            page_start=page_start,
                            page_end=page_end,
                            start_display_page=display_page(page_start),
                            end_display_page=display_page(page_end),
                            element_ids=[element_id],
                            xbrl_tags=tags,
                            child_key=f"glossary:{element_index}:{entry_index}:{part_index}",
                            text_kind="glossary",
                        )
                    )
            continue

        if kind != "table":
            section_title = extract_section_title(content)
            if section_title:
                current_section_title = section_title
            if is_heading_start(content) and narrative_buffer:
                emit_narrative_buffer()

            candidate = "\n\n".join(
                [*[str(item["content"]).strip() for item in narrative_buffer], content]
            )
            if narrative_buffer and token_count(candidate) > TEXT_MAX_TOKENS:
                emit_narrative_buffer()

            narrative_buffer.append(element)
            if (
                token_count("\n\n".join(str(item["content"]).strip() for item in narrative_buffer))
                >= TEXT_TARGET_TOKENS
            ):
                emit_narrative_buffer()
            continue

        emit_narrative_buffer()
        cell_matrices = table_grids_by_element.get(element_id) or parse_markdown_table_matrices(
            content
        )
        table_section = current_section_title or extract_table_title(content)
        table = build_table_record(
            element,
            filing,
            raw_sha256=raw_sha256,
            element_index=element_index,
            table_index=table_index,
            section_title=table_section,
            start_display_page=display_page(page_start),
            end_display_page=display_page(page_end),
            cell_matrices=cell_matrices,
        )
        tables.append(table)
        table_index += 1

        if not table["retrieval_eligible"]:
            continue

        if description_mode == "openai" and active_llm_client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "OpenAI descriptions require the optional 'openai' package."
                ) from error
            active_llm_client = OpenAI()

        description, provenance = describe_table(
            table,
            filing,
            mode=description_mode,
            llm_client=active_llm_client,
            llm_model=llm_model,
        )
        table["metadata"]["description_provenance"] = provenance
        chunks.append(
            _build_table_chunk(
                table,
                filing,
                raw_sha256=raw_sha256,
                description=description,
                provenance=provenance,
            )
        )

    emit_narrative_buffer()
    validate_corpus(chunks, tables, token_count=token_count)
    return chunks, tables
