from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

PARSER_NAME = "sec2md"
PARSER_VERSION = "0.1.23"
BUILTIN_CHUNKER_VERSION = "sec2md-built-in-v1"
STRUCTURE_CHUNKER_VERSION = "bankscope-structure-aware-v1"

TARGET_TOKENS = 600
MAX_TOKENS = 700
OVERLAP_TOKENS = 80

SEC_ITEM_PATTERN = re.compile(
    r"(?im)^(?:\*\*|__)?\s*item\s+(\d{1,2}[a-c]?)\s*[.:]",
)
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$",
)
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def deduplicate(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def make_chunk_id(
    *,
    accession_number: str,
    variant: str,
    content: str,
    element_ids: Sequence[str],
    child_key: str,
) -> str:
    identity = "\0".join(
        [
            accession_number,
            variant,
            child_key,
            ",".join(element_ids),
            content,
        ]
    )
    return sha256_text(identity)


def extract_explicit_sec_item(content: str) -> tuple[str | None, str, float]:
    match = SEC_ITEM_PATTERN.search(content[:500])

    if match is None:
        return None, "unassigned", 0.0

    return f"Item {match.group(1).upper()}", "explicit_heading_in_chunk", 1.0


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

    if not normalized:
        return True

    if re.fullmatch(r"\d{1,4}", normalized):
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


def build_embedding_text(
    filing: Mapping[str, Any],
    *,
    content: str,
    record_type: str,
    page_start: int,
    page_end: int,
    section_title: str | None,
) -> str:
    year = str(filing["report_date"])[:4]
    lines = [
        f"Bank: {filing['ticker']}",
        f"Entity: {filing.get('legal_name', 'JPMorgan Chase & Co.')}",
        f"Report: {year} 10-K",
        f"Evidence type: {record_type}",
        f"Internal pages: {page_start}-{page_end}",
    ]

    if section_title:
        lines.append(f"Section: {section_title}")

    return "\n".join([*lines, "", content])


def build_metadata(
    filing: Mapping[str, Any],
    *,
    variant: str,
    chunker_version: str,
    raw_sha256: str,
    content: str,
    record_type: str,
    page_start: int,
    page_end: int,
    start_display_page: int | None,
    end_display_page: int | None,
    element_ids: Sequence[str],
    xbrl_tags: Sequence[str],
    retrieval_eligible: bool,
    parent_id: str | None = None,
    table_element_id: str | None = None,
) -> Record:
    sec_item, sec_item_source, sec_item_confidence = extract_explicit_sec_item(content)
    section_title = extract_section_title(content)

    metadata: Record = {
        "ticker": str(filing["ticker"]),
        "cik": str(filing["cik"]),
        "accession_number": str(filing["accession_number"]),
        "filing_date": str(filing["filing_date"]),
        "report_date": str(filing["report_date"]),
        "source_url": str(filing["source_url"]),
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "chunk_variant": variant,
        "chunker_version": chunker_version,
        "raw_sha256": raw_sha256,
        "record_type": record_type,
        "page_start": page_start,
        "page_end": page_end,
        "element_ids": list(element_ids),
        "xbrl_tags": list(xbrl_tags),
        "retrieval_eligible": retrieval_eligible,
        "sec_item": sec_item,
        "sec_item_source": sec_item_source,
        "sec_item_confidence": sec_item_confidence,
        "section_title": section_title,
    }

    if start_display_page is not None:
        metadata["start_display_page"] = start_display_page

    if end_display_page is not None:
        metadata["end_display_page"] = end_display_page

    if parent_id is not None:
        metadata["parent_id"] = parent_id

    if table_element_id is not None:
        metadata["table_element_id"] = table_element_id

    return metadata


def build_record(
    filing: Mapping[str, Any],
    *,
    variant: str,
    chunker_version: str,
    raw_sha256: str,
    content: str,
    record_type: str,
    page_start: int,
    page_end: int,
    start_display_page: int | None,
    end_display_page: int | None,
    element_ids: Sequence[str],
    xbrl_tags: Sequence[str],
    child_key: str,
    retrieval_eligible: bool,
    parent_id: str | None = None,
    table_element_id: str | None = None,
) -> Record:
    content = content.strip()
    element_ids = deduplicate(element_ids)
    xbrl_tags = sorted(set(xbrl_tags))
    chunk_id = make_chunk_id(
        accession_number=str(filing["accession_number"]),
        variant=variant,
        content=content,
        element_ids=element_ids,
        child_key=child_key,
    )
    metadata = build_metadata(
        filing,
        variant=variant,
        chunker_version=chunker_version,
        raw_sha256=raw_sha256,
        content=content,
        record_type=record_type,
        page_start=page_start,
        page_end=page_end,
        start_display_page=start_display_page,
        end_display_page=end_display_page,
        element_ids=element_ids,
        xbrl_tags=xbrl_tags,
        retrieval_eligible=retrieval_eligible,
        parent_id=parent_id,
        table_element_id=table_element_id,
    )
    section_title = metadata.get("section_title")

    return {
        "record_id": f"{variant}::{chunk_id}",
        "record_type": record_type,
        "embedding_text": build_embedding_text(
            filing,
            content=content,
            record_type=record_type,
            page_start=page_start,
            page_end=page_end,
            section_title=str(section_title) if section_title else None,
        ),
        "document": content,
        "target_chunk_id": chunk_id,
        "metadata": metadata,
    }


def adapt_builtin_chunks(
    chunks: Sequence[Any],
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
) -> list[Record]:
    records: list[Record] = []

    for fallback_index, source_chunk in enumerate(chunks):
        chunk = as_dict(source_chunk)
        content = str(chunk.get("content") or "").strip()

        if not content:
            continue

        elements = [as_dict(element) for element in chunk.get("elements", [])]
        element_ids = deduplicate(
            chunk.get("element_ids", []) or [element.get("id") for element in elements]
        )
        xbrl_tags = deduplicate(
            chunk.get("tags", [])
            or [tag for element in elements for tag in element.get("tags") or []]
        )
        page_start = int(chunk.get("start_page") or chunk.get("page") or 1)
        page_end = int(chunk.get("end_page") or page_start)
        has_table = bool(chunk.get("has_table"))
        has_non_table = any(str(element.get("kind")) != "table" for element in elements)
        record_type = "mixed" if has_table and has_non_table else "table" if has_table else "text"
        retrieval_eligible = not (
            looks_like_navigation(content) or looks_like_page_furniture(content)
        )

        records.append(
            build_record(
                filing,
                variant="sec2md_builtin",
                chunker_version=BUILTIN_CHUNKER_VERSION,
                raw_sha256=raw_sha256,
                content=content,
                record_type=record_type,
                page_start=page_start,
                page_end=page_end,
                start_display_page=chunk.get("start_display_page"),
                end_display_page=chunk.get("end_display_page"),
                element_ids=element_ids,
                xbrl_tags=xbrl_tags,
                child_key=str(chunk.get("index", fallback_index)),
                retrieval_eligible=retrieval_eligible,
            )
        )

    return records


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
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
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
            candidate = "\n\n".join([*current, unit])

        current.append(unit)

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def parse_markdown_table_blocks(content: str) -> list[tuple[str, list[str]]]:
    lines = [line.rstrip() for line in content.splitlines()]
    blocks: list[tuple[str, list[str]]] = []
    context_lines: list[str] = []
    index = 0

    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            if lines[index].strip():
                context_lines.append(lines[index].strip())
            index += 1
            continue

        table_lines: list[str] = []

        while index < len(lines) and lines[index].lstrip().startswith("|"):
            table_lines.append(lines[index].strip())
            index += 1

        context = "\n\n".join(context_lines[-4:]).strip()
        blocks.append((context, table_lines))

    return blocks


def table_row_is_context(row: str) -> bool:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    nonempty_cells = [cell for cell in cells if cell]
    return len(nonempty_cells) <= 1


def make_table_child_content(
    *,
    context: str,
    header: str,
    separator: str,
    context_row: str | None,
    data_row: str,
    token_count: TokenCounter,
) -> str:
    table_lines = [header]

    if separator:
        table_lines.append(separator)

    if context_row and context_row != data_row:
        table_lines.append(context_row)

    table_lines.append(data_row)
    table = "\n".join(table_lines)
    context = context.strip()

    if context:
        candidate = f"{context}\n\n{table}"

        if token_count(candidate) <= MAX_TOKENS:
            return candidate

    return table


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

        for source_element in page.get("elements") or []:
            element = as_dict(source_element)
            element_id = str(element["id"])

            if element_id in seen_ids:
                continue

            seen_ids.add(element_id)
            elements.append(element)

    return elements


def build_structure_aware_records(
    pages: Sequence[Any],
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
    token_count: TokenCounter,
) -> tuple[list[Record], list[Record]]:
    display_pages, glossary_pages = page_maps(pages)
    elements = ordered_elements(pages)
    records: list[Record] = []
    table_parents: list[Record] = []
    narrative_buffer: list[Record] = []
    narrative_index = 0

    def display_page(page_number: int) -> int | None:
        value = display_pages.get(page_number)
        return int(value) if value is not None else None

    def emit_narrative_buffer() -> None:
        nonlocal narrative_buffer
        nonlocal narrative_index

        if not narrative_buffer:
            return

        combined = "\n\n".join(str(element["content"]).strip() for element in narrative_buffer)
        element_ids = deduplicate(element["id"] for element in narrative_buffer)
        tags = deduplicate(tag for element in narrative_buffer for tag in element.get("tags") or [])
        page_start = min(int(element["page_start"]) for element in narrative_buffer)
        page_end = max(int(element["page_end"]) for element in narrative_buffer)

        for part_index, part in enumerate(split_to_token_limit(combined, token_count)):
            retrieval_eligible = not (
                looks_like_navigation(part) or looks_like_page_furniture(part)
            )
            records.append(
                build_record(
                    filing,
                    variant="structure_aware",
                    chunker_version=STRUCTURE_CHUNKER_VERSION,
                    raw_sha256=raw_sha256,
                    content=part,
                    record_type="text",
                    page_start=page_start,
                    page_end=page_end,
                    start_display_page=display_page(page_start),
                    end_display_page=display_page(page_end),
                    element_ids=element_ids,
                    xbrl_tags=tags,
                    child_key=f"narrative:{narrative_index}:{part_index}",
                    retrieval_eligible=retrieval_eligible,
                )
            )

        narrative_index += 1
        narrative_buffer = []

    for element_index, element in enumerate(elements):
        content = str(element.get("content") or "").strip()

        if not content:
            continue

        element_id = str(element["id"])
        page_start = int(element["page_start"])
        page_end = int(element["page_end"])
        tags = deduplicate(element.get("tags") or [])
        kind = str(element.get("kind") or "text").casefold()

        if page_start in glossary_pages and kind == "text":
            emit_narrative_buffer()
            prefix, entries = split_glossary_entries(content)

            if prefix:
                narrative_buffer = [{**element, "content": prefix}]
                emit_narrative_buffer()

            for entry_index, entry in enumerate(entries):
                for part_index, part in enumerate(split_to_token_limit(entry, token_count)):
                    records.append(
                        build_record(
                            filing,
                            variant="structure_aware",
                            chunker_version=STRUCTURE_CHUNKER_VERSION,
                            raw_sha256=raw_sha256,
                            content=part,
                            record_type="glossary_child",
                            page_start=page_start,
                            page_end=page_end,
                            start_display_page=display_page(page_start),
                            end_display_page=display_page(page_end),
                            element_ids=[element_id],
                            xbrl_tags=tags,
                            child_key=(f"glossary:{element_index}:{entry_index}:{part_index}"),
                            retrieval_eligible=True,
                        )
                    )

            continue

        if kind != "table":
            if is_heading_start(content) and narrative_buffer:
                emit_narrative_buffer()

            candidate = "\n\n".join(
                [*[str(item["content"]).strip() for item in narrative_buffer], content]
            )

            if narrative_buffer and token_count(candidate) > MAX_TOKENS:
                emit_narrative_buffer()

            narrative_buffer.append(element)

            if (
                token_count("\n\n".join(str(item["content"]).strip() for item in narrative_buffer))
                >= TARGET_TOKENS
            ):
                emit_narrative_buffer()

            continue

        emit_narrative_buffer()
        parent_id = make_chunk_id(
            accession_number=str(filing["accession_number"]),
            variant="structure_aware_parent",
            content=content,
            element_ids=[element_id],
            child_key=f"table-parent:{element_index}",
        )
        parent_metadata = build_metadata(
            filing,
            variant="structure_aware",
            chunker_version=STRUCTURE_CHUNKER_VERSION,
            raw_sha256=raw_sha256,
            content=content,
            record_type="table_parent",
            page_start=page_start,
            page_end=page_end,
            start_display_page=display_page(page_start),
            end_display_page=display_page(page_end),
            element_ids=[element_id],
            xbrl_tags=tags,
            retrieval_eligible=False,
            parent_id=parent_id,
            table_element_id=element_id,
        )
        table_parents.append(
            {
                "parent_id": parent_id,
                "document": content,
                "metadata": parent_metadata,
            }
        )

        child_count_before = len(records)
        table_retrieval_eligible = not (
            looks_like_navigation(content) or looks_like_page_furniture(content)
        )

        for block_index, (context, table_lines) in enumerate(parse_markdown_table_blocks(content)):
            if not table_lines:
                continue

            header = table_lines[0]
            separator = (
                table_lines[1]
                if len(table_lines) > 1
                and MARKDOWN_TABLE_SEPARATOR_PATTERN.fullmatch(table_lines[1])
                else ""
            )
            data_start = 2 if separator else 1
            context_row: str | None = None

            for row_index, row in enumerate(table_lines[data_start:]):
                if not row.strip() or row.count("|") < 2:
                    continue

                if table_row_is_context(row):
                    context_row = row
                    continue

                child_content = make_table_child_content(
                    context=context,
                    header=header,
                    separator=separator,
                    context_row=context_row,
                    data_row=row,
                    token_count=token_count,
                )

                for part_index, part in enumerate(split_to_token_limit(child_content, token_count)):
                    records.append(
                        build_record(
                            filing,
                            variant="structure_aware",
                            chunker_version=STRUCTURE_CHUNKER_VERSION,
                            raw_sha256=raw_sha256,
                            content=part,
                            record_type="table_child",
                            page_start=page_start,
                            page_end=page_end,
                            start_display_page=display_page(page_start),
                            end_display_page=display_page(page_end),
                            element_ids=[element_id],
                            xbrl_tags=tags,
                            child_key=(
                                f"table:{element_index}:{block_index}:{row_index}:{part_index}"
                            ),
                            retrieval_eligible=table_retrieval_eligible,
                            parent_id=parent_id,
                            table_element_id=element_id,
                        )
                    )

        if len(records) == child_count_before:
            retrieval_eligible = not (
                looks_like_navigation(content) or looks_like_page_furniture(content)
            )
            records.append(
                build_record(
                    filing,
                    variant="structure_aware",
                    chunker_version=STRUCTURE_CHUNKER_VERSION,
                    raw_sha256=raw_sha256,
                    content=content,
                    record_type="table_child",
                    page_start=page_start,
                    page_end=page_end,
                    start_display_page=display_page(page_start),
                    end_display_page=display_page(page_end),
                    element_ids=[element_id],
                    xbrl_tags=tags,
                    child_key=f"table:{element_index}:whole",
                    retrieval_eligible=retrieval_eligible,
                    parent_id=parent_id,
                    table_element_id=element_id,
                )
            )

    emit_narrative_buffer()
    return records, table_parents


def validate_records(
    records: Sequence[Record],
    *,
    token_count: TokenCounter,
) -> None:
    if not records:
        raise ValueError("No records were generated.")

    record_ids = [str(record["record_id"]) for record in records]
    target_ids = [str(record["target_chunk_id"]) for record in records]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Generated record IDs are not unique.")

    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Generated target chunk IDs are not unique.")

    for record in records:
        record_id = str(record["record_id"])
        document = str(record.get("document") or "").strip()
        metadata = record.get("metadata")

        if not document:
            raise ValueError(f"Empty document: {record_id}.")

        if not isinstance(metadata, Mapping):
            raise ValueError(f"Missing metadata: {record_id}.")

        if int(metadata["page_start"]) <= 0 or int(metadata["page_end"]) <= 0:
            raise ValueError(f"Invalid page provenance: {record_id}.")

        if bool(metadata["retrieval_eligible"]) and token_count(document) > MAX_TOKENS:
            raise ValueError(f"Record exceeds {MAX_TOKENS} tokens: {record_id}.")


def eligible_records(records: Sequence[Record]) -> list[Record]:
    return [
        record for record in records if bool(record.get("metadata", {}).get("retrieval_eligible"))
    ]


def chunk_config_hash(variant: str) -> str:
    config = {
        "variant": variant,
        "target_tokens": TARGET_TOKENS,
        "max_tokens": MAX_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "chunker_version": (
            BUILTIN_CHUNKER_VERSION if variant == "sec2md_builtin" else STRUCTURE_CHUNKER_VERSION
        ),
    }
    return sha256_text(canonical_json(config))
