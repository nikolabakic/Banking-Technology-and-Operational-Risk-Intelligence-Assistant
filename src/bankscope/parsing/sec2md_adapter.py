from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from sec2md.table_parser import TableParser

PARSER_NAME = "sec2md"
PARSER_VERSION = "0.1.23"
BUILTIN_CHUNKER_VERSION = "sec2md-built-in-v1"
STRUCTURE_CHUNKER_VERSION = "bankscope-structure-aware-v2"

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
    logical_table_id: str | None = None,
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

    if logical_table_id is not None:
        metadata["logical_table_id"] = logical_table_id

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
    logical_table_id: str | None = None,
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
        logical_table_id=logical_table_id,
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


def extract_sec2md_table_grids(
    annotated_html: str,
) -> dict[str, list[list[list[str]]]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(annotated_html, "lxml")
    grids_by_element: dict[str, list[list[list[str]]]] = {}
    seen_tables: set[tuple[str, int]] = set()

    for source_node in soup.find_all(attrs={"data-sec2md-block": True}):
        element_id = normalize_space(source_node.get("data-sec2md-block"))

        if not element_id:
            continue

        table_nodes = (
            [source_node] if source_node.name == "table" else source_node.find_all("table")
        )

        for table_node in table_nodes:
            if not isinstance(table_node, Tag):
                continue

            table_key = (element_id, id(table_node))

            if table_key in seen_tables:
                continue

            seen_tables.add(table_key)
            parsed_table = TableParser(table_node)

            matrix = [
                [
                    normalize_space(grid_cell.cell.text) if grid_cell is not None else ""
                    for grid_cell in row
                ]
                for row in parsed_table.grid
            ]

            if not matrix:
                continue

            element_grids = grids_by_element.setdefault(element_id, [])

            if matrix not in element_grids:
                element_grids.append(matrix)

    return grids_by_element


def classify_table_parent(
    *,
    content: str,
    cell_matrices: Sequence[Sequence[Sequence[str]]],
) -> str:
    matrices = [matrix for matrix in cell_matrices if matrix]

    if not matrices:
        return "layout"

    cells = [
        normalize_space(cell)
        for matrix in matrices
        for row in matrix
        for cell in row
        if normalize_space(cell)
    ]

    if not cells:
        return "layout"

    normalized_content = normalize_space(content).casefold()
    header_text = " ".join(
        normalize_space(cell).casefold()
        for matrix in matrices
        for row in matrix[:3]
        for cell in row
        if normalize_space(cell)
    )

    glossary_markers = (
        "glossary of terms",
        "term definition",
        "acronym definition",
    )

    if any(marker in f"{header_text} {normalized_content}" for marker in glossary_markers):
        return "glossary"

    if (
        "table of contents" in normalized_content
        or "form 10-k index" in normalized_content
        or (
            re.search(r"\bitem\s+\d+[a-c]?\b", normalized_content)
            and re.search(r"\bpage\b", header_text)
        )
    ):
        return "index"

    layout_markers = (
        "commission file number",
        "state or other jurisdiction of incorporation",
        "i.r.s. employer identification",
        "address of principal executive offices",
        "registrant's telephone number",
        "securities registered pursuant",
        "indicate by check mark",
        "large accelerated filer",
        "emerging growth company",
    )

    if any(marker in normalized_content for marker in layout_markers):
        return "layout"

    row_count = max(len(matrix) for matrix in matrices)
    column_count = max(
        (len(row) for matrix in matrices for row in matrix),
        default=0,
    )

    word_counts = [len(cell.split()) for cell in cells]
    numeric_cell_count = sum(bool(re.search(r"\d|[$€£¥%]", cell)) for cell in cells)
    long_text_cell_count = sum(word_count >= 12 for word_count in word_counts)

    numeric_density = numeric_cell_count / len(cells)
    long_text_density = long_text_cell_count / len(cells)

    if row_count <= 1 or column_count <= 1:
        return "narrative_table" if long_text_density >= 0.5 else "layout"

    if numeric_density < 0.08 and long_text_density >= 0.35:
        return "narrative_table"

    return "data_table"


PERIOD_PATTERN = re.compile(
    r"(?i)\b(?:19|20)\d{2}\b|"
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b|"
    r"\b(?:first|second|third|fourth)\s+quarter\b"
)

VALUE_PATTERN = re.compile(
    r"""(?ix)
    ^\s*
    (?:[$€£¥]\s*)?
    \(?
    [-+]?
    \d[\d,\s]*
    (?:\.\d+)?
    \)?
    \s*(?:%|x|bps)?
    \s*(?:\([a-z]{1,3}\))?
    \s*$
    """
)


def is_period_label(value: str) -> bool:
    return bool(PERIOD_PATTERN.search(normalize_space(value)))


def is_table_value(value: str) -> bool:
    value = normalize_space(value)

    if not value or is_period_label(value):
        return False

    if value.casefold() in {
        "-",
        "—",
        "–",
        "nm",
        "n/m",
        "na",
        "n/a",
        "yes",
        "no",
    }:
        return True

    return bool(VALUE_PATTERN.fullmatch(value))


def extract_table_title(content: str) -> str | None:
    prefix = content.split("|", maxsplit=1)[0]
    candidates: list[str] = []

    for line in prefix.splitlines():
        candidate = normalize_space(re.sub(r"[*_#]+", "", line)).rstrip(".:")

        if candidate and len(candidate) <= 180:
            candidates.append(candidate)

    return candidates[-1] if candidates else None


def extract_table_unit(content: str) -> str | None:
    normalized = normalize_space(content)

    match = re.search(
        r"(?i)\b(?:dollars\s+)?in\s+"
        r"(millions|billions|thousands)\b",
        normalized,
    )

    if match:
        return match.group(1).casefold()

    if re.search(r"(?i)\bpercent(?:age)?\b", normalized):
        return "percent"

    if re.search(r"(?i)\bbasis points?\b", normalized):
        return "basis points"

    return None


def make_locator_prefix(
    filing: Mapping[str, Any],
    *,
    section_title: str | None,
    table_title: str | None,
    unit: str | None,
) -> list[str]:
    lines = [
        f"Bank: {filing['ticker']}",
        f"Entity: {filing.get('legal_name', 'JPMorgan Chase & Co.')}",
        f"Report date: {filing['report_date']}",
    ]

    if section_title:
        lines.append(f"Section: {section_title}")

    if table_title and table_title != section_title:
        lines.append(f"Table: {table_title}")

    if unit:
        lines.append(f"Unit: {unit}")

    return lines


def find_first_data_row(matrix: Sequence[Sequence[str]]) -> int:
    for row_index, row in enumerate(matrix):
        value_columns = [
            column_index for column_index, cell in enumerate(row) if is_table_value(cell)
        ]

        if not value_columns:
            continue

        first_value_column = min(value_columns)

        if any(normalize_space(cell) for cell in row[:first_value_column]):
            return row_index

    return min(1, len(matrix))


def build_data_locator_specs(
    matrix: Sequence[Sequence[str]],
    *,
    matrix_index: int,
    prefix: Sequence[str],
) -> list[Record]:
    if not matrix:
        return []

    data_start = find_first_data_row(matrix)
    header_rows = matrix[:data_start]
    specs: list[Record] = []
    active_row_context: list[str] = []

    for row_index in range(data_start, len(matrix)):
        row = list(matrix[row_index])
        value_columns = [
            column_index for column_index, cell in enumerate(row) if is_table_value(cell)
        ]

        if not value_columns:
            context_labels = deduplicate(
                cell for cell in row if normalize_space(cell) and not is_period_label(cell)
            )

            if context_labels:
                active_row_context = context_labels

            continue

        first_value_column = min(value_columns)
        row_labels = deduplicate(cell for cell in row[:first_value_column] if normalize_space(cell))
        row_path = deduplicate([*active_row_context, *row_labels])

        if not row_path:
            continue

        column_paths: list[Record] = []
        cell_coordinates: list[Record] = []

        for column_index in value_columns:
            path = deduplicate(
                header_row[column_index]
                for header_row in header_rows
                if column_index < len(header_row) and normalize_space(header_row[column_index])
            )

            if not path:
                path = [f"Column {column_index + 1}"]

            column_paths.append(
                {
                    "column_index": column_index,
                    "path": path,
                }
            )
            cell_coordinates.append(
                {
                    "matrix_index": matrix_index,
                    "row_index": row_index,
                    "column_index": column_index,
                }
            )

        column_text = " | ".join(" > ".join(column["path"]) for column in column_paths)

        document = "\n".join(
            [
                *prefix,
                f"Measure or row: {' > '.join(row_path)}",
                f"Column context: {column_text}",
            ]
        )

        specs.append(
            {
                "document": document,
                "locator_scope": "row",
                "row_path": row_path,
                "column_paths": column_paths,
                "cell_coordinates": cell_coordinates,
            }
        )

    if specs:
        return specs

    schema_labels = deduplicate(
        cell for row in matrix for cell in row if normalize_space(cell) and not is_table_value(cell)
    )

    if not schema_labels:
        return []

    return [
        {
            "document": "\n".join(
                [
                    *prefix,
                    f"Table fields: {' | '.join(schema_labels[:80])}",
                ]
            ),
            "locator_scope": "table_schema",
            "row_path": [],
            "column_paths": [],
            "cell_coordinates": [],
        }
    ]


def build_glossary_locator_specs(
    matrix: Sequence[Sequence[str]],
    *,
    matrix_index: int,
    prefix: Sequence[str],
) -> list[Record]:
    specs: list[Record] = []

    for row_index, row in enumerate(matrix):
        cells = deduplicate(row)

        if len(cells) < 2:
            continue

        document = "\n".join(
            [
                *prefix,
                f"Term: {cells[0]}",
                f"Definition: {' '.join(cells[1:])}",
            ]
        )

        specs.append(
            {
                "document": document,
                "locator_scope": "glossary_entry",
                "row_path": [cells[0]],
                "column_paths": [],
                "cell_coordinates": [
                    {
                        "matrix_index": matrix_index,
                        "row_index": row_index,
                        "column_index": column_index,
                    }
                    for column_index, cell in enumerate(row)
                    if normalize_space(cell)
                ],
            }
        )

    return specs


def build_narrative_locator_specs(
    matrix: Sequence[Sequence[str]],
    *,
    matrix_index: int,
    prefix: Sequence[str],
) -> list[Record]:
    labels = deduplicate(
        cell for row in matrix for cell in row if normalize_space(cell) and not is_table_value(cell)
    )

    if not labels:
        return []

    return [
        {
            "document": "\n".join(
                [
                    *prefix,
                    f"Table content: {' '.join(labels[:100])}",
                ]
            ),
            "locator_scope": "narrative",
            "row_path": [],
            "column_paths": [],
            "cell_coordinates": [],
        }
    ]


def build_table_locator_specs(
    filing: Mapping[str, Any],
    *,
    table_type: str,
    content: str,
    cell_matrices: Sequence[Sequence[Sequence[str]]],
    section_title: str | None,
) -> list[Record]:
    if table_type in {"layout", "index"}:
        return []

    prefix = make_locator_prefix(
        filing,
        section_title=section_title,
        table_title=extract_table_title(content),
        unit=extract_table_unit(content),
    )
    specs: list[Record] = []

    for matrix_index, matrix in enumerate(cell_matrices):
        if table_type == "glossary":
            matrix_specs = build_glossary_locator_specs(
                matrix,
                matrix_index=matrix_index,
                prefix=prefix,
            )
        elif table_type == "narrative_table":
            matrix_specs = build_narrative_locator_specs(
                matrix,
                matrix_index=matrix_index,
                prefix=prefix,
            )
        else:
            matrix_specs = build_data_locator_specs(
                matrix,
                matrix_index=matrix_index,
                prefix=prefix,
            )

        specs.extend(matrix_specs)

    return specs


def build_structure_aware_records(
    pages: Sequence[Any],
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
    token_count: TokenCounter,
    annotated_html: str | None = None,
) -> tuple[list[Record], list[Record]]:
    display_pages, glossary_pages = page_maps(pages)
    elements = ordered_elements(pages)
    table_grids_by_element = extract_sec2md_table_grids(annotated_html) if annotated_html else {}
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

    current_section_title: str | None = None
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
            section_title = extract_section_title(content)

            if section_title:
                current_section_title = section_title

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
        logical_table_id = parent_id
        cell_matrices = table_grids_by_element.get(element_id, [])
        table_type = classify_table_parent(content=content, cell_matrices=cell_matrices)
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
            logical_table_id=logical_table_id,
        )
        parent_metadata["table_type"] = table_type
        table_parents.append(
            {
                "parent_id": parent_id,
                "logical_table_id": logical_table_id,
                "table_type": table_type,
                "document": content,
                "cell_matrices": cell_matrices,
                "metadata": parent_metadata,
            }
        )

        locator_specs = build_table_locator_specs(
            filing,
            table_type=table_type,
            content=content,
            cell_matrices=cell_matrices,
            section_title=current_section_title,
        )

        for locator_index, locator_spec in enumerate(locator_specs):
            locator_document = str(locator_spec["document"])
            record = build_record(
                filing,
                variant="structure_aware",
                chunker_version=STRUCTURE_CHUNKER_VERSION,
                raw_sha256=raw_sha256,
                content=locator_document,
                record_type="table_locator",
                page_start=page_start,
                page_end=page_end,
                start_display_page=display_page(page_start),
                end_display_page=display_page(page_end),
                element_ids=[element_id],
                xbrl_tags=tags,
                child_key=f"table-locator:{element_index}:{locator_index}",
                retrieval_eligible=True,
                parent_id=parent_id,
                logical_table_id=logical_table_id,
                table_element_id=element_id,
            )
            record["metadata"].update(
                {
                    "table_type": table_type,
                    "locator_scope": locator_spec["locator_scope"],
                    "row_path": locator_spec["row_path"],
                    "column_paths": locator_spec["column_paths"],
                    "cell_coordinates": locator_spec["cell_coordinates"],
                    "evidence_ref": {
                        "parent_id": parent_id,
                        "logical_table_id": logical_table_id,
                        "cell_coordinates": locator_spec["cell_coordinates"],
                    },
                }
            )

            if current_section_title:
                record["metadata"]["section_title"] = current_section_title

            records.append(record)

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
