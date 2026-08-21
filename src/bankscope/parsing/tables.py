from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from sec2md.table_parser import TableParser

PARSER_NAME = "sec2md"
PARSER_VERSION = "0.1.23"
TABLE_STORE_VERSION = "bankscope-table-store-v1"
TABLE_DESCRIPTION_PROMPT_VERSION = "table-semantic-description-v1"
TABLE_DESCRIPTION_TIMEOUT_SECONDS = 30.0

MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")
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

Record = dict[str, Any]


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
    """Return the legacy structure-aware identity used by existing qrels."""
    identity = "\0".join([accession_number, variant, child_key, ",".join(element_ids), content])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def parse_markdown_table_matrices(content: str) -> list[list[list[str]]]:
    """Parse every markdown table block without splitting or changing the source."""
    lines = [line.rstrip() for line in content.splitlines()]
    matrices: list[list[list[str]]] = []
    index = 0

    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue

        table_lines: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            table_lines.append(lines[index].strip())
            index += 1

        matrix = [
            [normalize_space(cell) for cell in line.strip().strip("|").split("|")]
            for line in table_lines
            if not MARKDOWN_TABLE_SEPARATOR.fullmatch(line)
        ]
        if matrix:
            matrices.append(matrix)

    return matrices


def extract_sec2md_table_grids(
    annotated_html: str,
) -> dict[str, list[list[list[str]]]]:
    """Read sec2md's annotated HTML grids, including resolved row/column spans."""
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
            if matrix and matrix not in grids_by_element.setdefault(element_id, []):
                grids_by_element[element_id].append(matrix)

    return grids_by_element


def classify_table(
    *,
    content: str,
    cell_matrices: Sequence[Sequence[Sequence[str]]],
) -> str:
    """Classify tables only as far as retrieval filtering needs."""
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

    index_markers = (
        "table of contents",
        "form 10-k index",
        "cross-reference index",
        "documents incorporated by reference",
    )
    looks_like_index = any(marker in normalized_content for marker in index_markers)
    looks_like_index = looks_like_index or (
        "incorporated documents" in header_text and "where incorporated" in header_text
    )
    looks_like_index = looks_like_index or (
        "table description page" in header_text and "table reference" in normalized_content
    )
    looks_like_index = looks_like_index or (
        "notes to consolidated financial statements" in header_text
        and re.search(r"\bpage\b", header_text) is not None
    )
    looks_like_index = looks_like_index or (
        re.search(r"\bitem\s+\d+[a-c]?\b", normalized_content) is not None
        and re.search(r"\bpage\b", header_text) is not None
    )
    if looks_like_index:
        return "index"

    title = extract_table_title(content) or ""
    compact_title = re.sub(r"[^a-z]+", "", title.casefold())
    if any(
        marker in f"{header_text} {normalized_content}"
        for marker in ("glossary of terms", "term definition", "acronym definition")
    ) or any(marker in compact_title for marker in ("acronym", "glossary")):
        return "glossary"

    layout_markers = (
        "commission file",
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
    if all(
        marker in header_text
        for marker in ("title of each class", "trading symbol", "name of each exchange")
    ):
        return "layout"

    row_count = max(len(matrix) for matrix in matrices)
    column_count = max((len(row) for matrix in matrices for row in matrix), default=0)
    numeric_density = sum(bool(re.search(r"\d|[$€£¥%]", cell)) for cell in cells) / len(cells)
    long_text_density = sum(len(cell.split()) >= 12 for cell in cells) / len(cells)

    if row_count <= 1 or column_count <= 1:
        return "narrative_table" if long_text_density >= 0.5 else "layout"
    if numeric_density < 0.08 and long_text_density >= 0.35:
        return "narrative_table"
    return "data_table"


def is_period_label(value: str) -> bool:
    return bool(PERIOD_PATTERN.search(normalize_space(value)))


def is_table_value(value: str) -> bool:
    value = normalize_space(value)
    if not value or is_period_label(value):
        return False
    if value.casefold() in {"-", "—", "–", "nm", "n/m", "na", "n/a", "yes", "no"}:
        return True
    return bool(VALUE_PATTERN.fullmatch(value))


def find_first_data_row(matrix: Sequence[Sequence[str]]) -> int:
    for row_index, row in enumerate(matrix):
        value_columns = [index for index, cell in enumerate(row) if is_table_value(cell)]
        if not value_columns:
            continue
        if any(normalize_space(cell) for cell in row[: min(value_columns)]):
            return row_index
    return min(1, len(matrix))


def extract_table_title(content: str) -> str | None:
    prefix = content.split("|", maxsplit=1)[0]
    candidates: list[str] = []
    for line in prefix.splitlines():
        candidate = normalize_space(re.sub(r"[*_#]+", "", line)).rstrip(".:")
        if candidate and len(candidate) <= 180:
            candidates.append(candidate)
    if candidates:
        return candidates[-1]

    for line in content.splitlines():
        if line.lstrip().startswith("|"):
            first_cell = normalize_space(line.strip().strip("|").split("|", maxsplit=1)[0])
            if first_cell and not re.fullmatch(r":?-+:?", first_cell):
                return first_cell[:180]
    return None


def extract_table_unit(content: str) -> str | None:
    normalized = normalize_space(content)
    match = re.search(r"(?i)\b(?:dollars\s+)?in\s+(millions|billions|thousands)\b", normalized)
    if match:
        return match.group(1).casefold()
    if re.search(r"(?i)\bpercent(?:age)?\b", normalized):
        return "percent"
    if re.search(r"(?i)\bbasis points?\b", normalized):
        return "basis points"
    return None


def _table_introduction(content: str) -> str:
    prefix = content.split("|", maxsplit=1)[0]
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", prefix):
        cleaned = normalize_space(re.sub(r"[*_#]+", "", paragraph)).strip(" .:")
        if cleaned:
            paragraphs.append(cleaned)

    introduction = " ".join(paragraphs[-2:])
    if introduction:
        return introduction[:500].rstrip()
    return extract_table_title(content) or "No separate introduction was emitted with this table."


def _column_labels(cell_matrices: Sequence[Sequence[Sequence[str]]]) -> list[str]:
    labels: list[str] = []
    for matrix in cell_matrices:
        data_start = find_first_data_row(matrix)
        header_rows = matrix[:data_start]
        for column_index in range(max((len(row) for row in matrix), default=0)):
            path = deduplicate(
                row[column_index]
                for row in header_rows
                if column_index < len(row) and normalize_space(row[column_index])
            )
            if path:
                labels.append(" > ".join(path))
    return deduplicate(labels)[:16]


def _significant_row_labels(
    cell_matrices: Sequence[Sequence[Sequence[str]]],
) -> list[str]:
    labels: list[str] = []
    for matrix in cell_matrices:
        data_start = find_first_data_row(matrix)
        for row in matrix[data_start:]:
            value_columns = [index for index, cell in enumerate(row) if is_table_value(cell)]
            label_cells = row[: min(value_columns)] if value_columns else row[:1]
            labels.extend(
                normalize_space(cell)
                for cell in label_cells
                if normalize_space(cell) and not is_period_label(cell)
            )
    return deduplicate(labels)


def build_local_description(table: Mapping[str, Any], filing: Mapping[str, Any]) -> str:
    """Build a compact semantic description without copying numeric table rows."""
    matrices = table.get("cell_matrices") or []
    content = str(table["document"])
    lines = [f"Introduction: {_table_introduction(content)}"]

    columns = _column_labels(matrices)
    lines.append(f"Columns/periods: {'; '.join(columns) if columns else 'Not identified'}")

    unit = extract_table_unit(content)
    lines.append(f"Units: {unit or 'Not stated'}")

    row_labels = _significant_row_labels(matrices)
    lines.append(f"Significant rows: {'; '.join(row_labels) if row_labels else 'Not identified'}")

    return "\n".join(lines)


def _chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        return str(getattr(getattr(choices[0], "message", None), "content", None) or "").strip()
    if isinstance(response, Mapping):
        response_choices = response.get("choices")
        if isinstance(response_choices, Sequence) and response_choices:
            choice = response_choices[0]
            if isinstance(choice, Mapping):
                message = choice.get("message")
                if isinstance(message, Mapping):
                    return str(message.get("content") or "").strip()
    return ""


def build_openai_description(
    table: Mapping[str, Any],
    filing: Mapping[str, Any],
    *,
    client: Any,
    model: str,
) -> tuple[str, Record]:
    """Describe one table through Chat Completions, with no local fallback."""
    metadata = table["metadata"]
    instructions = (
        "Use the supplied bank, report, section, and page metadata as context. Describe this SEC "
        "filing table for semantic retrieval, focusing on its purpose and relationships that the "
        "deterministic column and row index may not capture. Do not repeat the supplied metadata. "
        "Do not reproduce the full table or enumerate its numeric values. Return plain text only."
    )
    prompt = "\n".join(
        [
            f"Prompt version: {TABLE_DESCRIPTION_PROMPT_VERSION}",
            f"Bank: {filing.get('ticker', metadata.get('ticker', ''))}",
            f"Report date: {filing.get('report_date', metadata.get('report_date', ''))}",
            f"Section: {metadata.get('section_title') or 'Unassigned'}",
            f"Internal pages: {metadata['page_start']}-{metadata['page_end']}",
            "",
            "Original markdown table:",
            str(table["document"]),
        ]
    )

    try:
        request_options: dict[str, Any] = {
            "max_tokens": 300,
            "temperature": 0,
            "timeout": TABLE_DESCRIPTION_TIMEOUT_SECONDS,
        }
        if "GPT_51" in model.strip().upper() or "GPT-5.1" in model.strip().upper():
            request_options = {
                "max_completion_tokens": 300,
                "timeout": TABLE_DESCRIPTION_TIMEOUT_SECONDS,
            }
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            **request_options,
        )
    except Exception as error:
        raise RuntimeError(
            f"OpenAI table description failed for table {table['table_id']}."
        ) from error

    description = _chat_completion_text(response)
    if not description:
        raise RuntimeError(f"OpenAI returned an empty description for table {table['table_id']}.")

    provenance: Record = {
        "mode": "openai",
        "provider": "openai",
        "api": "chat.completions",
        "model": model,
        "prompt_version": TABLE_DESCRIPTION_PROMPT_VERSION,
    }
    response_id = getattr(response, "id", None)
    if response_id:
        provenance["response_id"] = str(response_id)
    return description, provenance


def describe_table(
    table: Mapping[str, Any],
    filing: Mapping[str, Any],
    *,
    mode: str = "local",
    llm_client: Any = None,
    llm_model: str = "gpt-4o",
) -> tuple[str, Record]:
    if mode == "local":
        return build_local_description(table, filing), {
            "mode": "local",
            "generator": "bankscope-local-table-description-v1",
        }
    if mode != "openai":
        raise ValueError("description_mode must be 'local' or 'openai'.")

    client = llm_client
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI descriptions require the optional 'openai' package."
            ) from error
        client = OpenAI()

    local_description = build_local_description(table, filing)
    llm_description, provenance = build_openai_description(
        table, filing, client=client, model=llm_model
    )
    provenance["base_generator"] = "bankscope-local-table-description-v1"
    return f"{local_description}\nLLM synopsis: {llm_description}", provenance


def build_table_record(
    element: Mapping[str, Any],
    filing: Mapping[str, Any],
    *,
    raw_sha256: str,
    element_index: int,
    table_index: int,
    section_title: str | None,
    start_display_page: int | None,
    end_display_page: int | None,
    cell_matrices: Sequence[Sequence[Sequence[str]]],
) -> Record:
    """Store one complete sec2md table and retain its legacy parent identity."""
    content = str(element.get("content") or "").strip()
    element_id = str(element["id"])
    page_start = int(element["page_start"])
    page_end = int(element["page_end"])
    table_id = make_chunk_id(
        accession_number=str(filing["accession_number"]),
        variant="structure_aware_parent",
        content=content,
        element_ids=[element_id],
        child_key=f"table-parent:{element_index}",
    )
    matrices = [
        [[normalize_space(cell) for cell in row] for row in matrix] for matrix in cell_matrices
    ]
    table_type = classify_table(content=content, cell_matrices=matrices)
    retrieval_eligible = table_type not in {"layout", "index"}
    metadata: Record = {
        "ticker": str(filing.get("ticker", "")),
        "cik": str(filing.get("cik", "")),
        "accession_number": str(filing["accession_number"]),
        "filing_date": str(filing.get("filing_date", "")),
        "report_date": str(filing.get("report_date", "")),
        "source_url": str(filing.get("source_url", "")),
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "table_store_version": TABLE_STORE_VERSION,
        "raw_sha256": raw_sha256,
        "record_type": "table",
        "page_start": page_start,
        "page_end": page_end,
        "element_ids": [element_id],
        "xbrl_tags": sorted(set(str(tag) for tag in element.get("tags") or [])),
        "retrieval_eligible": retrieval_eligible,
        "section_title": section_title,
        "parent_id": table_id,
        "logical_table_id": table_id,
        "table_id": table_id,
        "table_element_id": element_id,
        "table_index": table_index,
        "source_element_index": element_index,
        "table_type": table_type,
    }
    if start_display_page is not None:
        metadata["start_display_page"] = start_display_page
    if end_display_page is not None:
        metadata["end_display_page"] = end_display_page

    return {
        "table_id": table_id,
        "table_index": table_index,
        "table_type": table_type,
        "retrieval_eligible": retrieval_eligible,
        "document": content,
        "cell_matrices": matrices,
        "metadata": metadata,
    }
