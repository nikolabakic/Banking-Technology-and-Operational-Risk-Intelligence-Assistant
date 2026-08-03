import argparse
import hashlib
import json
import re
from pathlib import Path

PROXY_VERSION = "deterministic-v1"

DEFAULT_INPUT = Path("data/processed/chunks/sec_10k_chunks.jsonl")
DEFAULT_OUTPUT = Path(
    "data/processed/table_proxies/sec_10k_table_proxies.jsonl"
)

PURE_NUMERIC_RE = re.compile(
    r"""
    ^
    [\$€£]?\s*
    \(?\s*
    [+\-−]?
    (?:
        \d{1,3}(?:,\d{3})*(?:\.\d+)?
        |
        \d+(?:\.\d+)?
        |
        \.\d+
    )
    \s*%?
    \s*\)?
    \s*(?:\([a-z0-9]+\))?
    $
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

SYMBOL_ONLY_RE = re.compile(
    r"^(?:[\$€£%]|[-–—−]+|n/?m)$",
    flags=re.IGNORECASE,
)

FOOTNOTE_ONLY_RE = re.compile(
    r"^(?:\([a-z0-9]+\)\s*)+$",
    flags=re.IGNORECASE,
)

YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

MISSING_VALUE_RE = re.compile(
    r"^(?:n/?a|n/?m|not applicable|not meaningful)$",
    flags=re.IGNORECASE,
)

def normalize_label(value: object) -> str:
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip(" |")


def unique_in_order(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = normalize_label(value)
        key = normalized.casefold()

        if normalized and key not in seen:
            unique_values.append(normalized)
            seen.add(key)

    return unique_values


def is_pure_numeric_cell(value: str) -> bool:
    normalized = normalize_label(value)

    if not normalized:
        return True

    # Years carry useful column semantics and must be preserved.
    if YEAR_RE.fullmatch(normalized):
        return False

    return bool(
        PURE_NUMERIC_RE.fullmatch(normalized)
        or SYMBOL_ONLY_RE.fullmatch(normalized)
    )


def is_ignored_label(value: str) -> bool:
    normalized = normalize_label(value)

    return bool(
        not normalized
        or is_pure_numeric_cell(normalized)
        or FOOTNOTE_ONLY_RE.fullmatch(normalized)
        or MISSING_VALUE_RE.fullmatch(normalized)
    )


def extract_cells(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []

    for line in table_text.splitlines():
        cells = [
            normalize_label(cell)
            for cell in line.split("|")
            if normalize_label(cell)
        ]

        if cells:
            rows.append(cells)

    return rows


def extract_column_labels(table_header: str) -> list[str]:
    labels: list[str] = []

    for row in extract_cells(table_header):
        for cell in row:
            if not is_ignored_label(cell):
                labels.append(cell)

    return unique_in_order(labels)


def extract_row_labels(
    table_text: str,
    table_header: str,
    column_labels: list[str],
) -> list[str]:
    header_lines = {
        normalize_label(line).casefold()
        for line in table_header.splitlines()
        if normalize_label(line)
    }
    column_keys = {label.casefold() for label in column_labels}

    labels: list[str] = []

    for line in table_text.splitlines():
        normalized_line = normalize_label(line)

        if not normalized_line:
            continue

        if normalized_line.casefold() in header_lines:
            continue

        for cell in line.split("|"):
            label = normalize_label(cell)

            if is_ignored_label(label):
                continue

            if label.casefold() in column_keys:
                continue

            labels.append(label)

    return unique_in_order(labels)


def clean_table_context(table_context: str) -> str:
    context_parts: list[str] = []

    for line in table_context.splitlines():
        normalized = normalize_label(line)

        if not normalized:
            continue

        prefix, separator, value = normalized.partition(":")
        prefix_key = prefix.casefold()

        if prefix_key in {"section", "unit"}:
            continue

        if separator and prefix_key in {"description", "context"}:
            normalized = normalize_label(value)

        if normalized:
            context_parts.append(normalized)

    return " ".join(unique_in_order(context_parts))


def extract_units(
    table_context: str,
    table_header: str,
    table_text: str,
) -> list[str]:
    source = " ".join(
        [table_context, table_header, table_text]
    ).casefold()

    units: list[str] = []

    if re.search(r"\bshares?\s+in\s+thousands\b", source):
        units.append("shares in thousands")

    if re.search(r"\bdollars?\s+per\s+share\b", source):
        units.append("dollars per share")

    if re.search(
        r"(?:\bdollars?\b|\$)\s*(?:amounts?\s+)?in\s+millions\b",
        source,
    ):
        units.append("dollars in millions")
    elif re.search(r"\bin\s+millions\b", source):
        units.append("in millions")

    if re.search(
        r"(?:\bdollars?\b|\$)\s*(?:amounts?\s+)?in\s+thousands\b",
        source,
    ):
        units.append("dollars in thousands")
    elif re.search(r"\bin\s+thousands\b", source):
        units.append("in thousands")

    if re.search(r"\bin\s+billions\b", source):
        units.append("in billions")

    if re.search(r"\bbasis\s+points?\b", source):
        units.append("basis points")

    if "%" in source or re.search(r"\bpercent(?:age)?\b", source):
        units.append("percent")

    return unique_in_order(units)


def build_proxy_text(
    ticker: str,
    report_date: str,
    sec_item: str,
    section_title: str,
    context: str,
    column_labels: list[str],
    row_labels: list[str],
    units: list[str],
) -> str:
    year_match = re.match(r"\d{4}", report_date)
    report_year = year_match.group() if year_match else report_date

    lines = [
        f"Bank: {ticker}",
        f"Report: {report_year} 10-K",
    ]

    if sec_item:
        lines.append(f"SEC item: {sec_item}")

    if section_title:
        lines.append(f"Section: {section_title}")

    if context:
        lines.append(f"Context: {context}")

    if column_labels:
        lines.append(f"Columns: {'; '.join(column_labels)}")

    if row_labels:
        lines.append(f"Rows: {'; '.join(row_labels)}")

    if units:
        lines.append(f"Units: {'; '.join(units)}")

    return "\n".join(lines)


def build_table_proxy(
    chunk: dict[str, object],
) -> dict[str, object]:
    if chunk.get("element_type") != "table":
        raise ValueError("Chunk is not a table")

    target_chunk_id = normalize_label(chunk.get("chunk_id"))
    ticker = normalize_label(chunk.get("ticker"))
    report_date = normalize_label(chunk.get("report_date"))

    if not target_chunk_id:
        raise ValueError("Missing chunk_id")

    if not ticker:
        raise ValueError(f"Missing ticker for chunk {target_chunk_id}")

    if not report_date:
        raise ValueError(
            f"Missing report_date for chunk {target_chunk_id}"
        )

    sec_item = normalize_label(chunk.get("sec_item"))
    section_title = normalize_label(chunk.get("section_title"))
    table_context = str(chunk.get("table_context") or "")
    table_header = str(chunk.get("table_header") or "")
    table_text = str(chunk.get("text") or "")

    column_labels = extract_column_labels(table_header)
    row_labels = extract_row_labels(
        table_text=table_text,
        table_header=table_header,
        column_labels=column_labels,
    )
    context = clean_table_context(table_context)
    units = extract_units(
        table_context=table_context,
        table_header=table_header,
        table_text=table_text,
    )

    proxy_text = build_proxy_text(
        ticker=ticker,
        report_date=report_date,
        sec_item=sec_item,
        section_title=section_title,
        context=context,
        column_labels=column_labels,
        row_labels=row_labels,
        units=units,
    )

    proxy_source = f"{PROXY_VERSION}\0{target_chunk_id}"
    proxy_id = hashlib.sha256(proxy_source.encode("utf-8")).hexdigest()

    return {
        "proxy_id": proxy_id,
        "target_chunk_id": target_chunk_id,
        "ticker": ticker,
        "report_date": report_date,
        "sec_item": sec_item or None,
        "section_title": section_title or None,
        "element_type": "table",
        "table_id": chunk.get("table_id"),
        "table_part_index": chunk.get("table_part_index"),
        "table_part_count": chunk.get("table_part_count"),
        "proxy_text": proxy_text,
        "proxy_version": PROXY_VERSION,
    }


def load_table_chunks(input_path: Path) -> list[dict[str, object]]:
    table_chunks: list[dict[str, object]] = []

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number}"
                )

            if record.get("element_type") == "table":
                table_chunks.append(record)

    return table_chunks


def load_existing_proxy_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    proxy_ids: set[str] = set()

    with output_path.open(encoding="utf-8") as output_file:
        for line in output_file:
            if not line.strip():
                continue

            record = json.loads(line)
            proxy_id = normalize_label(record.get("proxy_id"))

            if proxy_id:
                proxy_ids.add(proxy_id)

    return proxy_ids


def load_table_proxies(output_path: Path) -> list[dict[str, object]]:
    proxies: list[dict[str, object]] = []

    with output_path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number}"
                )

            proxies.append(record)

    return proxies


def validate_table_proxies(
    table_chunks: list[dict[str, object]],
    proxies: list[dict[str, object]],
) -> None:
    source_ids = [
        normalize_label(chunk.get("chunk_id"))
        for chunk in table_chunks
    ]
    proxy_ids = [
        normalize_label(proxy.get("proxy_id"))
        for proxy in proxies
    ]
    target_ids = [
        normalize_label(proxy.get("target_chunk_id"))
        for proxy in proxies
    ]

    if any(not source_id for source_id in source_ids):
        raise ValueError("Every table chunk must have a chunk_id")

    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Table chunk IDs must be unique")

    if len(table_chunks) != len(proxies):
        raise ValueError(
            "Expected one proxy per table chunk: "
            f"chunks={len(table_chunks)}, proxies={len(proxies)}"
        )

    if any(not proxy_id for proxy_id in proxy_ids):
        raise ValueError("Every table proxy must have a proxy_id")

    if len(set(proxy_ids)) != len(proxy_ids):
        raise ValueError("Proxy IDs must be unique")

    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Target chunk IDs must be unique")

    source_id_set = set(source_ids)
    target_id_set = set(target_ids)

    if target_id_set != source_id_set:
        missing = sorted(source_id_set - target_id_set)[:5]
        extra = sorted(target_id_set - source_id_set)[:5]
        raise ValueError(
            f"Proxy targets do not match table chunks: missing={missing}, extra={extra}"
        )

    source_by_id = dict(zip(source_ids, table_chunks, strict=True))
    metadata_fields = (
        "ticker",
        "report_date",
        "table_id",
        "table_part_index",
        "table_part_count",
    )

    for proxy in proxies:
        proxy_id = normalize_label(proxy.get("proxy_id"))
        target_id = normalize_label(proxy.get("target_chunk_id"))
        proxy_text = str(proxy.get("proxy_text") or "")
        proxy_version = normalize_label(proxy.get("proxy_version"))

        if proxy_version != PROXY_VERSION:
            raise ValueError(
                f"Unexpected proxy version for {target_id}: {proxy_version}"
            )

        expected_proxy_id = hashlib.sha256(
            f"{PROXY_VERSION}\0{target_id}".encode()
        ).hexdigest()

        if proxy_id != expected_proxy_id:
            raise ValueError(f"Invalid proxy_id for target {target_id}")

        if not proxy_text.strip():
            raise ValueError(f"Empty proxy_text for target {target_id}")

        if "SEC item: None" in proxy_text or "Section: None" in proxy_text:
            raise ValueError(f"Invalid None label in proxy_text for target {target_id}")

        source_chunk = source_by_id[target_id]

        for field in metadata_fields:
            if proxy.get(field) != source_chunk.get(field):
                raise ValueError(
                    f"Metadata mismatch for {target_id}, field={field}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic table proxies."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    return args


def main() -> None:
    args = parse_args()

    table_chunks = load_table_chunks(args.input)
    selected_chunks = (
        table_chunks[: args.limit]
        if args.limit is not None
        else table_chunks
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_proxy_ids = (
        set()
        if args.overwrite
        else load_existing_proxy_ids(args.output)
    )
    output_mode = "w" if args.overwrite else "a"

    generated = 0
    skipped = 0
    failed = 0

    with args.output.open(
        output_mode,
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for chunk in selected_chunks:
            try:
                proxy = build_table_proxy(chunk)
                proxy_id = str(proxy["proxy_id"])

                if proxy_id in existing_proxy_ids:
                    skipped += 1
                    continue

                output_file.write(
                    json.dumps(proxy, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                existing_proxy_ids.add(proxy_id)
                generated += 1
            except (KeyError, TypeError, ValueError) as error:
                failed += 1
                print(
                    "Failed chunk "
                    f"{chunk.get('chunk_id')}: {error}"
                )

    print(f"Input table chunks: {len(table_chunks)}")
    print(f"Selected table chunks: {len(selected_chunks)}")
    print(f"Generated proxies: {generated}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")

    if failed:
        raise SystemExit(1)

    if args.limit is None:
        proxies = load_table_proxies(args.output)
        validate_table_proxies(table_chunks, proxies)
        print(f"Validated proxies: {len(proxies)}")


if __name__ == "__main__":
    main()
