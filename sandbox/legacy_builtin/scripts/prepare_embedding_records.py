import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks/sec_10k_chunks.jsonl")
DEFAULT_PROXIES_PATH = Path("data/processed/table_proxies/sec_10k_table_proxies.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/embedding_records/sec_10k_embedding_records.jsonl")

METADATA_FIELDS = (
    "ticker",
    "cik",
    "accession_number",
    "filing_date",
    "report_date",
    "source_url",
    "sec_item",
    "section_title",
    "element_type",
    "order_start",
    "order_end",
    "token_count",
    "chunk_index",
    "table_id",
    "table_part_index",
    "table_part_count",
)

TABLE_LINK_FIELDS = (
    "ticker",
    "report_date",
    "sec_item",
    "section_title",
    "table_id",
    "table_part_index",
    "table_part_count",
)

Record = dict[str, Any]
Scalar = str | int | float | bool


def load_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}: {path}")

            records.append(record)

    return records


def required_text(
    record: Record,
    field: str,
    record_name: str,
) -> str:
    value = str(record.get(field) or "").strip()

    if not value:
        raise ValueError(f"Missing {field} for {record_name}")

    return value


def build_metadata(
    chunk: Record,
    proxy: Record | None = None,
) -> dict[str, Scalar]:
    metadata: dict[str, Scalar] = {}

    for field in METADATA_FIELDS:
        value = chunk.get(field)

        if isinstance(value, (str, int, float, bool)) and value != "":
            metadata[field] = value

    if proxy is not None:
        metadata["proxy_id"] = required_text(
            proxy,
            "proxy_id",
            "table proxy",
        )
        metadata["proxy_version"] = required_text(
            proxy,
            "proxy_version",
            "table proxy",
        )

    return metadata


def build_text_embedding_text(chunk: Record) -> str:
    chunk_id = required_text(chunk, "chunk_id", "text chunk")
    ticker = required_text(chunk, "ticker", chunk_id)
    report_date = required_text(chunk, "report_date", chunk_id)
    document = required_text(chunk, "text", chunk_id)

    lines = [
        f"Bank: {ticker}",
        f"Report: {report_date[:4]} 10-K",
    ]

    sec_item = str(chunk.get("sec_item") or "").strip()
    section_title = str(chunk.get("section_title") or "").strip()

    if sec_item:
        lines.append(f"SEC item: {sec_item}")

    if section_title:
        lines.append(f"Section: {section_title}")

    return "\n".join([*lines, "", document])


def index_chunks(chunks: list[Record]) -> dict[str, Record]:
    chunks_by_id: dict[str, Record] = {}

    for chunk in chunks:
        chunk_id = required_text(chunk, "chunk_id", "chunk")

        if chunk_id in chunks_by_id:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")

        chunks_by_id[chunk_id] = chunk

    return chunks_by_id


def index_proxies(
    proxies: list[Record],
) -> dict[str, Record]:
    proxies_by_target: dict[str, Record] = {}
    proxy_ids: set[str] = set()

    for proxy in proxies:
        proxy_id = required_text(proxy, "proxy_id", "table proxy")
        target_chunk_id = required_text(
            proxy,
            "target_chunk_id",
            "table proxy",
        )

        if proxy_id in proxy_ids:
            raise ValueError(f"Duplicate proxy_id: {proxy_id}")

        if target_chunk_id in proxies_by_target:
            raise ValueError(f"Duplicate target_chunk_id: {target_chunk_id}")

        proxy_ids.add(proxy_id)
        proxies_by_target[target_chunk_id] = proxy

    return proxies_by_target


def validate_proxy_link(
    chunk: Record,
    proxy: Record,
) -> None:
    chunk_id = required_text(chunk, "chunk_id", "table chunk")

    for field in TABLE_LINK_FIELDS:
        if proxy.get(field) != chunk.get(field):
            raise ValueError(
                f"Proxy metadata does not match table chunk: {chunk_id}, field={field}"
            )


def prepare_embedding_records(
    chunks: list[Record],
    proxies: list[Record],
) -> list[Record]:
    chunks_by_id = index_chunks(chunks)
    proxies_by_target = index_proxies(proxies)

    table_chunk_ids = {
        chunk_id for chunk_id, chunk in chunks_by_id.items() if chunk.get("element_type") == "table"
    }
    proxy_target_ids = set(proxies_by_target)

    if proxy_target_ids != table_chunk_ids:
        missing = sorted(table_chunk_ids - proxy_target_ids)[:5]
        extra = sorted(proxy_target_ids - table_chunk_ids)[:5]

        raise ValueError(f"Expected one proxy per table chunk: missing={missing}, extra={extra}")

    records: list[Record] = []

    for chunk in chunks:
        chunk_id = required_text(chunk, "chunk_id", "chunk")
        document = required_text(chunk, "text", chunk_id)
        required_text(chunk, "ticker", chunk_id)
        required_text(chunk, "report_date", chunk_id)

        record_type = chunk.get("element_type")

        if record_type == "text":
            record_id = f"text::{chunk_id}"
            embedding_text = build_text_embedding_text(chunk)
            metadata = build_metadata(chunk)

        elif record_type == "table":
            proxy = proxies_by_target[chunk_id]
            validate_proxy_link(chunk, proxy)

            proxy_id = required_text(proxy, "proxy_id", chunk_id)
            record_id = f"table::{proxy_id}"
            embedding_text = required_text(
                proxy,
                "proxy_text",
                chunk_id,
            )
            metadata = build_metadata(chunk, proxy)

        else:
            raise ValueError(f"Unsupported element_type for {chunk_id}: {record_type}")

        records.append(
            {
                "record_id": record_id,
                "record_type": record_type,
                "embedding_text": embedding_text,
                "document": document,
                "target_chunk_id": chunk_id,
                "metadata": metadata,
            }
        )

    record_ids = [str(record["record_id"]) for record in records]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Embedding record IDs must be unique")

    return records


def write_jsonl(
    records: list[Record],
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare text and table embedding records.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
    )
    parser.add_argument(
        "--proxies",
        type=Path,
        default=DEFAULT_PROXIES_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_jsonl(args.chunks)
    proxies = load_jsonl(args.proxies)

    records = prepare_embedding_records(chunks, proxies)
    write_jsonl(records, args.output, args.overwrite)

    text_count = sum(record["record_type"] == "text" for record in records)
    table_count = len(records) - text_count

    print(f"Text records: {text_count}")
    print(f"Table records: {table_count}")
    print(f"Total records: {len(records)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
