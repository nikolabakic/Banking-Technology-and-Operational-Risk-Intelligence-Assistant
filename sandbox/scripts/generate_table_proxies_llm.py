from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline

Record = dict[str, Any]

INPUT_PATH = Path("data/processed/chunks/sec_10k_chunks.jsonl")
OUTPUT_PATH = Path("data/processed/table_proxies/sec_10k_table_proxies.jsonl")

MODEL_ID = "Qwen/Qwen3.5-9B"
PROMPT_VERSION = "table_proxy_v1"
MAX_LIST_ITEMS = 40

SYSTEM_PROMPT = """
You create high-recall retrieval proxies for financial tables from SEC 10-K
filings.

Return exactly one valid JSON object with these keys:

- description: one or two sentences describing the purpose and content
  of the table;
- topics: important financial concepts, products, risks, entities, and
  metrics represented in the table;
- column_labels: meaningful column dimensions, dates, periods, scenarios,
  categories, and comparison groups;
- row_labels: meaningful row dimensions, line items, products, metrics,
  and categories;
- units: currencies, scales, percentages, ratios, or other units.

Rules:

- Use only information present in the supplied metadata and table.
- Preserve exact financial terminology.
- Preserve years, periods, scenario names, and category names.
- Do not include body cell values or numerical results.
- A number may appear only when it is part of a label, year, period,
  threshold category, or unit.
- Do not invent missing information.
- Remove duplicates.
- Each list may contain at most 40 short strings.
- Do not return Markdown, explanations, or code fences.
""".strip()

EXPECTED_JSON = """
{
  "description": "Short semantic description of the table.",
  "topics": ["topic"],
  "column_labels": ["column label"],
  "row_labels": ["row label"],
  "units": ["unit"]
}
""".strip()


class TableProxyContent(BaseModel):
    description: str = Field(min_length=1)
    topics: list[str] = Field(default_factory=list)
    column_labels: list[str] = Field(default_factory=list)
    row_labels: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def clean_description(cls, value: str) -> str:
        cleaned = " ".join(value.split())

        if not cleaned:
            raise ValueError("Description is empty.")

        if len(cleaned) > 600:
            raise ValueError("Description is too long.")

        return cleaned

    @field_validator(
        "topics",
        "column_labels",
        "row_labels",
        "units",
        mode="before",
    )
    @classmethod
    def clean_list(cls, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            raise ValueError("Expected a list.")

        cleaned_items: list[str] = []
        seen: set[str] = set()

        for item in value:
            cleaned = " ".join(str(item).split()).strip(" ;,")

            if not cleaned:
                continue

            normalized = cleaned.casefold()

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned_items.append(cleaned)

        return cleaned_items[:MAX_LIST_ITEMS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--exclude-ticker", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_table_chunks(path: Path) -> list[Record]:
    table_chunks: list[Record] = []

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("element_type") != "table":
                continue

            missing_fields = [
                field
                for field in ("chunk_id", "table_id", "text")
                if not record.get(field)
            ]

            if missing_fields:
                fields = ", ".join(missing_fields)
                raise ValueError(
                    f"Line {line_number} is missing required fields: {fields}"
                )

            table_chunks.append(record)

    return table_chunks


def create_proxy_id(
    chunk_id: str,
    model_id: str,
) -> str:
    identity = "\0".join(
        [
            chunk_id,
            model_id,
            PROMPT_VERSION,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_completed_proxy_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc

            proxy_id = record.get("proxy_id")

            if proxy_id:
                completed.add(str(proxy_id))

    return completed


def build_user_prompt(chunk: Record) -> str:
    metadata = {
        "bank": chunk.get("ticker"),
        "report_date": chunk.get("report_date"),
        "sec_item": chunk.get("sec_item"),
        "section_title": chunk.get("section_title"),
        "table_context": chunk.get("table_context"),
        "table_header": chunk.get("table_header"),
        "table_part_index": chunk.get("table_part_index"),
        "table_part_count": chunk.get("table_part_count"),
    }

    return "\n\n".join(
        [
            "Create a retrieval proxy for this table part.",
            "Required output format:",
            EXPECTED_JSON,
            "Metadata:",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            "Table:",
            str(chunk["text"]),
        ]
    )


def extract_json_object(text: str) -> Record:
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in model output.")

    result, _ = json.JSONDecoder().raw_decode(text[start:])

    if not isinstance(result, dict):
        raise ValueError("Model output is not a JSON object.")

    return result


def get_generated_text(result: Any) -> str:
    generated = result[0]["generated_text"]

    if isinstance(generated, str):
        return generated.strip()

    if isinstance(generated, list) and generated:
        last_message = generated[-1]

        if isinstance(last_message, dict):
            return str(last_message.get("content", "")).strip()

    raise ValueError("Unexpected model output format.")


def load_generator(model_id: str) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. Run this script on a Colab T4 runtime."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_id}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    return pipeline(
        task="text-generation",
        model=model_id,
        device_map="auto",
        dtype=torch.float16,
        model_kwargs={
            "quantization_config": quantization_config,
        },
    )


def generate_proxy_content(
    generator: Any,
    chunk: Record,
    max_new_tokens: int,
    max_attempts: int,
) -> TableProxyContent:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(chunk),
        },
    ]

    last_error: Exception | None = None

    for attempt in range(max_attempts):
        prompt = generator.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

        generated_text = get_generated_text(result)

        try:
            parsed = extract_json_object(generated_text)
            return TableProxyContent.model_validate(parsed)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_error = exc

            if attempt + 1 == max_attempts:
                break

            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    {
                        "role": "user",
                        "content": (
                            "The response was invalid. Return only one valid "
                            "JSON object matching the requested schema."
                        ),
                    },
                ]
            )

    raise RuntimeError(
        f"Could not generate a valid proxy: {last_error}"
    ) from last_error


def build_proxy_text(
    chunk: Record,
    content: TableProxyContent,
) -> str:
    lines = [
        f"Bank: {chunk['ticker']}",
        f"Report: 10-K for {chunk['report_date']}",
    ]

    if chunk.get("sec_item"):
        lines.append(f"SEC item: {chunk['sec_item']}")

    if chunk.get("section_title"):
        lines.append(f"Section: {chunk['section_title']}")

    part_count = chunk.get("table_part_count")

    if isinstance(part_count, int) and part_count > 1:
        part_index = int(chunk["table_part_index"]) + 1
        lines.append(f"Table part: {part_index} of {part_count}")

    lines.append(f"Description: {content.description}")

    fields = [
        ("Topics", content.topics),
        ("Columns", content.column_labels),
        ("Rows", content.row_labels),
        ("Units", content.units),
    ]

    for label, values in fields:
        if values:
            lines.append(f"{label}: {'; '.join(values)}")

    return "\n".join(lines)


def create_proxy_record(
    chunk: Record,
    content: TableProxyContent,
    model_id: str,
) -> Record:
    proxy: Record = {
        "proxy_id": create_proxy_id(
            str(chunk["chunk_id"]),
            model_id,
        ),
        "proxy_type": "table_part_summary",
        "target_chunk_id": chunk["chunk_id"],
        "table_id": chunk["table_id"],
        "proxy_text": build_proxy_text(chunk, content),
        "description": content.description,
        "topics": content.topics,
        "column_labels": content.column_labels,
        "row_labels": content.row_labels,
        "units": content.units,
        "generation_model": model_id,
        "prompt_version": PROMPT_VERSION,
    }

    metadata_fields = [
        "ticker",
        "cik",
        "accession_number",
        "filing_date",
        "report_date",
        "source_url",
        "sec_item",
        "section_title",
        "order_start",
        "order_end",
        "table_part_index",
        "table_part_count",
    ]

    for field in metadata_fields:
        if field in chunk:
            proxy[field] = chunk[field]

    return proxy


def append_jsonl(path: Path, record: Record) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def main() -> None:
    args = parse_args()

    table_chunks = load_table_chunks(args.input)

    included_tickers = {
        ticker.upper()
        for ticker in args.ticker or []
    }
    excluded_tickers = {
        ticker.upper()
        for ticker in args.exclude_ticker
    }

    if included_tickers:
        table_chunks = [
            chunk
            for chunk in table_chunks
            if str(chunk["ticker"]).upper() in included_tickers
        ]

    table_chunks = [
        chunk
        for chunk in table_chunks
        if str(chunk["ticker"]).upper() not in excluded_tickers
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    error_path = args.output.with_suffix(".errors.jsonl")

    if args.overwrite:
        args.output.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

    completed_proxy_ids = load_completed_proxy_ids(args.output)

    pending_chunks = [
        chunk
        for chunk in table_chunks
        if create_proxy_id(
            str(chunk["chunk_id"]),
            args.model,
        )
        not in completed_proxy_ids
    ]

    if args.limit is not None:
        pending_chunks = pending_chunks[: args.limit]

    logical_table_count = len(
        {
            str(chunk["table_id"])
            for chunk in pending_chunks
        }
    )

    print(
        f"Za obradu: {len(pending_chunks)} delova iz "
        f"{logical_table_count} logičkih tabela"
    )

    if not pending_chunks:
        return

    generator = load_generator(args.model)

    generated_count = 0
    failed_count = 0

    for chunk in tqdm(pending_chunks, desc="Table proxies"):
        try:
            content = generate_proxy_content(
                generator=generator,
                chunk=chunk,
                max_new_tokens=args.max_new_tokens,
                max_attempts=args.max_attempts,
            )
            proxy = create_proxy_record(
                chunk=chunk,
                content=content,
                model_id=args.model,
            )
            append_jsonl(args.output, proxy)
            generated_count += 1
        except Exception as exc:
            failed_count += 1
            append_jsonl(
                error_path,
                {
                    "target_chunk_id": chunk["chunk_id"],
                    "table_id": chunk["table_id"],
                    "ticker": chunk["ticker"],
                    "error": str(exc),
                },
            )

    print(f"Generisano: {generated_count}")
    print(f"Neuspešno: {failed_count}")
    print(f"Output: {args.output}")

    if failed_count:
        print(f"Greške: {error_path}")


if __name__ == "__main__":
    main()