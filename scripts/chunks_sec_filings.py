import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"

INPUT_DIR = Path("data/processed/elements")
OUTPUT_PATH = Path("data/processed/chunks/sec_10k_chunks.jsonl")

TARGET_TOKENS = 600
MAX_TOKENS = 700
OVERLAP_TOKENS = 80
MIN_TEXT_TOKENS = 80

Record = dict[str, Any]


def encode(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def split_text_element(
    element: Record,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, int, int]]:
    text = str(element["text"]).strip()
    order_index = int(element["order_index"])

    if len(encode(tokenizer, text)) <= MAX_TOKENS:
        return [(text, order_index, order_index)]

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=TARGET_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
        separators=[
            "\n\n",
            "\n",
            r"(?<=[.!?])\s+",
            "; ",
            ", ",
            " ",
            "",
        ],
        keep_separator="end",
        is_separator_regex=True,
    )

    return [
        (part.strip(), order_index, order_index)
        for part in splitter.split_text(text)
        if part.strip()
    ]


def split_text_elements(
    elements: list[Record],
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, int, int]]:
    pieces = [piece for element in elements for piece in split_text_element(element, tokenizer)]

    chunks: list[tuple[str, int, int]] = []
    current_texts: list[str] = []
    current_start = 0
    current_end = 0
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current_texts
        nonlocal current_start
        nonlocal current_end
        nonlocal current_tokens

        if not current_texts:
            return

        text = "\n\n".join(current_texts)

        if chunks and current_tokens < MIN_TEXT_TOKENS:
            previous_text, previous_start, _ = chunks[-1]
            merged_text = f"{previous_text}\n\n{text}"

            if len(encode(tokenizer, merged_text)) <= MAX_TOKENS:
                chunks[-1] = (
                    merged_text,
                    previous_start,
                    current_end,
                )
                current_texts = []
                current_tokens = 0
                return

        chunks.append((text, current_start, current_end))
        current_texts = []
        current_tokens = 0

    for text, order_start, order_end in pieces:
        if not current_texts:
            current_texts = [text]
            current_start = order_start
            current_end = order_end
            current_tokens = len(encode(tokenizer, text))
            continue

        candidate_text = "\n\n".join([*current_texts, text])
        candidate_tokens = len(encode(tokenizer, candidate_text))

        if current_tokens < TARGET_TOKENS and candidate_tokens <= MAX_TOKENS:
            current_texts.append(text)
            current_end = order_end
            current_tokens = candidate_tokens
            continue

        flush_current()

        current_texts = [text]
        current_start = order_start
        current_end = order_end
        current_tokens = len(encode(tokenizer, text))

    flush_current()
    return chunks


def split_table(
    element: Record,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    rows = [row.strip() for row in str(element["text"]).splitlines() if row.strip()]

    if not rows:
        return []

    full_text = "\n".join(rows)

    if len(encode(tokenizer, full_text)) <= MAX_TOKENS:
        return [full_text]

    header = rows[0]
    header_ids = encode(tokenizer, f"{header}\n")

    if len(header_ids) >= MAX_TOKENS:
        table_ids = encode(tokenizer, full_text)

        return [
            tokenizer.decode(
                table_ids[start : start + MAX_TOKENS],
                skip_special_tokens=True,
            ).strip()
            for start in range(0, len(table_ids), MAX_TOKENS)
        ]

    parts: list[str] = []
    current_rows = [header]

    def save_current() -> None:
        if len(current_rows) > 1:
            parts.append("\n".join(current_rows))

    for row in rows[1:]:
        candidate = "\n".join([*current_rows, row])

        if len(encode(tokenizer, candidate)) <= MAX_TOKENS:
            current_rows.append(row)
            continue

        save_current()

        row_ids = encode(tokenizer, row)
        available_tokens = MAX_TOKENS - len(header_ids)

        if len(row_ids) > available_tokens:
            for start in range(0, len(row_ids), available_tokens):
                row_part = tokenizer.decode(
                    row_ids[start : start + available_tokens],
                    skip_special_tokens=True,
                ).strip()

                parts.append(f"{header}\n{row_part}")

            current_rows = [header]
        else:
            current_rows = [header, row]

    save_current()
    return parts


def create_chunk(
    source: Record,
    text: str,
    element_type: str,
    order_start: int,
    order_end: int,
    tokenizer: PreTrainedTokenizerBase,
) -> Record:
    return {
        "ticker": source["ticker"],
        "cik": source["cik"],
        "accession_number": source["accession_number"],
        "filing_date": source["filing_date"],
        "report_date": source["report_date"],
        "source_url": source["source_url"],
        "sec_item": source.get("sec_item"),
        "element_type": element_type,
        "order_start": order_start,
        "order_end": order_end,
        "token_count": len(encode(tokenizer, text)),
        "text": text,
    }


def chunk_filing(
    records: list[Record],
    tokenizer: PreTrainedTokenizerBase,
) -> list[Record]:
    chunks: list[Record] = []
    text_group: list[Record] = []

    def flush_text_group() -> None:
        if not text_group:
            return

        source = text_group[0]

        for text, order_start, order_end in split_text_elements(
            text_group,
            tokenizer,
        ):
            chunks.append(
                create_chunk(
                    source=source,
                    text=text,
                    element_type="text",
                    order_start=order_start,
                    order_end=order_end,
                    tokenizer=tokenizer,
                )
            )

        text_group.clear()

    for record in records:
        if bool(record.get("is_navigation")):
            flush_text_group()
            continue
        if record["element_type"] == "table":
            flush_text_group()
            order_index = int(record["order_index"])

            for text in split_table(record, tokenizer):
                chunks.append(
                    create_chunk(
                        source=record,
                        text=text,
                        element_type="table",
                        order_start=order_index,
                        order_end=order_index,
                        tokenizer=tokenizer,
                    )
                )

            continue

        if text_group and record.get("sec_item") != text_group[-1].get("sec_item"):
            flush_text_group()

        text_group.append(record)

    flush_text_group()

    for chunk_index, chunk in enumerate(chunks):
        chunk["chunk_index"] = chunk_index

        identity = f"{chunk['accession_number']}\0{chunk_index}\0{chunk['text']}"

        chunk["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return chunks


def load_jsonl(path: Path) -> list[Record]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def print_summary(ticker: str, chunks: list[Record]) -> None:
    token_counts = sorted(int(chunk["token_count"]) for chunk in chunks)
    p95_index = max(0, int(0.95 * len(token_counts)) - 1)
    table_count = sum(chunk["element_type"] == "table" for chunk in chunks)

    print(
        f"{ticker}: chunks={len(chunks)}, "
        f"tables={table_count}, "
        f"tokens_min={token_counts[0]}, "
        f"tokens_mean={mean(token_counts):.1f}, "
        f"tokens_p95={token_counts[p95_index]}, "
        f"tokens_max={token_counts[-1]}"
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )
    all_chunks: list[Record] = []

    for path in sorted(INPUT_DIR.glob("*.jsonl")):
        records = load_jsonl(path)

        if not records:
            raise ValueError(f"{path} je prazan")

        chunks = chunk_filing(records, tokenizer)

        if not chunks:
            raise ValueError(f"{path}: nisu napravljeni chunkovi")

        print_summary(str(records[0]["ticker"]), chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"Nema JSONL fajlova u {INPUT_DIR}")

    if any(not chunk["text"] or int(chunk["token_count"]) > MAX_TOKENS for chunk in all_chunks):
        raise ValueError("Pronađen je prazan ili predugačak chunk")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in all_chunks) + "\n",
        encoding="utf-8",
    )

    print(f"Ukupno: {len(all_chunks)} chunkova -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
