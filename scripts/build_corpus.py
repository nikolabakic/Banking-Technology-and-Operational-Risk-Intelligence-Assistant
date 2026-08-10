from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

from bs4 import XMLParsedAsHTMLWarning
from sec2md import Parser

from bankscope.config.settings import get_settings
from bankscope.io import sha256_file, write_jsonl
from bankscope.parsing.corpus import (
    CORPUS_VERSION,
    MAX_EMBEDDING_TOKENS,
    build_corpus,
    validate_corpus,
)
from bankscope.parsing.tables import PARSER_NAME

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "data/filings.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed"
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"

Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the active sec2md narrative corpus and lossless table store."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument(
        "--ticker",
        action="append",
        help="Build one ticker (repeat the option or pass a comma-separated list).",
    )
    parser.add_argument(
        "--description-mode",
        choices=("local", "openai"),
        default="local",
    )
    parser.add_argument("--model", help="OpenAI model override (defaults to OPENAI_MODEL).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_raw_path(manifest_path: Path, filing: Record) -> Path:
    raw_path = Path(str(filing["local_html_path"]))
    if raw_path.is_absolute():
        return raw_path

    candidates = [ROOT / raw_path, manifest_path.parent / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_filings(path: Path) -> list[Record]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("filings")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("data/filings.json must be a list or an object with a 'filings' list.")
    return payload


def filter_filings(filings: list[Record], ticker_args: list[str] | None) -> list[Record]:
    if not ticker_args:
        return filings

    requested = {
        ticker.strip().upper()
        for argument in ticker_args
        for ticker in argument.split(",")
        if ticker.strip()
    }
    available = {str(filing.get("ticker", "")).upper() for filing in filings}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Tickers missing from data/filings.json: {missing}.")
    return [filing for filing in filings if str(filing.get("ticker", "")).upper() in requested]


def make_token_counter(tokenizer: Any):
    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return token_count


def manifest_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> None:
    args = parse_args()
    if args.ticker and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(
            "Filtered builds require an explicit --output-dir to protect the full corpus."
        )
    if args.description_mode == "local" and args.model:
        raise ValueError("--model is only valid with --description-mode openai.")

    output_paths = {
        "chunks": args.output_dir / "chunks.jsonl",
        "tables": args.output_dir / "tables.jsonl",
        "manifest": args.output_dir / "manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}. Use --overwrite.")

    filings = filter_filings(load_filings(args.manifest), args.ticker)
    if not filings:
        raise ValueError("No filings selected.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_count = make_token_counter(tokenizer)
    llm_client = None
    description_model = None
    if args.description_mode == "openai":
        settings = get_settings()
        from bankscope.llm import create_openai_client

        description_model = args.model or settings.openai_model
        llm_client = create_openai_client(settings)

    all_chunks: list[Record] = []
    all_tables: list[Record] = []
    filing_summaries: list[Record] = []

    for filing in filings:
        ticker = str(filing.get("ticker", ""))
        raw_path = resolve_raw_path(args.manifest, filing)
        if not raw_path.exists():
            raise FileNotFoundError(f"{ticker} raw filing not found: {raw_path}.")

        raw_bytes = raw_path.read_bytes()
        raw_sha256 = sha256_bytes(raw_bytes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            parser = Parser(raw_bytes.decode("utf-8", errors="replace"))
            pages = parser.get_pages(include_elements=True, include_images=False)
            annotated_html = str(parser.html())

        chunks, tables = build_corpus(
            pages,
            filing,
            raw_sha256,
            token_count,
            annotated_html=annotated_html,
            description_mode=args.description_mode,
            llm_client=llm_client,
            llm_model=description_model or "gpt-4o",
        )
        all_chunks.extend(chunks)
        all_tables.extend(tables)
        filing_summaries.append(
            {
                "ticker": ticker,
                "accession_number": str(filing["accession_number"]),
                "raw_path": raw_path.as_posix(),
                "raw_sha256": raw_sha256,
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "text_chunk_count": sum(chunk["record_type"] == "text" for chunk in chunks),
                "table_chunk_count": sum(chunk["record_type"] == "table" for chunk in chunks),
                "table_count": len(tables),
            }
        )
        print(f"{ticker}: chunks={len(chunks)}, tables={len(tables)}")

    validate_corpus(all_chunks, all_tables, token_count=token_count)
    write_jsonl(output_paths["chunks"], all_chunks)
    write_jsonl(output_paths["tables"], all_tables)

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "parser_name": PARSER_NAME,
        "parser_version": version("sec2md"),
        "tokenizer_name": args.tokenizer,
        "max_embedding_tokens": MAX_EMBEDDING_TOKENS,
        "observed_max_embedding_tokens": max(
            token_count(str(chunk["embedding_text"])) + 1 for chunk in all_chunks
        ),
        "description_mode": args.description_mode,
        "description_model": description_model,
        "filing_count": len(filings),
        "tickers": [str(filing.get("ticker", "")) for filing in filings],
        "chunk_count": len(all_chunks),
        "text_chunk_count": sum(chunk["record_type"] == "text" for chunk in all_chunks),
        "table_chunk_count": sum(chunk["record_type"] == "table" for chunk in all_chunks),
        "table_count": len(all_tables),
        "retrieval_eligible_table_count": sum(
            bool(table["retrieval_eligible"]) for table in all_tables
        ),
        "filings": filing_summaries,
        "outputs": {
            name: {
                "path": manifest_path(path),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
            if name != "manifest"
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Total: chunks={len(all_chunks)}, tables={len(all_tables)}")
    print(f"Manifest: {output_paths['manifest']}")


if __name__ == "__main__":
    main()
