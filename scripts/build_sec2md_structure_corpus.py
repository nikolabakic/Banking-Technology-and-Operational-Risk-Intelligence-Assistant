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
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from bankscope.parsing.sec2md_adapter import (
    PARSER_NAME,
    PARSER_VERSION,
    STRUCTURE_CHUNKER_VERSION,
    build_structure_aware_records,
    chunk_config_hash,
    eligible_records,
    validate_records,
)
from bankscope.sec.company_registry import load_bank_registry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "config/banks.yaml"
DEFAULT_MANIFEST_PATH = ROOT / "artifacts/manifests/filings.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/sec2md_structure_v3"
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"

Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 10-bank sec2md structure-aware corpus.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def resolve_raw_path(manifest_path: Path, filing: Record) -> Path:
    raw_path = Path(str(filing["local_html_path"]))

    if raw_path.is_absolute():
        return raw_path

    candidates = [ROOT / raw_path, manifest_path.parent.parent.parent / raw_path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def make_token_counter(tokenizer: PreTrainedTokenizerBase):
    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return token_count


def write_jsonl(path: Path, records: list[Record]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_corpus(records: list[Record], parents: list[Record], token_count) -> None:
    validate_records(records, token_count=token_count)
    parent_ids = [str(parent["parent_id"]) for parent in parents]

    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Table parent IDs are not unique.")

    parent_id_set = set(parent_ids)

    for record in records:
        metadata = record["metadata"]

        if record["record_type"] == "table_parent" or not metadata["retrieval_eligible"]:
            raise ValueError(f"Non-retrieval record in embedding corpus: {record['record_id']}.")

        if record["record_type"] != "table_locator":
            continue

        if metadata.get("parent_id") not in parent_id_set:
            raise ValueError(f"Missing table parent: {record['record_id']}.")

        if metadata.get("table_type") in {"layout", "index"}:
            raise ValueError(f"Non-retrieval table locator: {record['record_id']}.")

        if len(metadata.get("column_paths", [])) != len(
            metadata.get("cell_coordinates", [])
        ):
            raise ValueError(f"Column/coordinate mismatch: {record['record_id']}.")


def main() -> None:
    args = parse_args()
    output_paths = {
        "embedding_records": args.output_dir / "embedding_records.jsonl",
        "table_parents": args.output_dir / "table_parents.jsonl",
        "build_manifest": args.output_dir / "build_manifest.json",
    }

    existing = [path for path in output_paths.values() if path.exists()]

    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}. Use --overwrite.")

    installed_sec2md_version = version("sec2md")

    if installed_sec2md_version != PARSER_VERSION:
        raise ValueError(
            f"This build is pinned to sec2md=={PARSER_VERSION}; "
            f"installed={installed_sec2md_version}."
        )

    registry = load_bank_registry(args.registry)
    tickers = [bank.ticker for bank in registry.banks if bank.enabled]
    filings = json.loads(args.manifest.read_text(encoding="utf-8"))
    filings_by_ticker = {str(filing["ticker"]).upper(): filing for filing in filings}
    missing_tickers = [ticker for ticker in tickers if ticker not in filings_by_ticker]

    if missing_tickers:
        raise ValueError(f"Enabled banks missing from filing manifest: {missing_tickers}.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_count = make_token_counter(tokenizer)
    all_records: list[Record] = []
    all_parents: list[Record] = []
    bank_summaries: list[Record] = []

    for ticker in tickers:
        filing = filings_by_ticker[ticker]
        raw_path = resolve_raw_path(args.manifest, filing)

        if not raw_path.exists():
            raise FileNotFoundError(f"{ticker} raw filing not found: {raw_path}.")

        raw_bytes = raw_path.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            parser = Parser(raw_bytes.decode("utf-8", errors="replace"))
            pages = parser.get_pages(include_elements=True, include_images=False)
            annotated_html = parser.html()

        records, parents = build_structure_aware_records(
            pages,
            filing,
            raw_sha256=sha256_bytes(raw_bytes),
            token_count=token_count,
            annotated_html=annotated_html,
        )
        embeddings = eligible_records(records)
        validate_corpus(embeddings, parents, token_count)
        all_records.extend(embeddings)
        all_parents.extend(parents)
        bank_summaries.append(
            {
                "ticker": ticker,
                "raw_path": raw_path.as_posix(),
                "raw_sha256": sha256_bytes(raw_bytes),
                "page_count": len(pages),
                "embedding_record_count": len(embeddings),
                "text_record_count": sum(r["record_type"] != "table_locator" for r in embeddings),
                "table_locator_count": sum(r["record_type"] == "table_locator" for r in embeddings),
                "table_parent_count": len(parents),
            }
        )
        print(f"{ticker}: embeddings={len(embeddings)}, parents={len(parents)}")

    validate_corpus(all_records, all_parents, token_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_paths["embedding_records"], all_records)
    write_jsonl(output_paths["table_parents"], all_parents)

    build_manifest = {
        "parser_name": PARSER_NAME,
        "parser_version": installed_sec2md_version,
        "chunker_version": STRUCTURE_CHUNKER_VERSION,
        "chunk_config_hash": chunk_config_hash("structure_aware"),
        "tokenizer_name": args.tokenizer,
        "ticker_count": len(tickers),
        "tickers": tickers,
        "embedding_record_count": len(all_records),
        "table_parent_count": len(all_parents),
        "banks": bank_summaries,
        "outputs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
            if name != "build_manifest"
        },
    }
    output_paths["build_manifest"].write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Total: embeddings={len(all_records)}, parents={len(all_parents)}")
    print(f"Build manifest: {output_paths['build_manifest']}")


if __name__ == "__main__":
    main()
