from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sec2md import Chunker, Parser
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from bankscope.parsing.sec2md_adapter import (
    PARSER_VERSION,
    TABLE_TEXT_LOCATOR_MAX_TOKENS,
    TEXT_OVERLAP_TOKENS,
    TEXT_TARGET_TOKENS,
    adapt_builtin_chunks,
    build_structure_aware_records,
    chunk_config_hash,
    eligible_records,
    retrieval_token_limit,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "artifacts/manifests/filings.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/jpm_sec2md"
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"

Record = dict[str, Any]

QUERY_SPECS: tuple[Record, ...] = (
    {
        "query_id": "dev_jpm_cybersecurity_risk_definition_2025",
        "query": "How does JPMorgan Chase define cybersecurity risk in its 2025 Form 10-K?",
        "question_type": "narrative_risk",
        "status": "answerable",
        "gold_answer": (
            "Cybersecurity risk is the risk of harm or loss resulting from misuse or abuse "
            "of technology or the unauthorized disclosure of data."
        ),
        "anchors": (
            "Cybersecurity risk is the risk of harm or loss resulting from misuse or abuse "
            "of technology or the unauthorized disclosure of data",
        ),
    },
    {
        "query_id": "dev_jpm_standardized_cet1_ratio_2025",
        "query": (
            "What was JPMorgan Chase & Co.'s Standardized CET1 capital ratio on December 31, 2025?"
        ),
        "question_type": "table_exact_value",
        "status": "answerable",
        "gold_answer": "14.6%",
        "expected_value": 14.6,
        "expected_unit": "percent",
        "expected_period": "2025-12-31",
        "expected_entity": "JPMorgan Chase & Co.",
        "expected_variant": "Standardized",
        "anchors": (
            "December 31, 2025",
            "Standardized",
            "CET1",
            "14.6",
        ),
    },
    {
        "query_id": "dev_jpm_standardized_cet1_requirement_2025",
        "query": (
            "What was JPMorgan Chase & Co.'s Standardized CET1 capital ratio requirement, "
            "including regulatory buffers, on December 31, 2025?"
        ),
        "question_type": "entity_variant",
        "status": "answerable",
        "gold_answer": "11.5%",
        "expected_value": 11.5,
        "expected_unit": "percent",
        "expected_period": "2025-12-31",
        "expected_entity": "JPMorgan Chase & Co.",
        "expected_variant": "Standardized requirement including regulatory buffers",
        "anchors": (
            "Standardized CET1 capital ratio requirement, including regulatory buffers",
            "11.5%",
            "December 31, 2025",
        ),
    },
    {
        "query_id": "dev_jpm_bank_advanced_cet1_ratio_2025_metadata",
        "query": (
            "What was JPMorgan Chase Bank, N.A.'s Advanced CET1 capital ratio on December 31, 2025?"
        ),
        "question_type": "parser_metadata",
        "status": "answerable",
        "gold_answer": "15.8%",
        "expected_value": 15.8,
        "expected_unit": "percent",
        "expected_period": "2025-12-31",
        "expected_entity": "JPMorgan Chase Bank, N.A.",
        "expected_variant": "Advanced",
        "anchors": (
            "December 31, 2025",
            "Advanced",
            "JPMorgan Chase Bank, N.A.",
            "CET1",
            "15.8",
        ),
    },
    {
        "query_id": "dev_unsupported_jpm_standardized_cet1_2026",
        "query": (
            "What was JPMorgan Chase & Co.'s Standardized CET1 capital ratio on December 31, 2026?"
        ),
        "question_type": "unsupported_period",
        "status": "unsupported",
        "gold_answer": None,
        "expected_period": "2026-12-31",
        "anchors": (),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build both JPM sec2md chunk variants and their variant-specific qrels."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--raw-html",
        type=Path,
        help="Optional direct path to the JPM 10-K HTML; metadata still comes from the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_jpm_manifest(path: Path) -> Record:
    records = json.loads(path.read_text(encoding="utf-8"))

    for record in records:
        if str(record.get("ticker", "")).upper() == "JPM":
            return record

    raise ValueError(f"JPM is not present in the filing manifest: {path}.")


def resolve_raw_path(manifest_path: Path, filing: Record) -> Path:
    raw_path = Path(str(filing["local_html_path"]))

    if raw_path.is_absolute():
        return raw_path

    candidates = [ROOT / raw_path, manifest_path.parent.parent.parent / raw_path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"Unexpected tokenizer type: {type(tokenizer)!r}.")

    return tokenizer


def make_token_counter(tokenizer: PreTrainedTokenizerBase):
    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return token_count


def write_jsonl(path: Path, records: list[Record], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def record_matches(record: Record, anchors: tuple[str, ...]) -> bool:
    document = " ".join(str(record["document"]).split()).casefold()
    return all(" ".join(anchor.split()).casefold() in document for anchor in anchors)


def build_queries(records: list[Record], variant: str) -> tuple[list[Record], list[Record]]:
    queries: list[Record] = []
    audit_rows: list[Record] = []

    for spec in QUERY_SPECS:
        anchors = tuple(str(anchor) for anchor in spec["anchors"])
        relevant_records = (
            []
            if spec["status"] == "unsupported"
            else [record for record in records if record_matches(record, anchors)]
        )

        if spec["status"] == "answerable" and not relevant_records:
            raise ValueError(
                f"No direct evidence matched {spec['query_id']} for variant {variant}."
            )

        relevant_records.sort(
            key=lambda record: (
                len(str(record["document"])),
                str(record["target_chunk_id"]),
            )
        )
        relevant_ids = [str(record["target_chunk_id"]) for record in relevant_records]
        primary_target_id = relevant_ids[0] if relevant_ids else None
        query = {key: value for key, value in spec.items() if key not in {"anchors"}}
        query.update(
            {
                "ticker": "JPM",
                "entity": spec.get("expected_entity", "JPMorgan Chase & Co."),
                "report_date": "2025-12-31",
                "source_variant": variant,
                "relevant_target_chunk_ids": relevant_ids,
                "primary_target_chunk_id": primary_target_id,
                "annotation_notes": (
                    "Variant-specific qrels generated only from strict, auditable evidence anchors."
                ),
            }
        )
        queries.append(query)
        audit_rows.append(
            {
                "query_id": spec["query_id"],
                "variant": variant,
                "status": spec["status"],
                "anchors": list(anchors),
                "matched_count": len(relevant_records),
                "primary_target_chunk_id": primary_target_id,
                "matches": [
                    {
                        "target_chunk_id": record["target_chunk_id"],
                        "record_type": record["record_type"],
                        "page_start": record["metadata"]["page_start"],
                        "page_end": record["metadata"]["page_end"],
                        "preview": " ".join(str(record["document"]).split())[:900],
                    }
                    for record in relevant_records
                ],
            }
        )

    return queries, audit_rows


def summarize(records: list[Record], token_count) -> Record:
    eligible = eligible_records(records)
    token_counts = [token_count(str(record["document"])) for record in eligible]
    return {
        "all_records": len(records),
        "eligible_records": len(eligible),
        "ineligible_records": len(records) - len(eligible),
        "record_types": {
            record_type: sum(record["record_type"] == record_type for record in eligible)
            for record_type in sorted({str(record["record_type"]) for record in eligible})
        },
        "maximum_qwen_tokens": max(token_counts, default=0),
        "over_project_limit": sum(
            count > retrieval_token_limit(record)
            for record, count in zip(eligible, token_counts, strict=True)
        ),
        "missing_page_provenance": sum(
            not record["metadata"].get("page_start") or not record["metadata"].get("page_end")
            for record in eligible
        ),
    }


def main() -> None:
    args = parse_args()
    installed_sec2md_version = version("sec2md")

    if installed_sec2md_version != PARSER_VERSION:
        raise ValueError(
            f"This experiment is pinned to sec2md=={PARSER_VERSION}; "
            f"installed={installed_sec2md_version}."
        )

    filing = load_jpm_manifest(args.manifest)
    raw_path = (
        args.raw_html.expanduser().resolve()
        if args.raw_html is not None
        else resolve_raw_path(args.manifest, filing)
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"JPM raw filing not found: {raw_path}.")

    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_html = raw_bytes.decode("utf-8", errors="replace")
    tokenizer = load_tokenizer(args.tokenizer)
    token_count = make_token_counter(tokenizer)

    parser = Parser(raw_html)
    pages = parser.get_pages(include_elements=True, include_images=False)
    annotated_html = parser.html()
    markdown = "\n\n".join(page.content for page in pages if page.content)
    header = f"Bank: JPM\nReport: {filing['report_date'][:4]} 10-K"
    builtin_chunks = Chunker(
        chunk_size=TEXT_TARGET_TOKENS,
        chunk_overlap=TEXT_OVERLAP_TOKENS,
        max_table_tokens=TABLE_TEXT_LOCATOR_MAX_TOKENS,
    ).split(pages, header=header)

    builtin_all = adapt_builtin_chunks(
        builtin_chunks,
        filing,
        raw_sha256=raw_sha256,
    )
    structure_all, table_parents = build_structure_aware_records(
        pages, filing, raw_sha256=raw_sha256, token_count=token_count, annotated_html=annotated_html
    )

    # validate_records(builtin_all, token_count=token_count)
    validate_records(structure_all, token_count=token_count)

    variants = {
        "sec2md_builtin": builtin_all,
        "structure_aware": structure_all,
    }
    audit_rows: list[Record] = []
    output_hashes: dict[str, str] = {}

    markdown_path = args.output_dir / "jpm_sec2md.md"
    if markdown_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {markdown_path}. Use --overwrite.")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    output_hashes[markdown_path.relative_to(args.output_dir).as_posix()] = sha256_file(
        markdown_path
    )

    for variant, all_records in variants.items():
        variant_dir = args.output_dir / variant
        embeddings = eligible_records(all_records)
        queries, variant_audit = build_queries(embeddings, variant)
        audit_rows.extend(variant_audit)

        paths_and_values = (
            (variant_dir / "chunks.jsonl", all_records),
            (variant_dir / "embedding_records.jsonl", embeddings),
            (variant_dir / "queries.jsonl", queries),
        )

        for path, values in paths_and_values:
            write_jsonl(path, values, overwrite=args.overwrite)
            output_hashes[path.relative_to(args.output_dir).as_posix()] = sha256_file(path)

    parents_path = args.output_dir / "structure_aware/table_parents.jsonl"
    write_jsonl(parents_path, table_parents, overwrite=args.overwrite)
    output_hashes[parents_path.relative_to(args.output_dir).as_posix()] = sha256_file(parents_path)

    audit_path = args.output_dir / "qrels_audit.json"
    write_json(audit_path, audit_rows, overwrite=args.overwrite)
    output_hashes[audit_path.relative_to(args.output_dir).as_posix()] = sha256_file(audit_path)

    experiment_manifest = {
        "experiment": "jpm_sec2md_chunk_bakeoff_v1",
        "ticker": "JPM",
        "cik": filing["cik"],
        "accession_number": filing["accession_number"],
        "filing_date": filing["filing_date"],
        "report_date": filing["report_date"],
        "source_url": filing["source_url"],
        "raw_path": raw_path.as_posix(),
        "raw_sha256": raw_sha256,
        "parser_name": "sec2md",
        "parser_version": installed_sec2md_version,
        "tokenizer_name": args.tokenizer,
        "page_count": len(pages),
        "table_parent_count": len(table_parents),
        "chunk_config_hashes": {variant: chunk_config_hash(variant) for variant in variants},
        "variants": {
            variant: summarize(records, token_count) for variant, records in variants.items()
        },
        "output_sha256": output_hashes,
    }
    manifest_path = args.output_dir / "experiment_manifest.json"
    write_json(manifest_path, experiment_manifest, overwrite=args.overwrite)

    print(f"Raw JPM filing: {raw_path}")
    print(f"sec2md pages: {len(pages)}")
    print(f"Built-in eligible records: {len(eligible_records(builtin_all))}")
    print(f"Structure-aware eligible records: {len(eligible_records(structure_all))}")
    print(f"Table parents: {len(table_parents)}")
    print(f"Experiment manifest: {manifest_path}")


if __name__ == "__main__":
    main()
