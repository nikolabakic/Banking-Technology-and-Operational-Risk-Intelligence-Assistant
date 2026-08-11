"""Build versioned lexical-only glossary locators from the active table store."""

from __future__ import annotations

import argparse
from pathlib import Path

from bankscope.io import read_jsonl, write_jsonl
from bankscope.retrieval.glossary_locators import (
    GLOSSARY_LOCATOR_VERSION,
    build_glossary_locators,
    validate_glossary_locators,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/lexical_glossary_locators_v1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite.")

    records = read_jsonl(args.chunks)
    tables = read_jsonl(args.tables)
    locators = build_glossary_locators(records, tables)
    validate_glossary_locators(locators, records, tables)
    write_jsonl(args.output, locators)
    print(
        f"Glossary locators: version={GLOSSARY_LOCATOR_VERSION}, "
        f"count={len(locators)}, output={args.output}"
    )


if __name__ == "__main__":
    main()
