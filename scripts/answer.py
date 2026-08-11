"""Answer one single-bank question from hydrated mixed-retrieval evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bankscope.config.settings import get_settings
from bankscope.generation import SingleBankAnswerPipeline
from bankscope.generation.pipeline import (
    DEFAULT_CHUNKS,
    DEFAULT_GLOSSARY_LOCATORS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.llm import create_openai_client
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Single-bank question to answer.")
    parser.add_argument("--ticker", required=True, help="Required bank ticker, for example JPM.")
    parser.add_argument("--record-type", choices=("text", "table"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", help="Generation model override (defaults to OPENAI_MODEL).")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive.")
    if args.candidate_k < args.limit:
        raise ValueError("candidate-k must be at least limit.")
    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive.")

    settings = get_settings()
    generation_model = args.model or settings.openai_model
    llm_client = create_openai_client(settings)
    with SingleBankAnswerPipeline.from_paths(
        client=llm_client,
        generation_model=generation_model,
        temperature=settings.llm_temperature,
        chunks_path=args.chunks,
        tables_path=args.tables,
        glossary_locators_path=args.glossary_locators,
        qdrant_path=args.qdrant_path,
        qdrant_manifest_path=args.qdrant_manifest,
        collection_name=args.collection,
        bank_registry_path=settings.bank_registry_path,
    ) as pipeline:
        run = pipeline.answer(
            args.question,
            ticker=args.ticker,
            record_type=args.record_type,
            limit=args.limit,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
    print(json.dumps(run.output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
