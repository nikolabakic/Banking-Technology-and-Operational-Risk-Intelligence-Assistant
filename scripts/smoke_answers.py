"""Run one grounded application-pipeline answer for every active BankScope bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankscope.config.settings import get_settings  # noqa: E402
from bankscope.generation import GPT51_CANDIDATE_MODEL, BankAnswerPipeline  # noqa: E402
from bankscope.generation.pipeline import (  # noqa: E402
    DEFAULT_CHUNKS,
    DEFAULT_GLOSSARY_LOCATORS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.io import read_jsonl, sha256_file  # noqa: E402
from bankscope.llm import create_openai_client  # noqa: E402
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME  # noqa: E402

DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/application-smoke-10.json"
SMOKE_QUERY_IDS = (
    "dev_ally_operational_risk_definition_2025",
    "dev_bac_cyber_incident_impacts_2025",
    "dev_c_operational_risk_definition_2025",
    "dev_cof_cybersecurity_technology_risk_management_2025",
    "dev_gs_cybersecurity_risk_definition_2025",
    "dev_jpm_cybersecurity_risk_definition_2025",
    "dev_lob_bank_cet1_ratio_2024_split_table",
    "dev_pnc_operational_risk_definition_2025",
    "dev_stt_information_technology_risk_definition_2025",
    "dev_tfc_cyber_incident_response_team_2025",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=GPT51_CANDIDATE_MODEL)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def select_smoke_queries(queries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(query.get("query_id") or ""): dict(query) for query in queries}
    if missing := set(SMOKE_QUERY_IDS) - by_id.keys():
        raise ValueError(f"Smoke qrels are missing query IDs: {sorted(missing)}.")
    selected = [by_id[query_id] for query_id in SMOKE_QUERY_IDS]
    tickers = [str(query.get("ticker") or "").upper() for query in selected]
    if any(query.get("status") != "answerable" for query in selected):
        raise ValueError("Every application-smoke query must be answerable.")
    if any(not ticker for ticker in tickers) or len(tickers) != len(set(tickers)):
        raise ValueError("Application-smoke queries must cover ten unique explicit tickers.")
    return selected


def assess_smoke_answer(query: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    expected_ticker = str(query["ticker"]).upper()
    citations = output.get("citations")
    citations = (
        citations if isinstance(citations, Sequence) and not isinstance(citations, str) else []
    )
    citation_tickers = [
        str(citation.get("ticker") or "").upper()
        for citation in citations
        if isinstance(citation, Mapping)
    ]
    checks = {
        "ticker": output.get("ticker") == expected_ticker,
        "status": output.get("status") == "supported",
        "answer": bool(str(output.get("answer") or "").strip()),
        "citations": bool(citations),
        "citation_ownership": bool(citation_tickers)
        and all(ticker == expected_ticker for ticker in citation_tickers),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    queries = select_smoke_queries(read_jsonl(args.qrels))
    settings = get_settings()
    client = create_openai_client(settings)
    records: list[dict[str, Any]] = []
    with BankAnswerPipeline.from_paths(
        client=client,
        generation_model=args.model,
        temperature=settings.llm_temperature,
        chunks_path=args.chunks,
        tables_path=args.tables,
        glossary_locators_path=args.glossary_locators,
        qdrant_path=args.qdrant_path,
        qdrant_manifest_path=args.qdrant_manifest,
        collection_name=args.collection,
        bank_registry_path=settings.bank_registry_path,
    ) as pipeline:
        for query in queries:
            run = pipeline.answer(str(query["query"]), ticker=str(query["ticker"]))
            assessment = assess_smoke_answer(query, run.output)
            records.append(
                {
                    "query_id": query["query_id"],
                    "ticker": query["ticker"],
                    "assessment": assessment,
                    "output": run.output,
                }
            )

    passed = sum(record["assessment"]["passed"] for record in records)
    report = {
        "evaluation": "application-smoke-10",
        "created_at": datetime.now(UTC).isoformat(),
        "generation_model": args.model,
        "sources": {
            "qrels_sha256": sha256_file(args.qrels),
            "qdrant_manifest_sha256": sha256_file(args.qdrant_manifest),
        },
        "summary": {"queries": len(records), "passed": passed, "overall_pass": passed == 10},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["overall_pass"]:
        raise RuntimeError("Application smoke did not pass for every bank.")


if __name__ == "__main__":
    main()
