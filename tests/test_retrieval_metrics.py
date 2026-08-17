import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bankscope.evaluation.retrieval_metrics import evaluate_evidence_groups, evaluate_ranking
from scripts import evaluate as evaluate_script
from scripts import search as search_script

validate_qrels = evaluate_script.validate_qrels


def test_evaluate_ranking_deduplicates_targets() -> None:
    metrics = evaluate_ranking(["miss", "hit", "hit"], ["hit", "other"])

    assert metrics["first_relevant_rank"] == 2
    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_3"] == 1
    assert metrics["recall_at_3"] == 0.5
    assert metrics["reciprocal_rank_at_10"] == 0.5


def test_qrel_validation_skips_non_answerable_queries_without_loading_torch() -> None:
    records = [{"target_chunk_id": "known"}]
    queries = [
        {
            "query_id": "answerable",
            "query": "What is known?",
            "status": "answerable",
            "relevant_target_chunk_ids": ["known"],
        },
        {
            "query_id": "ambiguous",
            "query": "Which bank?",
            "status": "ambiguous",
            "relevant_target_chunk_ids": [],
        },
        {
            "query_id": "unsupported",
            "query": "What happens next year?",
            "status": "unsupported",
            "relevant_target_chunk_ids": [],
        },
    ]

    assert [query["query_id"] for query in validate_qrels(queries, records)] == ["answerable"]
    assert "torch" not in sys.modules


def test_qrel_validation_rejects_unknown_target_ids() -> None:
    query = {
        "query_id": "bad",
        "query": "Unknown evidence",
        "status": "answerable",
        "relevant_target_chunk_ids": ["missing"],
    }

    with pytest.raises(ValueError, match="unknown target IDs"):
        validate_qrels([query], [{"target_chunk_id": "known"}])


@pytest.mark.parametrize("tickers", [None, ["BAC"], ["BAC", "bac"]])
def test_cross_bank_qrels_require_two_to_four_unique_tickers(tickers: object) -> None:
    query = {
        "query_id": "cross",
        "query": "Compare banks",
        "ticker": None,
        "question_type": "cross_bank_coverage",
        "status": "answerable",
        "relevant_target_chunk_ids": ["known"],
    }
    if tickers is not None:
        query["tickers"] = tickers

    with pytest.raises(ValueError, match="tickers|two to four|duplicate"):
        validate_qrels([query], [{"target_chunk_id": "known"}])


def test_evidence_group_metrics_measure_complete_coverage() -> None:
    metrics = evaluate_evidence_groups(
        ["bank-a", "miss", "bank-b"], [["bank-a", "bank-a-alt"], ["bank-b"]]
    )

    assert metrics["group_recall_at_1"] == 0.5
    assert metrics["complete_group_hit_at_1"] == 0
    assert metrics["group_recall_at_3"] == 1.0
    assert metrics["complete_group_hit_at_3"] == 1


def test_summary_reports_group_metrics_only_for_grouped_queries() -> None:
    metrics = evaluate_ranking(["bank-a", "bank-b"], ["bank-a", "bank-b"])
    metrics.update(evaluate_evidence_groups(["bank-a", "bank-b"], [["bank-a"], ["bank-b"]]))

    summary = evaluate_script.summarize([{"method": "bm25", "metrics": metrics}])["bm25"]

    assert summary["grouped_query_count"] == 1
    assert summary["mean_group_recall_at_1"] == 0.5
    assert summary["complete_group_hit_rate_at_3"] == 1.0


@pytest.mark.parametrize(
    "groups",
    ["not-a-list", ["not-an-object"], [{}], [{"target_chunk_ids": []}]],
)
def test_qrel_validation_rejects_malformed_evidence_groups(groups: object) -> None:
    query = {
        "query_id": "grouped",
        "query": "Compare banks",
        "status": "answerable",
        "relevant_target_chunk_ids": ["known"],
        "required_evidence_groups": groups,
    }

    with pytest.raises(ValueError, match="evidence.group|Evidence group"):
        validate_qrels([query], [{"target_chunk_id": "known"}])


def test_search_bm25_skips_embedding_archive_and_emits_clean_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chunks = Path("chunks.jsonl")
    tables = Path("tables.jsonl")
    args = Namespace(
        query="operational risk",
        mode="bm25",
        limit=1,
        candidate_k=-1,
        ticker=None,
        record_type=None,
        rrf_k=-1,
        chunks=chunks,
        tables=tables,
        embeddings=Path("missing.npz"),
    )
    records = [
        {
            "record_id": "risk",
            "target_chunk_id": "risk",
            "record_type": "text",
            "embedding_text": "operational risk",
            "document": "Operational risk evidence.",
            "metadata": {"ticker": "JPM"},
        }
    ]
    monkeypatch.setattr(search_script, "parse_args", lambda: args)
    monkeypatch.setattr(search_script, "read_jsonl", lambda path: records if path == chunks else [])
    monkeypatch.setattr(
        search_script,
        "load_embedding_archive",
        lambda *args, **kwargs: pytest.fail("BM25 loaded the embedding archive"),
    )

    search_script.main()

    output = capsys.readouterr().out
    assert output.lstrip().startswith("{")
    assert '"target_chunk_id": "risk"' in output


def test_evaluator_bm25_skips_embedding_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = Path("chunks.jsonl")
    tables = Path("tables.jsonl")
    qrels = Path("queries.jsonl")
    locators = Path("locators.jsonl")
    reference = SimpleNamespace(read_text=lambda **_: '{"per_query": []}')
    captured: dict[str, object] = {}
    args = Namespace(
        qrels=qrels,
        chunks=chunks,
        tables=tables,
        glossary_locators=locators,
        embeddings=Path("missing.npz"),
        output=Path("ignored.json"),
        reference_results=reference,
        backend="mixed",
        methods=["bm25"],
        candidate_k=-1,
        rrf_k=-1,
    )
    records = [
        {
            "record_id": "risk",
            "target_chunk_id": "risk",
            "record_type": "text",
            "embedding_text": "operational risk",
            "document": "Operational risk evidence.",
            "metadata": {"ticker": "JPM"},
        }
    ]
    queries = [
        {
            "query_id": "q1",
            "query": "operational risk",
            "status": "answerable",
            "relevant_target_chunk_ids": ["risk"],
        }
    ]

    monkeypatch.setattr(evaluate_script, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluate_script,
        "read_jsonl",
        lambda path: records if path == chunks else queries if path == qrels else [],
    )
    monkeypatch.setattr(evaluate_script, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        evaluate_script,
        "load_embedding_archive",
        lambda *args, **kwargs: pytest.fail("BM25 loaded the embedding archive"),
    )
    monkeypatch.setattr(evaluate_script, "write_json", lambda path, value: captured.update(value))

    evaluate_script.main()

    corpus = captured["corpus"]
    assert isinstance(corpus, dict)
    assert corpus["embedding_model"] is None
    assert corpus["record_order_validated"] is False


def test_glossary_locator_gate_requires_hits_and_no_regressions() -> None:
    query_ids = ["dev_bac_bana_expansion_2025", "dev_pnc_gsib_expansion_2025"]
    replacement_ids = [
        "dev_cof_cybersecurity_technology_risk_management_2025",
        "dev_stt_information_technology_risk_definition_2025",
        "dev_cof_standardized_cet1_ratio_2025",
        "dev_stt_standardized_cet1_ratio_2025",
    ]
    candidate_rows = [
        {
            "query_id": query_id,
            "method_key": "mixed.hybrid",
            "metrics": {"first_relevant_rank": 1, "hit_at_5": 1, "hit_at_10": 1},
        }
        for query_id in query_ids
    ]
    candidate_rows.extend(
        {
            "query_id": query_id,
            "method_key": "mixed.hybrid",
            "metrics": {
                "first_relevant_rank": 2,
                "hit_at_5": 1,
                "hit_at_10": 1,
                **(
                    {
                        "required_evidence_group_count": 2,
                        "group_recall_at_10": 1.0,
                        "complete_group_hit_at_10": 1,
                    }
                    if query_id == "dev_stt_information_technology_risk_definition_2025"
                    else {}
                ),
            },
        }
        for query_id in replacement_ids
    )
    candidate_rows.extend(
        {
            "query_id": query_id,
            "method_key": "mixed.hybrid",
            "metrics": {
                "first_relevant_rank": 1,
                "hit_at_5": 1,
                "hit_at_10": 1,
                "required_evidence_group_count": 2,
                "group_recall_at_10": 1.0,
                "complete_group_hit_at_10": 1,
            },
        }
        for query_id in (
            "dev_cross_bac_c_standardized_cet1_2025",
            "dev_cross_c_jpm_operational_risk_definitions_2025",
            "dev_cross_pnc_tfc_cet1_2025",
        )
    )
    reference = {
        "per_query": [
            {
                "query_id": query_id,
                "method_key": "mixed.hybrid",
                "metrics": {"hit_at_5": 0, "hit_at_10": 0},
            }
            for query_id in query_ids
        ]
    }
    summary = {"mixed.hybrid": {"hit_count_at_5": 31, "hit_count_at_10": 32}}

    gate = evaluate_script.assess_glossary_locator_gate(candidate_rows, summary, reference)

    assert gate is not None
    assert gate["passed"] is True

    failing_rows = [dict(row) for row in candidate_rows]
    failing = next(
        row for row in failing_rows if row["query_id"] == "dev_cross_bac_c_standardized_cet1_2025"
    )
    failing["metrics"] = {
        **failing["metrics"],
        "group_recall_at_10": 0.5,
        "complete_group_hit_at_10": 0,
    }
    failed_gate = evaluate_script.assess_glossary_locator_gate(failing_rows, summary, reference)
    assert failed_gate is not None
    assert failed_gate["passed"] is False
    assert failed_gate["checks"]["cross_bank_evidence_groups_complete"] is False


def test_evaluator_rejects_duplicate_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_script, "parse_args", lambda: Namespace(methods=["bm25", "bm25"]))

    with pytest.raises(ValueError, match="duplicates"):
        evaluate_script.main()


def test_query_encoder_uses_known_revision_and_omits_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_options: list[dict[str, str]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **options: str) -> None:
            assert model_name == "model/name"
            model_options.append(options)

        def encode_query(self, texts: list[str], **options: object) -> np.ndarray:
            assert texts == ["query"]
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    search_script.encode_query("query", "model/name", "abc123")
    search_script.encode_query("query", "model/name", "unknown")

    assert model_options == [{"revision": "abc123"}, {}]
