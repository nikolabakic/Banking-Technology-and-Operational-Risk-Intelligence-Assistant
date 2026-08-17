from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from bankscope.generation.answer_generator import CITATION_PATTERN, generate_answer
from bankscope.generation.comparison_generator import synthesize_comparison
from bankscope.generation.contextualizer import contextualize_question
from bankscope.io import read_jsonl, sha256_file
from bankscope.retrieval.glossary_locators import validate_glossary_locators
from bankscope.retrieval.hybrid_retriever import HybridRetriever
from bankscope.retrieval.mixed_retriever import MixedRetriever
from bankscope.retrieval.qdrant_retriever import (
    DEFAULT_COLLECTION_NAME,
    QdrantRetriever,
    load_qdrant_manifest,
)
from bankscope.sec.bank_resolver import BankResolution, resolve_bank
from bankscope.sec.company_registry import load_bank_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = PROJECT_ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = PROJECT_ROOT / "data/processed/tables.jsonl"
DEFAULT_GLOSSARY_LOCATORS = PROJECT_ROOT / "data/processed/lexical_glossary_locators_v1.jsonl"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data/processed/qdrant"
DEFAULT_QDRANT_MANIFEST = PROJECT_ROOT / "data/processed/qdrant_manifest.json"
DEFAULT_BANK_REGISTRY = PROJECT_ROOT / "config/banks.yaml"


class QueryEncoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class SentenceTransformerQueryEncoder:
    """Load one SentenceTransformer and reuse it for every question in a run."""

    def __init__(self, model_name: str, model_revision: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_options: dict[str, Any] = {"local_files_only": True}
        if model_revision.strip().lower() != "unknown":
            model_options["revision"] = model_revision
        self.model = SentenceTransformer(model_name, **model_options)

    def encode(self, text: str) -> np.ndarray:
        vector = self.model.encode_query(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32)


@dataclass(frozen=True)
class AnswerRun:
    output: dict[str, Any]
    evidence: list[dict[str, Any]]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float


class BankAnswerPipeline:
    """Reusable grounded-answer pipeline for single-bank and comparison questions."""

    def __init__(
        self,
        *,
        retriever: Any,
        query_encoder: QueryEncoder,
        client: Any,
        generation_model: str,
        temperature: float = 0,
        close_callback: Any | None = None,
        dense_model: dict[str, str] | None = None,
        bank_names: dict[str, str] | None = None,
        bank_aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        if not generation_model.strip():
            raise ValueError("generation_model cannot be empty.")
        self.retriever = retriever
        self.query_encoder = query_encoder
        self.client = client
        self.generation_model = generation_model
        self.temperature = temperature
        self._close_callback = close_callback
        self._closed = False
        self.dense_model = dict(dense_model or {})
        self.bank_names = {
            ticker.strip().upper(): name.strip()
            for ticker, name in (bank_names or {}).items()
            if ticker.strip() and name.strip()
        }
        self.bank_aliases = {
            ticker.strip().upper(): tuple(alias.strip() for alias in aliases if alias.strip())
            for ticker, aliases in (bank_aliases or {}).items()
            if ticker.strip()
        }

    @classmethod
    def from_paths(
        cls,
        *,
        client: Any,
        generation_model: str,
        temperature: float = 0,
        chunks_path: str | Path = DEFAULT_CHUNKS,
        tables_path: str | Path = DEFAULT_TABLES,
        glossary_locators_path: str | Path = DEFAULT_GLOSSARY_LOCATORS,
        qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
        qdrant_manifest_path: str | Path = DEFAULT_QDRANT_MANIFEST,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        query_encoder: QueryEncoder | None = None,
        bank_registry_path: str | Path = DEFAULT_BANK_REGISTRY,
    ) -> BankAnswerPipeline:
        chunks_path = Path(chunks_path)
        tables_path = Path(tables_path)
        glossary_locators_path = Path(glossary_locators_path)
        manifest_path = Path(qdrant_manifest_path)
        manifest = load_qdrant_manifest(manifest_path)
        dense_model = manifest.get("dense_model")
        if not isinstance(dense_model, dict):
            raise ValueError("Qdrant manifest has no valid dense_model.")
        model_name = str(dense_model.get("name") or "")
        model_revision = str(dense_model.get("revision") or "")
        if not model_name or not model_revision:
            raise ValueError("Qdrant manifest has incomplete dense model metadata.")

        expected_chunks_hash = str(
            manifest.get("sources", {}).get("chunks", {}).get("sha256") or ""
        )
        if expected_chunks_hash != sha256_file(chunks_path):
            raise ValueError("chunks.jsonl does not match the Qdrant manifest.")

        tables = read_jsonl(tables_path)
        records = read_jsonl(chunks_path)
        glossary_locators = read_jsonl(glossary_locators_path)
        validate_glossary_locators(glossary_locators, records, tables)
        registry = load_bank_registry(bank_registry_path)
        enabled_banks = [bank for bank in registry.banks if bank.enabled]
        bank_names = {bank.ticker: bank.legal_name for bank in enabled_banks}
        bank_aliases = {bank.ticker: bank.aliases for bank in enabled_banks}
        qdrant = QdrantRetriever(
            qdrant_path,
            tables,
            manifest_path=manifest_path,
            collection_name=collection_name,
            tables_path=tables_path,
        )
        try:
            # Open the exclusive local store before loading the comparatively slow
            # embedding model, so a second BankScope process fails immediately.
            encoder = query_encoder or SentenceTransformerQueryEncoder(model_name, model_revision)
            retriever = MixedRetriever(
                qdrant,
                HybridRetriever(records, tables=tables, lexical_records=glossary_locators),
            )
            return cls(
                retriever=retriever,
                query_encoder=encoder,
                client=client,
                generation_model=generation_model,
                temperature=temperature,
                close_callback=qdrant.close,
                dense_model={"name": model_name, "revision": model_revision},
                bank_names=bank_names,
                bank_aliases=bank_aliases,
            )
        except Exception:
            qdrant.close()
            raise

    def close(self) -> None:
        if not self._closed and self._close_callback is not None:
            self._close_callback()
        self._closed = True

    def __enter__(self) -> BankAnswerPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        tickers: Sequence[str] = (),
        conversation_history: Sequence[Mapping[str, str]] = (),
        record_type: str | None = None,
        limit: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> AnswerRun:
        if self._closed:
            raise RuntimeError("The answer pipeline is closed.")
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if candidate_k < limit:
            raise ValueError("candidate_k must be at least limit.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        standalone_question = question
        contextualization_model: str | None = None
        contextualization_latency_ms = 0.0
        if conversation_history:
            if on_progress is not None:
                on_progress(
                    "contextualizing",
                    {"message": "Resolving the follow-up question..."},
                )
            contextualization = contextualize_question(
                question,
                conversation_history,
                client=self.client,
                model=self.generation_model,
                session_ticker=ticker,
                session_tickers=tickers,
            )
            standalone_question = contextualization.standalone_question
            contextualization_model = contextualization.model
            contextualization_latency_ms = contextualization.latency_ms
        contextualization_payload = {
            "applied": bool(conversation_history),
            "history_turns": len(conversation_history) // 2,
            "standalone_question": standalone_question,
            "model": contextualization_model,
            "latency_ms": contextualization_latency_ms,
        }

        if on_progress is not None:
            on_progress("resolving_bank", {"message": "Identifying the bank..."})
        resolution = resolve_bank(
            standalone_question,
            bank_names=self.bank_names,
            bank_aliases=self.bank_aliases,
            session_ticker=ticker,
            session_tickers=tickers,
        )
        if resolution.status in {"missing", "too_many"}:
            return self._ambiguous_bank_run(question, resolution, contextualization_payload)
        if resolution.status == "multiple":
            return self._comparison_run(
                question,
                standalone_question,
                resolution,
                contextualization_payload,
                record_type=record_type,
                limit=limit,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                on_progress=on_progress,
            )
        resolved_ticker = str(resolution.ticker)

        if on_progress is not None:
            on_progress(
                "embedding",
                {"message": "Encoding the question...", "ticker": resolved_ticker},
            )
        started = perf_counter()
        query_vector = self.query_encoder.encode(standalone_question)
        embedding_latency_ms = (perf_counter() - started) * 1000

        if on_progress is not None:
            on_progress("retrieving", {"message": "Searching indexed filings..."})
        started = perf_counter()
        evidence = self.retriever.search_hybrid(
            standalone_question,
            query_vector,
            limit=limit,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            ticker=resolved_ticker,
            record_type=record_type,
        )
        retrieval_latency_ms = (perf_counter() - started) * 1000

        if on_progress is not None:
            on_progress(
                "generating",
                {"message": "Generating a grounded answer...", "evidence_count": len(evidence)},
            )
        started = perf_counter()
        answer = generate_answer(
            question,
            evidence,
            client=self.client,
            model=self.generation_model,
            expected_ticker=resolved_ticker,
            expected_bank_name=self.bank_names.get(resolved_ticker, resolved_ticker),
            expected_record_type=record_type,
            temperature=self.temperature,
            resolved_question=standalone_question,
        )
        generation_latency_ms = (perf_counter() - started) * 1000
        if on_progress is not None:
            on_progress("validating", {"message": "Validating the grounded answer..."})
        output = {
            "question": question,
            "ticker": resolved_ticker,
            "contextualization": contextualization_payload,
            "bank_resolution": resolution.as_dict(),
            "retrieval": {
                "backend": "mixed",
                "mode": "hybrid",
                "evidence_count": len(evidence),
            },
            **answer,
        }
        return AnswerRun(
            output=output,
            evidence=evidence,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )

    @staticmethod
    def _relabel_bank_result(
        result: Mapping[str, Any], next_index: int
    ) -> tuple[dict[str, Any], int]:
        output = dict(result)
        mapping: dict[str, str] = {}
        citations: list[dict[str, Any]] = []
        for raw in result.get("citations") or []:
            citation = dict(raw)
            old_label = str(citation.get("label") or "").strip().upper()
            new_label = f"E{next_index}"
            next_index += 1
            mapping[old_label] = new_label
            citation["label"] = new_label
            citations.append(citation)
        output["citations"] = citations
        answer = str(output.get("answer") or "")
        output["answer"] = CITATION_PATTERN.sub(
            lambda match: f"[{mapping.get(match.group(1), match.group(1))}]", answer
        )
        return output, next_index

    def _comparison_run(
        self,
        question: str,
        standalone_question: str,
        resolution: BankResolution,
        contextualization: Mapping[str, Any],
        *,
        record_type: str | None,
        limit: int,
        candidate_k: int,
        rrf_k: int,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None,
    ) -> AnswerRun:
        selected_tickers = resolution.tickers
        if on_progress is not None:
            on_progress(
                "embedding",
                {"message": "Encoding the comparison question...", "tickers": selected_tickers},
            )
        started = perf_counter()
        query_vector = self.query_encoder.encode(standalone_question)
        embedding_latency_ms = (perf_counter() - started) * 1000

        all_evidence: list[dict[str, Any]] = []
        bank_results: list[dict[str, Any]] = []
        per_bank_retrieval: list[dict[str, Any]] = []
        if on_progress is not None:
            on_progress(
                "retrieving",
                {
                    "message": "Searching each selected bank filing...",
                    "tickers": selected_tickers,
                },
            )
        bank_searches = self.retriever.search_hybrid_by_ticker(
            standalone_question,
            query_vector,
            tickers=selected_tickers,
            limit_per_ticker=limit,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            record_type=record_type,
        )
        search_by_ticker = {search.ticker: search for search in bank_searches}
        if set(search_by_ticker) != set(selected_tickers):
            raise RuntimeError("Multi-bank retrieval did not return every selected ticker.")
        retrieval_latency_ms = sum(search.latency_ms for search in bank_searches)
        bank_generation_latency_ms = 0.0
        next_citation_index = 1
        for ticker in selected_tickers:
            bank_name = self.bank_names.get(ticker, ticker)
            bank_search = search_by_ticker[ticker]
            evidence = bank_search.results
            all_evidence.extend(evidence)
            per_bank_retrieval.append(
                {
                    "ticker": ticker,
                    "evidence_count": len(evidence),
                    "latency_ms": bank_search.latency_ms,
                }
            )

            if on_progress is not None:
                on_progress(
                    "generating",
                    {
                        "message": f"Generating the {ticker} answer...",
                        "ticker": ticker,
                        "evidence_count": len(evidence),
                    },
                )
            started = perf_counter()
            answer = generate_answer(
                question,
                evidence,
                client=self.client,
                model=self.generation_model,
                expected_ticker=ticker,
                expected_bank_name=bank_name,
                expected_record_type=record_type,
                temperature=self.temperature,
                resolved_question=standalone_question,
                comparison_scope=True,
            )
            bank_generation_latency_ms += (perf_counter() - started) * 1000
            relabeled, next_citation_index = self._relabel_bank_result(answer, next_citation_index)
            bank_results.append({"ticker": ticker, "bank_name": bank_name, **relabeled})

        supported_count = sum(result["status"] == "supported" for result in bank_results)
        if supported_count == len(bank_results):
            status = "supported"
        elif supported_count:
            status = "partial"
        else:
            status = "unsupported"

        if on_progress is not None and supported_count:
            on_progress(
                "synthesizing",
                {"message": "Synthesizing the bank comparison...", "tickers": selected_tickers},
            )
        started = perf_counter()
        synthesis = synthesize_comparison(
            question,
            bank_results,
            client=self.client,
            model=self.generation_model,
            resolved_question=standalone_question,
        )
        synthesis_latency_ms = (perf_counter() - started) * 1000
        generation_latency_ms = bank_generation_latency_ms + synthesis_latency_ms
        if on_progress is not None:
            on_progress("validating", {"message": "Validating the bank comparison..."})

        citations = [
            dict(citation) for result in bank_results for citation in result.get("citations") or []
        ]
        generation_request_count = sum(
            int((result.get("generation") or {}).get("request_count") or 0)
            for result in bank_results
        ) + int(synthesis["generation"].get("request_count") or 0)
        output = {
            "question": question,
            "mode": "comparison",
            "ticker": None,
            "tickers": list(selected_tickers),
            "contextualization": dict(contextualization),
            "bank_resolution": resolution.as_dict(),
            "retrieval": {
                "backend": "mixed",
                "mode": "hybrid",
                "evidence_count": len(all_evidence),
                "per_bank": per_bank_retrieval,
            },
            "status": status,
            "answer_type": "narrative",
            "answer": synthesis["answer"],
            "facts": None,
            "reason": (
                "All selected banks have grounded answers."
                if status == "supported"
                else "Only some selected banks have grounded answers."
                if status == "partial"
                else "No selected bank has sufficient evidence."
            ),
            "citations": citations,
            "bank_results": bank_results,
            "generation": {
                **synthesis["generation"],
                "request_count": generation_request_count,
                "bank_request_count": generation_request_count
                - int(synthesis["generation"].get("request_count") or 0),
                "synthesis_request_count": int(synthesis["generation"].get("request_count") or 0),
                "final_status": status,
            },
        }
        return AnswerRun(
            output=output,
            evidence=all_evidence,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )

    def _ambiguous_bank_run(
        self,
        question: str,
        resolution: BankResolution,
        contextualization: Mapping[str, Any],
    ) -> AnswerRun:
        if resolution.status == "too_many":
            answer = "Možete porediti najviše četiri podržane banke u jednom pitanju."
            reason_code = "too_many_banks_identified"
            reason = "Pitanje pominje više od četiri podržane banke."
        elif resolution.status == "multiple":
            names = [self.bank_names.get(ticker, ticker) for ticker in resolution.detected_tickers]
            answer = (
                "Trenutno mogu da odgovorim za jednu banku po pitanju. "
                f"Pronašao sam {', '.join(names)}; navedite jednu banku."
            )
            reason_code = "multiple_banks_identified"
            reason = "Pitanje pominje više podržanih banaka."
        else:
            answer = (
                "Nisam mogao da odredim na koju banku se pitanje odnosi. "
                "Navedite naziv ili ticker, na primer 'JPMorgan Chase' ili 'JPM'."
            )
            reason_code = "bank_not_identified"
            reason = "Pitanje ne sadrži prepoznat naziv ili ticker podržane banke."
        output = {
            "question": question,
            "ticker": None,
            "contextualization": dict(contextualization),
            "bank_resolution": resolution.as_dict(),
            "retrieval": {
                "backend": "mixed",
                "mode": "hybrid",
                "evidence_count": 0,
            },
            "status": "ambiguous",
            "answer_type": "narrative",
            "answer": answer,
            "facts": None,
            "reason": reason,
            "reason_code": reason_code,
            "citations": [],
            "generation": {
                "model": self.generation_model,
                "final_status": "ambiguous",
                "request_count": 0,
            },
        }
        return AnswerRun(
            output=output,
            evidence=[],
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
        )


# Backward-compatible import for existing scripts and third-party callers.
SingleBankAnswerPipeline = BankAnswerPipeline
