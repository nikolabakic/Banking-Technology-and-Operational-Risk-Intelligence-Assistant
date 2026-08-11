from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from bankscope.generation.answer_generator import generate_answer
from bankscope.io import read_jsonl, sha256_file
from bankscope.retrieval.glossary_locators import validate_glossary_locators
from bankscope.retrieval.hybrid_retriever import HybridRetriever
from bankscope.retrieval.mixed_retriever import MixedRetriever
from bankscope.retrieval.qdrant_retriever import (
    DEFAULT_COLLECTION_NAME,
    QdrantRetriever,
    load_qdrant_manifest,
)
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

        model_options = (
            {} if model_revision.strip().lower() == "unknown" else {"revision": model_revision}
        )
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


class SingleBankAnswerPipeline:
    """Reusable mixed-retrieval and grounded-generation pipeline for one bank."""

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
    ) -> SingleBankAnswerPipeline:
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
        bank_names = {bank.ticker: bank.legal_name for bank in registry.banks}
        encoder = query_encoder or SentenceTransformerQueryEncoder(model_name, model_revision)
        qdrant = QdrantRetriever(
            qdrant_path,
            tables,
            manifest_path=manifest_path,
            collection_name=collection_name,
            tables_path=tables_path,
        )
        try:
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
            )
        except Exception:
            qdrant.close()
            raise

    def close(self) -> None:
        if not self._closed and self._close_callback is not None:
            self._close_callback()
        self._closed = True

    def __enter__(self) -> SingleBankAnswerPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def answer(
        self,
        question: str,
        *,
        ticker: str,
        record_type: str | None = None,
        limit: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
    ) -> AnswerRun:
        if self._closed:
            raise RuntimeError("The answer pipeline is closed.")
        question = question.strip()
        ticker = ticker.strip().upper()
        if not question:
            raise ValueError("Question cannot be empty.")
        if not ticker:
            raise ValueError("ticker is required for the single-bank answer flow.")
        if self.bank_names and ticker not in self.bank_names:
            raise ValueError(f"Unknown bank ticker: {ticker}.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if candidate_k < limit:
            raise ValueError("candidate_k must be at least limit.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        started = perf_counter()
        query_vector = self.query_encoder.encode(question)
        embedding_latency_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        evidence = self.retriever.search_hybrid(
            question,
            query_vector,
            limit=limit,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            ticker=ticker,
            record_type=record_type,
        )
        retrieval_latency_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        answer = generate_answer(
            question,
            evidence,
            client=self.client,
            model=self.generation_model,
            expected_ticker=ticker,
            expected_bank_name=self.bank_names.get(ticker, ticker),
            expected_record_type=record_type,
            temperature=self.temperature,
        )
        generation_latency_ms = (perf_counter() - started) * 1000
        output = {
            "question": question,
            "ticker": ticker,
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
