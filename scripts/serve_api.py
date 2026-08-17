"""Serve the persistent local BankScope FastAPI application."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankscope.api import AppServices, QuestionRequest, create_app  # noqa: E402
from bankscope.chat import ChatStore, CitationSourceResolver  # noqa: E402
from bankscope.config.settings import get_settings  # noqa: E402
from bankscope.generation import BankAnswerPipeline  # noqa: E402
from bankscope.generation.answer_generator import GPT51_CANDIDATE_MODEL  # noqa: E402
from bankscope.generation.pipeline import (  # noqa: E402
    DEFAULT_CHUNKS,
    DEFAULT_GLOSSARY_LOCATORS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.llm import create_openai_client  # noqa: E402
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME  # noqa: E402

DEFAULT_CHAT_DB = Path("data/local/bankscope_chat.db")


class JsonLogFormatter(logging.Formatter):
    """Minimal structured logs without questions, answers, evidence, or credentials."""

    fields = (
        "request_id",
        "thread_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "error_code",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in self.fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model")
    parser.add_argument("--chat-db", type=Path, default=DEFAULT_CHAT_DB)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validated_request(payload: Any) -> tuple[str, str | None]:
    request = QuestionRequest.model_validate(payload)
    return request.question, request.session_ticker


def build_services(args: argparse.Namespace) -> AppServices:
    chunks = _project_path(args.chunks)
    tables = _project_path(args.tables)
    settings = get_settings()
    pipeline = BankAnswerPipeline.from_paths(
        client=create_openai_client(settings),
        generation_model=args.model or GPT51_CANDIDATE_MODEL,
        temperature=settings.llm_temperature,
        chunks_path=chunks,
        tables_path=tables,
        glossary_locators_path=_project_path(args.glossary_locators),
        qdrant_path=_project_path(args.qdrant_path),
        qdrant_manifest_path=_project_path(args.qdrant_manifest),
        collection_name=args.collection,
        bank_registry_path=_project_path(settings.bank_registry_path),
        agentic_rag_enabled=settings.agentic_rag_enabled,
    )
    store = ChatStore(_project_path(args.chat_db))
    store.initialize()
    sources = CitationSourceResolver.from_paths(chunks, tables)
    return AppServices(pipeline, store, sources, threading.Lock())


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535.")
    configure_logging()
    services = build_services(args)
    app = create_app(services)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        services.pipeline.close()


if __name__ == "__main__":
    main()
