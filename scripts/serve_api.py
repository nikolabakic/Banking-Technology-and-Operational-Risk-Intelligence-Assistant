"""Serve the BankScope answer pipeline to the local frontend."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankscope.config.settings import get_settings  # noqa: E402
from bankscope.generation import SingleBankAnswerPipeline  # noqa: E402
from bankscope.generation.answer_generator import (  # noqa: E402
    GPT51_CANDIDATE_MODEL,
    GenerationValidationError,
)
from bankscope.generation.pipeline import (  # noqa: E402
    DEFAULT_CHUNKS,
    DEFAULT_GLOSSARY_LOCATORS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.llm import create_openai_client  # noqa: E402
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME  # noqa: E402

MAX_REQUEST_BYTES = 32_768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        help=f"Generation model override (defaults to validated {GPT51_CANDIDATE_MODEL}).",
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _validated_request(payload: Any) -> tuple[str, str | None]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if len(question) > 4_000:
        raise ValueError("question must contain at most 4,000 characters.")
    session_ticker = payload.get("session_ticker")
    if session_ticker is not None and not isinstance(session_ticker, str):
        raise ValueError("session_ticker must be a string or null.")
    ticker = session_ticker.strip().upper() if isinstance(session_ticker, str) else None
    return question.strip(), ticker or None


class AnswerApiHandler(BaseHTTPRequestHandler):
    pipeline: ClassVar[SingleBankAnswerPipeline]
    pipeline_lock: ClassVar[threading.Lock] = threading.Lock()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(HTTPStatus.OK, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ready"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/answer":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError("Request body has an invalid size.")
            payload = json.loads(self.rfile.read(content_length))
            question, session_ticker = _validated_request(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        try:
            with self.pipeline_lock:
                run = self.pipeline.answer(question, ticker=session_ticker)
            self._send_json(HTTPStatus.OK, {**run.output, "evidence": run.evidence})
        except GenerationValidationError as error:
            cause = f" Validation details: {error.__cause__}" if error.__cause__ else ""
            self.log_error("Answer validation failed [%s]: %s.%s", error.code, error, cause)
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {
                    "error": (
                        "The model could not produce a valid grounded answer. Please try again."
                    ),
                    "code": error.code,
                },
            )
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self.log_error("Answer pipeline failed: %s", error)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "The answer pipeline failed. Check the API terminal for details."},
            )

    def log_message(self, message: str, *args: object) -> None:
        print(f"[BankScope API] {self.address_string()} - {message % args}")


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535.")

    settings = get_settings()
    pipeline = SingleBankAnswerPipeline.from_paths(
        client=create_openai_client(settings),
        generation_model=args.model or GPT51_CANDIDATE_MODEL,
        temperature=settings.llm_temperature,
        chunks_path=_project_path(args.chunks),
        tables_path=_project_path(args.tables),
        glossary_locators_path=_project_path(args.glossary_locators),
        qdrant_path=_project_path(args.qdrant_path),
        qdrant_manifest_path=_project_path(args.qdrant_manifest),
        collection_name=args.collection,
        bank_registry_path=_project_path(settings.bank_registry_path),
    )
    AnswerApiHandler.pipeline = pipeline
    server = ThreadingHTTPServer((args.host, args.port), AnswerApiHandler)
    print(f"BankScope API ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping BankScope API.")
    finally:
        server.server_close()
        pipeline.close()


if __name__ == "__main__":
    main()
