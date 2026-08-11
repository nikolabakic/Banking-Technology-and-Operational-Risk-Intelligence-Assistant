import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from bankscope.generation.answer_generator import GenerationValidationError
from scripts.serve_api import (
    PROJECT_ROOT,
    AnswerApiHandler,
    ThreadingHTTPServer,
    _json_default,
    _project_path,
    _validated_request,
)


def test_frontend_api_validates_question_and_normalizes_session_ticker() -> None:
    assert _validated_request(
        {"question": "  What was JPMorgan's CET1 ratio?  ", "session_ticker": " jpm "}
    ) == ("What was JPMorgan's CET1 ratio?", "JPM")


def test_frontend_api_resolves_config_paths_from_project_root() -> None:
    assert _project_path("config/banks.yaml") == PROJECT_ROOT / "config/banks.yaml"
    absolute = PROJECT_ROOT / "data/processed/chunks.jsonl"
    assert _project_path(absolute) == absolute


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"question": "   "},
        {"question": "Question", "session_ticker": 123},
        {"question": "x" * 4_001},
    ],
)
def test_frontend_api_rejects_invalid_requests(payload: object) -> None:
    with pytest.raises(ValueError):
        _validated_request(payload)


def test_frontend_api_serializes_numpy_evidence_values() -> None:
    payload = {"score": np.float32(0.75), "vector": np.asarray([1, 2])}

    assert json.loads(json.dumps(payload, default=_json_default)) == {
        "score": 0.75,
        "vector": [1, 2],
    }


def test_frontend_api_serves_pipeline_answers() -> None:
    calls = []

    class FakePipeline:
        def answer(self, question: str, *, ticker: str | None = None):
            calls.append((question, ticker))
            return SimpleNamespace(
                output={
                    "question": question,
                    "ticker": "JPM",
                    "status": "supported",
                    "answer_type": "narrative",
                    "answer": "Grounded answer [E1]",
                    "reason": "Direct support.",
                    "citations": [],
                },
                evidence=[{"target_chunk_id": "chunk-1", "score": np.float32(0.75)}],
            )

    AnswerApiHandler.pipeline = FakePipeline()  # type: ignore[assignment]
    server = ThreadingHTTPServer(("127.0.0.1", 0), AnswerApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/answer",
            data=json.dumps({"question": "What was the ratio?", "session_ticker": "jpm"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310
            payload = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert calls == [("What was the ratio?", "JPM")]
    assert payload["answer"] == "Grounded answer [E1]"
    assert payload["evidence"][0]["score"] == 0.75


def test_frontend_api_returns_stable_model_validation_error() -> None:
    class InvalidPipeline:
        def answer(self, question: str, *, ticker: str | None = None):
            raise GenerationValidationError("invalid_schema", "Invalid model response.")

    AnswerApiHandler.pipeline = InvalidPipeline()  # type: ignore[assignment]
    server = ThreadingHTTPServer(("127.0.0.1", 0), AnswerApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/answer",
            data=json.dumps({"question": "Question"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as captured:
            urlopen(request, timeout=5)  # noqa: S310
        payload = json.load(captured.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert captured.value.code == 422
    assert payload["code"] == "invalid_schema"
