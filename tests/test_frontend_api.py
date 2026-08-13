import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bankscope.api import AppServices, create_app
from bankscope.chat import ChatStore, CitationSourceResolver
from bankscope.generation.answer_generator import GenerationValidationError
from scripts.serve_api import PROJECT_ROOT, _json_default, _project_path, _validated_request


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, list[dict[str, str]]]] = []

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        conversation_history=(),
        on_progress=None,
    ):
        self.calls.append((question, ticker, list(conversation_history)))
        if on_progress:
            on_progress("resolving_bank", {"message": "Identifying the bank..."})
            on_progress("retrieving", {"message": "Searching indexed filings..."})
        return SimpleNamespace(
            output={
                "question": question,
                "ticker": "JPM",
                "status": "supported",
                "answer_type": "narrative",
                "answer": "Grounded answer [E1]",
                "reason": "Direct support.",
                "citations": [
                    {
                        "label": "E1",
                        "target_chunk_id": "chunk-1",
                        "ticker": "JPM",
                        "record_type": "text",
                    }
                ],
            },
            evidence=[{"target_chunk_id": "chunk-1", "score": np.float32(0.75)}],
        )


@pytest.fixture
def api(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    pipeline = FakePipeline()
    sources = CitationSourceResolver([], [], corpus_hash="corpus-v1")
    services = AppServices(pipeline, store, sources, threading.Lock())
    return TestClient(create_app(services)), pipeline, store


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
    [None, {}, {"question": "   "}, {"question": "x" * 4_001}],
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


def test_thread_answer_persists_messages_and_server_side_session(api) -> None:
    client, pipeline, _store = api
    thread = client.post("/api/threads", json={}).json()

    response = client.post(
        f"/api/threads/{thread['id']}/answers", json={"question": "What did JPM report?"}
    )
    assert response.status_code == 200
    assert response.json()["turn"]["response"]["citations"][0]["citation_id"]

    second = client.post(
        f"/api/threads/{thread['id']}/answers", json={"question": "What about the framework?"}
    )
    assert second.status_code == 200
    assert pipeline.calls[0] == ("What did JPM report?", None, [])
    assert pipeline.calls[1] == (
        "What about the framework?",
        "JPM",
        [
            {"role": "user", "content": "What did JPM report?"},
            {"role": "assistant", "content": "Grounded answer [E1]"},
        ],
    )

    history = client.get(f"/api/threads/{thread['id']}/messages").json()
    assert len(history["messages"]) == 4
    assert len(history["turns"]) == 2
    assert history["thread"]["title"] == "What did JPM report?"


def test_thread_crud_and_missing_thread(api) -> None:
    client, _pipeline, _store = api
    thread = client.post("/api/threads", json={"title": "Initial"}).json()
    renamed = client.patch(f"/api/threads/{thread['id']}", json={"title": "Renamed conversation"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed conversation"
    assert client.delete(f"/api/threads/{thread['id']}").status_code == 204
    assert client.get(f"/api/threads/{thread['id']}/messages").status_code == 404


def test_compatibility_answer_keeps_existing_contract(api) -> None:
    client, pipeline, _store = api
    response = client.post(
        "/api/answer", json={"question": "What was the ratio?", "session_ticker": "jpm"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer [E1]"
    assert response.json()["evidence"][0]["score"] == 0.75
    assert pipeline.calls == [("What was the ratio?", "JPM", [])]


def test_stream_emits_progress_answer_and_done(api) -> None:
    client, _pipeline, _store = api
    thread = client.post("/api/threads", json={}).json()
    with client.stream(
        "POST",
        f"/api/threads/{thread['id']}/stream",
        json={"question": "What did JPM report?"},
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert '"stage":"resolving_bank"' in body
    assert '"type":"answer"' in body
    assert '"type":"done"' in body


def test_citation_context_hydrates_persisted_source(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    pipeline = FakePipeline()
    sources = CitationSourceResolver(
        [
            {
                "record_id": "record-1",
                "target_chunk_id": "chunk-1",
                "record_type": "text",
                "document": "Canonical evidence",
                "metadata": {
                    "ticker": "JPM",
                    "accession_number": "filing-1",
                    "source_url": "https://example.com/filing",
                },
            }
        ],
        [],
        corpus_hash="corpus-v1",
    )
    client = TestClient(create_app(AppServices(pipeline, store, sources, threading.Lock())))
    thread = client.post("/api/threads", json={}).json()
    turn = client.post(
        f"/api/threads/{thread['id']}/answers", json={"question": "What did JPM report?"}
    ).json()["turn"]
    citation_id = turn["response"]["citations"][0]["citation_id"]

    context = client.get(f"/api/citations/{citation_id}/context")
    assert context.status_code == 200
    assert context.json()["chunks"][0]["document"] == "Canonical evidence"

    sources.corpus_hash = "changed"
    stale = client.get(f"/api/citations/{citation_id}/context")
    assert stale.status_code == 409
    assert stale.json()["code"] == "citation_corpus_mismatch"


def test_validation_error_is_stable_and_persisted(tmp_path) -> None:
    class InvalidPipeline:
        def answer(self, question: str, *, ticker=None, conversation_history=(), on_progress=None):
            raise GenerationValidationError(
                "contextualization_invalid_schema", "Invalid contextualization response."
            )

    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    services = AppServices(
        InvalidPipeline(),
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        threading.Lock(),
    )
    client = TestClient(create_app(services))
    thread = client.post("/api/threads", json={}).json()
    response = client.post(f"/api/threads/{thread['id']}/answers", json={"question": "Question"})
    assert response.status_code == 422
    assert response.json()["code"] == "contextualization_invalid_schema"
    turns = client.get(f"/api/threads/{thread['id']}/messages").json()["turns"]
    assert turns[0]["state"] == "error"
    assert turns[0]["error_code"] == "contextualization_invalid_schema"
