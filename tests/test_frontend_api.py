import asyncio
import json
import threading
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bankscope.api import AppServices, QuestionRequest, create_app
from bankscope.chat import ChatStore, CitationSourceResolver
from bankscope.generation.answer_generator import GenerationValidationError
from scripts.serve_api import PROJECT_ROOT, _json_default, _project_path, _validated_request


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        tickers=(),
        conversation_history=(),
        conversation_summary="",
        previous_answer=None,
        conversation_metadata=None,
        on_progress=None,
    ):
        self.calls.append((question, ticker, list(tickers), list(conversation_history)))
        if on_progress:
            on_progress("routing", {"message": "Routing the request..."})
            on_progress("resolving_bank", {"message": "Identifying the bank..."})
            on_progress("assessing_evidence", {"message": "Assessing retrieved evidence..."})
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
                "diagnostics": {
                    "route": "domain_rag",
                    "agentic_rag_enabled": False,
                    "outcome": "supported",
                    "failed_stage": None,
                    "error_code": None,
                    "stages": [],
                    "initial_evidence_count": 1,
                    "final_evidence_count": 1,
                    "model_request_count": 1,
                    "bank_plans": [],
                    "quality_gate": {"passed": True, "checks": {"pipeline_completed": True}},
                },
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
    request = QuestionRequest.model_validate(
        {"question": "Compare", "session_tickers": [" bac ", "c"]}
    )
    assert request.session_tickers == ["BAC", "C"]


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
    assert pipeline.calls[0] == ("What did JPM report?", None, [], [])
    assert pipeline.calls[1][:3] == ("What about the framework?", "JPM", ["JPM"])
    model_history = pipeline.calls[1][3]
    assert model_history[0] == {"role": "user", "content": "What did JPM report?"}
    assert model_history[1] == {
        "role": "assistant",
        "content": "Grounded answer [E1]",
    }

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


def test_thread_delete_waits_for_active_answer_before_removing_thread(tmp_path) -> None:
    class ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.waiter_started = threading.Event()

        def __enter__(self):
            if self._lock.locked():
                self.waiter_started.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_args: object) -> None:
            self._lock.release()

    class BlockingPipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def answer(self, question: str, **kwargs):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("Timed out waiting to release the active answer.")
            return super().answer(question, **kwargs)

    store = ChatStore(tmp_path / "delete-race-chat.db")
    store.initialize()
    thread = store.create_thread()
    pipeline = BlockingPipeline()
    generation_lock = ObservedLock()
    services = AppServices(
        pipeline,
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        generation_lock,  # type: ignore[arg-type]
    )
    client = TestClient(create_app(services))
    answer_result: list[tuple[dict[str, object], int]] = []
    answer_errors: list[BaseException] = []
    delete_responses = []

    def answer() -> None:
        try:
            answer_result.append(services.answer_thread(thread["id"], "Active question"))
        except BaseException as error:
            answer_errors.append(error)

    def delete() -> None:
        delete_responses.append(client.delete(f"/api/threads/{thread['id']}"))

    answer_worker = threading.Thread(target=answer, daemon=True)
    delete_worker = threading.Thread(target=delete, daemon=True)
    answer_worker.start()
    assert pipeline.started.wait(timeout=2)
    delete_worker.start()
    try:
        assert generation_lock.waiter_started.wait(timeout=2)
        assert delete_worker.is_alive()
        assert store.get_thread(thread["id"])["id"] == thread["id"]
    finally:
        pipeline.release.set()

    answer_worker.join(timeout=5)
    delete_worker.join(timeout=5)
    assert not answer_worker.is_alive()
    assert not delete_worker.is_alive()
    assert answer_errors == []
    assert answer_result[0][1] == 200
    assert delete_responses[0].status_code == 204
    with pytest.raises(KeyError):
        store.get_thread(thread["id"])


@pytest.mark.parametrize("outcome", ["supported", "validation_error", "value_error", "error"])
def test_cancelled_answer_result_or_recovery_never_persists(tmp_path, outcome: str) -> None:
    class BlockingPipeline(FakePipeline):
        generation_model = "test-model"

        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def answer(self, question: str, **kwargs):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("Timed out waiting to release the cancelled answer.")
            if outcome == "validation_error":
                raise GenerationValidationError(
                    "invalid_schema",
                    "Invalid grounded answer response.",
                    generation={"stage": "generating", "request_count": 1},
                )
            if outcome == "value_error":
                raise ValueError("invalid request")
            if outcome == "error":
                raise RuntimeError("provider failed")
            return super().answer(question, **kwargs)

    store = ChatStore(tmp_path / f"cancelled-{outcome}.db")
    store.initialize()
    thread = store.create_thread()
    pipeline = BlockingPipeline()
    services = AppServices(
        pipeline,
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        threading.Lock(),
    )
    cancellation_event = threading.Event()
    results: list[tuple[dict[str, object], int]] = []

    worker = threading.Thread(
        target=lambda: results.append(
            services.answer_thread(
                thread["id"],
                "Do not remember this question",
                cancellation_event=cancellation_event,
            )
        ),
        daemon=True,
    )
    worker.start()
    assert pipeline.started.wait(timeout=2)
    cancellation_event.set()
    pipeline.release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert results == [({"state": "cancelled"}, 499)]
    assert store.list_messages(thread["id"]) == []
    assert store.list_turns(thread["id"]) == []
    assert store.conversation_context(thread["id"])["unresolved_requests"] == []


def test_answer_cancelled_before_generation_does_not_call_pipeline_or_persist(tmp_path) -> None:
    class MustNotGenerate(FakePipeline):
        def answer(self, question: str, **kwargs):
            raise AssertionError("A pre-cancelled answer must not start generation.")

    store = ChatStore(tmp_path / "pre-cancelled.db")
    store.initialize()
    thread = store.create_thread()
    services = AppServices(
        MustNotGenerate(),
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        threading.Lock(),
    )
    cancellation_event = threading.Event()
    cancellation_event.set()

    result = services.answer_thread(
        thread["id"],
        "Do not start this question",
        cancellation_event=cancellation_event,
    )

    assert result == ({"state": "cancelled"}, 499)
    assert store.list_messages(thread["id"]) == []


def test_cancelled_stream_tombstones_worker_result(tmp_path) -> None:
    class BlockingPipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def answer(self, question: str, **kwargs):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("Timed out waiting to release the cancelled stream.")
            return super().answer(question, **kwargs)

    store = ChatStore(tmp_path / "cancelled-stream.db")
    store.initialize()
    thread = store.create_thread()
    pipeline = BlockingPipeline()
    generation_lock = threading.Lock()
    services = AppServices(
        pipeline,
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        generation_lock,
    )
    app = create_app(services)
    stream_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/threads/{thread_id}/stream"
        and "POST" in (getattr(route, "methods", None) or set())
    )

    async def cancel_after_generation_starts() -> None:
        response = await stream_endpoint(
            uuid.UUID(thread["id"]),
            QuestionRequest(question="Disconnect before remembering this"),
        )
        iterator = response.body_iterator
        connected = await anext(iterator)
        assert '"stage":"connected"' in connected
        assert await asyncio.to_thread(pipeline.started.wait, 2)

        pending_event = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        pending_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_event

        pipeline.release.set()

        def wait_for_worker() -> None:
            with generation_lock:
                pass

        await asyncio.wait_for(asyncio.to_thread(wait_for_worker), timeout=5)

    try:
        asyncio.run(cancel_after_generation_starts())
    finally:
        pipeline.release.set()

    assert store.list_messages(thread["id"]) == []
    assert store.list_turns(thread["id"]) == []


def test_compatibility_answer_keeps_existing_contract(api) -> None:
    client, pipeline, _store = api
    response = client.post(
        "/api/answer", json={"question": "What was the ratio?", "session_ticker": "jpm"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer [E1]"
    assert response.json()["evidence"][0]["score"] == 0.75
    assert pipeline.calls == [("What was the ratio?", "JPM", [], [])]


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
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    assert '"stage":"connected"' in body
    assert '"stage":"resolving_bank"' in body
    assert '"stage":"routing"' in body
    assert '"stage":"assessing_evidence"' in body
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


@pytest.mark.parametrize(
    "source_url",
    ["https://example.com/risk-update", "http://example.com/risk-update"],
)
def test_web_citation_context_returns_safe_persisted_anchor_without_corpus_lookup(
    tmp_path, source_url: str
) -> None:
    class CorpusResolverMustNotRun:
        corpus_hash = "different-current-corpus"

        def context(self, *_args, **_kwargs):
            raise AssertionError("Web citations must not be resolved against the filing corpus.")

    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    output = {
        "question": "What changed?",
        "ticker": None,
        "status": "supported",
        "answer_type": "narrative",
        "answer": "An updated risk notice was published [E1].",
        "reason": "Current web source.",
        "citations": [
            {
                "kind": "web",
                "label": "E1",
                "title": "  Official\n risk update  ",
                "snippet": "  The regulator\tpublished an updated risk notice.  ",
                "source_url": source_url,
                "untrusted_extra": "must not enter the synthetic anchor",
            }
        ],
    }
    turn = store.append_answer_turn(
        thread["id"], "What changed?", output, corpus_hash="corpus-at-answer-time"
    )
    citation_id = turn["response"]["citations"][0]["citation_id"]
    services = AppServices(FakePipeline(), store, CorpusResolverMustNotRun(), threading.Lock())
    client = TestClient(create_app(services))

    response = client.get(f"/api/citations/{citation_id}/context")

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_chunk_id"] == f"web:{citation_id}"
    assert payload["citation"]["target_chunk_id"] == f"web:{citation_id}"
    assert payload["record_type"] == "web"
    assert payload["source_url"] == source_url
    assert payload["corpus_hash"] == "corpus-at-answer-time"
    assert payload["citation"]["metadata"] == {
        "kind": "web",
        "citation_id": citation_id,
        "label": "E1",
        "source_url": source_url,
        "title": "Official risk update",
        "snippet": "The regulator published an updated risk notice.",
    }
    assert payload["chunks"] == [
        {
            "target_chunk_id": f"web:{citation_id}",
            "role": "anchor",
            "record_type": "web",
            "document": (
                "Official risk update\n\n"
                "The regulator published an updated risk notice.\n\n"
                f"Source: {source_url}"
            ),
            "metadata": payload["citation"]["metadata"],
        }
    ]


@pytest.mark.parametrize(
    "source_url",
    [
        "javascript:alert(1)",
        "file:///tmp/report",
        "//example.com/no-scheme",
        "https://example.com\\@unsafe.test/source",
        "https://example.com/source\x00suffix",
    ],
)
def test_web_citation_context_rejects_non_http_sources(tmp_path, source_url: str) -> None:
    class CorpusResolverMustNotRun:
        corpus_hash = "current-corpus"

        def context(self, *_args, **_kwargs):
            raise AssertionError("Invalid web citations must not reach the filing resolver.")

    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    output = {
        "answer": "Unsafe source [E1]",
        "citations": [
            {
                "kind": "web",
                "label": "E1",
                "title": "Unsafe source",
                "snippet": "Do not expose this as a link.",
                "source_url": source_url,
            }
        ],
    }
    turn = store.append_answer_turn(
        thread["id"], "Open it", output, corpus_hash="corpus-at-answer-time"
    )
    citation_id = turn["response"]["citations"][0]["citation_id"]
    services = AppServices(FakePipeline(), store, CorpusResolverMustNotRun(), threading.Lock())
    client = TestClient(create_app(services))

    response = client.get(f"/api/citations/{citation_id}/context")

    assert response.status_code == 422
    assert response.json() == {
        "error": "Web citation source URL is invalid.",
        "code": "invalid_citation_source_url",
    }
    assert source_url not in response.text


def test_validation_error_becomes_stable_conversation_turn(tmp_path) -> None:
    class InvalidPipeline:
        def answer(
            self,
            question: str,
            *,
            ticker=None,
            tickers=(),
            conversation_history=(),
            conversation_summary="",
            previous_answer=None,
            conversation_metadata=None,
            on_progress=None,
        ):
            raise GenerationValidationError(
                "contextualization_invalid_schema",
                "Invalid contextualization response.",
                generation={
                    "stage": "contextualizing",
                    "validation_errors": [{"location": "facts", "type": "model_type"}],
                },
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
    response = client.post(
        f"/api/threads/{thread['id']}/answers",
        json={"question": "Uporedi njihove pokazatelje"},
    )
    assert response.status_code == 200
    response_turn = response.json()["turn"]
    assert response_turn["state"] == "answered"
    assert response_turn["response"]["dialog_act"] == "retryable_error"
    assert response_turn["response"]["reason_code"] == "contextualization_invalid_schema"
    assert response_turn["response"]["answer"].startswith("Nisam uspeo")
    assert response_turn["response"]["diagnostics"]["error_code"] == (
        "contextualization_invalid_schema"
    )
    assert response_turn["response"]["diagnostics"]["failed_stage"] == "contextualizing"
    assert response_turn["response"]["diagnostics"]["validation_errors"] == [
        {"location": "facts", "type": "model_type"}
    ]
    turns = client.get(f"/api/threads/{thread['id']}/messages").json()["turns"]
    assert turns[0]["state"] == "answered"
    assert turns[0]["response"]["dialog_act"] == "retryable_error"


def test_recovery_preserves_generation_evidence_and_history_diagnostics(tmp_path) -> None:
    class InvalidPipeline:
        generation_model = "pipeline-model"
        agentic_rag_enabled = False

        def answer(
            self,
            question: str,
            *,
            ticker=None,
            tickers=(),
            conversation_history=(),
            conversation_summary="",
            previous_answer=None,
            conversation_metadata=None,
            on_progress=None,
        ):
            assert len(conversation_history) == 2
            if on_progress:
                on_progress("routing", {"message": "Routing the request..."})
                on_progress("retrieving", {"message": "Searching indexed filings..."})
                on_progress(
                    "generating",
                    {"message": "Generating a grounded answer...", "evidence_count": 4},
                )
            raise GenerationValidationError(
                "invalid_schema",
                "Invalid grounded answer response.",
                generation={
                    "stage": "generating",
                    "model": "actual-generation-model",
                    "request_count": 2,
                    "final_status": "validation_error",
                },
            )

    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    services = AppServices(
        FakePipeline(),
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        threading.Lock(),
    )
    client = TestClient(create_app(services))
    thread = client.post("/api/threads", json={}).json()
    first = client.post(
        f"/api/threads/{thread['id']}/answers",
        json={"question": "What did JPM report?"},
    )
    assert first.status_code == 200

    services.pipeline = InvalidPipeline()
    response = client.post(
        f"/api/threads/{thread['id']}/answers",
        json={"question": "How did that affect operational risk?"},
    )
    assert response.status_code == 200
    recovery = response.json()["turn"]["response"]
    diagnostics = recovery["diagnostics"]

    assert recovery["contextualization"]["history_turns"] == 1
    assert recovery["retrieval"]["evidence_count"] == 4
    assert recovery["generation"]["model"] == "actual-generation-model"
    assert recovery["generation"]["request_count"] == 2
    assert diagnostics["initial_evidence_count"] == 4
    assert diagnostics["final_evidence_count"] == 4
    assert diagnostics["generation_request_count"] == 2
    assert diagnostics["model_request_count"] == 2
    assert diagnostics["model_request_count_scope"] == "generation_only"
    assert diagnostics["history_turns"] == 1
    assert diagnostics["context_message_count"] == 2
    assert diagnostics["context_estimated_tokens"] > 0
    assert diagnostics["summary_used"] is False


def test_recovery_preserves_generation_metadata_from_request_exception(tmp_path) -> None:
    class RequestFailurePipeline:
        generation_model = "pipeline-model"
        agentic_rag_enabled = False

        def answer(self, question: str, *, on_progress=None, **_kwargs):
            if on_progress:
                on_progress("generating", {"message": "Generating...", "evidence_count": 3})
            error = RuntimeError("OpenAI answer generation failed.")
            error.generation = {
                "stage": "generating",
                "model": "request-model",
                "request_count": 1,
            }
            raise error

    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    services = AppServices(
        RequestFailurePipeline(),
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        threading.Lock(),
    )
    client = TestClient(create_app(services))
    thread = client.post("/api/threads", json={}).json()
    response = client.post(
        f"/api/threads/{thread['id']}/answers",
        json={"question": "How does Ally define operational risk?"},
    )
    recovery = response.json()["turn"]["response"]

    assert recovery["reason_code"] == "pipeline_failed"
    assert recovery["generation"]["model"] == "request-model"
    assert recovery["generation"]["request_count"] == 1
    assert recovery["retrieval"]["evidence_count"] == 3
    assert recovery["diagnostics"]["generation_request_count"] == 1


def test_same_thread_requests_load_context_and_persist_in_generation_order(tmp_path) -> None:
    class HandoffLock:
        """Release to an existing waiter before returning to the former owner."""

        def __init__(self) -> None:
            self._condition = threading.Condition()
            self._owner: int | None = None
            self._waiters = 0
            self._acquisitions = 0
            self.waiter_ready = threading.Event()

        def __enter__(self):
            identity = threading.get_ident()
            with self._condition:
                self._waiters += 1
                if self._owner is not None:
                    self.waiter_ready.set()
                while self._owner is not None:
                    self._condition.wait()
                self._waiters -= 1
                self._owner = identity
                self._acquisitions += 1
                self._condition.notify_all()
            return self

        def __exit__(self, *_args: object) -> None:
            identity = threading.get_ident()
            with self._condition:
                assert self._owner == identity
                acquisitions_before_release = self._acquisitions
                handoff_required = self._waiters > 0
                self._owner = None
                self._condition.notify_all()
                while handoff_required and self._acquisitions == acquisitions_before_release:
                    self._condition.wait()

    class BlockingPipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def answer(self, question: str, **kwargs):
            if question == "First question":
                self.first_started.set()
                if not self.release_first.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release the first generation.")
            return super().answer(question, **kwargs)

    store = ChatStore(tmp_path / "concurrent-chat.db")
    store.initialize()
    thread = store.create_thread()
    pipeline = BlockingPipeline()
    generation_lock = HandoffLock()
    services = AppServices(
        pipeline,
        store,
        CitationSourceResolver([], [], corpus_hash="corpus-v1"),
        generation_lock,  # type: ignore[arg-type]
    )
    errors: list[BaseException] = []

    def answer(question: str) -> None:
        try:
            services.answer_thread(thread["id"], question)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=answer, args=("First question",), daemon=True)
    second = threading.Thread(target=answer, args=("Second follow-up",), daemon=True)
    first.start()
    assert pipeline.first_started.wait(timeout=2)
    second.start()
    assert generation_lock.waiter_ready.wait(timeout=2)
    pipeline.release_first.set()

    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []

    assert [call[0] for call in pipeline.calls] == ["First question", "Second follow-up"]
    assert pipeline.calls[0][3] == []
    assert pipeline.calls[1][3] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Grounded answer [E1]"},
    ]
    assert [turn["question"] for turn in store.list_turns(thread["id"])] == [
        "First question",
        "Second follow-up",
    ]
