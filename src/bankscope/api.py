"""FastAPI application for the local BankScope chat product."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from bankscope.chat import ChatStore, CitationSourceResolver, StaleCitationError
from bankscope.generation.answer_generator import GenerationValidationError, question_language

LOGGER = logging.getLogger("bankscope.api")
GENERATION_ERROR_MESSAGE = "The model could not produce a valid grounded answer. Please try again."
PIPELINE_ERROR_MESSAGE = "The answer pipeline failed. Check the API terminal for details."


def _recovery_answer(question: str) -> str:
    if question_language(question) == "Serbian":
        return (
            "Nisam uspeo da završim pouzdanu pretragu za ovu poruku. Možeš da pokušaš "
            "ponovo ili da preformulišeš pitanje; prethodni kontekst razgovora je sačuvan."
        )
    return (
        "I couldn't complete reliable research for this message. Please try again or rephrase "
        "the question; the previous conversation context is still available."
    )


def _recovery_output(
    question: str,
    *,
    model: str,
    code: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Represent an expected pipeline failure as a normal assistant turn."""

    return {
        "question": question,
        "dialog_act": "retryable_error",
        "ticker": None,
        "contextualization": {
            "applied": False,
            "history_turns": 0,
            "standalone_question": question,
            "model": model,
            "latency_ms": 0.0,
            "source": "recovery",
            "fallback": True,
            "error_code": code,
            "skip_reason": "pipeline_recovery",
        },
        "bank_resolution": {
            "status": "not_required",
            "source": "recovery",
            "ticker": None,
            "detected_tickers": [],
        },
        "retrieval": {"backend": "none", "mode": "none", "evidence_count": 0},
        "status": "unsupported",
        "answer_type": "narrative",
        "answer": _recovery_answer(question),
        "facts": None,
        "reason": (
            "The current research attempt failed safely before a grounded answer was available."
        ),
        "reason_code": code,
        "citations": [],
        "generation": {
            "model": model,
            "final_status": "unsupported",
            "request_count": 0,
        },
        "diagnostics": dict(diagnostics),
    }


def jsonable(value: object) -> Any:
    def default(item: object) -> object:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.loads(json.dumps(value, ensure_ascii=False, default=default))


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    session_ticker: str | None = None
    session_tickers: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must be a non-empty string.")
        return normalized

    @field_validator("session_ticker")
    @classmethod
    def normalize_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().upper() or None

    @field_validator("session_tickers")
    @classmethod
    def normalize_tickers(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))
        if len(normalized) != len(values):
            raise ValueError("session_tickers must contain unique, non-empty tickers.")
        return normalized


class ThreadRequest(BaseModel):
    title: str | None = None


class RenameThreadRequest(BaseModel):
    title: str


@dataclass(slots=True)
class AppServices:
    pipeline: Any
    store: ChatStore
    sources: CitationSourceResolver
    pipeline_lock: threading.Lock

    def answer_thread(
        self,
        thread_id: str,
        question: str,
        *,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], int]:
        thread = self.store.get_thread(thread_id)
        conversation_history = self.store.conversation_history(thread_id)
        stage_trace: list[dict[str, Any]] = []
        latest_stage: str | None = None

        def tracked_progress(stage: str, details: Mapping[str, Any]) -> None:
            nonlocal latest_stage
            latest_stage = stage
            stage_trace.append({"stage": stage, "status": "started", **jsonable(dict(details))})
            if on_progress is not None:
                on_progress(stage, details)

        def error_diagnostics(
            code: str,
            failed_stage: str | None = None,
            generation: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            enabled = bool(getattr(self.pipeline, "agentic_rag_enabled", False))
            checks = {
                "pipeline_completed": False,
                "plan_schema": not code.startswith(
                    ("agentic_plan", "agentic_step", "agentic_verdict")
                ),
                "query_preservation": not any(
                    marker in code for marker in ("rewrite_lost", "search_lost", "added_numeric")
                ),
                "citation_contract": True,
                "bank_isolation": "crossed_bank" not in code,
                "action_budget": True,
                "request_budget": True,
            }
            diagnostics = {
                "route": "domain_rag",
                "agentic_rag_enabled": enabled,
                "outcome": "error",
                "failed_stage": failed_stage or latest_stage,
                "error_code": code,
                "stages": stage_trace,
                "initial_evidence_count": None,
                "final_evidence_count": None,
                "model_request_count": None,
                "bank_plans": [],
                "quality_gate": {"passed": False, "checks": checks},
            }
            validation_errors = (generation or {}).get("validation_errors")
            if isinstance(validation_errors, list):
                diagnostics["validation_errors"] = validation_errors
            return diagnostics

        try:
            with self.pipeline_lock:
                run = self.pipeline.answer(
                    question,
                    ticker=thread["session_ticker"],
                    tickers=thread["session_tickers"],
                    conversation_history=conversation_history,
                    on_progress=tracked_progress,
                )
            turn = self.store.append_answer_turn(
                thread_id,
                question,
                jsonable(run.output),
                corpus_hash=self.sources.corpus_hash,
            )
            return turn, 200
        except GenerationValidationError as error:
            LOGGER.exception(
                "generation_validation_failed",
                extra={"thread_id": thread_id, "error_code": error.code},
            )
            diagnostics = error_diagnostics(
                error.code,
                str(error.generation.get("stage") or "") or None,
                error.generation,
            )
            turn = self.store.append_answer_turn(
                thread_id,
                question,
                _recovery_output(
                    question,
                    model=str(getattr(self.pipeline, "generation_model", "unknown")),
                    code=error.code,
                    diagnostics=diagnostics,
                ),
                corpus_hash=self.sources.corpus_hash,
            )
            return turn, 200
        except ValueError:
            LOGGER.info(
                "answer_request_recovered",
                extra={"thread_id": thread_id, "error_code": "invalid_request"},
            )
            diagnostics = error_diagnostics("invalid_request")
            turn = self.store.append_answer_turn(
                thread_id,
                question,
                _recovery_output(
                    question,
                    model=str(getattr(self.pipeline, "generation_model", "unknown")),
                    code="invalid_request",
                    diagnostics=diagnostics,
                ),
                corpus_hash=self.sources.corpus_hash,
            )
            return turn, 200
        except Exception:
            LOGGER.exception("answer_pipeline_failed", extra={"thread_id": thread_id})
            diagnostics = error_diagnostics("pipeline_failed")
            turn = self.store.append_answer_turn(
                thread_id,
                question,
                _recovery_output(
                    question,
                    model=str(getattr(self.pipeline, "generation_model", "unknown")),
                    code="pipeline_failed",
                    diagnostics=diagnostics,
                ),
                corpus_hash=self.sources.corpus_hash,
            )
            return turn, 200


def _thread_payload(services: AppServices, thread_id: str) -> dict[str, Any]:
    return {
        "thread": services.store.get_thread(thread_id),
        "messages": services.store.list_messages(thread_id),
        "turns": services.store.list_turns(thread_id),
    }


def _sse(payload: Mapping[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _sse_comment(comment: str) -> str:
    return f": {comment}\n\n"


def create_app(services: AppServices) -> FastAPI:
    app = FastAPI(title="BankScope local API", version="0.2.0")
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            LOGGER.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/api/threads")
    def list_threads() -> dict[str, Any]:
        return {"threads": services.store.list_threads()}

    @app.post("/api/threads")
    def create_thread(body: ThreadRequest) -> dict[str, Any]:
        try:
            return services.store.create_thread(body.title)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/threads/{thread_id}/messages")
    def messages(thread_id: uuid.UUID) -> dict[str, Any]:
        try:
            return _thread_payload(services, str(thread_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Conversation not found.") from error

    @app.patch("/api/threads/{thread_id}")
    def rename_thread(thread_id: uuid.UUID, body: RenameThreadRequest) -> dict[str, Any]:
        try:
            return services.store.rename_thread(str(thread_id), body.title)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Conversation not found.") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/api/threads/{thread_id}", status_code=204)
    def delete_thread(thread_id: uuid.UUID) -> None:
        try:
            services.store.delete_thread(str(thread_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Conversation not found.") from error

    @app.post("/api/threads/{thread_id}/answers")
    def answer_thread(thread_id: uuid.UUID, body: QuestionRequest) -> JSONResponse:
        try:
            turn, status_code = services.answer_thread(str(thread_id), body.question)
            thread = services.store.get_thread(str(thread_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Conversation not found.") from error
        payload = {"thread": thread, "turn": turn}
        if status_code != 200:
            payload.update({"error": turn["error"], "code": turn["error_code"]})
        return JSONResponse(status_code=status_code, content=payload)

    @app.post("/api/threads/{thread_id}/stream")
    async def stream_thread(thread_id: uuid.UUID, body: QuestionRequest) -> StreamingResponse:
        try:
            services.store.get_thread(str(thread_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Conversation not found.") from error

        async def events():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            def progress(stage: str, details: Mapping[str, Any]) -> None:
                payload = {"type": "status", "stage": stage, **jsonable(dict(details))}
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            task = asyncio.create_task(
                asyncio.to_thread(
                    services.answer_thread,
                    str(thread_id),
                    body.question,
                    on_progress=progress,
                )
            )
            try:
                # Flush headers and a first event immediately. Agentic requests can otherwise
                # remain silent long enough for browsers or reverse proxies to drop the stream.
                yield _sse(
                    {
                        "type": "status",
                        "stage": "connected",
                        "message": "Connected to the answer service...",
                    }
                )
                while not task.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=10.0)
                    except TimeoutError:
                        yield _sse_comment("keep-alive")
                        continue
                    yield _sse(event)
                turn, status_code = await task
                if status_code == 200:
                    yield _sse({"type": "answer", "turn": turn})
                else:
                    yield _sse(
                        {
                            "type": "error",
                            "turn": turn,
                            "error": turn["error"],
                            "code": turn["error_code"],
                        }
                    )
                yield _sse({"type": "done", "status": status_code})
            except asyncio.CancelledError:
                # Persistence belongs to the worker and continues after a browser disconnect.
                task.add_done_callback(
                    lambda finished: finished.exception() if not finished.cancelled() else None
                )
                raise

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/citations/{citation_id}/context")
    def citation_context(
        citation_id: uuid.UUID, radius: int = Query(default=1, ge=0, le=3)
    ) -> JSONResponse:
        try:
            citation = services.store.get_citation(str(citation_id))
            context = services.sources.context(
                citation["target_chunk_id"],
                expected_corpus_hash=citation["corpus_hash"],
                radius=radius,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Citation source not found.") from error
        except StaleCitationError as error:
            return JSONResponse(
                status_code=409,
                content={"error": str(error), "code": "citation_corpus_mismatch"},
            )
        return JSONResponse(content={"citation": citation, **context})

    @app.post("/api/answer")
    def compatibility_answer(body: QuestionRequest) -> JSONResponse:
        try:
            with services.pipeline_lock:
                run = services.pipeline.answer(
                    body.question,
                    ticker=body.session_ticker,
                    tickers=body.session_tickers,
                )
            return JSONResponse(content=jsonable({**run.output, "evidence": run.evidence}))
        except GenerationValidationError as error:
            return JSONResponse(
                status_code=422,
                content={"error": GENERATION_ERROR_MESSAGE, "code": error.code},
            )
        except ValueError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        except Exception:
            LOGGER.exception("compatibility_answer_failed")
            return JSONResponse(
                status_code=500,
                content={"error": PIPELINE_ERROR_MESSAGE, "code": "pipeline_failed"},
            )

    return app
