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
from bankscope.generation.answer_generator import GenerationValidationError

LOGGER = logging.getLogger("bankscope.api")
GENERATION_ERROR_MESSAGE = "The model could not produce a valid grounded answer. Please try again."
PIPELINE_ERROR_MESSAGE = "The answer pipeline failed. Check the API terminal for details."


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
        try:
            with self.pipeline_lock:
                run = self.pipeline.answer(
                    question,
                    ticker=thread["session_ticker"],
                    on_progress=on_progress,
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
            turn = self.store.append_error_turn(
                thread_id,
                question,
                GENERATION_ERROR_MESSAGE,
                code=error.code,
            )
            return turn, 422
        except ValueError as error:
            turn = self.store.append_error_turn(
                thread_id, question, str(error), code="invalid_request"
            )
            return turn, 400
        except Exception:
            LOGGER.exception("answer_pipeline_failed", extra={"thread_id": thread_id})
            turn = self.store.append_error_turn(
                thread_id,
                question,
                PIPELINE_ERROR_MESSAGE,
                code="pipeline_failed",
            )
            return turn, 500


def _thread_payload(services: AppServices, thread_id: str) -> dict[str, Any]:
    return {
        "thread": services.store.get_thread(thread_id),
        "messages": services.store.list_messages(thread_id),
        "turns": services.store.list_turns(thread_id),
    }


def _sse(payload: Mapping[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


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
                while not task.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
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

        return StreamingResponse(events(), media_type="text/event-stream")

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
                run = services.pipeline.answer(body.question, ticker=body.session_ticker)
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
