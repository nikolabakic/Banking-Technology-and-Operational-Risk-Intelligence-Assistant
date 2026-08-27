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
from typing import Annotated, Any
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from bankscope.chat import ChatStore, CitationSourceResolver, StaleCitationError
from bankscope.generation.answer_generator import GenerationValidationError, question_language
from bankscope.generation.memory import CONVERSATION_SUMMARY_PROMPT_VERSION

LOGGER = logging.getLogger("bankscope.api")
GENERATION_ERROR_MESSAGE = "The model could not produce a valid grounded answer. Please try again."
PIPELINE_ERROR_MESSAGE = "The answer pipeline failed. Check the API terminal for details."
CANCELLED_ANSWER_STATUS_CODE = 499
MAX_WEB_CITATION_TITLE_LENGTH = 500
MAX_WEB_CITATION_SNIPPET_LENGTH = 4_000
MAX_WEB_CITATION_URL_LENGTH = 4_096


class InvalidWebCitationSourceError(ValueError):
    """Raised when persisted web-citation metadata cannot form a safe anchor."""


def _safe_web_citation_text(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_length].strip()


def _validated_web_citation_url(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidWebCitationSourceError("Web citation source URL is invalid.")
    source_url = value.strip()
    if (
        not source_url
        or len(source_url) > MAX_WEB_CITATION_URL_LENGTH
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127 or character == "\\"
            for character in source_url
        )
    ):
        raise InvalidWebCitationSourceError("Web citation source URL is invalid.")
    try:
        parsed = urlsplit(source_url)
        _ = parsed.port
    except ValueError as error:
        raise InvalidWebCitationSourceError("Web citation source URL is invalid.") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise InvalidWebCitationSourceError("Web citation source URL is invalid.")
    return source_url


def _web_citation_context(citation: Mapping[str, Any]) -> dict[str, Any]:
    raw_metadata = citation.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise InvalidWebCitationSourceError("Web citation metadata is invalid.")

    source_url = _validated_web_citation_url(raw_metadata.get("source_url"))
    title = _safe_web_citation_text(
        raw_metadata.get("title"), max_length=MAX_WEB_CITATION_TITLE_LENGTH
    )
    snippet = _safe_web_citation_text(
        raw_metadata.get("snippet"), max_length=MAX_WEB_CITATION_SNIPPET_LENGTH
    )
    citation_id = str(citation.get("id") or "")
    label = str(citation.get("label") or "")
    target_chunk_id = f"web:{citation_id}"

    safe_metadata = {
        "kind": "web",
        "citation_id": citation_id,
        "label": label,
        "source_url": source_url,
    }
    if title:
        safe_metadata["title"] = title
    if snippet:
        safe_metadata["snippet"] = snippet
    document_parts = [part for part in (title, snippet) if part]
    document_parts.append(f"Source: {source_url}")

    safe_citation = dict(citation)
    safe_citation["target_chunk_id"] = target_chunk_id
    safe_citation["metadata"] = safe_metadata
    return {
        "citation": safe_citation,
        "target_chunk_id": target_chunk_id,
        "record_type": "web",
        "ticker": "",
        "source_url": source_url,
        "corpus_hash": str(citation.get("corpus_hash") or ""),
        "chunks": [
            {
                "target_chunk_id": target_chunk_id,
                "role": "anchor",
                "record_type": "web",
                "document": "\n\n".join(document_parts),
                "metadata": safe_metadata,
            }
        ],
    }


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
    generation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Represent an expected pipeline failure as a normal assistant turn."""

    generation_metadata = dict(generation or {})
    request_count = _non_negative_count(generation_metadata.get("request_count")) or 0
    history_turns = _non_negative_count(diagnostics.get("history_turns")) or 0
    evidence_count = _non_negative_count(diagnostics.get("final_evidence_count")) or 0

    return {
        "question": question,
        "dialog_act": "retryable_error",
        "ticker": None,
        "contextualization": {
            "applied": False,
            "history_turns": history_turns,
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
        "retrieval": {"backend": "none", "mode": "none", "evidence_count": evidence_count},
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
            "model": str(generation_metadata.get("model") or model),
            "final_status": "unsupported",
            "request_count": request_count,
        },
        "diagnostics": dict(diagnostics),
    }


def _non_negative_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    count = int(value)
    return count if count >= 0 else None


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
        cancellation_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], int]:
        with self.pipeline_lock:
            return self._answer_thread_locked(
                thread_id,
                question,
                on_progress=on_progress,
                cancellation_event=cancellation_event,
            )

    def _answer_thread_locked(
        self,
        thread_id: str,
        question: str,
        *,
        on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], int]:
        def cancelled_result() -> tuple[dict[str, Any], int]:
            return {"state": "cancelled"}, CANCELLED_ANSWER_STATUS_CODE

        def is_cancelled() -> bool:
            return cancellation_event is not None and cancellation_event.is_set()

        if is_cancelled():
            return cancelled_result()

        thread = self.store.get_thread(thread_id)
        context = self.store.conversation_context(thread_id)
        summary_updated = False
        if context["needs_compaction"]:
            try:
                summary = self.pipeline.compact_conversation(
                    str(context["summary"]), context["compaction_messages"]
                )
                self.store.save_conversation_summary(
                    thread_id,
                    summary,
                    through_sequence=int(context["compaction_through_sequence"]),
                    prompt_version=CONVERSATION_SUMMARY_PROMPT_VERSION,
                )
                context = self.store.conversation_context(thread_id)
                summary_updated = True
            except Exception:
                LOGGER.exception("conversation_compaction_failed", extra={"thread_id": thread_id})
        stage_trace: list[dict[str, Any]] = []
        latest_stage: str | None = None

        def tracked_progress(stage: str, details: Mapping[str, Any]) -> None:
            nonlocal latest_stage
            if is_cancelled():
                return
            latest_stage = stage
            stage_trace.append({"stage": stage, "status": "started", **jsonable(dict(details))})
            if on_progress is not None:
                on_progress(stage, details)

        def persist_answer(output: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
            if is_cancelled():
                return cancelled_result()
            turn = self.store.append_answer_turn(
                thread_id,
                question,
                output,
                corpus_hash=self.sources.corpus_hash,
            )
            return turn, 200

        def error_diagnostics(
            code: str,
            failed_stage: str | None = None,
            generation: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            enabled = bool(getattr(self.pipeline, "agentic_rag_enabled", False))
            generation_request_count = _non_negative_count((generation or {}).get("request_count"))
            observed_evidence_count = next(
                (
                    count
                    for event in reversed(stage_trace)
                    if (count := _non_negative_count(event.get("evidence_count"))) is not None
                ),
                None,
            )
            history_message_count = len(context["messages"])
            history_turns = history_message_count // 2
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
                "initial_evidence_count": (observed_evidence_count if not enabled else None),
                "final_evidence_count": observed_evidence_count,
                "model_request_count": generation_request_count,
                "model_request_count_scope": "generation_only",
                "generation_request_count": generation_request_count,
                "history_turns": history_turns,
                "context_message_count": history_message_count,
                "context_estimated_tokens": int(context["estimated_tokens"]),
                "summary_used": bool(context["summary"]),
                "summary_updated": summary_updated,
                "bank_plans": [],
                "quality_gate": {"passed": False, "checks": checks},
            }
            validation_errors = (generation or {}).get("validation_errors")
            if isinstance(validation_errors, list):
                diagnostics["validation_errors"] = validation_errors
            citation_ids_received = (generation or {}).get("citation_ids_received")
            if isinstance(citation_ids_received, list):
                diagnostics["citation_ids_received"] = citation_ids_received[:20]
            return diagnostics

        if is_cancelled():
            return cancelled_result()

        try:
            run = self.pipeline.answer(
                question,
                ticker=thread["session_ticker"],
                tickers=thread["session_tickers"],
                conversation_history=context["messages"],
                conversation_summary=str(context["summary"]),
                previous_answer=context["previous_answer"],
                conversation_metadata={
                    "estimated_tokens": context["estimated_tokens"],
                    "summary_updated": summary_updated,
                    "unresolved_requests": context.get("unresolved_requests", []),
                },
                thread_id=thread_id,
                on_progress=tracked_progress,
            )
            return persist_answer(jsonable(run.output))
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
            return persist_answer(
                _recovery_output(
                    question,
                    model=str(getattr(self.pipeline, "generation_model", "unknown")),
                    code=error.code,
                    diagnostics=diagnostics,
                    generation=error.generation,
                )
            )
        except ValueError:
            LOGGER.info(
                "answer_request_recovered",
                extra={"thread_id": thread_id, "error_code": "invalid_request"},
            )
            diagnostics = error_diagnostics("invalid_request")
            return persist_answer(
                _recovery_output(
                    question,
                    model=str(getattr(self.pipeline, "generation_model", "unknown")),
                    code="invalid_request",
                    diagnostics=diagnostics,
                )
            )
        except Exception as error:
            LOGGER.exception("answer_pipeline_failed", extra={"thread_id": thread_id})
            raw_generation = getattr(error, "generation", None)
            generation = raw_generation if isinstance(raw_generation, Mapping) else None
            diagnostics = error_diagnostics("pipeline_failed", generation=generation)
            return persist_answer(
                _recovery_output(
                    question,
                    model=str(getattr(self.pipeline, "generation_model", "unknown")),
                    code="pipeline_failed",
                    diagnostics=diagnostics,
                    generation=generation,
                )
            )


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
            with services.pipeline_lock:
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
            cancellation_event = threading.Event()

            def progress(stage: str, details: Mapping[str, Any]) -> None:
                payload = {"type": "status", "stage": stage, **jsonable(dict(details))}
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            task = asyncio.create_task(
                asyncio.to_thread(
                    services.answer_thread,
                    str(thread_id),
                    body.question,
                    on_progress=progress,
                    cancellation_event=cancellation_event,
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
                elif status_code != CANCELLED_ANSWER_STATUS_CODE:
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
                cancellation_event.set()
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
            metadata = citation.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("kind") == "web":
                return JSONResponse(content=_web_citation_context(citation))
            if isinstance(metadata, Mapping) and metadata.get("source_kind") == "user_document":
                document_id = str(metadata.get("document_id") or "").strip()
                document = services.store.get_document(document_id)
                parsed_text = services.store.get_document_text(document_id)
                target_id = str(citation["target_chunk_id"])
                return JSONResponse(
                    content={
                        "citation": citation,
                        "target_chunk_id": target_id,
                        "record_type": "text",
                        "ticker": "UPLOAD",
                        "source_url": "",
                        "corpus_hash": citation["corpus_hash"],
                        "chunks": [
                            {
                                "target_chunk_id": target_id,
                                "role": "anchor",
                                "record_type": "text",
                                "document": parsed_text,
                                "metadata": {
                                    "source_kind": "user_document",
                                    "document_id": document_id,
                                    "filename": document["filename"],
                                    "section_title": document["filename"],
                                },
                            }
                        ],
                    }
                )
            context = services.sources.context(
                citation["target_chunk_id"],
                expected_corpus_hash=citation["corpus_hash"],
                radius=radius,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Citation source not found.") from error
        except InvalidWebCitationSourceError as error:
            return JSONResponse(
                status_code=422,
                content={"error": str(error), "code": "invalid_citation_source_url"},
            )
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

    @app.post("/api/documents/upload")
    async def upload_document(
        file: Annotated[UploadFile, File()],
        thread_id: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        """Upload a document to be stored in the database."""
        try:
            if not file.filename:
                raise ValueError("No filename provided.")

            content = await file.read()

            document = await run_in_threadpool(
                services.store.upload_document,
                thread_id=thread_id,
                filename=file.filename,
                content_type=file.content_type or "application/octet-stream",
                file_content=content,
                metadata={
                    "upload_filename": file.filename,
                    "content_type": file.content_type,
                },
            )

            return JSONResponse(content={"document": document})
        except ValueError as error:
            return JSONResponse(
                status_code=422,
                content={"error": str(error), "code": "invalid_document"},
            )
        except Exception:
            LOGGER.exception("document_upload_failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to upload document.", "code": "upload_failed"},
            )

    @app.get("/api/documents")
    def list_documents(thread_id: str | None = None) -> JSONResponse:
        """List all uploaded documents, optionally filtered by thread."""
        try:
            documents = services.store.list_documents(thread_id=thread_id)
            return JSONResponse(content={"documents": documents})
        except ValueError as error:
            LOGGER.exception("document_list_failed")
            return JSONResponse(
                status_code=422,
                content={"error": str(error), "code": "invalid_thread_id"},
            )
        except Exception:
            LOGGER.exception("document_list_failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to list documents.", "code": "list_failed"},
            )

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> JSONResponse:
        """Get document metadata."""
        try:
            document = services.store.get_document(str(document_id))
            return JSONResponse(content={"document": document})
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Document not found.") from error
        except Exception:
            LOGGER.exception("document_get_failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to get document.", "code": "get_failed"},
            )

    @app.get("/api/documents/{document_id}/content")
    def get_document_content(document_id: str) -> JSONResponse:
        """Get the raw content of a document."""
        try:
            content = services.store.get_document_content(str(document_id))
            document = services.store.get_document(str(document_id))
            return JSONResponse(
                content={
                    "content": content.decode("utf-8", errors="replace"),
                    "document": document,
                }
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Document not found.") from error
        except Exception:
            LOGGER.exception("document_content_failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to get document content.", "code": "content_failed"},
            )

    @app.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(document_id: str) -> None:
        """Delete a document."""
        try:
            services.store.delete_document(str(document_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Document not found.") from error
        except Exception as error:
            LOGGER.exception("document_delete_failed")
            raise HTTPException(status_code=500, detail="Failed to delete document.") from error

    return app
