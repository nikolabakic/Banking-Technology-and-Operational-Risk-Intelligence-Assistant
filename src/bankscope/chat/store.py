"""SQLite persistence for local BankScope conversations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
DEFAULT_THREAD_TITLE = "New conversation"
DEFAULT_MEMORY_MAX_TOKENS = 12_000
DEFAULT_MEMORY_MIN_RECENT_TURNS = 6


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _title_from_question(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 60 else f"{title[:57].rstrip()}..."


class ChatStore:
    """Small, versioned SQLite store using one connection per operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Chat database schema {version} is newer than supported {SCHEMA_VERSION}."
                )
            if version == 0:
                connection.executescript(
                    """
                    PRAGMA journal_mode = WAL;
                    CREATE TABLE chat_threads (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        session_ticker TEXT,
                        session_tickers_json TEXT NOT NULL DEFAULT '[]',
                        conversation_summary TEXT NOT NULL DEFAULT '',
                        summary_through_sequence INTEGER NOT NULL DEFAULT 0,
                        summary_prompt_version TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE chat_messages (
                        id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        status TEXT NOT NULL CHECK (status IN ('complete', 'error')),
                        content TEXT NOT NULL,
                        payload_json TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(thread_id, sequence)
                    );
                    CREATE INDEX ix_chat_messages_thread
                        ON chat_messages(thread_id, sequence);
                    CREATE TABLE message_citations (
                        id TEXT PRIMARY KEY,
                        message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
                        citation_index INTEGER NOT NULL,
                        label TEXT NOT NULL,
                        target_chunk_id TEXT NOT NULL,
                        corpus_hash TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        UNIQUE(message_id, citation_index)
                    );
                    CREATE INDEX ix_message_citations_message
                        ON message_citations(message_id, citation_index);
                    PRAGMA user_version = 3;
                    """
                )
                connection.commit()
            if version == 1:
                connection.execute(
                    "ALTER TABLE chat_threads ADD COLUMN "
                    "session_tickers_json TEXT NOT NULL DEFAULT '[]'"
                )
                connection.execute(
                    """UPDATE chat_threads
                       SET session_tickers_json = '["' || session_ticker || '"]'
                       WHERE session_ticker IS NOT NULL AND session_ticker != ''"""
                )
                connection.execute("PRAGMA user_version = 2")
                connection.commit()
                version = 2
            if version == 2:
                connection.execute(
                    "ALTER TABLE chat_threads ADD COLUMN "
                    "conversation_summary TEXT NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "ALTER TABLE chat_threads ADD COLUMN "
                    "summary_through_sequence INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute(
                    "ALTER TABLE chat_threads ADD COLUMN summary_prompt_version TEXT"
                )
                connection.execute("PRAGMA user_version = 3")
                connection.commit()

    @staticmethod
    def _thread(row: sqlite3.Row) -> dict[str, Any]:
        raw_tickers = json.loads(row["session_tickers_json"] or "[]")
        session_tickers = [
            str(value).strip().upper() for value in raw_tickers if str(value).strip()
        ]
        if not session_tickers and row["session_ticker"]:
            session_tickers = [str(row["session_ticker"]).strip().upper()]
        return {
            "id": row["id"],
            "title": row["title"],
            "session_ticker": row["session_ticker"],
            "session_tickers": session_tickers,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _conversation_state(self, thread_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT conversation_summary, summary_through_sequence,
                          summary_prompt_version
                   FROM chat_threads WHERE id = ?""",
                (thread_id,),
            ).fetchone()
        if row is None:
            raise KeyError(thread_id)
        return {
            "conversation_summary": str(row["conversation_summary"] or ""),
            "summary_through_sequence": int(row["summary_through_sequence"] or 0),
            "summary_prompt_version": row["summary_prompt_version"],
        }

    def create_thread(self, title: str | None = None) -> dict[str, Any]:
        normalized = (title or DEFAULT_THREAD_TITLE).strip()
        if not 1 <= len(normalized) <= 120:
            raise ValueError("title must contain between 1 and 120 characters.")
        now = _now()
        thread_id = str(uuid.uuid4())
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO chat_threads(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, normalized, now, now),
            )
            connection.commit()
        return self.get_thread(thread_id)

    def list_threads(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_threads ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._thread(row) for row in rows]

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM chat_threads WHERE id = ?", (thread_id,)
            ).fetchone()
        if row is None:
            raise KeyError(thread_id)
        return self._thread(row)

    def rename_thread(self, thread_id: str, title: str) -> dict[str, Any]:
        normalized = " ".join(title.split())
        if not 1 <= len(normalized) <= 120:
            raise ValueError("title must contain between 1 and 120 characters.")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE chat_threads SET title = ?, updated_at = ? WHERE id = ?",
                (normalized, _now(), thread_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(thread_id)
            connection.commit()
        return self.get_thread(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
            if cursor.rowcount == 0:
                raise KeyError(thread_id)
            connection.commit()

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"]) if row["payload_json"] else None
        return {
            "id": row["id"],
            "thread_id": row["thread_id"],
            "sequence": row["sequence"],
            "role": row["role"],
            "status": row["status"],
            "content": row["content"],
            "payload": payload,
            "created_at": row["created_at"],
        }

    def list_messages(self, thread_id: str) -> list[dict[str, Any]]:
        self.get_thread(thread_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY sequence",
                (thread_id,),
            ).fetchall()
        return [self._message(row) for row in rows]

    def list_turns(self, thread_id: str) -> list[dict[str, Any]]:
        messages = self.list_messages(thread_id)
        turns: list[dict[str, Any]] = []
        for index in range(0, len(messages), 2):
            user = messages[index]
            assistant = messages[index + 1] if index + 1 < len(messages) else None
            if user["role"] != "user":
                continue
            turn: dict[str, Any] = {
                "id": assistant["id"] if assistant else user["id"],
                "question": user["content"],
                "state": "error" if assistant and assistant["status"] == "error" else "answered",
                "created_at": user["created_at"],
            }
            if assistant and assistant["status"] == "error":
                turn["error"] = assistant["content"]
                turn["error_code"] = (assistant["payload"] or {}).get("code")
                turn["diagnostics"] = (assistant["payload"] or {}).get("diagnostics")
            elif assistant:
                turn["response"] = assistant["payload"]
            turns.append(turn)
        return turns

    def conversation_context(
        self,
        thread_id: str,
        *,
        max_tokens: int = DEFAULT_MEMORY_MAX_TOKENS,
        min_recent_turns: int = DEFAULT_MEMORY_MIN_RECENT_TURNS,
    ) -> dict[str, Any]:
        """Return raw model context and a compaction plan for one thread."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if min_recent_turns <= 0:
            raise ValueError("min_recent_turns must be positive.")

        memory_state = self._conversation_state(thread_id)
        messages = self.list_messages(thread_id)
        completed_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index in range(0, len(messages), 2):
            user = messages[index]
            assistant = messages[index + 1] if index + 1 < len(messages) else None
            if (
                user["role"] == "user"
                and assistant is not None
                and assistant["role"] == "assistant"
                and assistant["status"] == "complete"
            ):
                completed_pairs.append((user, assistant))

        checkpoint = int(memory_state["summary_through_sequence"] or 0)
        unsummarized = [
            pair for pair in completed_pairs if int(pair[1]["sequence"]) > checkpoint
        ]
        raw_messages = [
            {"role": message["role"], "content": str(message["content"])}
            for pair in unsummarized
            for message in pair
        ]
        summary = str(memory_state["conversation_summary"] or "")
        total_estimated_tokens = max(
            1,
            (len(summary) + sum(len(message["content"]) for message in raw_messages) + 3) // 4,
        )
        compact_pair_count = max(0, len(unsummarized) - min_recent_turns)
        compact_pairs = (
            unsummarized[:compact_pair_count]
            if total_estimated_tokens > max_tokens
            else []
        )
        compaction_messages = [
            {
                "role": message["role"],
                "content": str(message["content"]),
                "sequence": message["sequence"],
            }
            for pair in compact_pairs
            for message in pair
        ]
        recent_pairs = unsummarized[compact_pair_count:] if compact_pairs else unsummarized
        recent_messages = [
            {"role": message["role"], "content": str(message["content"])}
            for pair in recent_pairs
            for message in pair
        ]
        context_messages = raw_messages if not compact_pairs else recent_messages
        estimated_tokens = max(
            1,
            (
                len(summary)
                + sum(len(message["content"]) for message in context_messages)
                + 3
            )
            // 4,
        )
        previous_answer = None
        if completed_pairs:
            previous_user, previous_assistant = completed_pairs[-1]
            payload = previous_assistant.get("payload") or {}
            if str(payload.get("dialog_act") or "answer") in {
                "answer",
                "contextual_transform",
            }:
                previous_answer = {
                    "message_id": previous_assistant["id"],
                    "question": previous_user["content"],
                    "answer": previous_assistant["content"],
                    "citations": [dict(item) for item in payload.get("citations") or []],
                    "ticker": payload.get("ticker"),
                    "tickers": list(payload.get("tickers") or []),
                }
        return {
            "summary": summary,
            "summary_through_sequence": checkpoint,
            "summary_prompt_version": memory_state["summary_prompt_version"],
            "messages": context_messages,
            "estimated_tokens": estimated_tokens,
            "pre_compaction_estimated_tokens": total_estimated_tokens,
            "needs_compaction": bool(compact_pairs),
            "compaction_messages": compaction_messages,
            "compaction_through_sequence": (
                int(compaction_messages[-1]["sequence"]) if compaction_messages else checkpoint
            ),
            "previous_answer": previous_answer,
        }

    def save_conversation_summary(
        self,
        thread_id: str,
        summary: str,
        *,
        through_sequence: int,
        prompt_version: str,
    ) -> None:
        normalized = " ".join(summary.split())
        if not normalized:
            raise ValueError("conversation summary cannot be empty.")
        if through_sequence <= 0 or through_sequence % 2:
            raise ValueError("summary checkpoint must end on an assistant message.")
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE chat_threads
                   SET conversation_summary = ?, summary_through_sequence = ?,
                       summary_prompt_version = ?
                   WHERE id = ? AND summary_through_sequence < ?""",
                (normalized, through_sequence, prompt_version, thread_id, through_sequence),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT summary_through_sequence FROM chat_threads WHERE id = ?", (thread_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(thread_id)
                raise ValueError("conversation summary checkpoint did not advance.")
            connection.commit()

    def conversation_history(
        self,
        thread_id: str,
        *,
        max_tokens: int = DEFAULT_MEMORY_MAX_TOKENS,
    ) -> list[dict[str, str]]:
        """Backward-compatible raw history view used by older callers."""

        context = self.conversation_context(thread_id, max_tokens=max_tokens)
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in context["messages"]
        ]

    def append_answer_turn(
        self,
        thread_id: str,
        question: str,
        output: Mapping[str, Any],
        *,
        corpus_hash: str,
    ) -> dict[str, Any]:
        now = _now()
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response = dict(output)
        citations: list[dict[str, Any]] = []
        for raw in response.get("citations") or []:
            citation = dict(raw)
            citation_id = str(uuid.uuid4())
            citation["citation_id"] = citation_id
            citations.append(citation)
        response["citations"] = citations
        citations_by_label = {str(item.get("label") or ""): item for item in citations}
        bank_results: list[dict[str, Any]] = []
        for raw_result in response.get("bank_results") or []:
            result = dict(raw_result)
            result["citations"] = [
                citations_by_label.get(str(item.get("label") or ""), dict(item))
                for item in result.get("citations") or []
            ]
            bank_results.append(result)
        if bank_results:
            response["bank_results"] = bank_results

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            thread = connection.execute(
                "SELECT * FROM chat_threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if thread is None:
                raise KeyError(thread_id)
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM chat_messages WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, sequence, role, status, content, payload_json, created_at)
                   VALUES (?, ?, ?, 'user', 'complete', ?, NULL, ?)""",
                (user_id, thread_id, sequence + 1, question, now),
            )
            connection.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, sequence, role, status, content, payload_json, created_at)
                   VALUES (?, ?, ?, 'assistant', 'complete', ?, ?, ?)""",
                (
                    assistant_id,
                    thread_id,
                    sequence + 2,
                    str(response.get("answer") or ""),
                    _json(response),
                    now,
                ),
            )
            for index, citation in enumerate(citations, start=1):
                connection.execute(
                    """INSERT INTO message_citations
                       (id, message_id, citation_index, label, target_chunk_id,
                        corpus_hash, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        citation["citation_id"],
                        assistant_id,
                        index,
                        str(citation.get("label") or f"E{index}"),
                        str(citation.get("target_chunk_id") or ""),
                        corpus_hash,
                        _json(citation),
                    ),
                )
            title = thread["title"]
            if title == DEFAULT_THREAD_TITLE:
                title = _title_from_question(question)
            response_tickers = [
                str(value).strip().upper()
                for value in response.get("tickers") or []
                if str(value).strip()
            ]
            response_ticker = str(response.get("ticker") or "").strip().upper()
            if response_tickers:
                session_tickers = list(dict.fromkeys(response_tickers))
            elif response_ticker:
                session_tickers = [response_ticker]
            else:
                session_tickers = json.loads(thread["session_tickers_json"] or "[]")
                if not session_tickers and thread["session_ticker"]:
                    session_tickers = [str(thread["session_ticker"]).strip().upper()]
            ticker = session_tickers[0] if len(session_tickers) == 1 else None
            connection.execute(
                """UPDATE chat_threads
                   SET title = ?, session_ticker = ?, session_tickers_json = ?, updated_at = ?
                   WHERE id = ?""",
                (title, ticker, _json(session_tickers), now, thread_id),
            )
            connection.commit()
        return {
            "id": assistant_id,
            "question": question,
            "state": "answered",
            "response": response,
            "created_at": now,
        }

    def append_error_turn(
        self,
        thread_id: str,
        question: str,
        error: str,
        *,
        code: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            thread = connection.execute(
                "SELECT * FROM chat_threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if thread is None:
                raise KeyError(thread_id)
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM chat_messages WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, sequence, role, status, content, payload_json, created_at)
                   VALUES (?, ?, ?, 'user', 'complete', ?, NULL, ?)""",
                (user_id, thread_id, sequence + 1, question, now),
            )
            connection.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, sequence, role, status, content, payload_json, created_at)
                   VALUES (?, ?, ?, 'assistant', 'error', ?, ?, ?)""",
                (
                    assistant_id,
                    thread_id,
                    sequence + 2,
                    error,
                    _json({"code": code, "diagnostics": dict(diagnostics or {})}),
                    now,
                ),
            )
            title = thread["title"]
            if title == DEFAULT_THREAD_TITLE:
                title = _title_from_question(question)
            connection.execute(
                "UPDATE chat_threads SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, thread_id),
            )
            connection.commit()
        return {
            "id": assistant_id,
            "question": question,
            "state": "error",
            "error": error,
            "error_code": code,
            "diagnostics": dict(diagnostics or {}),
            "created_at": now,
        }

    def get_citation(self, citation_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM message_citations WHERE id = ?", (citation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(citation_id)
        return {
            "id": row["id"],
            "message_id": row["message_id"],
            "citation_index": row["citation_index"],
            "label": row["label"],
            "target_chunk_id": row["target_chunk_id"],
            "corpus_hash": row["corpus_hash"],
            "metadata": json.loads(row["metadata_json"]),
        }
