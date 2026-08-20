# Conversation state and citation sources

**Status:** active local persistence layer.

```mermaid
sequenceDiagram
    participant UI as React UI
    participant API as FastAPI
    participant DB as ChatStore / SQLite
    participant P as Answer pipeline
    participant S as CitationSourceResolver
    UI->>API: stream question for thread
    API->>DB: load completed same-thread history
    API->>P: answer(question, history, session scope)
    P-->>API: answer, clarification, direct response, or safe recovery
    API->>DB: atomically persist turn and citation metadata
    API-->>UI: answer/error SSE event
    UI->>API: open citation
    API->>DB: load citation metadata
    API->>S: resolve against current corpus hash
    S-->>UI: canonical table or narrative context
```

## Files and public API

| File | Public symbols | Responsibility |
|---|---|---|
| `store.py` | `ChatStore` | Initialize/migrate schema; manage threads; group turns; bound history; atomically persist answer/error turns; load citations |
| `sources.py` | `CitationSourceResolver`, `StaleCitationError` | Index corpus targets and reopen canonical narrative or complete-table evidence |

`ChatStore` exposes thread CRUD, `list_messages()`, `list_turns()`, bounded
`conversation_history()`, `append_answer_turn()`, `append_error_turn()`, and `get_citation()`.
Answered filing turns update the server-owned single/comparison bank scope. Direct responses,
clarifications, and recovery turns retain the existing scope. The conversation planner uses only
the newest two compact completed pairs for referential follow-ups; standalone questions receive no
history. Direct and recovery turns do not displace research context, while an out-of-scope turn
creates a barrier so a later pronoun cannot jump back to stale research.
Expected model and pipeline failures are stored as normal answered `retryable_error` turns, so the
user receives a conversational response and may retry or continue. The older error-turn contract
remains available for infrastructure-level failures. Diagnostics live in the existing
`payload_json`; no agentic-specific SQLite table or schema migration is required.

`CitationSourceResolver.context()` validates the citation's corpus hash. Text citations return an
anchor plus neighboring narrative chunks; table citations return the complete table. Stale or
missing targets fail closed.

## Invariants and changes

- SQLite transactions keep each user/assistant turn and its citations consistent.
- History is thread-isolated, bounded by turns/characters/estimated tokens, chronological, and
  limited to completed research or clarification turns.
- Stored assistant answers remain in SQLite for the UI, but model memory contains only compact
  dialog state without answer prose, facts, values, or citations.
- The original current message is authoritative; a contextualized search query is disposable.
- SQLite stores citation metadata, not a second canonical corpus.
- Runtime execution checks are persisted for observability but do not claim factual correctness.

Run `tests/test_chat_store.py`, `tests/test_chat_sources.py`, contextualizer tests, and frontend API
tests after changes.

[Package architecture](../README.md) · [Local data](../../../data/local/README.md)
