# Short-term conversation memory

**Status: accepted on 2026-08-13; selection and representation policy superseded by ADRs 013 and 014.**

## Context

BankScope retrieves evidence before generating an answer. Passing prior messages only to final
answer generation would leave follow-up queries such as “What about 2024?” too weak for reliable
retrieval. The SQLite chat store is already the authoritative source for server-owned thread state.

## Decision

- Load at most the four newest completed user/assistant pairs from the same thread, capped at
  12,000 characters. Never skip an oversized newest pair to include stale older pairs.
- Exclude incomplete and error turns. Deleting a thread cascade-deletes its messages.
- Rewrite follow-ups into one standalone question before bank resolution, embedding and hybrid
  retrieval. Use a deterministic JSON-mode request and validate its 4,000-character result with
  Pydantic.
- Preserve the current question's language and resolved entity, metric, period, approach and
  qualifiers. Explicit current-turn details override history.
- Strip prior `[E…]` markers. Assistant messages may resolve references but are never filing
  evidence. Final generation receives both original and resolved wording, while newly retrieved
  chunks remain its only factual evidence and citation source.
- Fail closed if contextualization is refused, truncated, malformed or unavailable, and persist an
  error turn instead of running context-blind retrieval.
- Keep `/api/answer` and direct pipeline callers stateless unless they pass
  `conversation_history`. Thread endpoint request shapes remain unchanged.
- Add no database migration, framework dependency, summary memory, or Responses API migration.

## Evaluation

The eight-case set in `data/evaluation/conversation_memory.jsonl` covers entity, metric, year,
qualifier and pronoun carryover, plus topic switching, bank switching and empty-history isolation.
`scripts/evaluate_conversation_memory.py` compares the current message with the contextualized
question using the pinned encoder, mixed hybrid retriever and frozen target-chunk judgments.

The accepted GPT-5.1 run on 2026-08-13 produced:

- stateless baseline Hit@5: **6/8**;
- contextualized Hit@5: **8/8**;
- rewrite-contract checks: **8/8**;
- isolation controls: **3/3**;
- overall gate: **pass**.

The gains were the metric-ellipsis and year-ellipsis cases; no case regressed. The generated result
is stored locally at `data/evaluation/results/conversation-memory-v1.json` and ignored by Git.

## Consequences

Every completed follow-up normally adds one model request before retrieval. First turns remain
unchanged. Memory survives refresh and restart but is strictly scoped to one SQLite thread. Four
turns and 12,000 characters are v1 bounds; summarization stays deferred until usage justifies it.

OpenAI conversation state and compaction remain possible migration paths, but do not replace the
need to contextualize BankScope's pre-generation retrieval query. This implementation stays on the
compatible Chat Completions client and validates structured output locally.

ADR 013 retains thread-isolated contextualization but makes it conditional on referential wording
and passes only the newest two completed pairs. The four-pair policy above remains the historical
evaluated v1 decision.

ADR 014 keeps the full transcript only for persistence/UI, while model routing sees compact
assistant state without prior answers, facts, numbers, or citations. Standalone questions receive
no history. Retryable-error and out-of-scope turns are excluded, and the compact window is bounded
by turns, characters, and an approximate token budget. The evaluation fixture also contains a long
mixed-topic thread to verify standalone topic isolation.
