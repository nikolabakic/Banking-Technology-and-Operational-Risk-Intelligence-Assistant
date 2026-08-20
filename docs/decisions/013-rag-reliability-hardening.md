# ADR 013: Deterministic RAG reliability boundaries

- Status: Accepted for the baseline path; agentic default remains gated
- Date: 2026-08-20
- Supersedes: ADR 007's always-on four-turn rewrite policy, ADR 008's shared comparison query, and
  ADR 012's current 6/4/2 agent budget

## Context

Real UI queries exposed four coupled failures: possessive bank spelling variants did not resolve,
cross-bank wording diluted per-bank retrieval, unrelated history contaminated new questions, and
malformed stream/persisted payloads could crash React. The optional agentic loop also ran a model
router before deterministic domain resolution and could discard baseline evidence.

## Decision

Bank identity stays deterministic. Multi-token and sufficiently specific single-token identifiers
accept omitted-apostrophe possessives, but short aliases keep strict word boundaries. A comparison
is decomposed into one peer-free query per ticker; retrieval and generation remain isolated until
validated results are synthesized.

Contextualization runs only for referential follow-ups and sees at most the newest two complete
user/assistant pairs. A rewrite must preserve explicit years/numeric qualifiers and cannot import
facts from assistant prose or banks outside user-authored thread scope. Standalone topic and bank
switches bypass memory.

Whole-filing summaries run five fixed, section-diverse searches and merge results round-robin.
Agentic retrieval remains optional and additive after baseline retrieval. It cannot remove baseline
evidence and is limited per bank to three orchestration requests, one tool action, and one verifier
request. The rollout evaluator therefore preserves baseline Top 5 and measures recovery in Top 10.

SSE endpoints emit an immediate status, periodic heartbeat comments, and proxy no-buffer headers.
The TypeScript client validates REST/SSE payloads and tolerates fragmented or malformed events.
Rendering uses defensive diagnostics and citation access plus a top-level recovery boundary.

## Consequences

Simple questions avoid an unnecessary contextualization request. Comparisons and full-filing
summaries perform more retrieval calls but each call has a clearer scope. The agentic experiment is
cheaper and cannot degrade the evidence ordering of the baseline path, but it remains disabled by
default until a fresh live quality report passes. See
[the hardening report](../reliability-hardening.md) for failure details and source research.

## Live evidence

The 2026-08-20 authorized live gate failed. Baseline Hit@5 was fully preserved and two genuine
baseline Top-10 misses were recovered, below the required three. Invalid structured action or
verifier responses affected 12 of 15 bank plans, and the unsupported control ended in a generation
schema error instead of a controlled abstention. The default therefore remains disabled. The same
run identified and fixed evaluator over-counting, request double-counting, and an outdated
multi-bank evaluation path; a new live decision requires a fresh report after schema reliability is
hardened.
