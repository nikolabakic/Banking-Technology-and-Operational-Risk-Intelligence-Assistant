# ADR 014: Conversational function orchestration

- Status: Accepted
- Date: 2026-08-20

## Context

BankScope treated a chat message primarily as a fully specified retrieval command. Greetings and
product help worked only through a narrow whitelist, short natural continuations were easy to miss,
and a contextualizer or generation-schema failure became an HTTP error turn with no useful
assistant response. The language model was asked to emit JSON-shaped actions, while runtime
function boundaries already existed but were not exposed through native function calling.

The product must remain evidence-first for bank-specific claims. Making it conversational must not
allow assistant history to become evidence, give the model control over ticker filters or canonical
IDs, or let a query rewrite silently change the user's bank, period, metric, or numeric qualifiers.

## Decision

Every threaded API request first passes deterministic scope and ambiguity guards. Standalone
questions receive no conversation history. Only referential follow-ups call the thin conversational
orchestrator with at most the newest two completed pairs, represented as compact routing state
rather than prior assistant prose. It must select exactly one strict native function:

- `respond_directly` for greetings, acknowledgements, product help, or a
  bank-independent conceptual explanation;
- `decline_out_of_scope` for unrelated requests; its answer is rendered by the server and retrieval
  is forbidden;
- `ask_clarification` when one missing bank, period, metric, or intent would materially change
  filing research;
- `research_filings` with a standalone internal search question for any bank-, filing-, metric-,
  risk-, disclosure-, comparison-, or period-specific claim.

The full transcript remains available to the UI in SQLite, but research memory stores only user
questions plus compact assistant state. Prior answers, facts, numeric values, and citations are
never sent back as research memory. The window is bounded by turns, characters, and a conservative
token estimate; retryable-error and out-of-scope turns are excluded.

The current user message remains authoritative. A research rewrite is validated against user-owned
context and allowed bank scope. If it changes or invents scope, the pipeline searches with the
original message. If orchestration is unavailable, deterministic handling covers greetings,
acknowledgements, capabilities, and natural continuations; otherwise the original question takes
the safe filing-research path. Capability text is rendered from the server-owned registry.

The bounded agentic retrieval loop also uses native strict functions for hybrid search, literal
search, canonical context reads, finish, and the independent evidence verdict. Function names encode
the action, all arguments are required, additional properties are forbidden, parallel calls are
disabled, and runtime still validates periods, numeric facts, bank isolation, canonical target IDs,
and request/action budgets before executing anything.

Final grounded generation uses four mutually exclusive strict functions for supported numeric,
supported narrative, ambiguous, and unsupported results. Cross-field requirements are encoded in
the selected JSON schema. Every property is required, nullable values are explicit, and additional
properties and parallel calls are disabled. Truncation and contract-shape failures receive at most
one repair attempt. Unsupported display text is rendered locally.

An explicit but vague CET1 “indicator/measure” request is clarified as capital amount versus capital
ratio before retrieval. Retrieval does not depend on a single model rewrite. Focused questions
search the validated
standalone form, the original user wording, and deterministic bank-scoped terminology for common
filing concepts such as operational risk, cybersecurity, third-party risk, and CET1. Recovery and
out-of-scope turns remain visible in the transcript but are excluded from research memory.

Threaded generation, validation, and pipeline failures are represented as normal answered turns
with `dialog_act=retryable_error`, a user-facing retry/rephrase message, stable diagnostics, and HTTP
200. `dialog_act=clarification` is likewise an answered turn. The compatibility `/api/answer`
endpoint retains its prior HTTP error contract for non-chat callers.

## Consequences

- Users can greet, thank, ask about capabilities, request general explanations, use pronouns, or say
  “tell me more” without formatting a complete domain command.
- Filing-specific statements still pass deterministic bank resolution, retrieval, answer-schema,
  citation ownership, and support validation.
- A router failure reduces search quality gracefully instead of suppressing the assistant turn.
- A router failure cannot turn an unrelated request into filing retrieval.
- The UI can label conversation, clarification, grounded research, and recovery without inferring
  intent from citation count or HTTP status.
- Referential threaded requests add one orchestration model call; deterministic scope decisions and
  standalone memory isolation do not. Diagnostics expose action, latency, fallback, and error code.

## Verification

Unit tests cover every conversation and answer function, strict schema properties, deterministic follow-up
fallback, invalid rewrite fallback, no-retrieval direct/clarification/scope turns, native agentic tool
calls, truncation and schema-repair retry, multi-query retrieval, recovery/scope-memory exclusion,
per-bank comparison isolation, long-history topic switches, and API recovery
persistence. Frontend parsing and rendering tests cover the optional
`dialog_act` while preserving old stored turns that do not contain it.
