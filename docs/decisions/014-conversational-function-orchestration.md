# ADR 014: Conversational function orchestration

- Status: Superseded in part by ADR 015 (general-chat scope, recovery memory, and web/tool behavior)
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

Every threaded API request passes a once-compiled LangGraph workflow:
`prepare -> route -> validate_route`. `prepare` always runs deterministic bank resolution first and
adds positive bank/domain/source signals. The model then selects one strict `RouteDecision`; code
validates that directive before any handler runs. Negative regexes are not pre-model vetoes.

`RouteDecision` has one of five actions:

- `direct_response` for greetings, acknowledgements, product help, or a
  bank-independent conceptual explanation;
- `out_of_scope` for clearly unrelated requests; its answer is rendered by the server and retrieval
  is forbidden;
- `clarification` when one missing bank, period, metric, or intent would materially change
  filing research;
- `filing_research` with a standalone internal search question for any bank-, filing-, metric-,
  risk-, disclosure-, comparison-, or period-specific claim;
- `web_research` for current, market, news, price, or external facts. Until a provider is connected,
  its registered handler returns the stable `web_search_unavailable` contract and never substitutes
  filing retrieval.

Every threaded question receives the available token-bounded conversation context. The router sees
the current summary, raw messages after its checkpoint, the immediately previous grounded answer
with allowed citation labels, deterministic bank scope, and the current question. The model, rather
than a pre-filter, decides whether the request is standalone, referential, or a transformation.
LangGraph orchestration uses `langgraph`, `langchain-core`, and `langchain-openai`; the existing raw
OpenAI-compatible client remains the grounded answer generator. Both clients share the configured
proxy URL, API key, and custom `x-*` headers.

SQLite schema v3 stores a thread-scoped model summary, its assistant-message checkpoint, and prompt
version. Below an estimated 12,000-token budget the router receives the full raw transcript. Above
the budget, one strict summary call compacts older complete pairs while retaining at least the six
newest pairs verbatim. The summary preserves preferences, topics, unresolved questions, and
referents, but is never filing evidence and carries no old citation labels.

`direct_response` includes contextual transformations such as shortening, simplifying,
reformatting, or translating the immediately previous grounded answer. The router emits the final
text and a citation-label subset in the same call. Post-model validation rejects new citation
labels, numbers, banks, or factual qualifiers; no retrieval is performed for a safe transformation.
Research actions instead return a standalone `search_question` and optional presentation guidance.
Only newly retrieved filing evidence may support new bank facts.

The current user message remains authoritative. A research rewrite is validated against user-owned
context and allowed bank scope. If it changes or invents scope, the pipeline searches with the
original message. If orchestration is unavailable, deterministic handling covers greetings,
acknowledgements, capabilities, clearly unrelated requests, supported-bank/domain questions, and
natural continuations. Ambiguous residue receives a clarification instead of an automatic
rejection. Capability text is rendered from the server-owned registry.
`CONVERSATION_ROUTER_BACKEND=legacy` temporarily bypasses LangGraph execution while retaining the
same route schema and validation policy; it does not restore the old hard scope veto.

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

An explicit but vague CET1 “indicator/measure” request may be clarified by the semantic router;
there is no pre-model special case. Retrieval does not depend on a single model rewrite. Focused questions
search the validated
standalone form, the original user wording, and deterministic bank-scoped terminology for common
filing concepts such as operational risk, cybersecurity, third-party risk, and CET1. Recovery and
out-of-scope turns remain visible in raw conversational context but never become filing evidence.

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
- A high-precision fallback prevents explicitly unrelated requests from becoming filing retrieval;
  ambiguous cases clarify instead of being rejected.
- Bank-specific filing, risk, and metric signals override unsafe direct/out-of-scope model routes.
- Source handlers are registered independently of the router, so a web provider can be added without
  changing the HTTP/SSE entrypoint.
- The UI can label conversation, clarification, grounded research, and recovery without inferring
  intent from citation count or HTTP status.
- Routed chat requests add one orchestration model call. Compaction adds a call only when the
  12,000-token budget is crossed. Diagnostics expose context message/token counts, summary use and
  updates, route action, presentation guidance, citation source, fallback, and error code.

## Verification

Unit tests cover every route, strict schema properties, deterministic follow-up fallback, invalid
rewrite fallback, no-retrieval direct/clarification/scope/web-unavailable turns, native agentic tool
calls, truncation and schema-repair retry, multi-query retrieval, recovery/scope-memory exclusion,
per-bank comparison isolation, long-history topic switches, and API recovery
persistence. A frozen 45-case routing evaluation requires 100% supported-bank filing recall, 100%
no-retrieval behavior for unrelated requests, at least 95% overall route accuracy, and zero scope
preservation violations. Frontend parsing and rendering tests cover the optional
`dialog_act` while preserving old stored turns that do not contain it.
