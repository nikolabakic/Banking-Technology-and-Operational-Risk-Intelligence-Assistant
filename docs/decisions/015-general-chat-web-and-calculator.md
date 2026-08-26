# ADR 015: General chat with optional web and calculator tools

- Status: Accepted; conversational scope superseded by ADR 016
- Date: 2026-08-24

## Context

BankScope had become less reliable and less conversational after short-term memory was introduced.
The reported Ally question failed three times even though the indexed 2025 filing contained the
definition and retrieval returned the correct evidence. Two failures were first turns in new
threads, so conversation memory was not the cause of this incident. The generator exposed both a
short citation label such as `E4` and the evidence's internal `target_chunk_id` to the model. The
model cited the correct evidence using both forms; local validation accepted only `E\d+`, retried
the same ambiguous contract, and eventually surfaced `invalid_schema` as “Research paused safely.”

Memory still had two secondary problems. Retryable server boilerplate was reintroduced as normal
assistant context, and `previous_answer` looked only at the immediately preceding pair. Thus
`answer -> thanks -> make it shorter` lost the grounded answer that should have been transformed.

The conversational router also explicitly rejected recipes, travel, sports, creative work, and
other benign requests merely because they were not banking questions. It authored direct answers
inside a 700-token routing request, web search was permanently registered as unavailable, and the
calculator existed only on the roadmap.

## Reference implementations reviewed

The reviewed repositories support an optional-tool design rather than making retrieval a
precondition for every answer:

- [daveebbelaar/ai-cookbook](https://github.com/daveebbelaar/ai-cookbook) is the clearest small
  example of a model choosing search only when needed and otherwise responding directly. This is
  the main orchestration reference.
- [avrabyt/RAG-Chatbot](https://github.com/avrabyt/RAG-Chatbot) is useful as a minimal document-Q&A
  example, but its document-first flow and lack of meaningful conversation memory are too narrow
  for BankScope's front door.
- [bragai/bRAG-langchain](https://github.com/bragai/bRAG-langchain) collects useful retrieval
  experiments—multi-query, routing, fusion, reranking, corrective RAG, and self-RAG—but is not a
  production chatbot architecture.
- [run-llama/rags](https://github.com/run-llama/rags) demonstrates a tool-centric RAG builder, but
  its base prompt requires a tool for every answer. BankScope deliberately does not copy that
  behavior because it recreates the product's current over-routing problem.
- [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) is the strongest product/RAG reference:
  hybrid retrieval, reranking, citations, source inspection, decomposition, and replaceable flows.
  It remains document-centric, so BankScope adopts its evidence UX ideas rather than its entire
  conversational boundary.

## Web-search options

OpenAI Responses with the built-in `web_search` tool remains the preferred first provider when the
configured endpoint supports it. The official guides allow the model to decide whether search is
needed and return URL citations and consulted sources in the response. See
[OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search) and
[OpenAI conversation state](https://developers.openai.com/api/docs/guides/conversation-state).

This choice reuses the existing authenticated client and lets one provider call search and
synthesize a cited answer. The live gateway smoke on 2026-08-24 returned HTTP 404 for
`/responses`, however, so it cannot be BankScope's only runtime option. The implemented fallback is
[Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search): it is designed
for agent/RAG results, returns ranked source snippets, and exposes useful search controls. The
`auto` chain tries OpenAI first, falls back to Tavily when a key is present, and remembers the last
successful provider so a known-bad endpoint is not called on every turn. The next A/B candidate is
[Brave Search API](https://api-dashboard.search.brave.com/app/documentation/web-search), including
its LLM-oriented context endpoint. Serper is acceptable for inexpensive link discovery but would
require more application-owned synthesis and citation work. Bing Search API is not a candidate
because Microsoft retired it on 2025-08-11; Azure-first deployments would instead need a separate
Foundry grounding design.

## Decision

BankScope's front door is a general conversational assistant with three optional tools:

1. indexed filing research for supported-bank claims that require filing evidence;
2. web search for current, changing, or external facts;
3. a deterministic calculator for arithmetic.

The model may answer benign requests directly, regardless of topic. Product-domain mismatch is no
longer a reason to refuse. The router still protects source selection: a supported-bank filing
claim cannot bypass filing retrieval, and a current supported-bank claim cannot be presented as a
stable ungrounded answer. Direct responses remain a single model request; tool work happens only
when selected. The conversational output budget is configurable and defaults to 1,600 tokens.

The calculator parses a bounded Python expression AST and evaluates it with `Decimal`. It never
uses `eval` or `exec`, rejects names, calls, attributes, non-arithmetic syntax, excessive nesting,
large exponents, non-finite values, and division by zero.

Web search is injected through `WebSearchProvider`. Once the conversation router selects the web
route, the OpenAI provider calls
`client.responses.create(..., tools=[{"type": "web_search"}], tool_choice="required")`, extracts the answer and URL
annotations/sources, validates `http`/`https` URLs, deduplicates them, and returns a stable typed
result. The Tavily provider calls the official Search endpoint with basic search, answer, and up to
five ranked sources by default; source snippets become evidence anchors. Web citations are
persisted with `kind=web`; the browser opens their validated source URL
instead of sending them to the filing-corpus drawer. Disabled, unsupported, timeout, malformed,
and empty-result states keep a specific web-search error contract rather than becoming a generic
pipeline failure. Ordered fallback results also carry their actual provider-attempt count, so
runtime diagnostics do not under-report model or search API calls.

Filing evidence remains fail-closed. Internal target IDs are no longer shown to the answer model.
If a model nevertheless returns an exact ID belonging to the current evidence set, it is mapped to
the corresponding short label before schema validation; unknown IDs remain invalid.

Retryable recovery turns remain visible to the user but their assistant boilerplate is excluded
from semantic model history. Up to three bounded trailing failed user requests are kept separately;
duplicate resubmissions and phrases such as "try again" do not replace the canonical request.
Acknowledgements and clarifications preserve it until a substantive answer succeeds. The store
also scans backward for the latest substantive answer (including filing, web, calculation, and
general chat), allowing acknowledgements between an answer and a later shorten/translate/reformat
request. A thread request is serialized from context load
through generation and persistence so concurrent turns cannot both route from the same stale
memory snapshot. Browser stream cancellation sets a cooperative tombstone: an in-flight blocking
model request may finish, but neither its result nor any recovery response is persisted or reused
as later conversation memory.

## Configuration

Web search is controlled by:

```dotenv
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=auto
WEB_SEARCH_MODEL=
WEB_SEARCH_TIMEOUT_SECONDS=45
WEB_SEARCH_CONTEXT_SIZE=medium
TAVILY_API_KEY=
TAVILY_MAX_RESULTS=5
```

A blank `WEB_SEARCH_MODEL` uses `OPENAI_MODEL`. `auto` uses Tavily only when `TAVILY_API_KEY` is
present; explicitly selecting `tavily` without a key fails fast at startup. `--model` remains a
runtime override; without it, the API server now honors the configured `OPENAI_MODEL` instead of
silently selecting a hard-coded candidate.

## Consequences

- Simple questions and benign general requests receive normal answers without retrieval.
- Arithmetic is reproducible and does not depend on model mental math.
- Current facts can be answered with external citations through OpenAI Responses or Tavily.
- Bank-specific filing claims retain deterministic bank scope, canonical retrieval, strict answer
  validation, and citation ownership.
- The exact Ally failure no longer discards a semantically correct grounded response because of an
  internal citation ID.
- A gateway that lacks the Responses endpoint automatically falls through to configured Tavily;
  Brave remains a future measured alternative.
- If no configured provider succeeds, the explicit unavailable state is returned; the application
  never silently pretends an answer is current.

## Verification

Regression coverage includes the exact Ally question and primary target hash, unknown-ID
rejection, benign general chat with stale bank session state, deterministic percentage arithmetic,
calculator safety limits, current-fact web routing, provider response/source parsing, web citation
rendering, OpenAI-to-Tavily sticky fallback, recovery provenance, unresolved-request recovery,
memory exclusion, latest-grounded-answer lookup, concurrent deletion, and cancelled-stream
non-persistence. The frozen
conversation-routing set now treats recipes, cooking, sports rules, and travel planning as direct
chat; weather and recent sports results as web research; and percentage arithmetic as calculator
work.
