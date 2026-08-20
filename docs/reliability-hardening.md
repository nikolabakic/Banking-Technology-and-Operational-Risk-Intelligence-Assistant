# RAG reliability hardening

**Status:** implemented on 2026-08-20; live model quality gates remain required before enabling
agentic RAG by default.

This change set addresses the reported bank-resolution, multi-bank retrieval, conversation-memory,
whole-filing summary, and blank-screen failures. It is based on local failure reproduction plus
patterns from maintained open-source RAG projects.

## Failure analysis and implemented controls

| Symptom | Root cause | Control |
|---|---|---|
| `JP Morgans` or `JPMorgans` failed to resolve | Alias matching understood apostrophe possessives but not an omitted apostrophe before `s` | Multi-token and sufficiently specific single-token identifiers accept the normalized possessive form while short aliases retain strict boundaries |
| A comparison mixed or weakened evidence | Every bank embedded the same cross-bank comparison sentence | The planner removes all peer-bank identifiers, creates one bank-owned subquestion per ticker, and independently embeds, retrieves, generates, then synthesizes |
| Follow-ups degraded after several turns | Every question was rewritten from full prior answers, including standalone topic/bank switches | Only referential follow-ups receive history; assistant prose/facts/citations are replaced by compact routing state, standalone questions receive no history, and period, numeric-fact, and bank-scope changes are rejected |
| A recipe was answered and searched against JPM filings | The router allowed out-of-scope prose and its failure fallback defaulted every unknown message to filing research | A deterministic scope guard and strict `decline_out_of_scope` action run before retrieval; refusal text is server-owned and declined turns are excluded from research memory |
| Repeated CET1 comparisons failed with root `invalid_schema` | One strict function still relied on cross-field Pydantic rules that JSON Schema did not encode | Four mutually exclusive answer functions encode numeric, narrative, ambiguous, and unsupported contracts; one repair retry is allowed and comparison failures are isolated per bank |
| A broad 10-K summary saw only generic Top-5 evidence | One generic query could not cover a whole filing | Whole-document summaries use five bounded section-oriented searches and round-robin evidence merging |
| Agentic mode could be slower and less reliable than baseline | The router ran before deterministic resolution; the loop allowed 6/4/2 requests/actions/verifications; an `unsupported` verdict could erase valid baseline evidence | Normal banking traffic always enters deterministic resolution, agentic retrieval is additive, baseline evidence remains first, and the per-bank budget is 3/1/1 |
| The UI sometimes became a blue/blank screen | Fragmented or malformed SSE/legacy diagnostics could reach unchecked React rendering; streams were silent during long model work | The API flushes a status event, sends heartbeat comments and no-buffer headers; the client validates payloads and tolerates fragmented SSE; diagnostics are defensive and a top-level error boundary offers reload recovery |

## External implementation patterns

- LangChain's official [RAG research agent template](https://github.com/langchain-ai/rag-research-agent-template)
  turns a question into a small research plan, runs targeted queries, and synthesizes only from
  retrieved documents. Its [retrieval graph](https://raw.githubusercontent.com/langchain-ai/rag-research-agent-template/main/src/retrieval_graph/graph.py)
  and [researcher graph](https://raw.githubusercontent.com/langchain-ai/rag-research-agent-template/main/src/retrieval_graph/researcher_graph/graph.py)
  motivated explicit per-bank subquestions and bounded, diverse summary searches.
- LlamaIndex's [SubQuestionQueryEngine](https://github.com/run-llama/llama_index/blob/9ba74b8628712e68d16955d9492b5192bd7e6f00/llama-index-core/llama_index/core/query_engine/sub_question_query_engine.py)
  follows the same decompose-then-synthesize boundary. Reported
  [multi-document comparison failures](https://github.com/run-llama/llama_index/discussions/16323)
  reinforce keeping document/bank stores separate rather than retrieving from one mixed pool.
- RAGFlow issue [#7885](https://github.com/infiniflow/ragflow/issues/7885) documents how combined
  multi-topic queries dilute hybrid-retrieval relevance and how separate targeted queries improve
  results. Its [RAG documentation](https://github.com/infiniflow/ragflow/blob/main/docs/basics/rag.md)
  also separates static document evidence from conversational memory.
- LangGraph's [memory guidance](https://github.com/langchain-ai/langgraphjs/blob/main/docs/docs/concepts/memory.md)
  warns that long histories distract models with stale or off-topic content and recommends trimming
  or summarization. BankScope trims to two pairs and skips rewriting standalone questions.
- Open WebUI issue [#16136](https://github.com/open-webui/open-webui/issues/16136) shows malformed
  SSE JSON causing a browser UI failure, while [#16747](https://github.com/open-webui/open-webui/issues/16747)
  records idle stream timeouts during long-running tools. These informed defensive SSE parsing,
  immediate events, heartbeats, and the React error boundary.

## Deliberately avoided pitfalls

- The LLM never chooses a ticker or expands bank scope.
- Prior assistant prose never becomes filing evidence.
- Out-of-scope requests never reach embedding, retrieval, or grounded generation.
- Multi-bank evidence is never pooled before each bank has a validated result.
- Agentic actions cannot replace or reorder validated baseline Top-5 evidence.
- Malformed transport or historical payloads cannot be rendered as trusted typed data.
- Agentic RAG remains `false` by default until the frozen and live gates pass.

## Verification contract

Run the complete Python suite and Ruff checks, then frontend lint, Vitest, and the production build.
The live `evaluate_agentic_rag.py` comparison is intentionally separate because it uses the model
gateway and local full corpus. It must preserve every baseline Hit@5, recover at least three known
baseline Top-10 misses in the additive Hit@10 window, pass runtime isolation/budget checks, avoid
needless expansion, and produce citation-free abstention for unsupported questions.

## Local verification result

The final deterministic gate on 2026-08-20 passed:

- Python: `216 passed` (one existing Starlette/httpx deprecation warning);
- Ruff lint and format checks: passed;
- frontend ESLint: passed;
- frontend Vitest: `15 passed` across two files;
- TypeScript and Vite production build: passed.

## Live agentic result

The explicitly authorized full-corpus/model run on 2026-08-20 executed all 12 baseline and agentic
pairs but **failed the rollout gate**:

- no baseline Hit@5 was lost;
- two genuine baseline Top-10 misses were recovered: Ally operational risk and Truist's incident
  response team (the raw report initially counted PNC/TFC as a third, but that row was already a
  baseline Top-10 hit; the evaluator has been corrected);
- the required minimum remains three genuine recoveries;
- 9/12 agentic rows failed the runtime contract and 12/15 per-bank plans encountered at least one
  invalid action/verifier schema response;
- the sufficient-initial-evidence case correctly used no extra tool action;
- the unsupported 2035 crypto-reserve case ended with `invalid_schema`, not a controlled
  `unsupported` result, although it emitted zero citations.

The run also exposed and fixed double-counted validation requests and an outdated multi-bank
retrieval-only path that had bypassed peer-free query decomposition. Those corrections improve the
next measurement but do not retroactively turn this run into a pass. Agentic RAG remains `false` by
default; another paid live run is justified only after improving structured action/verifier output
reliability.

A subsequent UI transcript exposed the compact spelling `JPMorgans`. It had resolved only the
explicit BAC peer, sending a cross-bank question into BAC's single-bank generator and correctly
failing schema validation. Resolver and query-decomposition variants now cover the compact spelling
without enabling the false-positive `Chase` → `chases` match.
