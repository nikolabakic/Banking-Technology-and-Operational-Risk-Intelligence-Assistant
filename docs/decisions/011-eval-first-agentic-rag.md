# ADR 011: Eval-first bounded agentic RAG

- Status: Superseded by ADR 012; experimental feature remains disabled by default
- Date: 2026-08-17

## Context

The existing retrieval path combines Qdrant dense retrieval and BM25S with RRF and has frozen retrieval, generation, memory, and comparison gates. The current corporate OpenAI-compatible gateway supports JSON-mode chat completions but did not expose a reliable Responses API or function-calling loop during the capability probe. An unrestricted agent or a filesystem-search replacement would therefore add risk without preserving the validated retrieval behavior.

The useful pattern from the reviewed agentic-RAG cookbook is its orchestration shape: search first, assess the evidence, and perform a bounded read or search only when needed. BankScope needs that pattern with bank, filing, request, and action isolation.

## Decision

BankScope keeps mixed Qdrant + BM25S + RRF as its initial retrieval. When `AGENTIC_RAG_ENABLED=true`, a structured JSON router first distinguishes narrow general product chat from domain RAG. After initial retrieval, a Pydantic-validated evidence planner independently chooses `generate`, `rewrite_search`, `expand_context`, or `abstain` for each bank.

The planner has a fixed v1 budget of one additional action per bank. Rewrite uses the same hybrid retrieval and fixed ticker/record filters. Expansion accepts only an initial narrative result as its anchor and reads at most the previous and next canonical chunks from the same ticker and accession. Tables remain whole and are not expanded. There is no second planning round and no arbitrary tool, filesystem, path, or shell access.

Every turn exposes and persists diagnostics in the existing message `payload_json`, including failures. Diagnostics are execution-contract checks, not a claim of factual correctness. Offline evaluation against qrels remains the factual quality gate.

## Acceptance and rollout

The default remains `AGENTIC_RAG_ENABLED=false`. `scripts/evaluate_agentic_rag.py` compares baseline and agentic runs over the 12-question annotated challenge set. Enabling by default requires all existing frozen gates plus the agentic gate: no lost baseline Hit@5, at least two recovered baseline misses, all plan/scope/budget/isolation contracts, no unnecessary action for sufficient initial evidence, and citation-free abstention for unsupported evidence.

If the gate fails, the implementation remains available for iteration but disabled. Changing the default requires a separate ADR with a recorded passing report.

## Consequences

The approach adds two model requests in the normal enabled single-bank domain path (route and evidence assessment), and independent assessments for comparisons. Latency and cost are visible in diagnostics and evaluation. The constrained design cannot perform iterative research, web search, attachments, long-term-memory lookup, reranking, or generic tools; those remain separate future decisions.

## First live gate result

The first representative live gate ran on 2026-08-17 with the same
`AZURE_GPT_51_2025_1113` candidate used by the application. The feature remains disabled.

- 20 of 24 baseline/agentic executions completed; one agentic plan failed schema validation and
  one comparison failed generation validation in both modes. The unsupported baseline also failed
  generation validation, while agentic mode correctly abstained without citations.
- Agentic mode preserved every completed baseline Hit@5 and recovered the annotated ALLY rewrite
  miss, but recovered only one miss; acceptance requires at least two.
- The sufficient-initial-evidence case correctly chose `generate` without extra retrieval.
- BAC context expansion used the expected initial narrative anchor and stayed within scope.
- Two comparison rewrites introduced numeric peer values that were not present in the original
  question. Fixed ticker filters prevented cross-bank evidence retrieval, but the v1 runtime
  `query_preservation` check did not detect this semantic mutation.

The recorded gate therefore failed `all_runs_executed`,
`at_least_two_baseline_misses_recovered`, and `all_runtime_contracts_pass`. Before another rollout
run, rewrite validation must reject newly introduced numeric facts and planner schema reliability
must be improved. No default change is authorized by this result.
