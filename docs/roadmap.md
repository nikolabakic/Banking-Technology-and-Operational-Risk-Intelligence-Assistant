# BankScope roadmap

The repository-overhaul phase was accepted on 2026-08-06 after the whole-table
corpus, full test suite and BM25 evaluation were recorded. The complete
dense/BM25/hybrid baseline and Qdrant comparison was recorded on 2026-08-10.

| Phase | Status | Exit condition |
|---|---|---|
| Registry and SEC acquisition | Complete | Ten complete primary filings; COF/STT replace partial USB/WFC filings |
| Parser selection | Complete | sec2md selected from recorded evaluation evidence |
| Repository overhaul | Complete | Five active commands, legacy isolated, tests and lint green |
| Whole-table corpus | Complete | One stored table and at most one description per parser-emitted table |
| Embeddings | Complete | Pinned Qwen archive contains 6,550 ordered, normalized vectors and matches the corpus hash |
| Retrieval evaluation | Complete | Balanced mixed retrieval passes 31/32 Top 5, 32/32 Top 10 and every grouped-evidence gate |
| Answer generation | Implemented | Single-bank answers use hydrated evidence, abstain and expose citations |
| Generation evaluation | Complete | GPT-5.1 passes 30/30 questions, including all numeric, narrative, variant and citation-contract gates |
| Generation hardening | Complete | The current manual audit accepts all 29 supported answers and the UI API uses the validated GPT-5.1 candidate |
| Automatic bank resolution | Complete | Names, aliases and tickers resolve before retrieval; missing/multiple banks fail locally |
| Durable local conversations | Complete | SQLite threads survive refresh/restart and retain server-owned bank context |
| Model-first conversation memory | Hardened | Every threaded turn receives bounded raw context; long threads use a summary plus six newest pairs, and prior grounded answers can be safely transformed |
| Streaming progress | Complete | The UI receives real bank-resolution, embedding, retrieval, generation and validation stages |
| Citation context | Complete | Persisted citations reopen canonical narrative/table evidence and fail closed after corpus changes |
| Local product interface | Complete | Routed chat history supports create, reopen, rename, delete and source inspection |
| Experimentation foundation | Implemented | FastAPI contracts, structured local logs and isolated quality gates support future experiments |
| Multi-bank questions | Hardened | Questions select 2-4 banks, build a peer-free subquestion for each bank, retrieve and validate independently, then synthesize with bank-owned citations |
| Reliability hardening | Implemented | Possessive aliases, diversified whole-filing summaries, guarded SSE payloads, heartbeats, and UI recovery have regression coverage |
| Optional agentic RAG | Experimental, live gate failed | The 2026-08-20 run preserved baseline Top 5 but recovered only 2/3 required genuine Top-10 misses, had widespread schema fallback, and failed controlled unsupported handling; default remains off |
| General chat and optional tools | Implemented | Benign requests answer directly; filing research, cited web search, and a safe Decimal calculator run only when selected |

## Acceptance gates

Before changing the active parser, chunking, table descriptions, embedding
model or retrieval method:

1. state the concrete problem;
2. make the smallest isolated change;
3. run unit/integrity checks;
4. run the frozen evaluation set;
5. keep the change only with a documented result and caveat.

The previous sec2md v3 hybrid result (`Hit@1 12/28`, `Hit@5 23/28`,
`MRR@10 0.589`) is the comparison point for the new simpler table design. A
regression is not hidden: it must either be fixed or explicitly accepted for a
clear simplicity benefit.

The accepted overhaul result and its compute caveat are recorded in
`docs/decisions/002-repository-overhaul.md`. The Qdrant implementation and its
measured full-backend result are recorded in
`docs/decisions/003-qdrant-local-retrieval.md`. The requirement-driven decision
to use Qdrant dense search with BM25S and application RRF is recorded in
`docs/decisions/004-mixed-vector-retrieval.md`. The current single-bank
generation-evaluation scope and its first measured baseline are recorded in
`docs/decisions/005-generation-evaluation.md`. The locally implemented hardening
candidate and its still-pending approval gate are described in
`docs/generation_hardening.md`. Automatic bank resolution and session fallback are
recorded in `docs/decisions/006-automatic-bank-resolution.md`.
Short-term conversation memory and its measured retrieval gate are recorded in
`docs/decisions/007-short-term-conversation-memory.md`.
The bounded multi-bank implementation and its accepted live evaluation gate are recorded in
`docs/decisions/008-multi-bank-comparisons.md`.
The complete-primary-filing replacement, GPU handoff and 34-question rebaseline are recorded in
`docs/decisions/009-complete-primary-filings.md`.
The accepted bank-balanced retrieval path and final ten-bank evaluation hashes are recorded in
`docs/decisions/010-bank-balanced-comparison-retrieval.md`.
The current resolver, comparison-query, selective-memory, additive-agentic, and UI transport
boundaries are recorded in `docs/decisions/013-rag-reliability-hardening.md`; supporting GitHub
research and reproduced failures are summarized in `docs/reliability-hardening.md`.
The Ally citation-schema repair, general-chat boundary, calculator, web-provider comparison, and
OpenAI/Tavily web-search decision are recorded in
`docs/decisions/015-general-chat-web-and-calculator.md`.

## Scope boundary

The active product remains a local single-user student RAG assistant for ten
banks. There is no deployment or final-report deadline. New ideas are added as
isolated experiments and become defaults only when their own tests and relevant
evaluation gates justify the change. Authentication, multi-user infrastructure
and cloud persistence remain outside the active scope.

Multi-bank questions support two to four configured banks. Comparisons above that bound,
unconfigured banks, authentication, multi-user infrastructure and cloud persistence remain outside
the active scope.
