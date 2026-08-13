# BankScope roadmap

The repository-overhaul phase was accepted on 2026-08-06 after the whole-table
corpus, full test suite and BM25 evaluation were recorded. The complete
dense/BM25/hybrid baseline and Qdrant comparison was recorded on 2026-08-10.

| Phase | Status | Exit condition |
|---|---|---|
| Registry and SEC acquisition | Complete | Ten configured banks and a reproducible filing manifest |
| Parser selection | Complete | sec2md selected from recorded evaluation evidence |
| Repository overhaul | Complete | Five active commands, legacy isolated, tests and lint green |
| Whole-table corpus | Complete | One stored table and at most one description per parser-emitted table |
| Embeddings | Implemented | Length/order/hash/model contracts and real smoke run pass |
| Retrieval evaluation | Complete | Frozen 30-question baseline/Qdrant comparison and backend decision recorded |
| Answer generation | Implemented | Single-bank answers use hydrated evidence, abstain and expose citations |
| Generation evaluation | Complete | Reusable 26-question evaluator and first baseline record deterministic metrics, advisory judge results and two explicit query errors |
| Generation hardening | Known citation caveat | Answer checks pass; citation audit is 24/25, the CLI default is unchanged and the UI API uses the validated GPT-5.1 candidate |
| Automatic bank resolution | Complete | Names, aliases and tickers resolve before retrieval; missing/multiple banks fail locally |
| Durable local conversations | Complete | SQLite threads survive refresh/restart and retain server-owned bank context |
| Short-term conversation memory | Implemented | Four bounded, completed turns from the current SQLite thread contextualize retrieval; the 8-case gate improved Hit@5 from 6/8 to 8/8 with all isolation controls passing |
| Streaming progress | Complete | The UI receives real bank-resolution, embedding, retrieval, generation and validation stages |
| Citation context | Complete | Persisted citations reopen canonical narrative/table evidence and fail closed after corpus changes |
| Local product interface | Complete | Routed chat history supports create, reopen, rename, delete and source inspection |
| Experimentation foundation | Implemented | FastAPI contracts, structured local logs and isolated quality gates support future experiments |
| Multi-bank questions | Future consideration | One question can explicitly select two or more indexed banks, retrieve evidence independently for each bank, produce a clearly structured comparison or synthesis, and preserve bank-specific citations without mixing evidence between entities |
| External tools and function calling | Future consideration | Assess web search, calculator and document lookup integrations to extend the assistant's capabilities |

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

## Scope boundary

The active product remains a local single-user student RAG assistant for ten
banks. There is no deployment or final-report deadline. New ideas are added as
isolated experiments and become defaults only when their own tests and relevant
evaluation gates justify the change. Authentication, multi-user infrastructure
and cloud persistence remain outside the active scope.

Multi-bank questions are not part of the current single-bank generation path.
The future experiment should support prompts such as comparing CET1 ratios or
operational-risk disclosures across several named banks in one request. Bank
resolution, retrieval and evidence validation must remain isolated per bank;
the final response may compare or synthesize the results only after every claim
can be traced to citations belonging to the correct bank.
