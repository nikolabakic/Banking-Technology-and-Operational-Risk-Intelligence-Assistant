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
| Generation hardening | Known citation caveat | Answer checks pass; citation audit is 24/25 and the default remains unchanged |
| Automatic bank resolution | Complete | Names, aliases and tickers resolve before retrieval; missing/multiple banks fail locally |
| Conversation history | Pending | Follow-ups inherit the resolved session bank without contaminating retrieval |
| Simple user interface | Pending | Local chat flow is usable by a reviewer |
| External tools and function calling | Future consideration | Assess web search, calculator and document lookup integrations to extend the assistant's capabilities |
| Final report | Pending | Retrieval and generation results are reported separately |

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
`docs/generation_hardening.md`.

## Scope boundary

The baseline is a clear student RAG assistant for ten banks. Multi-agent
orchestration, knowledge graphs, fine-tuning, production observability and
support for hundreds of banks are outside scope unless evaluation identifies a
specific need.
