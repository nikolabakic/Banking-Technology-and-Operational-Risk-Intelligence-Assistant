# BankScope roadmap

The repository-overhaul phase was accepted on 2026-08-06 after the whole-table
corpus, full test suite and BM25 evaluation were recorded. A GPU dense/hybrid
run is an explicit measurement follow-up, not an unresolved code migration.

| Phase | Status | Exit condition |
|---|---|---|
| Registry and SEC acquisition | Complete | Ten configured banks and a reproducible filing manifest |
| Parser selection | Complete | sec2md selected from recorded evaluation evidence |
| Repository overhaul | Complete | Five active commands, legacy isolated, tests and lint green |
| Whole-table corpus | Complete | One stored table and at most one description per parser-emitted table |
| Embeddings | Implemented | Length/order/hash/model contracts and real smoke run pass |
| Retrieval evaluation | BM25 complete | Frozen 30-question BM25 result recorded; GPU dense/hybrid run remains |
| Answer generation | Pending | Answers use hydrated evidence and expose citations |
| Conversation history | Pending | Follow-ups work without contaminating retrieval |
| Simple user interface | Pending | Local chat flow is usable by a reviewer |
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
`docs/decisions/002-repository-overhaul.md`.

## Scope boundary

The baseline is a clear student RAG assistant for ten banks. Multi-agent
orchestration, knowledge graphs, fine-tuning, production observability and
support for hundreds of banks are outside scope unless evaluation identifies a
specific need.
