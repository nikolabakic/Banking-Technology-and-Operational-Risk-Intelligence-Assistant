# Mixed VectorDB retrieval backend

**Status: accepted on 2026-08-10.** The active retrieval backend uses Qdrant for
dense vector search, BM25S for lexical search and application RRF for fusion.

## Context

The project requirement says to store embeddings in a VectorDB. The complete
Qdrant backend satisfies that requirement but its Qdrant/BM25 plus native-RRF
hybrid result had lower MRR and substantially higher latency than the baseline.
Qdrant dense retrieval, however, had metric parity with baseline dense search.

## Decision

- Store document embeddings in the persistent local Qdrant collection and use
  Qdrant for every dense branch of the active retrieval path.
- Retain BM25S for lexical retrieval and the deterministic application RRF for
  hybrid fusion.
- Make `mixed` the default search and evaluation backend.
- Retain `baseline` and full `qdrant` backends as explicit comparison options.
- Continue hydrating complete tables from the canonical `tables.jsonl` store.

This decision does not replace or reinterpret the frozen full-Qdrant benchmark
in decision 003. It changes the active architecture to meet the explicit
VectorDB requirement while preserving the strongest measured retrieval parts.

## Verification

The frozen 30-question evaluation completed on 2026-08-10. All 28 answerable
questions had identical baseline and mixed rankings in dense, BM25 and hybrid
modes.

| Hybrid backend | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Recall@10 | Mean retrieval latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 10/28 | 25/28 | 26/28 | 0.596 | 0.855 | 3.5 ms |
| Full Qdrant | 9/28 | 24/28 | 26/28 | 0.547 | 0.838 | 342.6 ms |
| Mixed | 10/28 | 25/28 | 26/28 | 0.596 | 0.855 | 84.5 ms |

The complete suite reports `66 passed`; Ruff lint and format checks pass. The
generated evaluation report remains under `data/evaluation/results/`.
