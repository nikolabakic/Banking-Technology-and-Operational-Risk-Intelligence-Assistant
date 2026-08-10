# Local Qdrant retrieval backend

**Status: accepted on 2026-08-10.** Qdrant Local Mode is a supported optional
backend. The existing NumPy dense, BM25S and application-RRF implementation
remains the default retrieval backend.

## Implemented outcome

- `qdrant-client[fastembed] 1.18.x` works on the project's Python 3.13 Windows
  environment with persistent close/reopen behavior.
- `bankscope_retrieval` contains all 5,565 retrieval records, including 1,556
  table descriptions, with 1,024-dimensional Qwen dense vectors.
- Sparse retrieval uses `Qdrant/bm25` with the IDF modifier; hybrid retrieval
  uses native RRF with 30 candidates per branch and `k=60`.
- Dense, BM25 and hybrid modes work through the existing search and evaluation
  CLIs, including ticker/type filters and whole-table hydration.
- Generated storage and its source/configuration manifest remain under the
  ignored `data/processed/` directory.

## Frozen evaluation

The single final comparison used all 30 frozen questions; 28 answerable
questions contributed to metrics.

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| Baseline dense | 10/28 | 21/28 | 23/28 | 0.521 | 0.669 |
| Qdrant dense | 10/28 | 21/28 | 23/28 | 0.521 | 0.669 |
| Baseline BM25S | 11/28 | 25/28 | 26/28 | 0.562 | 0.787 |
| Qdrant BM25 | 12/28 | 23/28 | 23/28 | 0.555 | 0.727 |
| Baseline hybrid | 10/28 | 25/28 | 26/28 | 0.596 | 0.855 |
| Qdrant hybrid | 9/28 | 24/28 | 26/28 | 0.547 | 0.838 |

Qdrant dense has exact aggregate parity. Qdrant hybrid retains Hit@10, stays
within the agreed Recall tolerance and loses neither a complete question
category nor a previously complete cross-bank evidence group. It fails the
agreed MRR gate of `0.584`, however, and its local mean retrieval latency was
about 395 ms versus about 4 ms for the baseline hybrid on this small corpus.

## Decision

Do not change the default backend. Qdrant remains available with
`--backend qdrant` for further development and as a storage option, while normal
search continues to use `--backend baseline` implicitly. Do not tune RRF or the
sparse model on the same frozen 30 questions. If a future answer-generation
phase demonstrates an operational need, evaluate a Qdrant-dense plus BM25S
mixed path as a separate, measured change.

## Verification

- Persistent dense, sparse, hybrid, filter and hydration integration test.
- `63 passed` in the complete pytest suite.
- Ruff lint passes for the implementation.
- Full generated report: `data/evaluation/results/retrieval.json`.
