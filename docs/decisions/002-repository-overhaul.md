# Repository overhaul and whole-table corpus

**Status: accepted on 2026-08-06.** The active repository is now the small
student-project baseline described below. Historical implementations are kept
under `sandbox/` and are not runtime fallbacks.

The USB/WFC corpus composition and 30-question baseline described here were superseded by
[ADR 009](009-complete-primary-filings.md) on 2026-08-17. The measurements below remain historical.

## Outcome

The active workflow was reduced from 18 overlapping scripts to five commands:

```text
download.py -> build_corpus.py -> embed.py -> search.py / evaluate.py
```

- `sec2md==0.1.23` is the only active SEC parser.
- The BeautifulSoup/builtin pipeline is in `sandbox/legacy_builtin/`.
- The sec2md-v3 locator pipeline and its frozen results are in
  `sandbox/legacy_v3/`.
- Notebooks, Docling/XBRL trials and one-off scaffolds are in
  `sandbox/experiments/`.
- Old generated artifacts are ignored under `sandbox/local_data/`.

Active code has no imports or runtime references into `sandbox/`.

## Table decision

Every table emitted by sec2md is stored once, in full Markdown form, in
`data/processed/tables.jsonl`. Layout/index tables remain auditable but are not
indexed. Each of the other tables has exactly one retrieval record containing:

- a deterministic index of context, columns, periods, units and every
  deduplicated row label;
- optionally, a short GPT-4o synopsis appended to that deterministic index;
- the stable whole-table ID used to hydrate a hit back to source evidence.

The optional LLM text is never returned as source evidence. Its model, API and
response provenance are recorded, and API failure does not silently fall back
to a mixed corpus.

## Corpus integrity

The local deterministic build over the ten filing records produced:

| Contract | Result |
|---|---:|
| Retrieval records | 5,565 |
| Narrative records | 4,009 |
| Table-description records | 1,556 |
| Complete stored tables | 1,783 |
| Tables excluded as layout/index | 227 |
| Duplicate record IDs | 0 |
| Missing table hydration targets | 0 |
| Tables with more than one description | 0 |
| Missing main/group qrel IDs | 0 |
| Longest embedding input | 1,049 / 2,048 tokens |

This replaces 16,419 sec2md-v3 row/cell locator records with 1,556 table
descriptions while retaining the complete 1,783-table evidence store.

## Retrieval check

The frozen set still contains 30 questions: 28 answerable and two diagnostic
ambiguous/unsupported questions. Table locator qrels were mapped to their
stable whole-table IDs. BM25 was rerun end to end on the new corpus:

| BM25 metric | Archived sec2md v3 | Whole-table corpus |
|---|---:|---:|
| Hit@1 | 7/28 | 11/28 |
| Hit@3 | 14/28 | 21/28 |
| Hit@5 | 16/28 | 25/28 |
| Hit@10 | 24/28 | 26/28 |
| MRR@10 | 0.409 | 0.562 |
| Mean required-group coverage@10 | 0.333 | 0.500 |
| Complete cross-bank questions@10 | 0/3 | 0/3 |

The same questions are used, but table relevance is now measured at whole-table
rather than row-locator granularity, so recall values are not a strict
like-for-like comparison. The result is sufficient to reject an obvious
lexical regression; cross-bank coverage remains a documented weakness.

The Qwen embedding path passed a real ten-record smoke run and validates input
length, record order, source hash, model name and model revision. A complete
5,565-record dense/hybrid run was not claimed in this local CPU-only session:
the ten-record encoding step alone took about five minutes. The archived
sec2md-v3 hybrid result remains the parser-selection evidence, while a future
GPU run can evaluate the new representation without another code change.

## Verification

- `58 passed` in the complete pytest suite.
- Ruff lint and formatting checks pass.
- All five CLI entry points pass `--help`.
- BM25 search works without an embedding archive and returns hydrated whole
  tables for table hits.
- Dense/hybrid search rejects stale hashes, mismatched record order and missing
  embeddings instead of producing silent results.
- Cross-bank evaluation reports required-group recall and complete-group hits.

## Remaining cautions

- The downloaded USB and WFC primary documents reference separate annual-report
  attachments, so those local corpora remain partial.
- The current BM25 run retrieves no complete two-bank evidence set in the top
  ten for the three cross-bank questions.
- A Hugging Face token appeared in an old Git commit. The current tree is clean,
  but that credential must be revoked/rotated; Git history was not rewritten.
