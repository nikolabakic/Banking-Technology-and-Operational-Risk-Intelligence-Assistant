# BankScope retrieval pipeline comparison

**Status: accepted on 2026-08-06.** The active parser is `sec2md==0.1.23`.
The original BeautifulSoup pipeline and the sec2md builtin-chunker experiment
are archived under `sandbox/`; the active retriever is dense + BM25S + RRF
without the evaluated reranker. The detailed JSON results remain in
`sandbox/legacy_v3/evaluation/` as historical evidence.

This document records the comparison that selected the parser. The subsequent
repository overhaul keeps sec2md but simplifies its 16,419 row/cell locators to
one complete table plus one description per retrieval-relevant table. That new
representation must be evaluated separately against the same frozen questions.

## Decision summary

The sec2md v3 corpus improves retrieval before reranking, including both table and
narrative questions. The current Qwen3 reranker reverses much of that improvement.

The comparison recommended **sec2md v3 dense + BM25S + RRF, without the current
reranker**. The completed overhaul then archived both evaluated implementations:
their results remain comparison evidence, but neither is an active runtime fallback.

Do not promote the evaluated sec2md + current-reranker combination as-is.

## Artifact validation

- Candidate embeddings: 20,428 x 1,024, float32, finite, non-zero, unit-normalized.
- Embedding record order and input SHA-256 match the final sec2md v3 JSONL exactly.
- Model: `Qwen/Qwen3-Embedding-0.6B`.
- Model revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Evaluation: 30 queries, of which 28 are answerable and 2 are diagnostic
  ambiguous/unsupported queries.
- Both corpora were evaluated with the same query set and four retrieval methods.

## Overall results (28 answerable queries)

| Pipeline | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Candidate Hit@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline hybrid | 8/28 | 19/28 | 20/28 | 23/28 | 0.494 | n/a |
| sec2md v3 hybrid | 12/28 | 20/28 | 23/28 | 24/28 | 0.589 | n/a |
| Baseline reranked | 13/28 | 17/28 | 22/28 | 25/28 | 0.588 | 26/28 |
| sec2md v3 reranked | 10/28 | 15/28 | 18/28 | 24/28 | 0.490 | 25/28 |

Equivalent hybrid-to-hybrid comparison shows the corpus gain:

- Hit@1: +4 queries.
- Hit@3: +1 query.
- Hit@5: +3 queries.
- Hit@10: +1 query.
- MRR@10: +0.095 (about +19%).

The sec2md hybrid pipeline is also effectively tied with the full baseline reranked
pipeline on MRR, with better Hit@3 and Hit@5, one fewer Hit@1, and one fewer Hit@10.

## Results by family

### Table questions (18)

| Pipeline | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline hybrid | 3/18 | 12/18 | 12/18 | 14/18 | 0.426 |
| sec2md v3 hybrid | 7/18 | 12/18 | 13/18 | 14/18 | 0.546 |
| Baseline reranked | 8/18 | 9/18 | 13/18 | 16/18 | 0.540 |
| sec2md v3 reranked | 6/18 | 9/18 | 11/18 | 15/18 | 0.464 |

The structured corpus materially improves early table ranking before reranking. The
current reranker then lowers every reported table metric except Hit@3, which is tied.

Notable sec2md reranked improvements over baseline reranked:

- Citigroup Standardized CET1: rank 4 to rank 1.
- PNC total deposits: rank 6 to rank 1.
- Citigroup Advanced CET1: rank 10 to rank 2.
- The corrected LOB Bank / 2024 CET1 evidence remains rank 1.

Notable regressions:

- Goldman Standardized CET1: rank 1 to rank 7.
- JPM CET1 requirement: rank 5 to rank 7.
- BAC Advanced CET1: rank 4 to rank 6.
- Ally comprehensive income is present at raw hybrid rank 5 but is removed from the
  parent-diversified candidate pool.

### Narrative questions (7)

| Pipeline | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline hybrid | 3/7 | 5/7 | 6/7 | 7/7 | 0.594 |
| sec2md v3 hybrid | 3/7 | 6/7 | 7/7 | 7/7 | 0.631 |
| Baseline reranked | 5/7 | 6/7 | 6/7 | 6/7 | 0.786 |
| sec2md v3 reranked | 3/7 | 5/7 | 5/7 | 6/7 | 0.562 |

Narrative retrieval remains at or above the equivalent baseline before reranking.
The current reranker causes the narrative regression, including moving Goldman cyber
risk from rank 2 to rank 10 and dropping JPM cyber risk outside the top 10 despite its
presence in the candidate pool.

### Cross-bank questions (3)

- sec2md hybrid improves union-qrel Hit@10 from 2/3 to 3/3 and MRR from 0.667 to
  0.750.
- Complete two-bank evidence coverage at top 10 remains only 1/3 for both hybrid and
  reranked candidate results. Union-qrel hits must not be interpreted as complete
  cross-bank answers.

## Candidate diversity and locator failure analysis

- Mean unique parent tables in the candidate pool: 7.11.
- Mean unique parent tables in reranked top 10: 4.54.
- The maximum candidate siblings from one parent is still 15 because the pool-filling
  fallback relaxes the cap when fewer diverse candidates are available.
- Ally comprehensive income proves a direct downside: without the cap its gold locator
  is candidate rank 5; with the cap it is removed.
- BANA and GSIB acronym-table qrels are absent even from the undiversified top 30. Their
  very broad `table_schema` locator text is a retrieval problem rather than a diversity
  problem.
- The BANA filing also contains direct narrative expansions that could answer the user,
  but the locked alias qrel intentionally tests the acronym-table locator. Therefore
  this miss measures schema-locator retrieval more strictly than end-answerability.

## Unsupported and ambiguous behavior

Neither pipeline abstains. Both return confident-looking evidence for:

- the ambiguous question with no bank/entity/approach; and
- the unsupported JPM December 31, 2026 question, using 2024/2025 evidence.

Retrieval scores are not calibrated answerability probabilities. A period/entity/
variant support check is still required before generation.

## Cost and latency

- Embedded records: 7,110 to 20,428 (2.87x).
- Embedding NPZ: 31.1 MB to 90.3 MB (2.90x).
- Embedding JSONL: 22.0 MB to 83.3 MB, plus 6.9 MB of non-embedded parent evidence.
- Measured mean sec2md hybrid retrieval: 0.0167 s/query.
- Measured mean sec2md reranking: 1.331 s/query.
- Measured mean baseline reranking: 1.681 s/query.

The structured corpus costs roughly 3x more storage but allows a competitive pipeline
without the approximately 1.3-second reranking stage.

## Historical decision options

1. **Selected at the time:** promote sec2md v3 hybrid retrieval without the current
   reranker and retain the baseline result for comparison.
2. Keep the baseline reranked pipeline as main because it still has the best Hit@1 and
   Hit@10, while retaining sec2md v3 as an experiment.
3. Select neither yet and authorize one bounded tuning cycle for schema/glossary
   locators, parent diversity, and reranking, followed by the same frozen 30-query test.
