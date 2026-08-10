# Single-bank generation evaluation

**Status: accepted on 2026-08-10.** The evaluator and first single-bank
generation baseline are recorded.

## Context

BankScope already had a frozen 30-question retrieval set and a grounded
single-bank answer command, but no end-to-end generation measurement. The
current answer contract requires an explicit ticker, so three cross-bank
questions and one ambiguous question without a bank cannot be evaluated without
expanding product scope.

## Decision

- Reuse the frozen `queries.jsonl`; do not modify retrieval qrels or its quality gate.
- Evaluate 26 in-scope questions: 25 answerable questions with a ticker and one
  unsupported-period question.
- Record the three cross-bank questions and one missing-ticker ambiguous question
  as explicit exclusions rather than silently dropping or mis-scoring them.
- Reuse one query encoder, Qdrant client and BM25S index for the complete run.
- Keep deterministic status, structured-field and citation metrics separate from
  an advisory semantic judge for non-numeric answers.
- Store model names, prompt version, source hashes, retrieval settings, timings,
  per-query answers and errors in `data/evaluation/results/generation.json`.

## Metric contract

Deterministic metrics cover expected answer status, expected numeric value,
unit, period, entity and variant matches, relevant-citation precision, relevant
citation hit and required evidence-group coverage. The semantic judge reports
correctness, completeness and groundedness against the gold answer and hydrated
evidence. Judge results are advisory because they are model-based; they do not
replace deterministic checks or retrieval metrics.

## Verification

Unit tests cover scope selection, pipeline reuse, status mapping, structured
values, citation failures, evidence-group coverage and invalid judge payloads.
Ruff and the focused generation-evaluation test suite pass.

An authorized one-query smoke run passed before the full run. The full evaluator
used `AZURE_GPT_4o_2024_1120` for both generation and advisory semantic judging,
mixed retrieval with `limit=5`, `candidate_k=30` and `rrf_k=60`, and the frozen
qrels hash `7aab627f...e873c01`.

| Metric | First baseline |
|---|---:|
| In-scope / explicitly excluded queries | 26 / 4 |
| Evaluated / query errors | 24 / 2 |
| Status accuracy on evaluated queries | 100% |
| Relevant citation hit, supported evaluated queries | 87.0% |
| Citation completeness, all evaluated queries | 87.5% |
| Mean relevant-citation precision | 76.4% |
| Exact expected-value match | 13/15 (86.7%) |
| Unit match | 13/15 (86.7%) |
| Period match | 8/15 (53.3%) |
| Entity match | 8/15 (53.3%) |
| Variant match | 5/9 (55.6%) |
| Advisory semantic correctness/completeness/groundedness | 8/8 for each |

The two explicit errors were
`dev_bac_cyber_incident_impacts_2025` (inconsistent citations) and
`dev_pnc_gsib_expansion_2025` (invalid model payload). The two value misses were
Citigroup answers rounded from `13.18%` to `13.2%` and from `11.93%` to `11.9%`.
Three answers cited retrieved evidence outside the manually listed qrels; this
may be a generation error or incomplete qrel coverage and requires source audit
before tuning.

The semantic judge is advisory and used the same model as generation, so its
perfect score is not independent evidence. Status accuracy excludes the two
errored queries, and deterministic period/entity/variant metrics measure whether
the generated text states those fields rather than inferring them from the
question.

Reproduce the baseline with:

```powershell
python scripts/evaluate_answers.py --query-id dev_jpm_standardized_cet1_ratio_2025
python scripts/evaluate_answers.py
```

The generated report remains at `data/evaluation/results/generation.json` and is
ignored by Git. Conversation history is now the next implementation phase; the
recorded failures should be audited without tuning the frozen evaluation set.
