# Multi-bank comparisons

**Status: accepted on 2026-08-14; query-planning detail superseded by ADR 013.**

## Context

BankScope previously rejected every question that named more than one configured bank. The frozen
evaluation data already contains three manually annotated two-bank questions with six independent
evidence groups, making a bounded comparison flow the next product increment.

## Decision

- Resolve one bank through the existing path and resolve two to four banks as an ordered comparison
  scope. More than four banks fail locally before embedding, retrieval, or model calls.
- Encode the standalone question once, then retrieve and generate independently for each ticker.
  Bank-specific evidence and citations cannot cross ticker boundaries.
- Relabel validated bank citations into one answer-wide `E1...En` namespace, then make one additional
  schema-validated synthesis request using bank results rather than raw filing evidence.
- Return `partial` when at least one, but not every, bank has a supported answer. Request and schema
  failures remain fail-closed; they are not converted into partial results.
- Persist the ordered bank set in SQLite schema v2. Explicit banks replace the session scope, while a
  bank-free follow-up inherits all session tickers through contextualization.
- Keep the single-bank response and `SingleBankAnswerPipeline` import compatible. The general class is
  named `BankAnswerPipeline`.

## Evaluation contract

`scripts/evaluate_comparisons.py` runs only the three frozen cross-bank questions. A passing v1 result
requires all six evidence groups, supported results for every expected ticker, zero cross-ticker
citation ownership violations, and passing correctness, completeness, and groundedness judgements.

The implementation passed the complete local unit, API, UI, lint, and build checks. After explicit
gateway approval, the GPT-5.1 generation run with the GPT-4o semantic judge produced:

- comparison questions: **3/3 supported**;
- required retrieval evidence groups: **6/6 hit**;
- cross-ticker citation ownership violations: **0**;
- semantic correctness, completeness, and groundedness: **3/3 pass**;
- overall gate: **pass**.

The ignored local result is stored at `data/evaluation/results/multi-bank-v1.json`. Each two-bank
question used two bank-specific generation requests plus one synthesis request. The frozen
single-bank generation run was not repeated.

## Consequences

A two-bank first turn normally makes three generation requests; four banks make five. Calls remain
sequential for deterministic behavior. Authentication, cloud persistence, unsupported banks, and
comparisons of more than four configured banks remain outside this phase.

ADR 013 keeps bank-isolated retrieval and synthesis but replaces the shared standalone embedding
with one peer-free, independently embedded subquestion per bank.
