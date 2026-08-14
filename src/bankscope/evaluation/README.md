# Evaluation primitives

**Status:** active metric and advisory-judge library.

```mermaid
flowchart TD
    Qrels[versioned queries/qrels] --> Ranking[evaluate_ranking]
    Qrels --> Evidence[evaluate_evidence_groups]
    Answers[structured answers] --> Metrics[evaluate_answer]
    Audit[manual citation audit] --> Metrics
    Answers --> Judge[judge_semantic_answer]
    Ranking --> Report[separate metric sections]
    Evidence --> Report
    Metrics --> Report
    Judge --> Report
```

## Files and public API

| File | Public symbols | Responsibility |
|---|---|---|
| `retrieval_metrics.py` | `deduplicate_ids()`, `evaluate_ranking()`, `evaluate_evidence_groups()` | Hit/rank/MRR and grouped multi-bank evidence coverage |
| `answer_metrics.py` | `expected_answer_status()`, `evaluate_answer()`, `summarize_answer_metrics()` | Deterministic status, entity, value, unit, period, and citation metrics |
| `semantic_judge.py` | `SemanticJudgement`, `judge_semantic_answer()` | Optional correctness/completeness/groundedness judgement over cited evidence only |

Deterministic metrics are authoritative for schema and exact contractual checks. The semantic judge
is advisory, versioned, and receives only evidence cited by the candidate answer. Structured facts
are preferred; text fallback exists only for historical baselines.

Retrieval, single-bank generation, conversation memory, and multi-bank comparison use distinct
denominators and reports. Do not combine them into one quality score or silently exclude errors.
Metric changes require matching tests and an update to the relevant ADR/report.

[Evaluation data](../../../data/evaluation/README.md) · [CLI evaluators](../../../scripts/README.md)

