# Evidence audit and final evaluation challenge

BankScope runs one optional, advisory evidence audit after a filing-grounded answer has passed the
existing generation, deterministic numeric, and citation validation. The audit receives only the
original question, final answer, and canonical evidence cited by that answer. It checks whether the
question is addressed, material claims are grounded, a claim contradicts the evidence, and citations
cover the material answer. It uses temperature 0, one request at most, and a versioned strict tool
schema with no numeric score.

`passed` requires every check to pass. `review_recommended` flags a potentially unsupported,
uncited, or contradictory material claim. Provider or schema failures produce neutral `unavailable`.
All three outcomes are advisory: the audit cannot change or remove the already validated answer,
status, or citations. Filing text is treated as untrusted data. Because the runtime audit has no
gold answer and is model-based, it does not prove absolute correctness or replace deterministic
validation and manual source review.

## Runtime diagnostics versus offline evaluation

Runtime **Diagnostics** explain execution: route, stages, evidence counts, request/tool budgets,
bank isolation, and validation completion. They monitor whether the bounded pipeline ran as
designed; they are not a factual-accuracy score. The evidence audit is a small post-answer review
against cited evidence. Offline evaluation answers a different question: how outputs perform across
versioned, manually annotated cases using deterministic status, numeric field, citation, evidence
group, and trap checks. Do not combine these into one score.

## Separate challenge set

`data/evaluation/evidence_audit_challenge_v1.jsonl` adds 10 manually reviewable cases without
changing the frozen 34-query `queries.jsonl` contract or its denominators:

- two unsupported or missing-period questions;
- two ambiguous questions;
- two exact numeric table questions with entity, period, and variant expectations;
- two multi-claim narrative questions, including one requiring two evidence groups;
- two citation/evidence traps where a topically similar canonical chunk is explicitly insufficient.

Each answerable case has a manually written gold answer and locally verified canonical target IDs.
Trap IDs are stored separately and must never overlap relevant qrels. `required_claims` remain a
transparent checklist for manual narrative review; an LLM is not allowed to create or certify the
gold contract.

## Reproduction and reporting

Run the unchanged frozen evaluations with:

```powershell
python scripts/evaluate.py
python scripts/evaluate_answers.py --model AZURE_GPT_51_2025_1113
```

Run the separate descriptive challenge with an available local Qdrant store, cached embedding
model, configured OpenAI-compatible endpoint, and explicit model:

```powershell
python scripts/evaluate_evidence_audit_challenge.py --model AZURE_GPT_51_2025_1113
```

The challenge writes `data/evaluation/results/evidence-audit-challenge-v1.json` and deliberately
defines no new rollout threshold. Reportable results, once actually executed, are: evaluated/error
counts; deterministic status accuracy; exact value, unit, period, entity, and variant match rates;
relevant-citation hit, completeness, and precision; citation trap avoidance; evidence-audit status
distribution; model provenance; and source hashes. Multi-claim checklists and any
`review_recommended` cases should be manually reviewed and described rather than converted into an
invented aggregate score.

## Recorded live run

The separate challenge was executed on 2026-08-31 at 08:55 UTC with
`AZURE_GPT_51_2025_1113`, the command above, the local Qdrant store, and the cached embedding
model. All 10 cases completed with no runtime errors. These descriptive results do not change a
frozen baseline or create a rollout gate:

- status accuracy: 8/10 (80%);
- relevant-citation hit rate for expected-supported cases: 5/6 (83.3%);
- citation completeness across all cases: 8/10 (80%);
- mean citation precision: 90%;
- exact value, unit, period, entity, and variant match: 1/2 (50%) for each dimension;
- citation/evidence trap avoidance: 2/2 (100%);
- runtime audit outcomes: nine `passed`, zero `review_recommended`, zero `unavailable`, and one
  pre-retrieval ambiguous response for which the audit was intentionally not run.

The two status misses are important report caveats. The ambiguous Truist question was answered for
Truist Financial Corporation even though the filing also contains a different Truist Bank value.
The JPMorgan Chase Bank Advanced-ratio case returned `unsupported` because its manually relevant
table was not retrieved in the final evidence set. The audit returned `passed` for both outputs: it
can assess only the final answer and its cited evidence, so it cannot resolve an omitted entity
alternative or detect missing evidence that retrieval never supplied. This measured result
demonstrates why the audit is advisory and why gold-backed offline evaluation remains necessary.

The versioned source hashes for this run are:

- challenge: `8ec8634b70fb43ac6d5316f4f069638f51151462e6f28bdf632aa31b4851c145`;
- chunks: `ac17ae4cbfc2b22bec99d77792f7ad9cc9f35d6f5540525f128b86fb5e779b65`;
- tables: `78d9b301f66b034a7dd9f1347bb7305c8340a7aa72f8f8feadaa06125ec58715`;
- Qdrant manifest: `5076df7b23b528a2ba4bd052fad576a943202de0ab4345a7a60389ffe56bd84c`.

The detailed machine-readable output is
`data/evaluation/results/evidence-audit-challenge-v1.json`. The results directory is ignored by Git,
so rerun the documented command when reproducing the report from a fresh checkout.
