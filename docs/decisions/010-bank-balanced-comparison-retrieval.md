# Bank-balanced multi-bank retrieval

**Status: accepted on 2026-08-17.**

## Context

The ten-bank corpus and dense index were valid, but globally fused cross-bank searches could let
one bank consume most of Top 10. Changing RRF weights or increasing `candidate_k` through 100 did
not recover every required evidence group; the `candidate_k=60` experiment also reduced aggregate
Top-5 coverage to 30/32. The BAC/C comparison was the reproducible failure: relevant Citi evidence
could crowd out the required Bank of America evidence.

## Decision

- For a request containing two to four unique tickers, run the existing mixed hybrid search once
  per bank with `limit=5`, `candidate_k=30`, and `rrf_k=60`.
- Give generation the isolated per-bank result lists. For frozen retrieval metrics, deterministically
  interleave those lists into a deduplicated, bank-balanced Top 10.
- Use this same multi-bank path in the comparison pipeline without changing the HTTP/API contract.
- Record explicit ticker pairs in the three cross-bank qrels: `BAC/C`, `C/JPM`, and `PNC/TFC`.
- Keep evidence acceptance manual. The added BAC target directly states its Standardized CET1 ratio;
  thematically similar or rounded evidence remains unacceptable.

## Measured result

The final mixed-hybrid gate passed with Hit@5 31/32, Hit@10 32/32, MRR@10 0.63854, all four
COF/STT questions, both glossary questions, no regression on the original 28 answerable questions,
all three cross-bank questions and 6/6 cross-bank evidence groups, and all four grouped questions
with 8/8 groups.

Generation passed 30/30 evaluated questions with 29 supported answers, 17/17 numeric contracts,
11/11 variant checks, 12/12 grounded narratives, and all supported citations inside the manually
accepted qrel/audit contract. Comparison passed 6/6 evidence groups, semantic judging, and citation
ownership with zero violations. The ten-bank application smoke passed 10/10.

Tracked and generated provenance hashes:

- qrels: `51ecfc1df72a78a718f9f9137878cefca7ab8c61501f05177e54e9cb7889686a`;
- citation audit: `2bceb8879bf536698536fe6e26bf6fe2e7f6f25baecea76ec335d4ac5eab3805`;
- retrieval result: `557463824a0143f22f024f24e8cf7de9a8b1310c0d9a793922663d3c0791a531`;
- generation result: `f7558c20c681e0d0e77646b81d92d5cc96fe213729b2efc08bc1fda3d722f710`;
- comparison result: `60eceeab85631a4826430e302bce682233cac9344d96b5cf4d32ccaf70d60db7`;
- application-smoke result: `a46aa0ba600361a3b40da5d43361db7f42c8ccea948c56205591e9aed72dcacb`.

## Consequences

Single-bank retrieval is unchanged. Comparison work grows linearly with two to four selected banks,
but evidence ownership and per-bank quotas are explicit and testable. Generated result files remain
local and ignored; this ADR preserves the accepted metrics and hashes.
