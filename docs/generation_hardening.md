# Generation hardening candidate

**Status: v2 frozen generation run completed; the quality gate did not pass because one
rounded extra citation remains outside the exact-support contract.**

The candidate keeps one Chat Completions request per retrieval-supported question, with no
retry. `AZURE_GPT_51_2025_1113` is selected explicitly through `--model`; the configured
application default remains unchanged until the quality gate passes.

## Runtime contract

- Chat Completions uses `response_format={"type":"json_object"}` and an explicit JSON
  instruction. Prompt v4 states that numeric `facts` is exactly one object, never an array,
  and separates the base `metric` from an approach/method `variant`.
- GPT-5.1 requests use `max_completion_tokens`; no reasoning parameter is sent.
- Pydantic validates `status`, `answer_type`, `answer`, `reason`, `citation_ids`, and optional
  numeric `facts` (`entity`, `metric`, optional `variant`, `period`, `value_text`, `unit`).
- The bank's full legal name comes from `config/banks.yaml` and is included in the prompt.
- Numeric answers are rendered locally as
  `Entity — Metric — Variant — Period: Value Unit [citations]`.
- Citation labels must resolve to supplied evidence. A numeric value must equal a complete
  numeric token in a cited document after only whitespace, currency, and thousands-separator
  normalization. There is no rounding tolerance.
- Invalid JSON, schema, citations, refusals, content filters, and truncation fail closed with a
  stable `GenerationValidationError` code. Provenance records contract versions, response
  format, model, final status, request count, latency, finish reason, and available token usage.

## Evaluation contract

Frozen retrieval qrels remain unchanged. `data/evaluation/generation_citation_audit_v2.jsonl`
retains the three v1 outcomes and accepts five additional directly supporting chunks from the
recorded v1 run:

- the BAC BANA regulatory-capital table is accepted as direct alternative evidence;
- the Citi `13.2%` narrative is relevant but insufficient for the exact `13.18%` contract;
- the BAC BANA dividend passage is rejected because it does not expand the abbreviation.
- the additional BAC cyber passage and BAC `11.4%` narrative are accepted;
- Truist Corporation Table 37 and its `10.8%` narrative are accepted;
- Truist Bank Table 36 is accepted for `11.8%`.
- the JPM regulatory-capital note is accepted because it directly explains the buffers included
  in the separately table-supported `11.5%` Standardized requirement;
- the PNC `$440.9 billion` liquidity narrative is relevant but rejected for the exact
  `$440,866 million` answer because it is rounded.

The retrieval candidate adds a versioned lexical-only glossary locator file. Locator hits are
deduplicated by parent table ID and hydrate to the unchanged complete table evidence. The dense
corpus, embeddings and Qdrant collection are unchanged.

Existing qrel citation metrics remain available. Support-aware citation metrics use the union
of qrels and manually accepted alternatives; reviewed but rejected evidence is not included.
Structured metrics read `facts` first and use answer text only for historical artifacts. The
GPT-4o semantic judge receives only the evidence cited by a supported narrative answer.
The evaluator writes an explicit pass/fail gate with every required denominator. Filtered runs
must use an explicit output path, so they cannot replace the complete candidate artifact.

## Recorded approval-gated run

The following command was run once after explicit approval. It made external model calls and must
not be repeated without a new approval. The v1 compatibility probe was not repeated.

```powershell
python scripts/evaluate_answers.py --model AZURE_GPT_51_2025_1113
```

The frozen run permits at most 25 generation requests because the unsupported-period question
abstains locally; semantic
judge requests are separate. The result is written to
`data/evaluation/results/generation-gpt51-json-v2.json`, leaving the original
`generation.json` and v1 candidate intact.

The candidate passes only if all 26 questions have a result without schema/format errors,
status is 26/26, value/unit/period/entity are each 15/15, variant is 9/9, every numeric answer
has exact cited support, all narrative answers pass cited-evidence groundedness, and no citation
falls outside the qrel/audit contract. Only then may GPT-5.1 become the default and decision 005
be superseded. The remaining citation issue is tracked but no longer blocks bank resolution or
conversation-history work.

## Recorded GPT-5.1 v2 run — 2026-08-11

The single authorized frozen run completed all 26 questions without schema or format errors and
wrote `generation-gpt51-json-v2.json` (SHA-256
`bc0ae67f7306867ec5abc313359c303dc52d01f928936d679248db9a8b4bdada`). It made exactly 25
generation requests; the unsupported-period question abstained locally. The ten separate GPT-4o
judge calls received only cited evidence. No retry or compatibility probe was performed.

All answer-quality checks passed:

- status: 26/26;
- value, unit, period, and entity: 15/15 each;
- variant: 9/9;
- exact numeric cited support: 15/15;
- narrative correctness, completeness, and groundedness: 10/10 each;
- generation budget: 25/25 requests.

The frozen artifact reports citation-contract precision for 23/25 supported answers against the
pre-run audit-v2 hash. Post-run manual review accepted the new JPM buffer note and rejected the
new PNC liquidity narrative: the latter reports only the rounded `$440.9 billion`, not the exact
`$440,866 million`. The updated audit-v2 SHA-256 is
`b8716ca3f1f4663b041161150b25a3a6cb73740f3ec82d09b6315aa5db23367a`. The audited result is
therefore 24/25, so the overall gate still fails. The gateway exposed response IDs and zero
reasoning tokens, but not input/output/total token counts. Mean generation latency was 2.801
seconds across the 25 requests (70.019 seconds total).

GPT-5.1 remains a candidate, and the application default and decision 005 remain unchanged.
The rounded extra-citation issue is deferred and does not block product work. A new frozen run
requires separate
approval; this result is not iteratively tuned or overwritten.

## Recorded GPT-5.1 run — 2026-08-11

The first synthetic probe returned `facts` as an array and failed Pydantic validation. After the
single approved prompt-v3 correction, the second one-call synthetic probe passed. The one
authorized frozen run then completed without schema or format errors and wrote
`generation-gpt51-json-v1.json` (SHA-256
`6e257d65e5b455cdd187a9f9cbf0b59809204b422056e09b9db24159b6990b97`).

The quality gate did not pass:

- results/errors: 26/26 and 0 errors;
- status: 24/26; the BANA and GSIB expansion questions abstained because their direct qrel
  evidence was absent from the retrieved top five;
- value, unit, period, and entity: 15/15 each;
- exact numeric cited support: 15/15;
- variant: 0/9 because every model response folded the approach into `metric` and left the
  optional `variant` field null;
- narrative groundedness: 8/8 judged answers, short of the required 10 because of the two
  abstentions;
- citation contract: 19/25 supported questions; four supported answers included additional
  cited chunks not present in qrels or audit v1;
- generation requests: 25, within budget.

No retry or post-run tuning was performed. GPT-5.1 therefore remains a candidate rather than the
default, and decision 005 remains unchanged.

## Local v2 retrieval result — 2026-08-11

The v2 glossary locator artifact contains 299 deterministic definition records. The extra 61
records beyond the initial prototype come from an existing `Term | Definition` table that was
already classified as a glossary. No numeric row locators were added.

The formal mixed-hybrid frozen run is recorded separately in
`retrieval-glossary-locators-v1.json` and passes its local gate:

- Hit@5: 27/28; Hit@10: 28/28;
- BANA first relevant rank: 3; GSIB first relevant rank: 5;
- no Hit@5 or Hit@10 regressions against the historical mixed result;
- the historical chunks, embeddings and Qdrant hashes are unchanged.

Prompt/schema v4, unit rendering and citation audit v2 are locally implemented. The single v2
generation run and its cited-evidence semantic judging are recorded above.
