# BankScope RAG Assistant

BankScope is a student RAG project for searching the latest downloaded 10-K
filings of ten U.S. banks. 

## Current design

The active pipeline has six commands:

```text
download.py -> build_corpus.py -> embed.py -> search.py / evaluate.py -> answer.py
```

- `sec2md==0.1.23` is the only active filing parser.
- Narrative text is split into bounded, overlapping chunks.
- A parser-emitted table is never split again. The complete Markdown table is
  stored once in `tables.jsonl`.
- Each retrieval-relevant table gets one compact description in `chunks.jsonl`.
  A table hit is resolved back to the complete table before evidence is shown.
- Acronym and glossary tables additionally get small lexical-only definition
  locators. They share the parent table target ID, are deduplicated before the
  output limit and always hydrate back to the complete table.
- The default mixed backend retrieves dense candidates from persistent Qdrant,
  retrieves lexical candidates with BM25S and combines both rankings with
  application reciprocal-rank fusion (RRF).
- Answer requests resolve a configured bank deterministically from its legal name,
  common alias or ticker before retrieval. Missing or multiple banks return an
  `ambiguous` result without embedding, retrieval or model calls.
- Persistent Qdrant Local Mode is available as an optional backend. Its dense,
  BM25 and native-RRF paths are also implemented, but full Qdrant hybrid did not
  pass the MRR quality gate.

The parser decision is based on the frozen 30-question comparison: sec2md
hybrid improved Hit@1 from 8/28 to 12/28 and MRR@10 from 0.494 to 0.589 over
the original parser pipeline. See
[`docs/decisions/001-parser-selection.md`](docs/decisions/001-parser-selection.md).

## Setup

Requirements: Python 3.13, Git and a local environment file.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `SEC_USER_AGENT` to an application name and a contact email. Credentials
belong only in `.env` or a secret manager; never commit them.

## Run the pipeline

Run commands from the repository root:

```powershell
python scripts/download.py
python scripts/build_corpus.py --overwrite
python scripts/embed.py --overwrite
python scripts/build_qdrant.py
python scripts/search.py "How does JPMorgan Chase define cybersecurity risk?" --ticker JPM
python scripts/answer.py "How does JPMorgan Chase define cybersecurity risk?"
python scripts/evaluate.py
python scripts/evaluate_answers.py --model AZURE_GPT_51_2025_1113
```

The default search uses Qdrant for dense retrieval and BM25S plus application
RRF for hybrid retrieval. Baseline and full-Qdrant comparisons remain available:

```powershell
python scripts/search.py "operational risk capital" --backend baseline --mode hybrid --ticker JPM
python scripts/search.py "operational risk capital" --backend qdrant --mode hybrid --ticker JPM
python scripts/evaluate.py --backend all
```

Use `build_qdrant.py --recreate` only when intentionally rebuilding the existing
`bankscope_retrieval` collection.

Useful smoke runs:

```powershell
python scripts/download.py --ticker JPM
python scripts/build_corpus.py --ticker JPM --output-dir data/processed/smoke-jpm --overwrite
python scripts/embed.py --limit 10
python scripts/evaluate_answers.py --model AZURE_GPT_51_2025_1113 `
  --query-id dev_jpm_standardized_cet1_ratio_2025 `
  --output artifacts/generation-gpt51-smoke.json
```

Filtered corpus builds require their own output directory so a smoke run cannot
replace the complete ten-bank corpus by accident.

`build_corpus.py` writes the glossary locator artifact together with chunks and
tables. To regenerate only that small artifact from existing processed data, run
`python scripts/build_glossary_locators.py --overwrite`.

Generated files are local and ignored by Git:

```text
data/raw/sec/                 downloaded filing HTML
data/processed/chunks.jsonl   text chunks and table descriptions
data/processed/tables.jsonl   complete tables and stable table IDs
data/processed/lexical_glossary_locators_v1.jsonl lexical-only definition locators
data/processed/manifest.json  parser and corpus provenance
data/processed/embeddings.npz vectors joined to chunks by record order
data/processed/qdrant/        generated persistent local Qdrant database
data/processed/qdrant_manifest.json Qdrant source hashes and vector configuration
data/evaluation/results/generation.json generation metrics, answers and provenance
data/evaluation/results/generation-gpt51-json-v1.json hardened candidate baseline
data/evaluation/results/retrieval-glossary-locators-v1.json v2 retrieval gate result
data/evaluation/results/generation-gpt51-json-v2.json reserved v2 generation result
```

Table descriptions are deterministic by default. GPT-4o descriptions are an
explicit optional enrichment and never replace either the deterministic table
index or the source table as evidence:

```powershell
python -m pip install -e ".[dev,llm]"
python scripts/build_corpus.py --description-mode openai --overwrite
```

This mode obtains the authenticated corporate client through
`model_access.access_model()`. `OPENAI_MODEL` is configurable in `.env` and
defaults to `AZURE_GPT_4o_2024_1120`. It makes one API request per
retrieval-eligible table, so the local mode should be used for normal development runs.

`answer.py` performs the default mixed hybrid retrieval and passes only hydrated
evidence to the configured OpenAI-compatible Chat Completions endpoint. Generation
uses JSON mode followed by a strict Pydantic contract. Numeric answers are rendered
locally from validated `facts`, and their exact numeric token must occur in at least
one cited evidence document. The flow makes at most one generation request per
question and never retries. The first generation slice intentionally requires
`--ticker`; unsupported periods fail before the model call, while ambiguous or
insufficient evidence produces an abstention. Custom gateways can be configured by
the internal `model_access` package. The bank normally comes from the question;
`--ticker` remains an optional session/evaluation fallback for compatibility.

`evaluate_answers.py` reuses the same long-lived answer pipeline and evaluates
the 26 frozen questions that fit the current single-bank contract: 25 answerable
questions and one unsupported-period question. Three cross-bank questions and
the ambiguous question without a ticker are listed as explicit scope exclusions.
Deterministic status, structured-field and citation metrics are reported
separately from the advisory semantic judge. Metrics read structured `facts` first
and retain a text fallback for the historical baseline. Qrel citation metrics stay
unchanged; support-aware metrics additionally use the versioned manual citation
audit. The GPT-4o semantic judge receives only evidence actually cited by the
answer. Use `--skip-judge` to omit the additional evaluation-only judge calls.

The first recorded 26-question generation baseline completed 24 queries and
captured two model-format/citation errors. Among completed queries, answer
status accuracy was 100%, exact expected-value accuracy was 86.7%, relevant
citation hit rate was 87.0%, and all eight advisory semantic judgements passed.
See decision 005 for denominators and caveats; retrieval metrics remain separate.

The hardened GPT-5.1 v2 frozen run completed all 26 questions without schema or
format errors and passed every answer-quality check, including variant 9/9 and
grounded narratives 10/10. It did not pass the overall gate because one extra PNC
citation gives only the rounded `$440.9 billion`, not exact support for
`$440,866 million`; the post-run citation audit is therefore 24/25. GPT-5.1 is not
the default, and the citation issue is deferred while bank resolution and conversation
history proceed. The frozen run must not be
repeated without separate approval. The historical `generation.json` and v1
candidate remain unchanged. See [`docs/generation_hardening.md`](docs/generation_hardening.md)
for the recorded result and audit contract.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

`sandbox/` contains superseded code, notebooks and experiments. Nothing under
it is imported by the active project. See [`docs/data_pipeline.md`](docs/data_pipeline.md)
for schemas and invariants,
[`docs/decisions/002-repository-overhaul.md`](docs/decisions/002-repository-overhaul.md)
for the completed migration result,
[`docs/decisions/003-qdrant-local-retrieval.md`](docs/decisions/003-qdrant-local-retrieval.md)
for the Qdrant evaluation decision,
[`docs/decisions/004-mixed-vector-retrieval.md`](docs/decisions/004-mixed-vector-retrieval.md)
for the active mixed-backend decision,
[`docs/decisions/005-generation-evaluation.md`](docs/decisions/005-generation-evaluation.md)
for the generation-evaluation contract, and [`docs/roadmap.md`](docs/roadmap.md)
for the next project phases.

## Known corpus limitations

- The downloaded USB and WFC primary filings point to separate annual-report
  attachments, so their local content is partial.
- A table that sec2md itself emits as separate continuation elements remains
  separate; BankScope only guarantees that each emitted table is not split
  further.
- Retrieval scores are not answerability probabilities. Ambiguous and
  unsupported questions still need checks before answer generation.
