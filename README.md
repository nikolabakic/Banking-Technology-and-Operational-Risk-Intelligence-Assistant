# BankScope RAG Assistant

BankScope is a student RAG project for searching the latest downloaded 10-K
filings of ten U.S. banks. The repository favors a short, reproducible local
pipeline over production infrastructure.

## Current design

The active pipeline has five commands:

```text
download.py -> build_corpus.py -> embed.py -> search.py / evaluate.py
```

- `sec2md==0.1.23` is the only active filing parser.
- Narrative text is split into bounded, overlapping chunks.
- A parser-emitted table is never split again. The complete Markdown table is
  stored once in `tables.jsonl`.
- Each retrieval-relevant table gets one compact description in `chunks.jsonl`.
  A table hit is resolved back to the complete table before evidence is shown.
- Hybrid retrieval is dense search plus BM25S combined with reciprocal-rank
  fusion (RRF). The previous reranker is archived because it reduced the
  measured sec2md result.

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
python scripts/search.py "How does JPMorgan Chase define cybersecurity risk?" --ticker JPM
python scripts/evaluate.py
```

Useful smoke runs:

```powershell
python scripts/download.py --ticker JPM
python scripts/build_corpus.py --ticker JPM --output-dir data/processed/smoke-jpm --overwrite
python scripts/embed.py --limit 10
```

Filtered corpus builds require their own output directory so a smoke run cannot
replace the complete ten-bank corpus by accident.

Generated files are local and ignored by Git:

```text
data/raw/sec/                 downloaded filing HTML
data/processed/chunks.jsonl   text chunks and table descriptions
data/processed/tables.jsonl   complete tables and stable table IDs
data/processed/manifest.json  parser and corpus provenance
data/processed/embeddings.npz vectors joined to chunks by record order
```

Table descriptions are deterministic by default. GPT-4o descriptions are an
explicit optional enrichment and never replace either the deterministic table
index or the source table as evidence:

```powershell
python -m pip install -e ".[dev,llm]"
python scripts/build_corpus.py --description-mode openai --overwrite
```

This mode requires `OPENAI_API_KEY`; `OPENAI_MODEL` is configurable in `.env`
and defaults to `gpt-4o`. It makes one API request per retrieval-eligible table,
so the local mode should be used for normal development runs.

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
for the completed migration result, and [`docs/roadmap.md`](docs/roadmap.md) for
the next project phases.

## Known corpus limitations

- The downloaded USB and WFC primary filings point to separate annual-report
  attachments, so their local content is partial.
- A table that sec2md itself emits as separate continuation elements remains
  separate; BankScope only guarantees that each emitted table is not split
  further.
- Retrieval scores are not answerability probabilities. Ambiguous and
  unsupported questions still need checks before answer generation.
