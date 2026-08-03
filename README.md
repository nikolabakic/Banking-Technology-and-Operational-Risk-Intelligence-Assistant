# BankScope RAG Assistant

BankScope is a Retrieval-Augmented Generation assistant for exploring SEC 10-K
filings of ten publicly listed U.S. banks. The project is developed as a
student and internship project, with emphasis on a clear, reproducible
pipeline rather than production-scale infrastructure.

## Current status

Completed:

- registry and acquisition of the latest 10-K filings for ten banks;
- SEC HTML inspection and parsing;
- structure-aware text and table chunking;
- deterministic semantic proxy generation for tables;
- structural validation of the generated table proxies.

Last validated corpus:

| Record type | Count |
|---|---:|
| Text chunks | 3,890 |
| Table chunks | 3,220 |
| All chunks | 7,110 |
| Table proxies | 3,220 |

The next phase is embeddings. The embedding model, supporting libraries and
vector storage are intentionally not selected in advance. Each new phase
starts with a short comparison of current suitable tools, followed by a
project-specific decision and a small verified baseline.

## Development setup

Requirements:

- Python 3.13
- Git
- VS Code

Create and activate the virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

Set `SEC_USER_AGENT` to an application name and contact email. Never commit
`.env` or credentials.

## Data pipeline

Run commands from the repository root:

```powershell
python scripts/download_sec_filings.py
python scripts/inspect_sec_filings.py
python scripts/parse_sec_filings.py
python scripts/chunk_sec_filings.py
python scripts/generate_table_proxies.py --overwrite
```

Generated data is stored under `data/raw/` and `data/processed/`. These files
are local pipeline artifacts and are not committed.

Superseded scripts, notebooks, and other development artifacts are preserved
under `sandbox/` for mentor review. They are not part of the active pipeline.

Run code checks with:

```powershell
python -m pytest
python -m ruff check .
```

## Known corpus limitation

The primary USB and WFC filings point to separate Annual Report attachments,
so their current local corpus is substantially smaller than the other eight
banks. They remain useful for available-content retrieval, but should not be
used to conclude that information is absent from the complete Annual Report.
The downloader will be revisited only if evaluation shows that this limitation
blocks the project.

See [data pipeline](docs/data_pipeline.md) for the current processing design
and [roadmap](docs/roadmap.md) for phase status and decision gates.
