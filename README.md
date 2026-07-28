# BankScope RAG Assistant

BankScope is a Retrieval-Augmented Generation assistant for exploring the latest SEC 10-K filings of publicly listed U.S. banks.

The project is being implemented incrementally, starting with a validated 10-bank development corpus. SEC data acquisition, document processing, retrieval, citations, conversation history and evaluation will be added in later phases.

## Development setup

Requirements:

- Python 3.13.14
- Git
- VS Code

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install the locked dependencies:

python -m pip install -r requirements.lock

Create the local environment file:

Copy-Item .env.example .env

Fill in the required values locally. Never commit .env or secret credentials.

Validation

Run the project setup checks:

python scripts/run_smoke_test.py
python -m pytest
python -m ruff check .
Current status

Project skeleton, application settings, structured logging and initial tests are complete. SEC data acquisition and RAG functionality have not been implemented yet.