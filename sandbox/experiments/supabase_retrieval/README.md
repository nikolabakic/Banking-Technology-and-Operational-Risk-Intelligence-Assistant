# Isolated Supabase retrieval experiment

This directory benchmarks Supabase's retrieval primitives without integrating
them into BankScope. Nothing under `src/`, `scripts/`, or `pyproject.toml`
depends on this experiment.

The benchmark uses the same frozen corpus, embeddings, qrels, candidate count,
and RRF constant as the active retrieval evaluation:

- dense: PostgreSQL `pgvector` cosine search with an HNSW index;
- lexical: PostgreSQL full-text search (`websearch_to_tsquery`, GIN index);
- hybrid: reciprocal-rank fusion of 30 dense and 30 full-text candidates.

It creates only the dedicated schema `bankscope_supabase_experiment`. The
database connection must be supplied through `SUPABASE_DB_URL`; do not commit
that value.

## Run

Install the experiment-only driver without changing project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r sandbox/experiments/supabase_retrieval/requirements.txt
$env:SUPABASE_DB_URL = "postgresql://..."
.\.venv\Scripts\python.exe sandbox/experiments/supabase_retrieval/evaluate.py --recreate
```

The script writes `results.json` beside itself. `--recreate` drops and recreates
only the dedicated experimental schema.

For a hosted Supabase project, use a direct or session-pooler connection while
loading/indexing the corpus. The transaction pooler can be used for queries but
is not appropriate for every database administration operation.
