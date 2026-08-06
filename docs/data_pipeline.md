# Data pipeline

## One active path

```text
config/banks.yaml
  -> scripts/download.py
  -> data/filings.json + data/raw/sec/**
  -> scripts/build_corpus.py
  -> data/processed/chunks.jsonl
     data/processed/tables.jsonl
     data/processed/manifest.json
  -> scripts/embed.py
  -> data/processed/embeddings.npz
  -> scripts/search.py / scripts/evaluate.py
```

`sec2md==0.1.23` parses the SEC HTML into ordered pages and elements. The old
local HTML parser and the sec2md builtin chunker are historical material under
`sandbox/`.

## Narrative chunks

Narrative elements remain in source order. Headings and the configured token
limit determine chunk boundaries; a small overlap preserves context. Each
record in `chunks.jsonl` contains:

- `record_id`: stable identity for vector order;
- `target_chunk_id`: evidence ID used by qrels;
- `record_type`: `text` or `table`;
- `embedding_text`: metadata-enriched retrieval input;
- `document`: narrative evidence, or a table description before hydration;
- `metadata`: bank, filing, section, page and source provenance.

## Tables

BankScope does not split a table emitted by sec2md. `tables.jsonl` stores one
record per emitted table with a stable `table_id`, filing-local `table_index`,
classification, complete Markdown, cell matrices and source metadata.

Obvious layout and index tables are retained for auditability but are not
retrieval eligible. Every eligible table has exactly one `record_type=table`
description in `chunks.jsonl`, and its `target_chunk_id` equals `table_id`.

```text
description is searched/embedded
        -> table_id
        -> complete table is returned as evidence
```

The local description is deterministic and uses filing metadata, nearby
introductory text, section/title, periods, units, columns and all deduplicated
row labels. Numeric table bodies remain only in the table store.

The optional OpenAI mode appends a short synopsis to that deterministic index.
It is explicit, records its model/source provenance and fails rather than
silently mixing description methods. An LLM description is retrieval metadata,
never evidence.

## Embeddings and retrieval

`embed.py` validates and embeds `embedding_text` in JSONL order with
`Qwen/Qwen3-Embedding-0.6B`, normalizes vectors and stores the record IDs,
model revision and input SHA-256 in the NPZ archive. The active 2,048-token
budget is checked before encoding. Dense and hybrid search refuse an archive
whose IDs, order or source hash do not match the active chunks; BM25 can run
without an embedding archive.

Dense and BM25 rankings are combined with RRF. The active baseline has no
reranker. `evaluate.py` uses `data/evaluation/queries.jsonl`; answerable queries
contribute to metrics, while ambiguous and unsupported queries are recorded as
diagnostics. Cross-bank qrels additionally report evidence-group coverage, so a
result is complete only when every required entity is represented.

## Invariants

- IDs and record order are unique and deterministic for the same raw filing
  and pinned parser version.
- Every table description references exactly one existing table.
- Complete table Markdown is never placed in embedding text.
- All evidence retains ticker, accession, report date, pages and source URL.
- Generated artifacts are ignored; `data/filings.json`, evaluation qrels and
  decision records are the small tracked data contracts.
