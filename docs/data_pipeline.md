# Data pipeline

## Corpus

The corpus contains the latest downloaded primary 10-K filing for:

`ALLY`, `BAC`, `C`, `GS`, `JPM`, `LOB`, `PNC`, `TFC`, `USB` and `WFC`.

The processing flow is:

```text
SEC primary HTML
-> ordered parsed elements
-> text and table chunks
-> deterministic table proxies
-> embedding records (next phase)
```

## Acquisition and parsing

`scripts/download_sec_filings.py` reads `config/banks.yaml`, downloads each
latest primary 10-K HTML document and writes the filing manifest.

`scripts/parse_sec_filings.py` uses
`src/bankscope/parsing/sec_html_parser.py` to:

- remove scripts, styles, hidden content and navigation-like elements;
- normalize problematic invisible Unicode characters;
- preserve document order;
- classify headings, paragraphs, lists and data tables;
- attach SEC Item, section title and source metadata when available;
- linearize table rows without discarding numeric values.

## Chunking

`scripts/chunk_sec_filings.py` writes
`data/processed/chunks/sec_10k_chunks.jsonl`.

Accepted configuration:

- soft target: 600 tokens;
- hard maximum: 700 tokens;
- recursive overlap: 80 tokens;
- minimum text target: 80 tokens;
- tokenizer used for the completed chunking run:
  `Qwen/Qwen3-Embedding-0.6B`.

Last validated output:

| Ticker | All chunks | Table chunks |
|---|---:|---:|
| ALLY | 786 | 247 |
| BAC | 900 | 508 |
| C | 1,015 | 454 |
| GS | 1,151 | 531 |
| JPM | 1,423 | 765 |
| LOB | 497 | 176 |
| PNC | 628 | 263 |
| TFC | 520 | 234 |
| USB | 97 | 19 |
| WFC | 93 | 23 |
| Total | 7,110 | 3,220 |

There were no empty chunks, duplicate chunk IDs or chunks above the accepted
hard maximum in the validated run.

## Table proxies

`scripts/generate_table_proxies.py` creates one `deterministic-v1` proxy for
each table chunk. A proxy uses available bank, year, SEC Item, section,
context, row labels, column labels and units. It omits pure numeric table
values because the original table remains the evidence returned after
retrieval.

Relationship:

```text
proxy_text -> embedding and search
target_chunk_id -> original table chunk and citation evidence
```

Last validated output contained 3,220 unique proxies for 3,220 unique table
chunks. The generator now performs corpus-size-independent structural checks
during a full run.

## Known limitations

- USB and WFC primary filing documents reference separate Annual Report
  attachments, so their present corpus is partial.
- `sec_item` and `section_title` are useful retrieval metadata but are not
  reliable enough to be mandatory filters for every filing.
- Parser and chunking changes should now be driven by concrete retrieval
  failures, not another general refactor.
