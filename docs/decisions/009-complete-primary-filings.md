# Complete primary filings and ten-bank rebaseline

**Status: accepted and fully rebaselined on 2026-08-17.**

## Context

The active USB and WFC primary 10-K documents were only filing shells: material Parts I, II, or IV
content lived in separate annual-report attachments. That made their local corpora materially
smaller than the other eight banks and unsuitable for a ten-bank filing RAG baseline.

## Decision

- Replace USB and WFC with Capital One Financial Corporation (`COF`, CIK `0000927628`) and State
  Street Corporation (`STT`, CIK `0000093751`). Their SEC primary 2025 10-K HTML documents contain
  risk disclosures, MD&A, financial statements, notes and auditor reports in-document.
- Treat a proxy-only Part III incorporation as complete. Reject a primary document that delegates
  Parts I, II, or IV to an Annual Report/Exhibit 13, or lacks the core 10-K sections and financial
  statement/auditor evidence.
- Pin Qwen3-Embedding-0.6B revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Use CUDA for the full embedding pass and keep parsing,
  Qdrant construction and evaluation local CPU steps.
- Expand the frozen evaluation contract from 30 to 34 questions with one narrative and one exact
  standardized-CET1 question for each replacement bank. The resulting contract has 32 answerable
  questions and the single-bank generation gate has 30 eligible questions.

## Local deterministic result

The validated primary-document build contains exactly ten tickers and no USB/WFC records. It
produced 6,550 retrieval chunks and 2,085 complete stored tables. COF contributed 563 chunks and
212 tables; STT contributed 560 chunks and 147 tables. Their corpus sizes are consistent with the
eight retained complete filings rather than the retired filing shells.

The replacement archive contains 6,550 ordered 1,024-dimensional vectors and the rebuilt Qdrant
collection contains the same 6,550 records. The accepted artifact hashes are:

- chunks: `ac17ae4cbfc2b22bec99d77792f7ad9cc9f35d6f5540525f128b86fb5e779b65`;
- tables: `78d9b301f66b034a7dd9f1347bb7305c8340a7aa72f8f8feadaa06125ec58715`;
- glossary locators: `27dbc121f3d6fdc3cdce28f3cde22a9839111464d47229fd1282ee6b8afc817c`;
- embeddings: `3b8f49d77214af347d369b5b111b87ca8bed5d5c7fdb738148a75f216beb069f`;
- Qdrant manifest: `5076df7b23b528a2ba4bd052fad576a943202de0ab4345a7a60389ffe56bd84c`.

The completed retrieval, generation, comparison and ten-bank application results are recorded in
ADR 010. GPU was used only to create the already accepted embedding archive; the final retrieval
repair and all acceptance runs used local CPU plus the configured generation/judge APIs.
