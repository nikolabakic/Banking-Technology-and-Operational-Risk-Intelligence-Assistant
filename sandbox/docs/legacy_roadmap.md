# BankScope RAG Assistant
## Complete implementation roadmap for a five-week project

**Project type:** Retrieval-Augmented Generation assistant  
**Primary corpus:** latest SEC 10-K filings of publicly listed U.S. banks  
**Development corpus:** 10 banks  
**Scale target:** approximately 100 banks after the baseline is validated  
**Time constraint:** 5 weeks  
**Primary interface:** Gradio  
**Primary vector store:** Qdrant Local Mode  
**Core framework stack:** LangChain Core, LangGraph, Pydantic, DeepEval, Langfuse  
**Implementation principle:** build a deterministic, evaluated RAG workflow before adding agentic behavior

---

# 1. Purpose of this document

This document is the complete implementation roadmap for the project. It is designed so that each numbered implementation phase can be handled in a separate ChatGPT conversation.

Each phase contains:

- objective,
- prerequisites,
- implementation tasks,
- recommended technical decisions,
- expected outputs,
- validation checks,
- completion criteria,
- common failure modes,
- a suggested prompt for starting a dedicated chat.

The roadmap is intentionally stricter than a simple tutorial. A phase is considered complete only when its outputs are saved, reproducible and validated.

---

# 2. Assignment requirements and project interpretation

The faculty assignment requires the following:

1. Obtain the latest 10-K filing for 10 public companies through the SEC EDGAR API.
2. Save the filings as local HTML files.
3. Inspect and process documents containing text, tables, headings, footers and other structural elements.
4. Split the processed documents into chunks suitable for embedding.
5. Generate vector embeddings.
6. Store chunks and embeddings in a vector database.
7. Connect the retrieval system to the provided language model.
8. Build a conversational interface.
9. Return references or citations whenever possible.
10. Preserve conversation history for multi-turn interactions.
11. Consider external tools or function calling.
12. Evaluate retrieval and generation quality.
13. Build a small test set from the extracted documents.

This implementation narrows the company domain to publicly listed U.S. banks. This is a useful domain because:

- filings are structurally similar enough to support systematic parsing,
- vocabulary is specialized and retrieval is non-trivial,
- users can ask both single-bank and cross-bank questions,
- citations and metadata filtering are important,
- the system can be meaningfully scaled from 10 to approximately 100 banks.

The latest 10-K filing is the required corpus. Historical filings are not part of the baseline.

---

# 3. Final product definition

The final system should allow a user to ask questions such as:

- What cybersecurity risks does JPMorgan disclose?
- Which section discusses credit-loss provisions?
- Compare liquidity-risk disclosures for three selected banks.
- What changed in a follow-up question referring to the previously discussed bank?
- Which source passages support the answer?
- What is the reported value of a selected structured financial fact?

The system must:

1. identify relevant bank, year and SEC section constraints,
2. retrieve supporting passages,
3. reject unsupported questions when evidence is insufficient,
4. generate a grounded answer,
5. cite only retrieved sources,
6. preserve conversation context,
7. expose retrieved evidence in the interface,
8. record evaluation and latency information.

The system is not intended to provide investment advice or guarantee the interpretation of financial disclosures.

---

# 4. Scope

## 4.1 Required baseline

The required baseline contains:

- 10 selected publicly listed U.S. banks,
- latest available 10-K filing per bank,
- SEC API downloader,
- raw HTML preservation,
- filing manifest,
- custom SEC-aware parsing,
- section-aware and token-aware chunking,
- local embedding model,
- persistent Qdrant collection,
- dense semantic retrieval,
- metadata filters,
- connection to the provided language model,
- grounded prompting,
- validated citations,
- multi-turn history,
- Gradio interface,
- manually built evaluation dataset,
- retrieval metrics,
- generation and citation evaluation,
- automated smoke tests.

## 4.2 Recommended portfolio extensions

Implement only after the baseline passes its completion criteria:

- sparse or BM25 retrieval,
- Reciprocal Rank Fusion,
- CrossEncoder reranking,
- LangGraph workflow,
- Langfuse tracing,
- DeepEval component-level evaluation,
- SEC Company Facts tool,
- calculator tool,
- scaling from 10 to approximately 100 banks,
- latency and indexing benchmarks.

## 4.3 Explicitly out of scope

These features are excluded unless every required phase is already complete:

- multi-agent orchestration,
- unrestricted autonomous agent loops,
- GraphRAG,
- knowledge graph construction,
- multiple years of filings for every bank,
- distributed processing,
- Kubernetes,
- self-hosted Langfuse infrastructure,
- authentication and user management,
- unrestricted web browsing,
- portfolio recommendations,
- production-grade financial analytics,
- training or fine-tuning an LLM,
- comparing many vector databases,
- supporting every SEC filing type,
- perfect extraction of every complex financial table.

---

# 5. Success criteria

## 5.1 Functional criteria

The system is functionally complete when:

- all 10 required banks have a valid latest 10-K filing,
- all raw filings are stored locally,
- every processed filing has a manifest entry,
- every indexed chunk has deterministic identifiers and complete metadata,
- retrieval supports filters by bank, filing year and SEC item,
- the assistant produces citations tied to retrieved chunks,
- invalid citation identifiers cannot reach the UI,
- multi-turn follow-up questions work,
- conversation state survives at least the active application session,
- the evaluation pipeline runs from one command,
- the Gradio application runs without notebook state.

## 5.2 Quality targets

Targets are directional, not guarantees:

- retrieval Recall@10 should be high on the curated test set,
- bank-filter accuracy should be close to perfect,
- section-filter accuracy should be close to perfect,
- unknown citation rate must be zero,
- citation metadata completeness must be 100%,
- unsupported questions should usually trigger refusal or uncertainty,
- answers should not introduce claims absent from the retrieved context,
- end-to-end latency should remain practical for an interactive demo.

## 5.3 Reproducibility criteria

- dependencies are pinned,
- configuration is externalized,
- downloads are cached,
- index construction is deterministic,
- repeated indexing does not create duplicates,
- evaluation results are saved with configuration metadata,
- all notebooks can be deleted without breaking the application.

---

# 6. Architecture principles

## 6.1 Deterministic workflow before agentic behavior

The central pipeline is:

```text
user query
    ↓
input validation
    ↓
follow-up contextualization
    ↓
metadata-filter resolution
    ↓
candidate retrieval
    ↓
optional fusion and reranking
    ↓
evidence sufficiency check
    ↓
grounded answer generation
    ↓
citation validation
    ↓
response persistence and tracing
    ↓
Gradio output
```

The LLM must not freely choose arbitrary tools or repeat steps indefinitely.

## 6.2 Separation of offline and online paths

### Offline path

```text
SEC discovery
→ download
→ raw storage
→ parsing
→ structural normalization
→ chunking
→ embedding
→ vector indexing
→ evaluation corpus preparation
```

### Online path

```text
query
→ query processing
→ metadata filters
→ retrieval
→ reranking
→ context construction
→ generation
→ citation validation
→ history persistence
```

The UI must never perform document parsing or index construction.

## 6.3 Explicit contracts

Every boundary should use validated Pydantic models:

- downloaded filing,
- parsed element,
- document section,
- chunk,
- retrieval filter,
- retrieved chunk,
- grounded answer,
- citation,
- evaluation case,
- application settings.

## 6.4 Replaceable infrastructure

The following components should be behind small project-owned interfaces:

- embedding provider,
- vector store,
- language model client,
- reranker,
- tracing provider.

This prevents the complete application from depending directly on a framework-specific object.

## 6.5 Evidence-first responses

The system should answer only from retrieved filing evidence or a clearly identified deterministic tool result.

When evidence is insufficient, the expected behavior is:

- state that the available documents do not support a reliable answer,
- optionally identify which bank, year or section is missing,
- do not fill gaps from general model knowledge.

---

# 7. Recommended technology stack

## 7.1 Runtime and development

```text
Python 3.12
VS Code
virtual environment: .venv
Git
pytest
ruff
mypy or pyright
```

Python 3.12 is preferred over very new Python releases because ML, vector-database and evaluation libraries usually support it reliably.

## 7.2 Core stack

| Responsibility | Choice | Usage policy |
|---|---|---|
| HTTP and SEC access | `httpx`, `tenacity` | Direct SEC client owned by the project |
| HTML parsing | `BeautifulSoup`, `lxml` | Main parser implementation |
| Validation | Pydantic v2 | Mandatory at all important boundaries |
| Configuration | `pydantic-settings` | Mandatory |
| RAG utilities | LangChain Core | Use components, not opaque end-to-end chains |
| Workflow | LangGraph | Add only after the plain RAG baseline works |
| Embeddings | Sentence Transformers | Local and reproducible |
| Vector database | Qdrant Local Mode | Persistent local development and easy scale path |
| Sparse retrieval | Qdrant sparse vectors or `rank-bm25` | Extension after dense retrieval |
| Reranking | Sentence Transformers CrossEncoder | Optional quality improvement |
| UI | Gradio | Required final interface |
| Conversation state | LangGraph SQLite checkpointer | Final history implementation |
| Evaluation | custom metrics + DeepEval | Deterministic metrics remain primary |
| Tracing | Langfuse Cloud free tier | Optional and disable-able |
| Tests | pytest | Unit, integration and smoke tests |

## 7.3 Libraries intentionally not combined

Avoid redundant framework overlap:

- do not use PydanticAI together with LangGraph for orchestration,
- do not use LangSmith together with Langfuse,
- do not maintain separate DeepEval and RAGAS pipelines,
- do not use both ChromaDB and Qdrant in the final application,
- do not create both custom history storage and multiple framework memory systems.

## 7.4 Fallback decisions

If an advanced component causes schedule risk:

| Preferred component | Fallback |
|---|---|
| Qdrant Local Mode | ChromaDB |
| hybrid retrieval | dense retrieval only |
| CrossEncoder | no reranker |
| LangGraph | direct Python orchestration service |
| Langfuse | structured local logs |
| DeepEval judge metrics | manual scoring + deterministic metrics |
| Company Facts tool | omit tool calling |
| 100 banks | remain at 10 banks with a documented scale test |

---

# 8. Corpus strategy

## 8.1 Development corpus

Use 10 publicly listed U.S. banks.

Selection criteria:

- active public company,
- valid CIK,
- available latest 10-K,
- variation in bank size and business model,
- filings that can be legally downloaded from SEC EDGAR,
- no duplicate holding-company identity.

Store the bank registry in a version-controlled configuration file:

```yaml
banks:
  - ticker: JPM
    cik: "0000019617"
    legal_name: JPMorgan Chase & Co.
    enabled: true
```

CIK values must always be normalized to 10 digits when used in SEC API paths.

## 8.2 Scale corpus

After the 10-bank system is stable:

- extend the registry to approximately 100 banks,
- download only the latest 10-K,
- use the same parser, metadata schema and collection,
- benchmark incremental indexing,
- inspect retrieval balance across banks.

Do not create one vector collection per bank.

## 8.3 Corpus versioning

A corpus version should be derivable from:

- registry version,
- accession numbers,
- parser version,
- chunking configuration,
- embedding model,
- embedding normalization setting.

Example:

```text
corpus_version = banks_v2__parser_v3__chunk550_80__minilm_v1
```

---

# 9. Data model

## 9.1 Filing manifest

```python
class FilingManifest(BaseModel):
    document_id: str
    cik: str
    ticker: str
    bank_name: str
    form: str
    accession_number: str
    filing_date: date
    report_date: date | None
    source_url: str
    local_html_path: str
    downloaded_at: datetime
    content_sha256: str
    parser_version: str | None = None
    processing_status: str
    error_message: str | None = None
```

## 9.2 Parsed element

```python
class ParsedElement(BaseModel):
    element_id: str
    document_id: str
    element_type: Literal[
        "heading",
        "paragraph",
        "table",
        "list",
        "footnote",
        "other",
    ]
    text: str
    html: str | None
    order_index: int
    sec_item: str | None
    section_title: str | None
    parent_heading: str | None
    metadata: dict[str, Any]
```

## 9.3 Document chunk

```python
class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    chunk_index: int
    cik: str
    ticker: str
    bank_name: str
    filing_year: int
    accession_number: str
    sec_item: str | None
    section_title: str | None
    element_type: Literal["text", "table"]
    source_url: str
    content_sha256: str
```

## 9.4 Retrieval filters

```python
class RetrievalFilters(BaseModel):
    tickers: list[str] | None = None
    ciks: list[str] | None = None
    filing_years: list[int] | None = None
    sec_items: list[str] | None = None
    element_types: list[str] | None = None
```

## 9.5 Retrieved chunk

```python
class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    reranker_score: float | None = None
    final_rank: int
    retrieval_method: str
```

## 9.6 Grounded answer

```python
class GroundedAnswer(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    is_answerable: bool
    insufficient_evidence_reason: str | None = None
```

## 9.7 Citation

```python
class Citation(BaseModel):
    citation_label: str
    chunk_id: str
    bank_name: str
    filing_year: int
    accession_number: str
    sec_item: str | None
    section_title: str | None
    source_url: str
    supporting_excerpt: str
```

## 9.8 Deterministic identifiers

Recommended formulas:

```text
document_id =
SHA-256(cik + accession_number + content_sha256)

element_id =
SHA-256(document_id + order_index + normalized_element_text)

chunk_id =
SHA-256(document_id + chunk_index + normalized_chunk_text)
```

---

# 10. Metadata schema

Every indexed point must include:

```text
document_id
chunk_id
cik
ticker
bank_name
filing_year
form
accession_number
sec_item
section_title
element_type
chunk_index
token_count
source_url
content_sha256
parser_version
chunker_version
embedding_model
```

Recommended Qdrant payload indexes:

```text
cik
ticker
filing_year
sec_item
accession_number
element_type
document_id
```

A bank is a filterable document entity, not a tenant. Cross-bank comparison must remain possible.

---

# 11. Chunking policy

## 11.1 Structural order

Use this hierarchy:

```text
filing
→ SEC Item
→ subsection or heading
→ HTML element
→ token-aware chunk
```

Never split the entire filing using a fixed character window before identifying sections.

## 11.2 Initial configuration

```text
target_chunk_tokens = 550
maximum_chunk_tokens = 750
overlap_tokens = 80
minimum_chunk_tokens = 80
```

Treat these as an experimental baseline, not universal constants.

## 11.3 Text rules

- never mix two SEC Items in one chunk,
- include the bank, filing year, SEC Item and section heading in the chunk prefix,
- merge short adjacent paragraphs under the same heading,
- split long paragraphs by sentence boundaries,
- preserve ordered and unordered lists,
- remove navigation boilerplate and repeated headers,
- keep meaningful footnotes only when linked to nearby content,
- do not remove domain terminology as stopwords,
- do not lemmatize filing text before embedding.

## 11.4 Table rules

For each table:

- preserve title or nearby heading,
- preserve column headers,
- convert to a readable Markdown-like representation,
- repeat headers when splitting by rows,
- include nearby explanatory text,
- save the original HTML separately,
- mark the chunk as `element_type="table"`.

Large tables should be split by logical row groups. Do not guarantee exact numerical question answering from table embeddings alone.

---

# 12. Retrieval design

## 12.1 Baseline

```text
query
→ optional metadata filters
→ dense embedding
→ Qdrant top-15 candidates
→ optional score threshold
→ final top-5 to top-7 context chunks
```

## 12.2 Recommended advanced retrieval

```text
dense top-30
+
sparse/BM25 top-30
→ Reciprocal Rank Fusion
→ top-20
→ CrossEncoder reranking
→ final top-5 to top-7
```

## 12.3 Cross-bank comparison

For a query involving multiple named banks:

```text
retrieve top-k separately for each bank
→ merge candidates
→ rerank globally
→ enforce minimum evidence coverage per requested bank
```

A global top-k without per-bank balancing may return evidence for only one dominant bank.

## 12.4 Query processing

Query processing may:

- resolve tickers and legal names,
- detect explicit bank filters,
- detect SEC Item references,
- contextualize follow-up questions,
- preserve the original query,
- return structured filters.

It must not silently invent a bank or year when ambiguity is material.

## 12.5 Evidence sufficiency

The system should treat a query as unsupported when:

- no candidate passes the minimum quality rule,
- retrieved evidence belongs to the wrong bank,
- a requested comparison lacks evidence for one or more banks,
- retrieved passages do not contain information relevant to the requested claim,
- the question requires data outside the available filing set.

---

# 13. Citation policy

## 13.1 Citation format

Do not invent page numbers for HTML filings.

Recommended display:

```text
[JPMorgan Chase, 2025 10-K, Item 1A, “Cybersecurity Risk”, chunk 017]
```

## 13.2 Generation-time citation labels

The LLM receives context with short labels:

```text
[S1]
[S2]
[S3]
```

The response schema returns cited labels or chunk IDs.

## 13.3 Citation validation

Before displaying an answer:

1. parse every citation label,
2. verify that it maps to a retrieved chunk,
3. reject unknown labels,
4. remove duplicate citations,
5. build display metadata from project data, not LLM text,
6. verify that at least one citation exists for a supported factual answer,
7. optionally allow one repair generation.

Unknown citation rate must be zero.

## 13.4 Source panel

The UI should display:

- citation label,
- bank,
- filing year,
- SEC Item,
- section title,
- excerpt,
- SEC source URL,
- retrieval score or rank if useful for debugging.

---

# 14. Conversation-history policy

## 14.1 Required behavior

The system should support:

- conversation IDs,
- thread IDs,
- recent messages,
- follow-up contextualization,
- clearing a conversation,
- preserving citations with assistant messages.

## 14.2 LangGraph state

Suggested state:

```python
class RagState(TypedDict):
    thread_id: str
    messages: list
    original_query: str
    standalone_query: str
    filters: RetrievalFilters
    retrieved_chunks: list[RetrievedChunk]
    evidence_sufficient: bool
    answer: GroundedAnswer | None
    citations: list[Citation]
    errors: list[str]
```

## 14.3 Checkpointing

Use a SQLite LangGraph checkpointer for the final local application.

Do not implement a separate complex ORM history layer unless the assignment later requires browsing and managing many saved conversations.

## 14.4 Context window policy

- include only the recent turns needed for contextualization,
- do not append the full conversation to every generation prompt,
- use a separate contextualization step for follow-up queries,
- summarize only if conversation length becomes a measured problem.

---

# 15. Tool policy

## 15.1 Recommended tools

### SEC Company Facts tool

Use for structured XBRL facts when a question asks for a reported numerical value that is poorly represented by text chunks.

Input:

```text
CIK
taxonomy
concept
unit
period
```

Output:

- value,
- unit,
- filing reference,
- period,
- source endpoint.

### Calculator tool

Use only for deterministic arithmetic over explicitly retrieved values.

## 15.2 Tool routing

Tool routing should be deterministic or tightly constrained:

```text
narrative filing question → RAG
structured reported fact → Company Facts
arithmetic over known values → calculator
unsupported current information → refusal
```

## 15.3 Excluded tools

Do not add unrestricted web search to the baseline. It weakens source control and makes evaluation less clear.

---

# 16. Observability policy

## 16.1 Local logs

Always maintain structured local logs for:

- run ID,
- thread ID,
- query,
- resolved filters,
- retrieval latency,
- generation latency,
- number of candidates,
- selected chunk IDs,
- errors,
- model and configuration identifiers.

Do not log secrets.

## 16.2 Langfuse

Use Langfuse only after the application works locally.

Recommended spans:

```text
rag_request
├── contextualize_query
├── resolve_filters
├── dense_retrieval
├── sparse_retrieval
├── fusion
├── reranking
├── evidence_check
├── generation
├── citation_validation
└── persistence
```

Langfuse must be optional through configuration. The application should still run when tracing is disabled.

---

# 17. Evaluation strategy

## 17.1 Evaluation dataset

Create a curated dataset with approximately 50 to 70 questions for the final system.

Suggested distribution:

| Category | Approximate count |
|---|---:|
| direct single-bank questions | 15 |
| section-specific questions | 10 |
| terminology and exact-phrase questions | 8 |
| multi-bank comparison questions | 10 |
| table or numerical questions | 5 |
| follow-up questions | 5 |
| deliberately unanswerable questions | 7 |

Each case should include:

```text
case_id
question
conversation_context
expected_bank_ids
expected_sec_items
relevant_chunk_ids or evidence descriptions
reference_answer
answerable
notes
```

## 17.2 Retrieval metrics

Primary deterministic metrics:

- Recall@5,
- Recall@10,
- Mean Reciprocal Rank,
- NDCG@10,
- bank-filter accuracy,
- SEC-item-filter accuracy,
- cross-bank coverage,
- candidate diversity,
- retrieval latency p50 and p95.

## 17.3 Generation metrics

Use a combination of:

- manual factual correctness,
- answer completeness,
- answer relevance,
- unsupported-claim count,
- refusal correctness,
- DeepEval Faithfulness,
- DeepEval Answer Relevancy.

## 17.4 Citation metrics

- citation presence rate,
- valid citation rate,
- unknown citation rate,
- citation metadata completeness,
- citation support rate,
- DeepEval Citation Faithfulness if practical.

## 17.5 Operational metrics

- download success rate,
- parsing success rate,
- chunk count by bank,
- embedding throughput,
- index construction time,
- index size,
- retrieval latency,
- reranking latency,
- generation latency,
- total latency,
- failure rate.

## 17.6 Evaluation discipline

- deterministic metrics are primary,
- LLM-as-a-judge metrics are secondary,
- manually inspect a representative answer subset,
- record judge model and prompt,
- save every evaluation configuration,
- do not optimize repeatedly against a tiny test set without holding out cases.

---

# 18. Target repository structure

```text
bankscope-rag/
├── app.py
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── uv.lock or requirements.lock
├── .env.example
├── .gitignore
│
├── config/
│   ├── banks.yaml
│   └── logging.yaml
│
├── data/
│   ├── raw/
│   │   └── sec/
│   ├── processed/
│   │   ├── elements/
│   │   └── chunks/
│   ├── evaluation/
│   └── samples/
│
├── artifacts/
│   ├── qdrant/
│   ├── cache/
│   ├── manifests/
│   ├── logs/
│   ├── checkpoints/
│   └── evaluation_results/
│
├── src/
│   └── bankscope/
│       ├── __init__.py
│       ├── config/
│       │   └── settings.py
│       ├── domain/
│       │   ├── models.py
│       │   ├── enums.py
│       │   └── exceptions.py
│       ├── sec/
│       │   ├── client.py
│       │   ├── company_registry.py
│       │   ├── filing_discovery.py
│       │   ├── downloader.py
│       │   ├── company_facts.py
│       │   └── rate_limit.py
│       ├── parsing/
│       │   ├── html_cleaner.py
│       │   ├── element_extractor.py
│       │   ├── item_detector.py
│       │   ├── table_parser.py
│       │   ├── normalizer.py
│       │   └── pipeline.py
│       ├── chunking/
│       │   ├── tokenizer.py
│       │   ├── text_chunker.py
│       │   ├── table_chunker.py
│       │   ├── validation.py
│       │   └── pipeline.py
│       ├── embeddings/
│       │   ├── base.py
│       │   ├── sentence_transformer.py
│       │   └── service.py
│       ├── vector_store/
│       │   ├── base.py
│       │   ├── qdrant_store.py
│       │   └── collection.py
│       ├── retrieval/
│       │   ├── filters.py
│       │   ├── query_processor.py
│       │   ├── dense.py
│       │   ├── sparse.py
│       │   ├── fusion.py
│       │   ├── reranker.py
│       │   └── service.py
│       ├── llm/
│       │   ├── base.py
│       │   ├── provided_model.py
│       │   ├── prompts.py
│       │   └── structured_output.py
│       ├── rag/
│       │   ├── context_builder.py
│       │   ├── evidence.py
│       │   ├── citations.py
│       │   ├── workflow.py
│       │   └── service.py
│       ├── tools/
│       │   ├── calculator.py
│       │   ├── company_facts.py
│       │   └── router.py
│       ├── history/
│       │   └── checkpointer.py
│       ├── evaluation/
│       │   ├── dataset.py
│       │   ├── retrieval_metrics.py
│       │   ├── generation_metrics.py
│       │   ├── citation_metrics.py
│       │   ├── deepeval_runner.py
│       │   └── report.py
│       ├── observability/
│       │   ├── logging.py
│       │   └── langfuse.py
│       └── ui/
│           ├── gradio_app.py
│           ├── callbacks.py
│           └── formatting.py
│
├── scripts/
│   ├── download_filings.py
│   ├── inspect_corpus.py
│   ├── process_filings.py
│   ├── build_index.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_end_to_end.py
│   ├── scale_corpus.py
│   └── run_smoke_test.py
│
├── notebooks/
│   ├── 01_corpus_inspection.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   └── 04_evaluation_analysis.ipynb
│
└── tests/
    ├── unit/
    ├── integration/
    ├── evaluation/
    └── fixtures/
```

Not every optional module must be created on day one. Create a module only when its phase begins.

---

# 19. Configuration

Suggested `.env.example`:

```env
APP_ENV=development
LOG_LEVEL=INFO

SEC_USER_AGENT=BankScopeRAG your_email@example.com
SEC_REQUESTS_PER_SECOND=4
SEC_MAX_CONCURRENCY=3
SEC_TIMEOUT_SECONDS=30

BANK_REGISTRY_PATH=config/banks.yaml
RAW_DATA_DIR=data/raw/sec
PROCESSED_DATA_DIR=data/processed
MANIFEST_DIR=artifacts/manifests

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
NORMALIZE_EMBEDDINGS=true

CHUNK_TARGET_TOKENS=550
CHUNK_MAX_TOKENS=750
CHUNK_OVERLAP_TOKENS=80
CHUNK_MIN_TOKENS=80

QDRANT_PATH=artifacts/qdrant
QDRANT_COLLECTION=bankscope_10k
DENSE_CANDIDATE_K=30
SPARSE_CANDIDATE_K=30
FINAL_CONTEXT_K=6
MIN_RETRIEVAL_SCORE=

RERANKER_ENABLED=false
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

LLM_MODEL=
LLM_BASE_URL=
LLM_API_KEY=
LLM_TEMPERATURE=0.0
MAX_CONTEXT_TOKENS=7000

LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

CHECKPOINT_DB_PATH=artifacts/checkpoints/langgraph.sqlite

DEEPEVAL_MODEL=
DEEPEVAL_API_KEY=
```

Never commit real secrets.

---

# 20. Five-week execution plan

## Week 1: corpus and parsing

- project setup,
- Pydantic models,
- bank registry,
- SEC client,
- latest 10-K discovery,
- downloader and manifest,
- corpus inspection,
- SEC Item parser,
- table handling,
- parser validation.

**Week 1 gate:** 10 raw filings and 10 validated processed-document outputs.

## Week 2: chunking, embeddings and baseline retrieval

- chunking implementation,
- chunk statistics,
- chunking comparison,
- local embeddings,
- Qdrant collection,
- deterministic indexing,
- dense retrieval,
- metadata filters,
- initial retrieval test set.

**Week 2 gate:** reproducible index and working retrieval without an LLM or UI.

## Week 3: RAG, citations and conversation history

- provided-model adapter,
- grounded prompt,
- structured answer,
- citation validator,
- evidence sufficiency,
- direct orchestration baseline,
- LangGraph conversion,
- SQLite checkpointing,
- basic Gradio UI.

**Week 3 gate:** end-to-end cited multi-turn assistant.

## Week 4: retrieval improvement and evaluation

- sparse retrieval,
- fusion,
- reranking experiment,
- final evaluation dataset,
- deterministic metrics,
- DeepEval,
- Langfuse,
- error analysis,
- configuration selection.

**Week 4 gate:** selected configuration supported by saved evaluation results.

## Week 5: scale, hardening and presentation

- scale to approximately 100 banks,
- incremental indexing,
- benchmark storage and latency,
- Company Facts tool if time permits,
- tests,
- UI cleanup,
- README,
- architecture diagram,
- limitations,
- demo scenarios,
- final report.

**Week 5 gate:** reproducible portfolio-ready project.

---

# 21. Chat-by-chat implementation phases

Each phase below is intended to be used in a separate chat.

---

## Chat 01: Project skeleton and dependency strategy

### Objective

Create a reproducible Python project without prematurely implementing RAG logic.

### Prerequisites

- Python 3.12 installed,
- VS Code,
- Git.

### Tasks

1. Create the repository structure needed for the first two weeks.
2. Choose dependency management: `uv` is recommended, `pip` with a lock file is acceptable.
3. Add `pyproject.toml`.
4. Add development dependencies.
5. Add `.gitignore`.
6. Add `.env.example`.
7. Implement `ApplicationSettings` with `pydantic-settings`.
8. Add structured logging.
9. Create a smoke-test script.
10. Add basic CI-compatible commands.

### Deliverables

- importable `src/bankscope` package,
- configuration module,
- `pyproject.toml`,
- initial README,
- smoke test.

### Completion criteria

```bash
python scripts/run_smoke_test.py
pytest
ruff check .
```

all complete without errors.

### Common mistakes

- installing into the wrong Python interpreter,
- creating every future module before it is needed,
- hardcoding local Windows paths,
- committing `.env`,
- using Python 3.14 before dependency support is confirmed.

### Prompt for the dedicated chat

> We are implementing Chat 01 from `BANKSCOPE_RAG_COMPLETE_ROADMAP.md`. Help me create the project skeleton, `pyproject.toml`, Pydantic settings, logging and smoke test. Work step by step and do not implement SEC or RAG functionality yet.

---

## Chat 02: Bank registry and SEC identity validation

### Objective

Define the 10-bank development corpus and validate company identities.

### Tasks

1. Select 10 publicly listed U.S. banks.
2. Record ticker, legal name and CIK.
3. Normalize CIK to 10 digits.
4. Detect duplicate parent holding companies.
5. Save the registry to `config/banks.yaml`.
6. Implement a Pydantic registry model.
7. Validate every configured company against SEC submissions data.
8. Produce a validation report.

### Deliverables

- `config/banks.yaml`,
- registry loader,
- identity-validation script,
- report with valid and invalid entries.

### Completion criteria

- every enabled bank resolves to the expected legal entity,
- every bank has at least one 10-K,
- no duplicated CIK exists,
- the registry can later be extended without code changes.

### Common mistakes

- using a subsidiary instead of the public holding company,
- dropping leading CIK zeros,
- mixing tickers from another exchange or old company names,
- manually embedding the list into Python code.

### Prompt

> We are implementing Chat 02 from the roadmap. Help me choose and validate 10 publicly listed U.S. banks, build `config/banks.yaml`, normalize CIKs and create the Pydantic registry loader and validation script.

---

## Chat 03: SEC client and latest 10-K discovery

### Objective

Build a compliant and testable SEC EDGAR client.

### Tasks

1. Implement an `httpx` client with a clear User-Agent.
2. Add rate limiting below SEC limits.
3. Add bounded concurrency.
4. Add retry with exponential backoff.
5. Read SEC submissions JSON.
6. identify the latest valid 10-K for each bank,
7. exclude amendments unless intentionally required,
8. construct the filing-document URL,
9. return a validated filing descriptor.

### Deliverables

- `sec/client.py`,
- `sec/rate_limit.py`,
- `sec/filing_discovery.py`,
- fixture-based tests,
- discovery report for 10 banks.

### Completion criteria

- latest 10-K is consistently discovered for all selected banks,
- errors identify the bank and endpoint,
- repeated calls reuse cache when possible,
- tests do not depend entirely on live network access.

### Common mistakes

- selecting 10-Q,
- treating 10-K/A as the main filing,
- issuing requests without a valid User-Agent,
- adding excessive concurrency,
- assuming the primary document filename is constant.

### Prompt

> We are implementing Chat 03. Build a direct SEC EDGAR client with `httpx`, rate limiting, retry, caching and latest 10-K discovery. Use Pydantic models and fixture-based tests. Do not download filing HTML yet.

---

## Chat 04: Filing downloader and manifest

### Objective

Download raw filing HTML exactly once and create an auditable manifest.

### Tasks

1. Download the primary filing HTML.
2. Preserve raw bytes without cleaning.
3. Use deterministic file paths.
4. Calculate SHA-256.
5. Save manifest records.
6. Skip unchanged existing files.
7. support resume after failure,
8. write a summary report.

### Deliverables

```text
data/raw/sec/<cik>/<accession>/<primary_document>.html
artifacts/manifests/filings.jsonl
artifacts/manifests/download_report.csv
```

### Completion criteria

- 10 valid HTML filings exist,
- every file has a manifest row,
- hashes match saved content,
- rerunning performs no unnecessary downloads,
- partial failures do not delete successful files.

### Common mistakes

- overwriting files silently,
- storing only cleaned text,
- naming files only by ticker,
- failing to save accession number,
- making the downloader dependent on a notebook.

### Prompt

> We are implementing Chat 04. Create a resumable SEC filing downloader that preserves raw HTML, calculates checksums and maintains a JSONL manifest. It must skip unchanged files and generate a clear download report.

---

## Chat 05: Corpus inspection and parsing specification

### Objective

Inspect actual filings before writing the final parser.

### Tasks

1. Measure file sizes and HTML complexity.
2. Inspect representative documents manually.
3. identify inline XBRL elements,
4. identify table patterns,
5. inspect heading and SEC Item patterns,
6. inspect repeated headers, TOC and navigation,
7. define parser acceptance criteria,
8. create a small set of manually annotated parser fixtures.

### Deliverables

- corpus-inspection notebook or script,
- `source_inventory.csv`,
- parser-fixture HTML fragments,
- documented parsing rules,
- list of known exceptions.

### Completion criteria

- at least three structurally different filings are inspected,
- the parser specification states what will be kept and removed,
- representative expected section boundaries are manually recorded.

### Common mistakes

- coding the parser from one filing,
- assuming every SEC Item is represented by the same tag,
- deleting tables before deciding how to use them,
- treating the table of contents as body content.

### Prompt

> We are implementing Chat 05. Inspect the downloaded SEC HTML corpus and help me produce a parsing specification, source inventory and representative fixtures. Do not jump directly to a complete parser before we understand the structures.

---

## Chat 06: SEC-aware HTML parser

### Objective

Convert raw filing HTML into ordered, validated structural elements.

### Tasks

1. remove scripts, style and non-content elements,
2. normalize whitespace without destroying table structure,
3. extract headings, paragraphs, lists and tables in document order,
4. detect SEC Item boundaries,
5. remove or mark the table of contents,
6. attach active section metadata to each element,
7. preserve original table HTML,
8. save parsed elements as JSONL or Parquet,
9. generate parser diagnostics.

### Deliverables

- parsing pipeline,
- parsed elements for all 10 filings,
- section coverage report,
- extraction warning report,
- unit tests using fixtures.

### Completion criteria

- element order is preserved,
- important SEC sections are detected,
- TOC duplicates are not mistaken for body sections,
- no filing silently produces empty content,
- parser warnings are visible and attributable.

### Common mistakes

- using one regex over the complete HTML text,
- losing document order,
- splitting tables into unrelated text fragments,
- assigning the wrong SEC Item to elements,
- hiding parser errors.

### Prompt

> We are implementing Chat 06. Build the SEC-aware HTML parser from the inspection specification. Preserve ordered elements, detect Item sections, handle tables separately and produce diagnostics and tests.

---

## Chat 07: Chunking strategy and experiment

### Objective

Implement and validate section-aware text and table chunking.

### Tasks

1. implement tokenizer-based length measurement,
2. implement text chunking within section boundaries,
3. implement table chunking,
4. add heading and filing context to chunks,
5. generate deterministic chunk IDs,
6. test at least two reasonable configurations,
7. inspect chunk distributions,
8. manually evaluate representative chunks,
9. choose and record the baseline configuration.

### Deliverables

- text chunker,
- table chunker,
- chunk validation,
- chunk corpus,
- comparison report,
- selected configuration.

### Metrics

- chunks per bank,
- token distribution,
- percentage below minimum,
- percentage above maximum,
- duplicated chunk percentage,
- section-boundary violations,
- table-chunk count.

### Completion criteria

- no chunk crosses SEC Item boundaries,
- no chunk exceeds the configured hard maximum,
- chunk metadata is complete,
- representative chunks remain understandable without nearby chunks,
- the chosen configuration is justified by evidence.

### Common mistakes

- selecting chunk size only by intuition,
- measuring characters instead of model tokens,
- adding excessive overlap,
- repeating large metadata prefixes,
- evaluating only average length.

### Prompt

> We are implementing Chat 07. Build section-aware text and table chunkers, compare two configurations and select the baseline using token statistics, boundary checks and manual inspection.

---

## Chat 08: Embedding model and embedding pipeline

### Objective

Create a local, reproducible embedding pipeline.

### Baseline recommendation

```text
sentence-transformers/all-MiniLM-L6-v2
```

Use a different model only when there is a documented reason and it fits local compute constraints.

### Tasks

1. implement an embedding-provider interface,
2. load the model once,
3. batch chunks,
4. normalize embeddings if cosine similarity is used,
5. cache or reuse completed embeddings,
6. validate vector shape and numerical health,
7. save embedding metadata,
8. benchmark throughput on the development corpus.

### Deliverables

- embedding service,
- embedding manifest,
- validation report,
- throughput benchmark.

### Completion criteria

- number of vectors equals number of chunks,
- dimensions are consistent,
- no NaN or infinite values,
- vector norms match configuration,
- rerunning does not recompute unchanged chunks.

### Common mistakes

- embedding raw filing HTML,
- loading the model inside a loop,
- mixing vectors from two models in one collection,
- using inconsistent normalization,
- failing to include the embedding model in the corpus version.

### Prompt

> We are implementing Chat 08. Create the local Sentence Transformers embedding pipeline with batching, normalization, caching, validation and throughput measurement. Keep the embedding provider replaceable.

---

## Chat 09: Qdrant collection and deterministic indexing

### Objective

Create a persistent vector index with strong metadata support.

### Tasks

1. create Qdrant Local Mode storage,
2. create one collection for the selected embedding model,
3. define vector distance,
4. map chunk metadata to payload,
5. create payload indexes,
6. implement deterministic upsert,
7. implement deletion of stale document versions,
8. create collection health checks,
9. create index-build report.

### Deliverables

- Qdrant repository,
- collection initializer,
- index builder,
- health-check script,
- indexing report.

### Completion criteria

- point count equals unique chunk count,
- no duplicate chunk IDs,
- metadata filters work,
- restart preserves the collection,
- rebuilding produces the same logical index.

### Common mistakes

- one collection per bank,
- random point IDs,
- failing to delete stale chunks,
- storing only text and vector without filing metadata,
- coupling all application logic to Qdrant classes.

### Prompt

> We are implementing Chat 09. Build a persistent Qdrant Local Mode collection, payload indexes, deterministic upserts, stale-document cleanup and health checks for the BankScope chunks.

---

## Chat 10: Dense retrieval baseline

### Objective

Implement retrieval independently of generation.

### Tasks

1. embed user queries,
2. apply structured metadata filters,
3. retrieve top candidates,
4. return validated `RetrievedChunk` objects,
5. expose scores and ranks,
6. implement bank-name and ticker resolution,
7. build a command-line retrieval inspection tool,
8. create the first retrieval evaluation cases.

### Deliverables

- dense retriever,
- filter resolver,
- CLI inspection script,
- initial retrieval dataset,
- baseline metrics.

### Completion criteria

- exact bank filters never return another bank,
- SEC Item filters work,
- retrieved text and metadata match stored chunks,
- retrieval can be evaluated without an LLM,
- failure states are explicit.

### Common mistakes

- testing retrieval only through chatbot answers,
- mixing filter extraction with vector-store code,
- ignoring empty results,
- selecting top-k without inspecting rankings,
- using one similarity threshold for every query type without evidence.

### Prompt

> We are implementing Chat 10. Build and evaluate the dense retrieval baseline with query embeddings, bank and SEC Item filters, CLI inspection and deterministic retrieval metrics. Do not connect the LLM yet.

---

## Chat 11: Retrieval evaluation dataset

### Objective

Build a defensible test set from the actual filings.

### Tasks

1. define question categories,
2. create questions from known source sections,
3. record expected banks and SEC Items,
4. identify relevant chunk IDs after chunking,
5. add comparison and unanswerable questions,
6. define train/development and final holdout subsets,
7. validate cases manually,
8. save dataset in JSONL.

### Deliverables

- versioned evaluation dataset,
- annotation guidelines,
- dataset validation script,
- category summary.

### Completion criteria

- each answerable case has evidence,
- each unanswerable case is genuinely unsupported by the corpus,
- no case depends on undocumented external knowledge,
- multi-bank cases specify coverage expectations,
- the final holdout is not used repeatedly for tuning.

### Common mistakes

- generating all questions with an LLM and not checking them,
- writing questions before seeing the corpus,
- storing only reference answers without evidence labels,
- evaluating on fewer than a handful of trivial questions.

### Prompt

> We are implementing Chat 11. Help me build a manually validated retrieval and RAG test set from the processed 10-K corpus, including direct, section-specific, comparison, follow-up and unanswerable cases.

---

## Chat 12: Hybrid retrieval and reranking experiment

### Objective

Determine whether advanced retrieval provides measurable improvement.

### Tasks

1. implement BM25 or sparse retrieval,
2. implement Reciprocal Rank Fusion,
3. optionally add a CrossEncoder,
4. compare:
   - dense only,
   - sparse only,
   - hybrid,
   - hybrid plus reranker,
5. measure quality and latency,
6. inspect regressions by category,
7. select the final configuration.

### Deliverables

- sparse retriever,
- fusion module,
- optional reranker,
- comparison report,
- final retrieval configuration.

### Completion criteria

- the advanced configuration is retained only if improvement justifies complexity,
- cross-bank coverage is measured,
- latency is reported,
- reranking operates on a bounded candidate set,
- the baseline remains available as fallback.

### Common mistakes

- assuming hybrid is automatically better,
- tuning on the final holdout,
- reranking the complete corpus,
- comparing configurations with different evaluation cases,
- reporting only aggregate averages.

### Prompt

> We are implementing Chat 12. Add sparse retrieval, RRF and optionally CrossEncoder reranking. Compare all configurations on the same retrieval dataset and keep only improvements that justify their latency and complexity.

---

## Chat 13: Provided-model adapter and grounded prompt

### Objective

Connect the provided LLM through a small project-owned interface.

### Tasks

1. define an LLM client protocol,
2. implement the provided-model adapter,
3. support timeouts and explicit errors,
4. create context formatting,
5. build grounded system and user prompts,
6. set deterministic generation parameters where supported,
7. define Pydantic structured output,
8. test answerable and unanswerable contexts.

### Deliverables

- LLM adapter,
- prompt templates,
- context builder,
- structured output parser,
- generation tests.

### Completion criteria

- no provider-specific code leaks into retrieval or UI,
- the prompt forbids unsupported external knowledge,
- retrieved sources have stable labels,
- malformed structured output is handled,
- an empty context does not produce a confident factual answer.

### Common mistakes

- pasting every retrieved field into the prompt,
- instructing the model to invent citations,
- allowing free-form output with no validation,
- mixing LLM API calls into Gradio callbacks,
- hiding provider failures as generic answers.

### Prompt

> We are implementing Chat 13. Create a replaceable adapter for the provided model, grounded prompts, labeled context and a Pydantic `GroundedAnswer`. Test unsupported and malformed-output behavior.

---

## Chat 14: Citation manager and evidence validation

### Objective

Guarantee that displayed citations refer to actual retrieved chunks.

### Tasks

1. assign stable source labels,
2. parse cited labels from structured output,
3. validate labels against retrieved context,
4. create citation objects from stored metadata,
5. check citation presence,
6. implement one optional repair attempt,
7. implement evidence-sufficiency rules,
8. add citation unit tests.

### Deliverables

- citation manager,
- answer validator,
- citation formatting,
- failure and repair paths,
- tests.

### Completion criteria

- unknown citation rate is zero,
- citation metadata is never authored by the LLM,
- unsupported answers are marked unanswerable,
- source excerpts correspond to cited chunks,
- citation behavior is tested independently of the UI.

### Common mistakes

- trusting source names produced by the model,
- inventing HTML page numbers,
- accepting citations that were not in the context,
- displaying all retrieved sources as if they were cited,
- using citation presence as proof of support.

### Prompt

> We are implementing Chat 14. Build the citation manager, evidence-sufficiency rules and answer validator. Citations must map only to retrieved chunk IDs, and invalid citations must never reach the user.

---

## Chat 15: Direct end-to-end RAG service

### Objective

Create the complete RAG request flow as ordinary testable Python before LangGraph.

### Tasks

1. validate input,
2. contextualize or pass through query,
3. resolve filters,
4. retrieve candidates,
5. build context,
6. check evidence,
7. call LLM,
8. validate citations,
9. return one `AssistantResponse`,
10. record latency by stage.

### Deliverables

- RAG service,
- integration tests,
- CLI chat or single-query script,
- latency report.

### Completion criteria

- the complete system works without Gradio,
- each stage is independently replaceable,
- errors preserve diagnostic information,
- retrieval evidence can be inspected,
- unsupported cases produce controlled responses.

### Common mistakes

- introducing LangGraph before the logic works,
- putting the complete workflow in one 300-line function,
- catching every exception and returning a generic string,
- losing original and contextualized queries.

### Prompt

> We are implementing Chat 15. Combine the validated components into a direct, testable end-to-end RAG service before using LangGraph. Include stage-level latency and integration tests.

---

## Chat 16: LangGraph workflow and conversation history

### Objective

Convert the stable RAG service into an explicit state graph with checkpointed conversation history.

### Nodes

```text
validate_input
contextualize_followup
resolve_filters
retrieve_candidates
rerank_candidates
check_evidence
generate_answer
validate_citations
repair_once
persist_result
```

### Tasks

1. define `RagState`,
2. implement nodes as thin wrappers around tested services,
3. add conditional edges,
4. enforce a maximum of one citation-repair attempt,
5. configure SQLite checkpointing,
6. use `thread_id`,
7. test two-turn and multi-turn conversations,
8. test conversation reset.

### Deliverables

- compiled LangGraph workflow,
- SQLite checkpointer,
- conversation tests,
- graph diagram.

### Completion criteria

- graph state does not contain non-serializable heavy objects,
- a follow-up query resolves correctly,
- separate thread IDs do not share history,
- no unbounded loop exists,
- the direct service remains understandable beneath the graph.

### Common mistakes

- turning every small function into a graph node,
- storing vector-store clients in state,
- relying on full history instead of contextualization,
- confusing user sessions with bank filters,
- implementing autonomous tool selection.

### Prompt

> We are implementing Chat 16. Convert the working RAG service into a deterministic LangGraph workflow with SQLite checkpointing, thread IDs, follow-up contextualization and at most one repair attempt.

---

## Chat 17: Gradio interface

### Objective

Expose the system through a simple and inspectable chat application.

### Required UI elements

- chat panel,
- message input,
- send action,
- clear conversation action,
- optional bank filter control,
- source panel,
- expandable retrieved-context diagnostics,
- visible errors,
- loading state.

### Tasks

1. create Gradio Blocks application,
2. connect callbacks to the RAG service,
3. create or reuse thread ID,
4. format citations,
5. display evidence,
6. handle errors without crashing,
7. avoid reloading models on every message,
8. test application startup from a clean process.

### Deliverables

- `app.py`,
- UI modules,
- run instructions,
- screenshot or demo recording.

### Completion criteria

- the app starts from the terminal,
- multiple messages work,
- clearing the chat creates clean state,
- source links and excerpts are visible,
- the UI does not build the index,
- exceptions produce useful messages.

### Common mistakes

- putting all backend logic in `app.py`,
- using global mutable conversation lists,
- reinitializing Qdrant and the embedding model in callbacks,
- hiding sources in plain model-generated text,
- depending on notebook variables.

### Prompt

> We are implementing Chat 17. Build the Gradio chat interface around the existing LangGraph RAG workflow. Include clear citations, an evidence panel, thread handling and robust error states without moving backend logic into the UI.

---

## Chat 18: DeepEval and end-to-end evaluation

### Objective

Evaluate generation and citation behavior while retaining deterministic retrieval metrics.

### Tasks

1. convert evaluation cases to DeepEval format,
2. add Faithfulness,
3. add Answer Relevancy,
4. add contextual metrics where expected evidence exists,
5. add citation faithfulness if practical,
6. preserve project-owned deterministic metrics,
7. run manual review on a representative subset,
8. generate an evaluation report.

### Deliverables

- evaluation runner,
- saved raw results,
- summary CSV,
- category breakdown,
- manual-review sheet,
- final retrieval and generation comparison.

### Completion criteria

- judge model and prompts are recorded,
- evaluation failures do not disappear into averages,
- unsupported questions are evaluated,
- results are reproducible from one command,
- deterministic and judge-based metrics are clearly separated.

### Common mistakes

- treating LLM judge scores as ground truth,
- evaluating generation without storing retrieved context,
- using a different corpus version from the index,
- reporting only one average score,
- repeatedly tuning against the same final cases.

### Prompt

> We are implementing Chat 18. Build the final evaluation pipeline using deterministic retrieval and citation metrics plus selective DeepEval generation metrics. Save raw results, category breakdowns and a manual-review subset.

---

## Chat 19: Langfuse tracing

### Objective

Add optional observability without making it a runtime dependency.

### Tasks

1. create a no-op tracing interface,
2. add Langfuse implementation,
3. trace each workflow stage,
4. attach model and retrieval configuration,
5. attach latency and selected chunk IDs,
6. avoid recording secrets,
7. allow tracing to be disabled,
8. verify that failure to reach Langfuse does not break the app.

### Deliverables

- tracing abstraction,
- Langfuse integration,
- example traces,
- privacy and logging notes.

### Completion criteria

- local app works with `LANGFUSE_ENABLED=false`,
- traces show complete request timing,
- no API keys or full environment data are logged,
- a tracing outage does not block answers.

### Common mistakes

- adding observability before debugging the baseline,
- sending secrets,
- depending on cloud tracing for application state,
- self-hosting a heavy stack for a five-week project.

### Prompt

> We are implementing Chat 19. Add optional Langfuse tracing behind a no-op interface. Trace RAG stages, configuration and latency, but ensure the assistant works normally when Langfuse is disabled or unavailable.

---

## Chat 20: Company Facts and calculator tools

### Objective

Add a narrow, useful function-calling extension for structured numerical questions.

### Tasks

1. implement Company Facts SEC API client,
2. define strict Pydantic tool schemas,
3. implement deterministic routing rules,
4. implement calculator over explicit inputs,
5. return citations or filing references for tool data,
6. log tool calls,
7. test invalid concepts and missing facts,
8. prevent unrestricted tool loops.

### Deliverables

- Company Facts tool,
- calculator,
- router,
- tool audit record,
- tests.

### Completion criteria

- tools are used only for supported query classes,
- every numerical result identifies source concept, unit and period,
- arithmetic is deterministic,
- missing facts do not become invented values,
- tool use is optional to the main RAG path.

### Common mistakes

- treating Company Facts as a general financial database,
- letting the LLM invent taxonomy concepts,
- calculating over values that were not retrieved,
- adding web search because function calling is available.

### Prompt

> We are implementing Chat 20. Add a constrained SEC Company Facts tool and calculator with Pydantic schemas, deterministic routing, source metadata and tests. Do not create an autonomous agent loop.

---

## Chat 21: Scale from 10 to approximately 100 banks

### Objective

Demonstrate that the architecture scales through data-driven indexing rather than architectural rewrites.

### Tasks

1. expand the bank registry,
2. validate identities in batches,
3. download missing filings incrementally,
4. process only new or changed documents,
5. index only changed chunks,
6. benchmark each offline stage,
7. inspect chunk distribution by bank,
8. test multi-bank retrieval,
9. measure collection size and latency,
10. document bottlenecks.

### Deliverables

- expanded registry,
- scale report,
- indexing benchmark,
- storage report,
- retrieval comparison,
- failure list.

### Completion criteria

- scale requires configuration changes, not new code branches per bank,
- failed banks are reported without stopping the batch,
- indexing is resumable,
- no duplicate points exist,
- retrieval latency remains practical,
- cross-bank coverage is tested.

### Common mistakes

- scaling before the parser is stable,
- one collection per bank,
- redownloading every filing,
- rerunning all embeddings for one changed document,
- claiming scalability only from vector-count estimates.

### Prompt

> We are implementing Chat 21. Extend the validated 10-bank pipeline to approximately 100 banks using incremental acquisition, processing and indexing. Benchmark every stage and test cross-bank retrieval without redesigning the system.

---

## Chat 22: Testing and hardening

### Objective

Make the project robust enough for repeatable demonstration and grading.

### Test groups

#### Unit tests

- CIK normalization,
- accession formatting,
- SEC Item detection,
- text normalization,
- deterministic IDs,
- token limits,
- filter construction,
- citation mapping,
- evidence rules.

#### Integration tests

- fixture HTML to parsed elements,
- parsed elements to chunks,
- chunks to Qdrant,
- query to retrieved chunks,
- context to grounded response,
- multi-turn graph workflow.

#### Smoke tests

- configuration,
- directories,
- model loading,
- Qdrant health,
- optional LLM connectivity,
- optional Langfuse connectivity.

### Tasks

1. add fixtures,
2. remove network dependence from most tests,
3. add temporary Qdrant paths,
4. test corrupt and empty documents,
5. test unavailable model endpoint,
6. test missing environment variables,
7. test application startup,
8. generate coverage summary.

### Deliverables

- complete test suite,
- fixture corpus,
- smoke-test command,
- failure-handling documentation.

### Completion criteria

- critical transformations have tests,
- test runs are deterministic,
- live SEC access is not required for ordinary unit tests,
- important errors have actionable messages,
- a clean clone can follow the README successfully.

### Prompt

> We are implementing Chat 22. Harden the complete BankScope project with unit, integration and smoke tests, fixture SEC HTML, temporary Qdrant storage and explicit tests for failure cases.

---

## Chat 23: Final analysis, README and presentation

### Objective

Present the project as an engineering and ML system, not merely a collection of code.

### README sections

1. project goal,
2. assignment mapping,
3. architecture,
4. corpus,
5. installation,
6. environment variables,
7. data acquisition,
8. index construction,
9. running the application,
10. evaluation,
11. results,
12. limitations,
13. ethical and financial disclaimer,
14. repository structure,
15. future work.

### Required result tables

- corpus statistics,
- chunking statistics,
- retrieval comparison,
- generation evaluation,
- citation evaluation,
- latency,
- 10-bank versus scale-corpus indexing.

### Demo scenarios

1. direct answer with citations,
2. SEC Item filtered question,
3. follow-up question,
4. cross-bank comparison,
5. unanswerable question,
6. optional numerical tool question.

### Deliverables

- final README,
- architecture diagram,
- evaluation report,
- known-limitations section,
- demo script,
- cleaned repository.

### Completion criteria

- all claims in the README are supported by saved results,
- commands are tested from a clean environment,
- no secrets or unnecessary data are committed,
- optional features are clearly labeled,
- limitations are explicit,
- the five-minute demo is reproducible.

### Prompt

> We are implementing Chat 23. Review the final repository and help me create the README, architecture diagram, results tables, limitations and a concise demo script. Verify that every claim is supported by project artifacts.

---

# 22. Phase dependency graph

```text
01 Project setup
    ↓
02 Bank registry
    ↓
03 SEC discovery
    ↓
04 Download and manifest
    ↓
05 Corpus inspection
    ↓
06 Parsing
    ↓
07 Chunking
    ↓
08 Embeddings
    ↓
09 Qdrant indexing
    ↓
10 Dense retrieval
    ↓
11 Evaluation dataset
    ↓
12 Hybrid and reranking
    ↓
13 LLM adapter
    ↓
14 Citations
    ↓
15 Direct RAG service
    ↓
16 LangGraph and history
    ↓
17 Gradio
    ↓
18 DeepEval
    ↓
19 Langfuse
    ↓
20 Optional tools
    ↓
21 Scale test
    ↓
22 Hardening
    ↓
23 Final documentation
```

Parallel work that is safe:

- Chat 11 can begin during Chats 07 to 10.
- Chat 13 can begin after the provided model interface is known.
- Chat 19 can be added after Chat 16.
- Chat 20 is independent after the main RAG system is stable.
- Chat 22 should run continuously, with a final hardening pass near the end.

---

# 23. Critical path and cut order

When time becomes limited, cut features in this order:

1. unrestricted or additional tools,
2. Company Facts integration,
3. Langfuse,
4. CrossEncoder reranker,
5. sparse retrieval,
6. scale from 100 banks down to a smaller demonstrated subset,
7. advanced UI diagnostics,
8. conversation summarization.

Do not cut:

- SEC acquisition,
- raw-file preservation,
- parsing validation,
- chunk metadata,
- deterministic IDs,
- dense retrieval,
- citations,
- evaluation dataset,
- retrieval evaluation,
- basic generation evaluation,
- conversation history,
- Gradio interface.

---

# 24. Expected time budget

Approximate focused work:

| Area | Hours |
|---|---:|
| setup and SEC acquisition | 8–12 |
| inspection and parsing | 14–20 |
| chunking, embeddings and indexing | 10–14 |
| retrieval and evaluation set | 10–14 |
| generation, citations and history | 12–16 |
| UI | 4–7 |
| evaluation and observability | 8–12 |
| scaling and hardening | 8–14 |
| documentation and demo | 5–8 |

**Recommended portfolio scope:** approximately 75–95 focused hours.

The schedule is realistic only if advanced features are added after baseline gates are met.

---

# 25. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| inconsistent SEC HTML | high | high | fixtures, diagnostics, fallback rules |
| incorrect Item boundaries | high | high | TOC detection, manual validation |
| table extraction complexity | high | medium | separate table path, limited claims |
| redundant language across banks | high | high | metadata filters, hybrid retrieval, reranking |
| LLM hallucinated citations | medium | high | structured output and strict validation |
| framework version changes | medium | medium | pinned dependencies and project-owned interfaces |
| long evaluation runtime | medium | medium | small curated set and cached results |
| judge-model bias | high | medium | deterministic metrics and manual review |
| 100-bank scale delay | medium | medium | keep 10-bank baseline complete |
| model API instability | medium | high | adapter, timeout, clear fallback behavior |
| overengineering | high | high | phase gates and explicit cut order |

---

# 26. Definition of done

The project is done when:

- the latest 10-K filings for the required 10 public banks are acquired through the SEC EDGAR API,
- documents are processed with explicit handling of headings, text and tables,
- chunks and metadata are validated,
- embeddings are generated and persistently indexed,
- retrieval works with bank and section filters,
- the provided model produces grounded answers,
- citations map to real retrieved chunks,
- conversation history supports follow-up questions,
- the Gradio interface works from a clean process,
- a curated evaluation dataset exists,
- retrieval and generation results are saved,
- tests cover critical paths,
- the README reproduces setup and execution,
- optional advanced features are clearly separated from required functionality,
- scale claims are supported by measurements rather than assumptions.

---

# 27. Final architectural decision summary

Use:

- direct SEC API access,
- custom SEC-aware parser,
- Pydantic at every major data boundary,
- section-aware chunking,
- local Sentence Transformers embeddings,
- one Qdrant collection with metadata filters,
- dense retrieval as the mandatory baseline,
- hybrid retrieval and reranking only if metrics support them,
- LangChain components selectively,
- LangGraph for deterministic workflow and history,
- strict project-owned citation validation,
- Gradio for the user interface,
- deterministic metrics plus DeepEval,
- optional Langfuse tracing,
- Company Facts and calculator as narrow optional tools.

Do not optimize the project around the number of libraries used. Optimize it around:

- reproducibility,
- evidence quality,
- citation correctness,
- measurable retrieval performance,
- explicit failure behavior,
- controlled scaling from 10 to approximately 100 banks.
