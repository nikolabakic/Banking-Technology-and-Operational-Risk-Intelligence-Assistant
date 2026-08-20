# Architecture decision records

**Status:** accepted historical decisions for the active system.

```mermaid
flowchart LR
    Parser[001 parser] --> Overhaul[002 repository/corpus]
    Overhaul --> Qdrant[003 local Qdrant]
    Qdrant --> Mixed[004 mixed retrieval]
    Mixed --> Generation[005 generation evaluation]
    Generation --> Resolver[006 bank resolution]
    Resolver --> Memory[007 conversation memory]
    Memory --> Comparison[008 multi-bank comparisons]
    Comparison --> CompleteFilings[009 complete primary filings]
    CompleteFilings --> Balanced[010 bank-balanced retrieval]
    Balanced --> Agentic[011 eval-first agentic RAG]
    Agentic --> Loop[012 bounded hybrid agent loop]
    Loop --> Reliability[013 reliability boundaries]
    Reliability --> Conversation[014 conversational orchestration]
```

| ADR | Decision |
|---|---|
| [001](001-parser-selection.md) | Select sec2md using the frozen retrieval comparison |
| [002](002-repository-overhaul.md) | Adopt the active src/scripts/data structure and whole-table corpus |
| [003](003-qdrant-local-retrieval.md) | Accept local Qdrant for persistent dense retrieval |
| [004](004-mixed-vector-retrieval.md) | Use Qdrant dense + BM25S + application RRF by default |
| [005](005-generation-evaluation.md) | Separate deterministic answer metrics from advisory judging |
| [006](006-automatic-bank-resolution.md) | Resolve banks before retrieval with bounded session fallback |
| [007](007-short-term-conversation-memory.md) | Introduce same-thread contextualization; rewrite policy superseded by ADR 013 |
| [008](008-multi-bank-comparisons.md) | Introduce bank-isolated comparison synthesis; shared query superseded by ADR 013 |
| [009](009-complete-primary-filings.md) | Replace partial USB/WFC filings with complete COF/STT filings |
| [010](010-bank-balanced-comparison-retrieval.md) | Balance cross-bank retrieval before comparison synthesis |
| [011](011-eval-first-agentic-rag.md) | Add bounded, diagnostics-first agentic RAG behind a disabled rollout flag |
| [012](012-bounded-hybrid-agent-loop.md) | Historical bounded hybrid loop; current budget superseded by ADR 013 |
| [013](013-rag-reliability-hardening.md) | Add deterministic resolution, query, memory, agentic, and UI reliability boundaries |
| [014](014-conversational-function-orchestration.md) | Make every threaded request a conversational action backed by strict native functions and normal recovery turns |

ADRs are append-only records of a decision at a point in time. If a decision is reversed, add a new
ADR and mark the old one superseded rather than rewriting its measured history.

[Documentation index](../README.md)
