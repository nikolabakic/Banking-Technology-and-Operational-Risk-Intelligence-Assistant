# Documentation index

**Status:** active documentation registry.

The primary explanation of each subsystem lives beside that subsystem in its README. This folder
contains cross-cutting decisions, current status, and durable evaluation reports.

## Current reference

| Document | Purpose |
|---|---|
| [evidence_audit_evaluation.md](evidence_audit_evaluation.md) | Advisory runtime evidence audit, separate 10-case final challenge, reproduction, and reporting limits |
| [roadmap.md](roadmap.md) | Current phase status, scope boundary, and acceptance gates |
| [generation_hardening.md](generation_hardening.md) | Frozen GPT-5.1 candidate result and citation caveat |
| [reliability-hardening.md](reliability-hardening.md) | Current failure analysis, GitHub research, controls, and verification contract |
| [ADR 013](decisions/013-rag-reliability-hardening.md) | Current deterministic RAG and UI reliability boundaries |
| [ADR 016](decisions/016-finance-technology-conversation-scope.md) | Current conversation scope for direct answers and web search |
| [ADR 015](decisions/015-general-chat-web-and-calculator.md) | Calculator, web-search, and Ally reliability decision; general-chat scope superseded by ADR 016 |
| [ADR 012](decisions/012-bounded-hybrid-agent-loop.md) | Superseded bounded hybrid agentic retrieval design |
| [ADR 011](decisions/011-eval-first-agentic-rag.md) | Superseded one-step experiment and measured evaluation result |

Detailed runtime contracts are linked from the [repository guide](../README.md), especially the
[data](../data/README.md), [scripts](../scripts/README.md), and
[BankScope package](../src/bankscope/README.md) guides.

## Decisions

[`decisions/`](decisions/README.md) contains ADR-style records for accepted parser, repository,
retrieval, generation, bank-resolution, memory, and comparison choices. ADRs explain why a choice
was made and preserve the measured outcome; they are not tutorials or future plans. ADR 011
records the first one-step agentic experiment and its failed gate. ADR 012 introduced the hybrid
search/read/verifier loop. ADR 013 keeps it disabled by default while tightening bank resolution,
comparison decomposition, memory selection, evidence ordering, budgets, and UI transport safety.
ADR 015 records why the reported Ally failure was a citation-schema ambiguity rather than a memory
failure and compares the selected web-search provider with the fallback options. ADR 016 supersedes
its general-chat boundary with the active finance-and-technology scope.

## Archived material

Superseded roadmaps and speculative plans live in [`sandbox/docs/`](../sandbox/docs/README.md).
They may contain obsolete paths, versions, and assumptions and are never authoritative for current
behavior.

## Documentation maintenance checklist

When behavior, a file path, or a public interface changes:

1. Update the nearest functional README and its file/API table.
2. Update diagrams when data flow, ownership, or ordering changes.
3. Update root documentation only when onboarding or system-level architecture changes.
4. Add or amend an ADR when measured evidence changes an accepted architecture decision.
5. Keep generated outputs out of documentation; record commands, versions, hashes, and summaries.
6. Verify relative links and Mermaid rendering in the GitHub preview.
7. Run documented `--help`, test, lint, and build commands before merging.

[Back to repository guide](../README.md)
