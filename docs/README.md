# Documentation index

**Status:** active documentation registry.

The primary explanation of each subsystem lives beside that subsystem in its README. This folder
contains cross-cutting decisions, current status, and durable evaluation reports.

## Current reference

| Document | Purpose |
|---|---|
| [roadmap.md](roadmap.md) | Current phase status, scope boundary, and acceptance gates |
| [generation_hardening.md](generation_hardening.md) | Frozen GPT-5.1 candidate result and citation caveat |

Detailed runtime contracts are linked from the [repository guide](../README.md), especially the
[data](../data/README.md), [scripts](../scripts/README.md), and
[BankScope package](../src/bankscope/README.md) guides.

## Decisions

[`decisions/`](decisions/README.md) contains ADR-style records for accepted parser, repository,
retrieval, generation, bank-resolution, memory, and comparison choices. ADRs explain why a choice
was made and preserve the measured outcome; they are not tutorials or future plans.

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

