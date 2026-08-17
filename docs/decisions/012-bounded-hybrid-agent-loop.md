# ADR 012: Bounded hybrid agentic retrieval loop

- Status: Accepted, experimental feature disabled by default
- Date: 2026-08-17

## Context

The first ADR 011 live gate preserved baseline hits but recovered only one annotated miss. Serbian
paraphrases frequently needed canonical English filing terminology, the one-shot planner could
abstain before trying a targeted search, radius-one context reads missed complete definitions, and
generation-schema failures obscured otherwise valid retrieval results.

The reviewed agentic-RAG cookbook demonstrates a reusable search/read loop with bounded outputs,
agent-readable tool errors, structured results, and a hard request cap. BankScope still benefits
from semantic retrieval over filing prose, so a grep-only replacement would discard a validated
strength of the current system.

## Decision

Initial Qdrant dense + BM25S + RRF retrieval remains mandatory. When agentic mode is enabled, each
bank gets an isolated loop with four runtime-scoped actions: hybrid search, literal exact search,
canonical context read, and finish. The loop permits at most six orchestration requests, four tool
actions, two independent evidence-verifier requests, and context windows of at most three chunks
per side. Runtime owns ticker, accession, record filters, limits, and canonical IDs.

Search queries may add English filing terminology but must retain explicit periods and cannot add
numeric facts. Exact search accepts literal phrases rather than regular expressions. Tool failures
are returned in the trace so the model can recover. Repeated actions consume the budget and return
no new evidence. Two consecutive schema failures terminate safely with existing evidence or an
unsupported result.

`BankAnswerPipeline.retrieve_evidence()` is the authoritative retrieval-only interface. The agentic
evaluator obtains Hit@5/10 and runtime-contract results from it and records end-to-end generation
separately, so generation JSON failures cannot erase retrieval measurements.

## Rollout

The feature stays disabled by default. Rollout requires all existing frozen gates, no lost baseline
Hit@5, at least three recovered known misses, bank/accession isolation, per-bank request/action
budgets, no unnecessary tool action for sufficient initial evidence, and citation-free unsupported
answers. A passing live report and a separate default-change decision are still required.
