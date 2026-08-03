# BankScope roadmap

This roadmap records project phases, not predetermined technology choices.

Before every new phase:

1. define the concrete project requirement and constraints;
2. review suitable current libraries, models and software;
3. compare realistic candidates against the existing code and data;
4. agree on one minimal baseline;
5. implement a small smoke test before a full run;
6. keep the decision only if evaluation supports it.

## Phase status

| Phase | Status | Exit condition |
|---|---|---|
| Project setup and bank registry | Complete | Ten configured banks validate correctly |
| SEC acquisition | Complete | Latest primary 10-K HTML and manifest exist for all banks |
| HTML parsing | Complete | Ordered text, headings, tables and citation metadata are extracted |
| Chunking | Complete | Text and tables satisfy the accepted `600/700/80` policy |
| Table proxies | Complete | One deterministic, validated proxy exists per table chunk |
| Repository cleanup | Complete | Active paths contain current code; older artifacts are labeled in `sandbox/` |
| Embeddings | Next | Chosen baseline passes a small test and full corpus generation |
| Vector storage | Pending | Records can be persisted, filtered and reconstructed reliably |
| Retrieval | Pending | Evaluation queries retrieve supporting chunks with citations |
| RAG generation | Pending | Answers remain grounded and expose their sources |
| Conversation history | Pending | Follow-up questions work without contaminating retrieval |
| User interface | Pending | A simple usable chat interface runs locally |
| Final evaluation and report | Pending | Retrieval and generation are evaluated separately |

## Embeddings decision gate

No model or framework is currently selected for this phase. The discussion
must cover at least:

- retrieval quality for financial and SEC language;
- handling of narrative text and deterministic table proxies;
- model context length relative to the real chunk distribution;
- local CPU, Colab T4 and memory constraints;
- license, download size, runtime and reproducibility;
- query/document instruction requirements;
- compatibility with the vector storage options considered in the next phase;
- a small project-specific comparison before full generation.

The existing `scripts/generate_embeddings.py` is an experiment to inspect, not
an accepted architecture. It must not be used for the full corpus until this
decision gate is complete.

## Scope boundary

The required result is a clear RAG assistant for ten banks. Multi-agent
orchestration, knowledge graphs, fine-tuning, support for 100 banks and
production observability are outside the baseline unless evaluation reveals a
specific need.
