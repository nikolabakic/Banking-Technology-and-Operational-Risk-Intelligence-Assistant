# Active prompt inventory

Every active model call has a version, bounded inputs, an explicit output contract, and a test or
evaluation gate. Pydantic schemas define structure; prompts retain only semantic and safety rules.

| Call | Prompt version | Inputs | Output contract | Evaluator / gate |
|---|---|---|---|---|
| Conversation router | `conversation-langgraph-router-v4-model-context` | Summary, bounded raw history, previous grounded answer, deterministic bank scope, current question | Strict `RouteDecision` function | `evaluate_conversation_routing.py`; ≥95% accuracy, 100% bank filing recall, 100% unrelated no-retrieval, zero scope violations |
| Conversation compaction | `conversation-summary-tool-v1` | Prior summary and old complete pairs | Strict `ConversationSummary` function | Memory unit tests; checkpoint and six-pair retention tests |
| Grounded answer | `generation-grounded-tool-v7-presentation` | Current/resolved question, one-bank retrieved evidence, style guidance | Four mutually exclusive strict answer functions | Answer, citation, numeric, bank-isolation, and generation gates |
| Comparison synthesis | `generation-comparison-synthesis-v5-presentation` | Validated per-bank answers and owned citation labels | Strict `ComparisonSynthesis` function | Comparison schema, coverage, citation ownership, and pipeline tests |
| Agent retrieval step | `agentic-rag-loop-v3-native-tools` | Current bank, question, bounded evidence preview, remaining budgets | Strict search/read/finish functions | Agentic challenge and action/request-budget tests |
| Evidence verifier | `agentic-rag-verifier-v2-native-tools` | Question and bounded evidence preview | Strict `EvidenceVerdict` function | Agentic evidence and bank-isolation tests |
| Legacy standalone rewrite | `conversation-standalone-question-v2-native-tool` | Complete raw history pairs, session scope, current question | Strict `StandaloneQuestion` function | Conversation-memory regression suite |
| Semantic judge | `generation-semantic-judge-v2-native-tool` | Question, reference, generated answer, cited evidence | Strict `SemanticJudgement` function | Semantic-judge tests; advisory metrics only |
| Table description | `table-semantic-description-v1` | Filing metadata and one table | Plain text; 30-second timeout and deterministic parameters | Table parsing tests and non-empty-output validation |

The obsolete agentic `route_question` and `plan_evidence` model paths were removed. Conversation
routing is owned by `ConversationGraph`; evidence iteration is owned by the strict agent step and
verifier tools.

## Candidate acceptance

`scripts/evaluate_conversation_routing.py` accepts `--baseline <prior-result.json>`. It records the
candidate prompt version and rejects a candidate if it misses an absolute gate or regresses on any
baseline acceptance metric. This provides the baseline/candidate/evaluator/no-regression loop used
by prompt-optimization systems without adding Promptim, Prompt Ops, or DSPy as runtime dependencies.
