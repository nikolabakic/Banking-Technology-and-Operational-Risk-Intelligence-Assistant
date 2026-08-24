# Runtime settings

**Status:** active configuration adapter.

`settings.py` centralizes environment-backed configuration with Pydantic Settings.

```mermaid
flowchart LR
    Env[.env and process environment] --> Settings[ApplicationSettings]
    Settings --> SEC[download client]
    Settings --> Paths[data paths]
    Settings --> Model[LLM client]
    Settings --> Web[Optional web-search provider]
```

## Public API

| Symbol | Contract |
|---|---|
| `ApplicationSettings` | Typed SEC identity/rate, registry/raw paths, and optional model credentials |
| `get_settings()` | Cached process-wide settings instance |

The SEC user agent must include an email address and the request rate must remain positive and no
greater than the SEC limit enforced by the model. Secrets use `SecretStr` to reduce accidental
logging. `.env.example` is documentation; `.env` is local and ignored.

`AGENTIC_RAG_ENABLED` is a boolean rollout flag with the safe default `false`. Settings are cached
and the pipeline is constructed once, so changing the value requires an API restart. For a local
session, either set `AGENTIC_RAG_ENABLED=true` in `.env` or set the process environment before
starting BankScope; process environment values take precedence over `.env`.

`CONVERSATION_ROUTER_BACKEND` defaults to `langgraph`. The temporary `legacy` value bypasses graph
execution but retains the same strict route schema, validation policy, and non-veto fallback; it is
intended only as a short-lived rollback switch while the LangGraph path is stabilized.

`LLM_CONVERSATION_MAX_OUTPUT_TOKENS` and `LLM_REQUEST_TIMEOUT_SECONDS` configure the natural-chat
budget and adapter timeout. `WEB_SEARCH_ENABLED`, `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_MODEL`,
`WEB_SEARCH_TIMEOUT_SECONDS`, and `WEB_SEARCH_CONTEXT_SIZE` configure the optional OpenAI provider.
`TAVILY_API_KEY` and `TAVILY_MAX_RESULTS` configure the BYO fallback. Provider `auto` tries OpenAI
first and includes Tavily when its key is present; a blank web model inherits `OPENAI_MODEL`.
Settings changes require restart.

When adding a setting, give it a safe default where possible, add it to `.env.example` when users
must know it, and cover validation/caching in `tests/test_settings.py`.

[Package architecture](../README.md) · [Static configuration](../../../config/README.md)
