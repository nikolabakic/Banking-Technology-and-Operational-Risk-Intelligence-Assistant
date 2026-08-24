# LLM client adapter

**Status:** active gateway boundary.

```mermaid
flowchart LR
    Settings[ApplicationSettings] --> Factory[create_openai_client]
    Corporate[model_access.access_model] --> Factory
    Factory --> Client[OpenAI-compatible client]
    Client --> Generation[structured generation]
    Client --> Web[Responses web_search]
    Settings --> Tavily[Tavily Search fallback]
```

`create_openai_client(settings)` is the package-level factory. `model_access.access_model()` is the
internal corporate gateway hook and reads the primary/fallback environment values expected by the
local environment. Keeping this adapter small prevents gateway details from leaking into parsing,
retrieval, or API code.

The same native client is injected into `OpenAIWebSearchProvider`, while LangChain receives the
same gateway parameters for conversation routing. The gateway must implement the endpoint being
called. In `auto` mode a configured Tavily provider catches a Responses failure and becomes the
remembered web provider; if all providers fail, the error remains a specific tool state rather
than being hidden by a chat-completions fallback.

Client construction must not log credentials. Model-specific request options and response
validation belong in the generation package, not here. Cover changes with `tests/test_llm_client.py`
and `tests/test_model_access.py`.

[Package architecture](../README.md) · [Generation](../generation/README.md)
