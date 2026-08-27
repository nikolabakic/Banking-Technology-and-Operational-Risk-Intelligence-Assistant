from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import scripts.serve_api as serve_api


def _service_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        openai_model="configured-model",
        llm_temperature=0.0,
        bank_registry_path=tmp_path / "banks.yaml",
        agentic_rag_enabled=False,
        conversation_router_backend="langgraph",
        web_search_enabled=True,
        web_search_provider="openai",
        web_search_model=None,
        web_search_timeout_seconds=12.5,
        web_search_context_size="high",
    )


def _service_args(tmp_path: Path, *, model: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        chunks=tmp_path / "chunks.jsonl",
        tables=tmp_path / "tables.jsonl",
        glossary_locators=tmp_path / "glossary.json",
        qdrant_path=tmp_path / "qdrant",
        qdrant_manifest=tmp_path / "manifest.json",
        collection="bankscope",
        chat_db=tmp_path / "chat.db",
    )


@pytest.mark.parametrize(
    ("cli_model", "expected_model"),
    [(None, "configured-model"), ("cli-model", "cli-model")],
)
def test_build_services_uses_cli_override_or_configured_model(
    monkeypatch, tmp_path: Path, cli_model: str | None, expected_model: str
) -> None:
    settings = _service_settings(tmp_path)
    captured: dict[str, object] = {}
    pipeline = object()
    client = object()
    conversation_model = object()

    def from_paths(**kwargs):
        captured["pipeline_kwargs"] = kwargs
        return pipeline

    def create_conversation_model(_settings, *, model):
        captured["conversation_model_name"] = model
        return conversation_model

    class FakeStore:
        def __init__(self, path: Path) -> None:
            captured["chat_db"] = path

        def initialize(self) -> None:
            captured["store_initialized"] = True

    sources = object()
    monkeypatch.setattr(serve_api, "get_settings", lambda: settings)
    monkeypatch.setattr(serve_api, "create_openai_client", lambda _settings: client)
    monkeypatch.setattr(serve_api, "create_langchain_chat_model", create_conversation_model)
    monkeypatch.setattr(serve_api, "BankAnswerPipeline", SimpleNamespace(from_paths=from_paths))
    monkeypatch.setattr(serve_api, "ChatStore", FakeStore)
    monkeypatch.setattr(
        serve_api,
        "CitationSourceResolver",
        SimpleNamespace(from_paths=lambda _chunks, _tables: sources),
    )

    args = _service_args(tmp_path, model=cli_model)
    services = serve_api.build_services(args)

    pipeline_kwargs = captured["pipeline_kwargs"]
    assert isinstance(pipeline_kwargs, dict)
    assert pipeline_kwargs["generation_model"] == expected_model
    assert pipeline_kwargs["conversation_model"] is conversation_model
    web_provider = pipeline_kwargs["web_search_provider"]
    assert isinstance(web_provider, serve_api.OpenAIWebSearchProvider)
    assert web_provider.client is client
    assert web_provider.model == expected_model
    assert web_provider.timeout_seconds == 12.5
    assert web_provider.search_context_size == "high"
    assert captured["conversation_model_name"] == expected_model
    assert services.pipeline is pipeline
    assert services.sources is sources
    assert captured["store_initialized"] is True


@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_build_services_closes_web_provider_when_pipeline_construction_fails(
    monkeypatch, tmp_path: Path, cleanup_raises: bool
) -> None:
    startup_error = ValueError("pipeline construction failed")

    class Provider:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if cleanup_raises:
                raise RuntimeError("provider cleanup failed")

    provider = Provider()
    monkeypatch.setattr(serve_api, "get_settings", lambda: _service_settings(tmp_path))
    monkeypatch.setattr(serve_api, "create_openai_client", lambda _settings: object())
    monkeypatch.setattr(
        serve_api, "create_langchain_chat_model", lambda _settings, *, model: object()
    )
    monkeypatch.setattr(
        serve_api,
        "_build_web_search_provider",
        lambda *_args, **_kwargs: provider,
    )

    def fail_pipeline_construction(**_kwargs):
        raise startup_error

    monkeypatch.setattr(
        serve_api,
        "BankAnswerPipeline",
        SimpleNamespace(from_paths=fail_pipeline_construction),
    )

    with pytest.raises(ValueError, match="pipeline construction failed") as exc_info:
        serve_api.build_services(_service_args(tmp_path))

    assert exc_info.value is startup_error
    assert provider.close_calls == 1


@pytest.mark.parametrize("failure_stage", ["store", "sources"])
def test_build_services_cleans_up_constructed_resources_when_later_startup_fails(
    monkeypatch, tmp_path: Path, failure_stage: str
) -> None:
    startup_error = ValueError(f"{failure_stage} construction failed")

    class Pipeline:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("pipeline cleanup failed")

    class Store:
        def __init__(self, _path: Path) -> None:
            pass

        def initialize(self) -> None:
            if failure_stage == "store":
                raise startup_error

    pipeline = Pipeline()
    monkeypatch.setattr(serve_api, "get_settings", lambda: _service_settings(tmp_path))
    monkeypatch.setattr(serve_api, "create_openai_client", lambda _settings: object())
    monkeypatch.setattr(
        serve_api, "create_langchain_chat_model", lambda _settings, *, model: object()
    )
    monkeypatch.setattr(serve_api, "_build_web_search_provider", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        serve_api,
        "BankAnswerPipeline",
        SimpleNamespace(from_paths=lambda **_kwargs: pipeline),
    )
    monkeypatch.setattr(serve_api, "ChatStore", Store)

    def build_sources(_chunks: Path, _tables: Path) -> object:
        if failure_stage == "sources":
            raise startup_error
        return object()

    monkeypatch.setattr(
        serve_api,
        "CitationSourceResolver",
        SimpleNamespace(from_paths=build_sources),
    )

    with pytest.raises(ValueError, match=f"{failure_stage} construction failed") as exc_info:
        serve_api.build_services(_service_args(tmp_path))

    assert exc_info.value is startup_error
    assert pipeline.close_calls == (1 if failure_stage == "sources" else 0)


def test_auto_web_provider_uses_sticky_tavily_fallback_when_key_is_configured() -> None:
    settings = SimpleNamespace(
        web_search_enabled=True,
        web_search_provider="auto",
        web_search_model="web-model",
        web_search_timeout_seconds=14,
        web_search_context_size="low",
        tavily_api_key=SecretStr("tvly-test-secret"),
        tavily_max_results=7,
    )

    provider = serve_api._build_web_search_provider(
        settings,
        client=object(),
        generation_model="generation-model",
    )

    assert isinstance(provider, serve_api.FallbackWebSearchProvider)
    assert isinstance(provider._providers[0], serve_api.OpenAIWebSearchProvider)
    tavily = provider._providers[1]
    assert isinstance(tavily, serve_api.TavilyWebSearchProvider)
    assert tavily.timeout_seconds == 14
    assert tavily.max_results == 7
    provider.close()


def test_explicit_tavily_provider_requires_key() -> None:
    settings = SimpleNamespace(
        web_search_enabled=True,
        web_search_provider="tavily",
        tavily_api_key=None,
    )

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        serve_api._build_web_search_provider(
            settings,
            client=object(),
            generation_model="generation-model",
        )


@pytest.mark.parametrize(
    "settings",
    [
        SimpleNamespace(web_search_enabled=False, web_search_provider="auto"),
        SimpleNamespace(web_search_enabled=True, web_search_provider="disabled"),
    ],
)
def test_web_provider_can_be_disabled(settings: SimpleNamespace) -> None:
    assert (
        serve_api._build_web_search_provider(
            settings,
            client=object(),
            generation_model="generation-model",
        )
        is None
    )
