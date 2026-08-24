import sys
from types import SimpleNamespace

from bankscope.config.settings import ApplicationSettings
from bankscope.llm.client import create_langchain_chat_model, create_openai_client


def test_client_uses_internal_model_access(monkeypatch) -> None:
    expected_client = object()
    calls = []

    def access_model():
        calls.append(True)
        return expected_client

    monkeypatch.setattr("bankscope.llm.client.access_model", access_model)
    settings = ApplicationSettings(
        _env_file=None,
        sec_user_agent="BankScopeRAG test@example.com",
    )

    client = create_openai_client(settings)

    assert client is expected_client
    assert calls == [True]


def test_langchain_chat_model_reuses_proxy_credentials_and_headers(monkeypatch) -> None:
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    monkeypatch.setattr(
        "bankscope.llm.client.model_access_parameters",
        lambda: {
            "api_key": "secret",
            "base_url": "https://proxy.example/v1",
            "default_headers": {"x-app-id": "app", "x-api-version": "v1"},
        },
    )
    settings = ApplicationSettings(
        _env_file=None,
        sec_user_agent="BankScopeRAG test@example.com",
    )

    create_langchain_chat_model(settings)

    assert captured == {
        "api_key": "secret",
        "base_url": "https://proxy.example/v1",
        "default_headers": {"x-app-id": "app", "x-api-version": "v1"},
        "model": "AZURE_GPT_51_2025_1113",
        "timeout": 45.0,
        "max_retries": 1,
        "max_completion_tokens": 1600,
    }
