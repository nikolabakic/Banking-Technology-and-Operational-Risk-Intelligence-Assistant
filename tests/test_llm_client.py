from bankscope.config.settings import ApplicationSettings
from bankscope.llm.client import create_openai_client


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
