import sys
from types import SimpleNamespace

from bankscope.llm.model_access import access_model


def test_access_model_supports_existing_openai_environment_names(monkeypatch) -> None:
    captured = {}

    def openai_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_factory))
    dotenv_calls = []

    def load_test_dotenv(path, *, override):
        dotenv_calls.append((path, override))

    monkeypatch.setattr("bankscope.llm.model_access.load_dotenv", load_test_dotenv)
    values = {
        "OPENAI_API_KEY": "secret",
        "OPENAI_API_BASE_URL": "https://gateway.example/openai",
        "OPENAI_X_APP_ID": "app",
        "OPENAI_X_USER_ID": "user",
        "OPENAI_X_COMPANY_ID": "company",
        "OPENAI_X_API_VERSION": "2023-12-06",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    access_model()

    assert dotenv_calls[0][1] is False
    assert dotenv_calls[0][0].name == ".env"
    assert captured == {
        "api_key": "secret",
        "base_url": "https://gateway.example/openai",
        "default_headers": {
            "x-app-id": "app",
            "x-user-id": "user",
            "x-company-id": "company",
            "x-api-version": "2023-12-06",
        },
    }
