from types import SimpleNamespace

import pytest

from bankscope.generation.answer_generator import GenerationValidationError
from bankscope.generation.contextualizer import contextualize_question


class FakeCompletions:
    def __init__(self, content: str, *, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_calls = []
        if self.content:
            function = SimpleNamespace(
                name="submit_standalone_question",
                arguments=self.content,
            )
            tool_calls = [SimpleNamespace(function=function)]
        message = SimpleNamespace(content=None, refusal=None, tool_calls=tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)]
        )


def test_contextualizer_returns_valid_question_and_removes_old_citations() -> None:
    completions = FakeCompletions(
        '{"standalone_question":"What was JPMorgan Chase CET1 ratio in 2024?"}'
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = contextualize_question(
        "What about 2024?",
        [
            {"role": "user", "content": "What was JPM's CET1 ratio in 2025?"},
            {"role": "assistant", "content": "It was 14.6% [E1]."},
        ],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        session_ticker="JPM",
    )

    assert result.standalone_question == "What was JPMorgan Chase CET1 ratio in 2024?"
    request = completions.calls[0]
    assert request["max_completion_tokens"] == 300
    assert "temperature" not in request
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False
    assert "[E1]" not in request["messages"][1]["content"]


@pytest.mark.parametrize(
    ("content", "finish_reason", "code"),
    [
        ("not json", "stop", "contextualization_invalid_schema"),
        ("", "stop", "contextualization_empty"),
        ('{"standalone_question":"Question"}', "length", "contextualization_truncated"),
    ],
)
def test_contextualizer_fails_closed(content: str, finish_reason: str, code: str) -> None:
    completions = FakeCompletions(content, finish_reason=finish_reason)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(GenerationValidationError) as error:
        contextualize_question(
            "Follow-up?",
            [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
            client=client,
            model="test-model",
        )

    assert error.value.code == code


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "user", "content": "Unpaired question"}],
        [
            {"role": "assistant", "content": "Wrong first role"},
            {"role": "user", "content": "Wrong second role"},
        ],
    ],
)
def test_contextualizer_rejects_incomplete_or_misordered_history(history) -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions('{"standalone_question":"Question"}'))
    )

    with pytest.raises(ValueError, match="turn pairs"):
        contextualize_question("Follow-up?", history, client=client, model="test-model")


def test_contextualizer_rejects_blank_standalone_question() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions('{"standalone_question":"   "}'))
    )

    with pytest.raises(GenerationValidationError) as error:
        contextualize_question(
            "Follow-up?",
            [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
            client=client,
            model="test-model",
        )

    assert error.value.code == "contextualization_invalid_schema"
