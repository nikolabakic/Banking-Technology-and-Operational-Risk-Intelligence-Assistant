import json
from types import SimpleNamespace

from bankscope.generation.memory import (
    CONVERSATION_SUMMARY_PROMPT_VERSION,
    summarize_conversation,
)


class SummaryCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        function = SimpleNamespace(
            name="save_conversation_summary",
            arguments=json.dumps(
                {
                    "summary": (
                        "The user prefers concise answers. Current topic: JPMorgan "
                        "operational risk. Open request: compare the framework with Citi."
                    )
                }
            ),
        )
        message = SimpleNamespace(
            refusal=None,
            tool_calls=[SimpleNamespace(function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )


def test_conversation_compaction_uses_one_strict_tool_call() -> None:
    completions = SummaryCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    summary = summarize_conversation(
        "",
        [
            {"role": "user", "content": "Keep answers concise."},
            {"role": "assistant", "content": "Understood."},
        ],
        client=client,
        model="gpt-5.1",
    )

    assert summary.startswith("The user prefers concise answers")
    request = completions.calls[0]
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False
    assert request["tools"][0]["function"]["strict"] is True
    assert request["max_completion_tokens"] == 1_500
    payload = json.loads(request["messages"][1]["content"])
    assert payload["prompt_version"] == CONVERSATION_SUMMARY_PROMPT_VERSION
    assert payload["older_messages"][0]["content"] == "Keep answers concise."
