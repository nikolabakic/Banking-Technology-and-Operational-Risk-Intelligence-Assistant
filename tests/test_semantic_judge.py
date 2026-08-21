from types import SimpleNamespace

import pytest

from bankscope.evaluation.semantic_judge import (
    SEMANTIC_JUDGE_PROMPT_VERSION,
    judge_semantic_answer,
)


class MockCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        function = SimpleNamespace(
            name="submit_semantic_judgement",
            arguments=self.content,
        )
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(function=function)],
        )
        return SimpleNamespace(id="judge-1", choices=[SimpleNamespace(message=message)])


def client_with(content: str):
    completions = MockCompletions(content)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_semantic_judge_returns_validated_advisory_result() -> None:
    client, completions = client_with(
        '{"correctness":true,"completeness":true,"groundedness":true,"reason":"The claims match."}'
    )

    result = judge_semantic_answer(
        question="What was the ratio?",
        gold_answer="14.6%",
        generated_answer="The ratio was 14.6% [E1].",
        evidence=[{"target_chunk_id": "chunk-1", "evidence": "The ratio was 14.6%."}],
        client=client,
        model="judge-model",
    )

    assert result["correctness"] is True
    assert result["prompt_version"] == SEMANTIC_JUDGE_PROMPT_VERSION
    assert result["response_id"] == "judge-1"
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["tool_choice"] == "required"
    assert call["parallel_tool_calls"] is False
    assert call["tools"][0]["function"]["strict"] is True
    assert "target_chunk_id=chunk-1" in call["messages"][1]["content"]


def test_semantic_judge_rejects_invalid_payload() -> None:
    client, _ = client_with('{"correctness":true}')

    with pytest.raises(RuntimeError, match="invalid semantic-judge payload"):
        judge_semantic_answer(
            question="Question",
            gold_answer="Reference",
            generated_answer="Answer",
            evidence=[],
            client=client,
            model="judge-model",
        )
