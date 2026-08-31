from types import SimpleNamespace

import pytest

from bankscope.evaluation.semantic_judge import (
    EVIDENCE_AUDIT_PROMPT_VERSION,
    EVIDENCE_AUDIT_SCHEMA_VERSION,
    SEMANTIC_JUDGE_PROMPT_VERSION,
    audit_evidence_answer,
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


class AuditCompletions:
    def __init__(self, content: str, *, fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider unavailable")
        function = SimpleNamespace(name="submit_evidence_audit", arguments=self.content)
        message = SimpleNamespace(tool_calls=[SimpleNamespace(function=function)])
        return SimpleNamespace(id="audit-1", choices=[SimpleNamespace(message=message)])


def audit_client(content: str, *, fail: bool = False):
    completions = AuditCompletions(content, fail=fail)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.mark.parametrize(
    ("content", "expected_status"),
    [
        (
            '{"status":"passed","question_addressed":true,"grounded":true,'
            '"citation_coverage_ok":true,"contradiction_found":false,'
            '"summary":"All material claims are cited and supported."}',
            "passed",
        ),
        (
            '{"status":"review_recommended","question_addressed":true,"grounded":false,'
            '"citation_coverage_ok":false,"contradiction_found":false,'
            '"summary":"One material claim is unsupported."}',
            "review_recommended",
        ),
        (
            '{"status":"review_recommended","question_addressed":true,"grounded":false,'
            '"citation_coverage_ok":true,"contradiction_found":true,'
            '"summary":"A material claim contradicts the evidence."}',
            "review_recommended",
        ),
    ],
)
def test_evidence_audit_returns_strict_advisory_status(content: str, expected_status: str) -> None:
    client, completions = audit_client(content)

    result = audit_evidence_answer(
        question="What was the ratio?",
        generated_answer="The ratio was 14.6% [E1].",
        evidence=[
            {
                "audit_label": "E1",
                "target_chunk_id": "chunk-1",
                "ticker": "JPM",
                "record_type": "table",
                "evidence": "The ratio was 14.6%.",
            }
        ],
        client=client,
        model="judge-model",
    )

    assert result["status"] == expected_status
    assert result["metadata"]["prompt_version"] == EVIDENCE_AUDIT_PROMPT_VERSION
    assert result["metadata"]["schema_version"] == EVIDENCE_AUDIT_SCHEMA_VERSION
    assert result["metadata"]["response_id"] == "audit-1"
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["parallel_tool_calls"] is False
    assert call["tools"][0]["function"]["strict"] is True
    assert "[E1] target_chunk_id=chunk-1" in call["messages"][1]["content"]
    assert "chain-of-thought" in call["messages"][0]["content"]


def test_evidence_audit_provider_failure_is_unavailable() -> None:
    client, completions = audit_client("", fail=True)

    result = audit_evidence_answer(
        question="Question",
        generated_answer="Answer",
        evidence=[],
        client=client,
        model="judge-model",
    )

    assert result["status"] == "unavailable"
    assert result["metadata"]["error_code"] == "provider_failure"
    assert result["metadata"]["request_count"] == 1
    assert len(completions.calls) == 1


def test_evidence_audit_uses_gpt51_completion_token_parameter() -> None:
    client, completions = audit_client(
        '{"status":"passed","question_addressed":true,"grounded":true,'
        '"citation_coverage_ok":true,"contradiction_found":false,'
        '"summary":"The answer is supported."}'
    )

    result = audit_evidence_answer(
        question="Question",
        generated_answer="Answer [E1].",
        evidence=[{"audit_label": "E1", "evidence": "Answer."}],
        client=client,
        model="AZURE_GPT_51_2025_1113",
    )

    assert result["status"] == "passed"
    call = completions.calls[0]
    assert call["max_completion_tokens"] == 400
    assert "max_tokens" not in call
    assert call["temperature"] == 0


def test_evidence_audit_malformed_payload_is_unavailable() -> None:
    client, _ = audit_client(
        '{"status":"passed","question_addressed":true,"grounded":false,'
        '"citation_coverage_ok":true,"contradiction_found":false,"summary":"Inconsistent."}'
    )

    result = audit_evidence_answer(
        question="Question",
        generated_answer="Answer",
        evidence=[],
        client=client,
        model="judge-model",
    )

    assert result["status"] == "unavailable"
    assert result["metadata"]["error_code"] == "invalid_payload"
