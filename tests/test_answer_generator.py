import json
from types import SimpleNamespace
from typing import Any

import pytest

from bankscope.generation.answer_generator import (
    GenerationValidationError,
    _question_language,
    generate_answer,
)


class MockCompletions:
    def __init__(
        self,
        payload: str | list[str],
        *,
        finish_reason: str | list[str] = "stop",
        refusal: str | None = None,
    ) -> None:
        self.payload = payload
        self.finish_reason = finish_reason
        self.refusal = refusal
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = (
            self.payload[min(len(self.calls) - 1, len(self.payload) - 1)]
            if isinstance(self.payload, list)
            else self.payload
        )
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            parsed = {}
        if parsed.get("status") == "ambiguous":
            function_name = "submit_ambiguous_answer"
        elif parsed.get("status") == "unsupported":
            function_name = "submit_unsupported_answer"
        elif parsed.get("answer_type") == "narrative":
            function_name = "submit_supported_narrative_answer"
        else:
            function_name = "submit_supported_numeric_answer"
        function = SimpleNamespace(name=function_name, arguments=payload)
        message = SimpleNamespace(
            content=None,
            refusal=self.refusal,
            tool_calls=[SimpleNamespace(function=function)],
        )
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        finish_reason = (
            self.finish_reason[min(len(self.calls) - 1, len(self.finish_reason) - 1)]
            if isinstance(self.finish_reason, list)
            else self.finish_reason
        )
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(id="chatcmpl_answer_1", choices=[choice], usage=usage)


def mock_client(
    payload: str | list[str], *, finish_reason: str | list[str] = "stop", refusal: str | None = None
) -> tuple[Any, MockCompletions]:
    completions = MockCompletions(payload, finish_reason=finish_reason, refusal=refusal)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def model_payload(**changes: Any) -> str:
    value: dict[str, Any] = {
        "status": "supported",
        "answer_type": "numeric",
        "answer": "The ratio was 15%.",
        "facts": {
            "entity": "JPMorgan Chase & Co.",
            "metric": "CET1 capital ratio",
            "variant": "Standardized",
            "period": "2025-12-31",
            "value_text": "15",
            "unit": "percent",
        },
        "citation_ids": ["E1"],
        "reason": "Directly stated.",
    }
    value.update(changes)
    return json.dumps(value)


def evidence(
    *, ticker: str = "JPM", year: str = "2025", document: str | None = None
) -> dict[str, Any]:
    return {
        "target_chunk_id": "chunk-1",
        "record_type": "text",
        "ticker": ticker,
        "evidence": document or f"The bank reported a CET1 ratio of 15% in {year}.",
        "metadata": {
            "ticker": ticker,
            "report_date": f"{year}-12-31",
            "filing_date": "2026-02-13",
            "section_title": "Capital",
            "page_start": 42,
            "page_end": 42,
            "source_url": "https://www.sec.gov/example.htm",
        },
    }


def test_numeric_answer_uses_strict_function_facts_and_verified_citation() -> None:
    client, completions = mock_client(model_payload())

    result = generate_answer(
        "What was JPM's CET1 ratio in 2025?",
        [evidence()],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
        expected_bank_name="JPMorgan Chase & Co.",
    )

    assert result["status"] == "supported"
    assert result["answer_type"] == "numeric"
    assert result["answer"] == (
        "JPMorgan Chase & Co. — CET1 capital ratio — Standardized — 2025-12-31: 15 percent [E1]"
    )
    assert result["facts"]["value_text"] == "15"
    assert result["citations"][0]["target_chunk_id"] == "chunk-1"
    call = completions.calls[0]
    assert "response_format" not in call
    assert call["tool_choice"] == "required"
    assert call["parallel_tool_calls"] is False
    assert {tool["function"]["name"] for tool in call["tools"]} == {
        "submit_supported_numeric_answer",
        "submit_supported_narrative_answer",
        "submit_ambiguous_answer",
        "submit_unsupported_answer",
    }
    assert all(tool["function"]["strict"] is True for tool in call["tools"])
    for tool in call["tools"]:
        parameters = tool["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])
    schema = call["tools"][0]["function"]["parameters"]
    assert set(schema["required"]) == set(schema["properties"])
    numeric_schema = schema["$defs"]["NumericFacts"]
    assert set(numeric_schema["required"]) == set(numeric_schema["properties"])
    assert schema["additionalProperties"] is False
    assert numeric_schema["additionalProperties"] is False
    assert call["max_completion_tokens"] == 1_600
    assert "max_tokens" not in call
    assert "temperature" not in call
    assert "Expected bank: JPMorgan Chase & Co." in call["messages"][1]["content"]
    assert "REQUIRED OUTPUT LANGUAGE: English" in call["messages"][0]["content"]
    assert "Required output language: English" in call["messages"][1]["content"]
    assert "never a JSON array or list" not in call["messages"][0]["content"]
    assert "Required JSON schema" not in call["messages"][0]["content"]
    assert '"variant":"Standardized"' not in call["messages"][0]["content"]
    assert "Base measure only" in numeric_schema["properties"]["metric"]["description"]
    assert len(completions.calls) == 1
    assert result["generation"]["request_count"] == 1
    assert result["generation"]["final_status"] == "supported"
    assert result["generation"]["latency_ms"] >= 0
    assert result["generation"]["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "input_tokens": 100,
        "output_tokens": 20,
    }


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("What were Citigroup's material cybersecurity risks in 2025?", "English"),
        ("Koji su glavni operativni rizici banke?", "Serbian"),
        ("Који су главни оперативни ризици банке?", "Serbian"),
        ("¿Cuáles fueron los riesgos de ciberseguridad del banco?", "Spanish"),
    ],
)
def test_question_language_is_determined_before_generation(question: str, language: str) -> None:
    assert _question_language(question) == language


def test_narrative_answer_keeps_natural_text_and_adds_valid_markers() -> None:
    client, _ = mock_client(
        model_payload(
            answer_type="narrative",
            answer="Operational risk includes failed processes.",
            facts=None,
        )
    )

    result = generate_answer(
        "How does JPM define operational risk?",
        [evidence(document="Operational risk includes failed processes.")],
        client=client,
        model="AZURE_GPT_4o_2024_1120",
        expected_ticker="JPM",
    )

    assert result["answer"] == "Operational risk includes failed processes. [E1]"
    assert result["facts"] is None


def test_numeric_renderer_does_not_duplicate_percent_unit() -> None:
    facts = json.loads(model_payload())["facts"]
    facts["value_text"] = "11.4 %"
    client, _ = mock_client(model_payload(facts=facts))

    result = generate_answer(
        "What was JPM's Standardized CET1 ratio in 2025?",
        [evidence(document="The Standardized CET1 ratio was 11.4 percent in 2025.")],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
    )

    assert result["answer"].endswith("11.4 percent [E1]")
    assert "% percent" not in result["answer"]


def test_redundant_citation_id_format_is_normalized_before_validation() -> None:
    client, _ = mock_client(model_payload(citation_ids=["[e1], E1"]))

    result = generate_answer(
        "What was JPM's CET1 ratio in 2025?",
        [evidence()],
        client=client,
        model="test-model",
        expected_ticker="JPM",
    )

    assert [item["label"] for item in result["citations"]] == ["E1"]


def test_ally_internal_target_id_is_hidden_and_normalized_to_short_citation_label() -> None:
    ally_target_id = "3036bb6433b0efa7c29c84a13ff4236f19d89c426d2721857a74028a4586caac"
    client, completions = mock_client(
        model_payload(
            answer_type="narrative",
            answer=(
                "Operational risk is loss or harm arising from inadequate or failed processes, "
                "systems, people, or external events [E1]."
            ),
            facts=None,
            citation_ids=[ally_target_id, "E1"],
        )
    )
    ally_evidence = evidence(
        ticker="ALLY",
        document=(
            "Operational risk is the risk of loss or harm arising from inadequate or failed "
            "processes or systems, human factors, or external events."
        ),
    )
    ally_evidence["target_chunk_id"] = ally_target_id

    result = generate_answer(
        "How does Ally Financial define operational risk in its 2025 Form 10-K?",
        [ally_evidence],
        client=client,
        model="test-model",
        expected_ticker="ALLY",
    )

    assert [item["label"] for item in result["citations"]] == ["E1"]
    assert "target_chunk_id" not in completions.calls[0]["messages"][1]["content"]
    assert ally_target_id not in completions.calls[0]["messages"][1]["content"]


def test_unknown_internal_target_id_still_fails_closed() -> None:
    client, _ = mock_client(
        model_payload(
            answer_type="narrative",
            answer="Operational risk includes failed processes [E1].",
            facts=None,
            citation_ids=["unknown-e1-internal-id", "E1"],
        )
    )

    with pytest.raises(GenerationValidationError) as captured:
        generate_answer(
            "How does JPM define operational risk in 2025?",
            [evidence(document="Operational risk includes failed processes in 2025.")],
            client=client,
            model="test-model",
            expected_ticker="JPM",
        )

    assert captured.value.code == "invalid_schema"


def test_model_can_abstain_as_ambiguous() -> None:
    client, _ = mock_client(
        model_payload(
            status="ambiguous",
            answer_type="narrative",
            answer="Please clarify which ratio you mean.",
            facts=None,
            citation_ids=[],
            reason="The question does not identify a ratio.",
        )
    )

    result = generate_answer(
        "What was JPM's ratio?",
        [evidence()],
        client=client,
        model="test-model",
        expected_ticker="JPM",
    )

    assert result["status"] == "ambiguous"
    assert result["citations"] == []


def test_missing_requested_period_abstains_without_api_call() -> None:
    client, completions = mock_client("not used")

    result = generate_answer(
        "What was JPM's CET1 ratio in 2027?",
        [evidence(year="2025")],
        client=client,
        model="test-model",
        expected_ticker="JPM",
    )

    assert result["status"] == "unsupported"
    assert result["answer_type"] == "narrative"
    assert result["generation"]["request_count"] == 0
    assert result["generation"]["final_status"] == "unsupported"
    assert "2027" in result["reason"]
    assert completions.calls == []


def test_mismatched_entity_or_evidence_type_fails_closed() -> None:
    client, _ = mock_client("not used")

    with pytest.raises(ValueError, match="does not match requested JPM"):
        generate_answer(
            "What was the ratio?",
            [evidence(ticker="BAC")],
            client=client,
            model="test-model",
            expected_ticker="JPM",
        )
    with pytest.raises(ValueError, match="type text does not match requested table"):
        generate_answer(
            "What was the ratio?",
            [evidence()],
            client=client,
            model="test-model",
            expected_ticker="JPM",
            expected_record_type="table",
        )


def test_unknown_numeric_citation_is_reconciled_to_exact_value_evidence() -> None:
    client, _ = mock_client(model_payload(citation_ids=["E2"]))

    result = generate_answer(
        "What was JPM's ratio?",
        [evidence()],
        client=client,
        model="test-model",
        expected_ticker="JPM",
    )

    assert [citation["label"] for citation in result["citations"]] == ["E1"]


def test_unknown_numeric_citation_without_exact_value_is_rejected() -> None:
    client, _ = mock_client(model_payload(citation_ids=["E2"]))

    with pytest.raises(GenerationValidationError) as captured:
        generate_answer(
            "What was JPM's ratio?",
            [
                evidence(
                    document="The filing discusses CET1 but does not state the requested value."
                )
            ],
            client=client,
            model="test-model",
            expected_ticker="JPM",
        )

    assert captured.value.code == "invalid_citations"


@pytest.mark.parametrize("value_text", ["13.18", "13.18%", "$ 13.18"])
def test_exact_numeric_value_accepts_safe_formatting_differences(value_text: str) -> None:
    client, _ = mock_client(
        model_payload(
            facts={
                "entity": "Citigroup Inc.",
                "metric": "CET1 capital ratio",
                "variant": "Standardized",
                "period": "2025-12-31",
                "value_text": value_text,
                "unit": "percent",
            }
        )
    )

    result = generate_answer(
        "What was Citi's ratio in 2025?",
        [evidence(ticker="C", document="The exact ratio was 13.18 %.")],
        client=client,
        model="test-model",
        expected_ticker="C",
    )

    assert result["facts"]["value_text"] == value_text


@pytest.mark.parametrize(
    ("value_text", "document"),
    [("13.2", "The exact ratio was 13.18%."), ("11.9", "The exact ratio was 11.93%.")],
)
def test_rounded_numeric_value_is_rejected(value_text: str, document: str) -> None:
    facts = json.loads(model_payload())["facts"]
    facts["value_text"] = value_text
    client, completions = mock_client(model_payload(facts=facts))

    with pytest.raises(GenerationValidationError) as captured:
        generate_answer(
            "What was the ratio in 2025?",
            [evidence(document=document)],
            client=client,
            model="test-model",
            expected_ticker="JPM",
        )

    assert captured.value.code == "numeric_value_not_in_cited_evidence"
    assert len(completions.calls) == 1


def test_invalid_schema_refusal_and_truncation_have_stable_codes() -> None:
    invalid_client, _ = mock_client('{"status":"supported"}')
    with pytest.raises(GenerationValidationError) as invalid:
        generate_answer(
            "Question",
            [evidence()],
            client=invalid_client,
            model="test-model",
            expected_ticker="JPM",
        )
    assert invalid.value.code == "invalid_schema"
    assert invalid.value.generation["request_count"] == 2
    assert invalid.value.generation["final_status"] == "validation_error"
    assert invalid.value.generation["usage"]["input_tokens"] == 100
    assert invalid.value.generation["validation_errors"]

    refusal_client, _ = mock_client("{}", refusal="Cannot comply")
    with pytest.raises(GenerationValidationError) as refusal:
        generate_answer(
            "Question",
            [evidence()],
            client=refusal_client,
            model="test-model",
            expected_ticker="JPM",
        )
    assert refusal.value.code == "response_refused"

    truncated_client, _ = mock_client("{}", finish_reason="length")
    with pytest.raises(GenerationValidationError) as truncated:
        generate_answer(
            "Question",
            [evidence()],
            client=truncated_client,
            model="test-model",
            expected_ticker="JPM",
        )
    assert truncated.value.code == "response_truncated"
    assert truncated.value.generation["request_count"] == 2


def test_numeric_facts_array_is_rejected_after_one_repair_retry() -> None:
    payload = json.loads(model_payload())
    payload["facts"] = [payload["facts"]]
    client, completions = mock_client(json.dumps(payload))

    with pytest.raises(GenerationValidationError) as captured:
        generate_answer(
            "What was JPM's CET1 ratio in 2025?",
            [evidence()],
            client=client,
            model="AZURE_GPT_51_2025_1113",
            expected_ticker="JPM",
        )

    assert captured.value.code == "invalid_schema"
    assert len(completions.calls) == 2


def test_semantic_schema_failure_is_repaired_once() -> None:
    client, completions = mock_client(['{"status":"supported"}', model_payload()])

    result = generate_answer(
        "What was JPM's CET1 ratio in 2025?",
        [evidence()],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
    )

    assert result["status"] == "supported"
    assert result["generation"]["request_count"] == 2
    assert "failed local contract validation" in (
        completions.calls[1]["messages"][0]["content"].casefold()
    )


def test_explicit_presentation_word_limit_is_repaired_once() -> None:
    long_payload = model_payload(
        answer=" ".join(["supported"] * 12) + " [E1]",
        citation_ids=["E1"],
    )
    short_payload = model_payload(answer="Short supported answer. [E1]", citation_ids=["E1"])
    client, completions = mock_client([long_payload, short_payload])

    result = generate_answer(
        "What was JPM's CET1 ratio in 2025?",
        [evidence()],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
        presentation_guidance="Keep the answer under 8 words.",
    )

    assert result["status"] == "supported"
    assert result["generation"]["request_count"] == 2
    assert "at most 8 words" in completions.calls[1]["messages"][0]["content"]


def test_narrative_citation_array_is_reconciled_to_known_inline_markers() -> None:
    client, _ = mock_client(
        model_payload(
            answer="Short supported answer. [E1]",
            answer_type="narrative",
            facts=None,
            citation_ids=["E1", "E2"],
        )
    )

    result = generate_answer(
        "What does JPM disclose?",
        [evidence()],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
    )

    assert [citation["label"] for citation in result["citations"]] == ["E1"]


def test_evidence_recheck_adds_bank_only_fail_closed_instruction() -> None:
    client, completions = mock_client(model_payload())

    generate_answer(
        "JPMorgan CET1 ratio for 2025",
        [evidence()],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
        comparison_scope=True,
        evidence_recheck=True,
    )

    system = completions.calls[0]["messages"][0]["content"]
    assert "bounded evidence recheck" in system
    assert "Do not require evidence for peer banks" in system
    assert "Continue to abstain" in system


def test_unsupported_model_text_is_replaced_by_local_abstention() -> None:
    client, _ = mock_client(
        model_payload(
            status="unsupported",
            answer_type="narrative",
            answer="Here is a complete apple pie recipe.",
            facts=None,
            citation_ids=[],
            reason="Unrelated request.",
        )
    )

    result = generate_answer(
        "Koji podatak JPM navodi?",
        [evidence()],
        client=client,
        model="test-model",
        expected_ticker="JPM",
    )

    assert result["status"] == "unsupported"
    assert "apple pie" not in result["answer"].casefold()
    assert "apple pie" not in result["reason"].casefold()
    assert "dokazi" in result["answer"].casefold()


def test_truncation_retries_once_with_larger_budget_and_concise_instruction() -> None:
    client, completions = mock_client(model_payload(), finish_reason=["length", "stop"])

    result = generate_answer(
        "What was JPM's CET1 ratio in 2025?",
        [evidence()],
        client=client,
        model="AZURE_GPT_51_2025_1113",
        expected_ticker="JPM",
    )

    assert result["status"] == "supported"
    assert result["generation"]["request_count"] == 2
    assert completions.calls[1]["max_completion_tokens"] == 2_000
    assert "previous attempt reached the output limit" in (
        completions.calls[1]["messages"][0]["content"].casefold()
    )


def test_non_gpt51_model_keeps_compatible_completion_parameters() -> None:
    client, completions = mock_client(model_payload())

    generate_answer(
        "Question",
        [evidence()],
        client=client,
        model="AZURE_GPT_4o_2024_1120",
        expected_ticker="JPM",
        temperature=0,
    )

    call = completions.calls[0]
    assert call["max_tokens"] == 1_600
    assert call["temperature"] == 0
    assert "max_completion_tokens" not in call
