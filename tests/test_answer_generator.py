from types import SimpleNamespace
from typing import Any

import pytest

from bankscope.generation.answer_generator import generate_answer


class MockCompletions:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.payload)
        return SimpleNamespace(id="chatcmpl_answer_1", choices=[SimpleNamespace(message=message)])


def mock_client(payload: str) -> tuple[Any, MockCompletions]:
    completions = MockCompletions(payload)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def evidence(*, ticker: str = "JPM", year: str = "2025") -> dict[str, Any]:
    return {
        "target_chunk_id": "chunk-1",
        "record_type": "text",
        "ticker": ticker,
        "evidence": f"The bank reported a CET1 ratio of 15% in {year}.",
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


def test_supported_answer_uses_hydrated_evidence_and_verified_citation() -> None:
    client, completions = mock_client(
        '{"status":"supported","answer":"The CET1 ratio was 15% [E1].",'
        '"citation_ids":["E1"],"reason":"Directly stated."}'
    )

    result = generate_answer(
        "What was JPM's CET1 ratio in 2025?",
        [evidence()],
        client=client,
        model="AZURE_GPT_4o_2024_1120",
        expected_ticker="JPM",
    )

    assert result["status"] == "supported"
    assert result["citations"][0] == {
        "label": "E1",
        "target_chunk_id": "chunk-1",
        "ticker": "JPM",
        "record_type": "text",
        "report_date": "2025-12-31",
        "filing_date": "2026-02-13",
        "section_title": "Capital",
        "page_start": 42,
        "page_end": 42,
        "display_page_start": None,
        "display_page_end": None,
        "source_url": "https://www.sec.gov/example.htm",
    }
    assert "document:\nThe bank reported" in completions.calls[0]["messages"][1]["content"]
    assert completions.calls[0]["model"] == "AZURE_GPT_4o_2024_1120"
    assert result["generation"]["api"] == "chat.completions"


def test_model_can_abstain_as_ambiguous() -> None:
    client, _ = mock_client(
        '{"status":"ambiguous","answer":"Please clarify which ratio you mean.",'
        '"citation_ids":[],"reason":"The question does not identify a ratio."}'
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


def test_unknown_or_missing_inline_citation_is_rejected() -> None:
    client, _ = mock_client(
        '{"status":"supported","answer":"The ratio was 15% [E2].",'
        '"citation_ids":["E2"],"reason":"Directly stated."}'
    )

    with pytest.raises(RuntimeError, match="unverifiable"):
        generate_answer(
            "What was JPM's ratio?",
            [evidence()],
            client=client,
            model="test-model",
            expected_ticker="JPM",
        )


def test_missing_reason_and_inline_markers_are_normalized_without_changing_citations() -> None:
    client, _ = mock_client(
        '{"status":"supported","answer":"The ratio was 15%","citation_ids":["E1"]}'
    )

    result = generate_answer(
        "What was JPM's ratio?",
        [evidence()],
        client=client,
        model="test-model",
        expected_ticker="JPM",
    )

    assert result["answer"] == "The ratio was 15% [E1]"
    assert result["reason"] == "The cited filing evidence supports the answer."
    assert result["citations"][0]["label"] == "E1"
