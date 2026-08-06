from types import SimpleNamespace
from typing import Any

import pytest

from bankscope.parsing.corpus import build_corpus


def make_filing() -> dict[str, Any]:
    return {
        "ticker": "BAC",
        "cik": "0000070858",
        "legal_name": "Bank of America Corporation",
        "accession_number": "0000070858-26-000001",
        "filing_date": "2026-02-20",
        "report_date": "2025-12-31",
        "source_url": "https://example.test/bac.htm",
    }


def word_count(text: str) -> int:
    return max(1, len(text.split()))


def table_page(content: str) -> list[dict[str, Any]]:
    return [
        {
            "number": 42,
            "display_page": 40,
            "content": content,
            "elements": [
                {
                    "id": "table-42",
                    "kind": "table",
                    "content": content,
                    "page_start": 42,
                    "page_end": 42,
                    "tags": [],
                }
            ],
        }
    ]


def data_table() -> str:
    return (
        "Credit exposure by category. Dollars in millions.\n\n"
        "| Category | 2025 | 2024 |\n"
        "| --- | --- | --- |\n"
        "| Loans | 125 | 110 |\n"
        "| Securities | 80 | 75 |"
    )


def test_every_parser_table_is_stored_but_layout_and_index_tables_are_not_chunked() -> None:
    layout = "**Table of Contents**\n\n| Item | Page |\n| --- | --- |\n| Item 1. Business | 4 |"
    pages = table_page(layout)
    pages[0]["elements"].append(
        {
            "id": "table-data",
            "kind": "table",
            "content": data_table(),
            "page_start": 42,
            "page_end": 42,
            "tags": [],
        }
    )

    chunks, tables = build_corpus(pages, make_filing(), "raw", word_count)

    assert len(tables) == 2
    assert [table["table_index"] for table in tables] == [0, 1]
    assert tables[0]["table_type"] == "index"
    assert tables[0]["retrieval_eligible"] is False
    assert tables[0]["document"] == layout
    table_chunks = [chunk for chunk in chunks if chunk["record_type"] == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0]["target_chunk_id"] == tables[1]["table_id"]


def test_cover_page_commission_file_table_is_layout_only() -> None:
    cover = (
        "**FORM 10-K**\n\n"
        "| For the fiscal year ended | Commission file | |\n"
        "| --- | --- | --- |\n"
        "| December 31, 2025 | number | 1-5805 |"
    )

    chunks, tables = build_corpus(table_page(cover), make_filing(), "raw", word_count)

    assert tables[0]["table_type"] == "layout"
    assert tables[0]["retrieval_eligible"] is False
    assert not [chunk for chunk in chunks if chunk["record_type"] == "table"]


class MockResponses:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("test outage")
        return SimpleNamespace(
            id="resp_test_1",
            output_text="BAC 2025 credit exposure table covering Loans and Securities.",
        )


def test_openai_mode_uses_responses_api_and_records_provenance() -> None:
    responses = MockResponses()
    client = SimpleNamespace(responses=responses)

    chunks, tables = build_corpus(
        table_page(data_table()),
        make_filing(),
        "raw",
        word_count,
        description_mode="openai",
        llm_client=client,
        llm_model="gpt-4o-test",
    )

    assert len(responses.calls) == 1
    assert responses.calls[0]["model"] == "gpt-4o-test"
    assert responses.calls[0]["max_output_tokens"] == 300
    assert "Original markdown table" in responses.calls[0]["input"]
    chunk = next(chunk for chunk in chunks if chunk["record_type"] == "table")
    assert "Significant rows: Loans; Securities" in chunk["document"]
    assert chunk["document"].endswith(
        "LLM synopsis: BAC 2025 credit exposure table covering Loans and Securities."
    )
    provenance = chunk["metadata"]["description_provenance"]
    assert provenance == {
        "mode": "openai",
        "provider": "openai",
        "api": "responses",
        "model": "gpt-4o-test",
        "response_id": "resp_test_1",
        "base_generator": "bankscope-local-table-description-v1",
    }
    assert tables[0]["metadata"]["description_provenance"] == provenance


def test_openai_failure_does_not_fall_back_to_local_description() -> None:
    client = SimpleNamespace(responses=MockResponses(fail=True))

    with pytest.raises(RuntimeError, match="OpenAI table description failed"):
        build_corpus(
            table_page(data_table()),
            make_filing(),
            "raw",
            word_count,
            description_mode="openai",
            llm_client=client,
        )
