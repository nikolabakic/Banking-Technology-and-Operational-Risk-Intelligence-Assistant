import hashlib
from typing import Any

import pytest

from bankscope.parsing.corpus import build_corpus, validate_corpus


def make_filing() -> dict[str, Any]:
    return {
        "ticker": "JPM",
        "cik": "0000019617",
        "legal_name": "JPMorgan Chase & Co.",
        "accession_number": "0001628280-26-008131",
        "filing_date": "2026-02-13",
        "report_date": "2025-12-31",
        "source_url": "https://example.test/jpm-20251231.htm",
    }


def word_count(text: str) -> int:
    return max(1, len(text.split()))


def legacy_id(*, content: str, variant: str, child_key: str, element_ids: list[str]) -> str:
    identity = "\0".join(
        [
            make_filing()["accession_number"],
            variant,
            child_key,
            ",".join(element_ids),
            content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def sample_pages() -> list[dict[str, Any]]:
    narrative = "**Credit risk**\n\nThe bank monitors concentrations and loan performance."
    table = (
        "The following table presents loans. Dollars in millions.\n\n"
        "| Portfolio | 2025 | 2024 |\n"
        "| --- | --- | --- |\n"
        "| Commercial | 125 | 110 |\n"
        "| Consumer | 80 | 75 |"
    )
    return [
        {
            "number": 10,
            "display_page": 8,
            "content": f"{narrative}\n\n{table}",
            "elements": [
                {
                    "id": "text-1",
                    "kind": "text",
                    "content": narrative,
                    "page_start": 10,
                    "page_end": 10,
                    "tags": [],
                },
                {
                    "id": "table-1",
                    "kind": "table",
                    "content": table,
                    "page_start": 10,
                    "page_end": 10,
                    "tags": ["us-gaap:LoansReceivableNet"],
                },
            ],
        }
    ]


def test_build_corpus_preserves_legacy_narrative_and_table_target_ids() -> None:
    pages = sample_pages()
    chunks, tables = build_corpus(pages, make_filing(), "raw", word_count)

    narrative = pages[0]["elements"][0]
    table = pages[0]["elements"][1]
    expected_text_id = legacy_id(
        content=narrative["content"],
        variant="structure_aware",
        child_key="narrative:0:0",
        element_ids=["text-1"],
    )
    expected_table_id = legacy_id(
        content=table["content"],
        variant="structure_aware_parent",
        child_key="table-parent:1",
        element_ids=["table-1"],
    )

    text_chunk = next(chunk for chunk in chunks if chunk["record_type"] == "text")
    table_chunk = next(chunk for chunk in chunks if chunk["record_type"] == "table")
    assert text_chunk["target_chunk_id"] == expected_text_id
    assert tables[0]["table_id"] == expected_table_id
    assert table_chunk["target_chunk_id"] == expected_table_id


def test_local_table_description_is_compact_and_source_table_is_lossless() -> None:
    pages = sample_pages()
    chunks, tables = build_corpus(pages, make_filing(), "raw", word_count)

    source_markdown = pages[0]["elements"][1]["content"]
    table = tables[0]
    table_chunks = [chunk for chunk in chunks if chunk["record_type"] == "table"]

    assert table["document"] == source_markdown
    assert table["cell_matrices"][0][-1] == ["Consumer", "80", "75"]
    assert len(table_chunks) == 1
    description = table_chunks[0]["document"]
    embedding_text = table_chunks[0]["embedding_text"]
    assert embedding_text.count("Bank: JPM") == 1
    assert embedding_text.count("Entity: JPMorgan Chase & Co.") == 1
    assert embedding_text.count("Report: 2025 10-K") == 1
    assert embedding_text.count("Section: Credit risk") == 1
    assert embedding_text.count("Internal pages: 10-10") == 1
    assert "Bank: JPM" not in description
    assert "Introduction: The following table presents loans" in description
    assert "Columns/periods: Portfolio; 2025; 2024" in description
    assert "Units: millions" in description
    assert "Significant rows: Commercial; Consumer" in description
    assert "125" not in description
    assert "| Commercial |" not in description
    assert table_chunks[0]["metadata"]["description_provenance"]["mode"] == "local"


def test_local_table_description_keeps_the_final_row_label() -> None:
    rows = "\n".join(f"| Metric {index} | {index} |" for index in range(1, 16))
    table = f"Risk metrics.\n\n| Metric | 2025 |\n| --- | --- |\n{rows}"

    pages = [
        {
            "number": 10,
            "display_page": 8,
            "content": table,
            "elements": [
                {
                    "id": "table-many-rows",
                    "kind": "table",
                    "content": table,
                    "page_start": 10,
                    "page_end": 10,
                    "tags": [],
                }
            ],
        }
    ]
    chunks, _ = build_corpus(pages, make_filing(), "raw", word_count)

    description = next(chunk for chunk in chunks if chunk["record_type"] == "table")["document"]
    assert "Metric 15" in description


def test_build_corpus_is_deterministic() -> None:
    first = build_corpus(sample_pages(), make_filing(), "raw", word_count)
    second = build_corpus(sample_pages(), make_filing(), "raw", word_count)

    assert first == second


def test_global_validation_rejects_duplicate_filing_records() -> None:
    chunks, tables = build_corpus(sample_pages(), make_filing(), "raw", word_count)

    with pytest.raises(ValueError, match="record IDs must be unique"):
        validate_corpus([*chunks, *chunks], [*tables, *tables])


def test_validation_rejects_embedding_input_over_budget() -> None:
    chunks, tables = build_corpus(sample_pages(), make_filing(), "raw", word_count)
    oversized = [{**chunks[0], "embedding_text": "token " * 2048}, *chunks[1:]]

    with pytest.raises(ValueError, match="Embedding input exceeds"):
        validate_corpus(oversized, tables, token_count=word_count)
