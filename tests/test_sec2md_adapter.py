from typing import Any

from bankscope.parsing.sec2md_adapter import (
    adapt_builtin_chunks,
    build_structure_aware_records,
    eligible_records,
    validate_records,
)


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


def test_builtin_adapter_marks_toc_as_not_retrieval_eligible() -> None:
    chunks = [
        {
            "index": 0,
            "content": "**Form 10-K Index**\n\n| Item | Page |\n| --- | --- |\n| 1 | 1 |",
            "has_table": True,
            "start_page": 2,
            "end_page": 2,
            "element_ids": ["toc"],
            "tags": [],
            "elements": [{"id": "toc", "kind": "table"}],
        },
        {
            "index": 1,
            "content": "Cybersecurity risk is direct evidence.",
            "has_table": False,
            "start_page": 150,
            "end_page": 150,
            "element_ids": ["cyber"],
            "tags": [],
            "elements": [{"id": "cyber", "kind": "text"}],
        },
    ]

    records = adapt_builtin_chunks(chunks, make_filing(), raw_sha256="raw")

    assert len(records) == 2
    assert len(eligible_records(records)) == 1
    assert eligible_records(records)[0]["metadata"]["page_start"] == 150


def test_structure_aware_table_child_repeats_header_and_links_parent() -> None:
    table_content = (
        "The following table presents capital ratios.\n\n"
        "| December 31, 2025 | Standardized — JPMorgan Chase & Co. | "
        "Advanced — JPMorgan Chase Bank, N.A. |\n"
        "| --- | --- | --- |\n"
        "| Risk-based capital metrics | | |\n"
        "| CET1 capital ratio | 14.6% | 15.8% |"
    )
    pages = [
        {
            "number": 296,
            "display_page": 294,
            "content": table_content,
            "elements": [
                {
                    "id": "table-1",
                    "content": table_content,
                    "kind": "table",
                    "page_start": 296,
                    "page_end": 296,
                    "tags": ["us-gaap:CommonStockValue"],
                }
            ],
        }
    ]

    records, parents = build_structure_aware_records(
        pages,
        make_filing(),
        raw_sha256="raw",
        token_count=word_count,
    )

    assert len(parents) == 1
    assert len(records) == 1
    child = records[0]
    assert child["record_type"] == "table_child"
    assert "December 31, 2025" in child["document"]
    assert "Risk-based capital metrics" in child["document"]
    assert "CET1 capital ratio" in child["document"]
    assert "14.6%" in child["document"]
    assert "15.8%" in child["document"]
    assert child["metadata"]["parent_id"] == parents[0]["parent_id"]
    assert child["metadata"]["start_display_page"] == 294


def test_structure_aware_glossary_has_one_pair_per_child() -> None:
    glossary = (
        "**Glossary of Terms and Acronyms**\n\n"
        "**BANA:** Bank of America, National Association\n\n"
        "**GSIB:** Global systemically important bank"
    )
    pages = [
        {
            "number": 322,
            "display_page": 320,
            "content": glossary,
            "elements": [
                {
                    "id": "glossary-1",
                    "content": glossary,
                    "kind": "text",
                    "page_start": 322,
                    "page_end": 322,
                    "tags": [],
                }
            ],
        }
    ]

    records, parents = build_structure_aware_records(
        pages,
        make_filing(),
        raw_sha256="raw",
        token_count=word_count,
    )

    assert parents == []
    assert len(records) == 2
    assert {record["record_type"] for record in records} == {"glossary_child"}
    assert "GSIB" not in records[0]["document"]
    assert "BANA" not in records[1]["document"]


def test_structure_aware_ids_are_deterministic_and_records_validate() -> None:
    pages = [
        {
            "number": 150,
            "display_page": 148,
            "content": "**Cybersecurity risk**\n\nDirect definition.",
            "elements": [
                {
                    "id": "cyber-1",
                    "content": "**Cybersecurity risk**\n\nDirect definition.",
                    "kind": "text",
                    "page_start": 150,
                    "page_end": 150,
                    "tags": [],
                }
            ],
        }
    ]

    first, _ = build_structure_aware_records(
        pages,
        make_filing(),
        raw_sha256="raw",
        token_count=word_count,
    )
    second, _ = build_structure_aware_records(
        pages,
        make_filing(),
        raw_sha256="raw",
        token_count=word_count,
    )

    validate_records(first, token_count=word_count)
    assert first == second
