import pytest

from scripts.download import (
    find_latest_10_k,
    merge_manifest_records,
    validate_primary_10_k_html,
)


def complete_10_k(incorporation: str = "Portions of the proxy statement - Part III") -> bytes:
    body = f"""
    <html><body>
    DOCUMENTS INCORPORATED BY REFERENCE {incorporation}
    TABLE OF CONTENTS
    ITEM 1. BUSINESS
    ITEM 1A. RISK FACTORS
    ITEM 1C. CYBERSECURITY
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
    ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES
    ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
    CONSOLIDATED BALANCE SHEET
    REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM
    ITEM 9A. CONTROLS AND PROCEDURES
    ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES
    </body></html>
    """
    return body.encode()


def test_find_latest_10_k_ignores_amendments_and_uses_latest_date() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-K/A", "10-K"],
                "accessionNumber": ["old", "amended", "new"],
                "filingDate": ["2024-02-01", "2025-03-01", "2025-02-01"],
                "reportDate": ["2023-12-31", "2024-12-31", "2024-12-31"],
                "primaryDocument": ["old.htm", "amended.htm", "new.htm"],
            }
        }
    }

    filing = find_latest_10_k(submissions)

    assert filing["accession_number"] == "new"
    assert filing["primary_document"] == "new.htm"


def test_single_ticker_update_preserves_and_orders_manifest() -> None:
    existing = [
        {"ticker": "BAC", "report_date": "2024-12-31"},
        {"ticker": "JPM", "report_date": "2024-12-31"},
        {"ticker": "DISABLED", "report_date": "2024-12-31"},
    ]
    updated = [{"ticker": "JPM", "report_date": "2025-12-31"}]

    merged = merge_manifest_records(existing, updated, ["JPM", "BAC"])

    assert [record["ticker"] for record in merged] == ["JPM", "BAC"]
    assert merged[0]["report_date"] == "2025-12-31"
    assert merged[1]["report_date"] == "2024-12-31"


def test_find_latest_10_k_rejects_windows_path_traversal() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["0000000000-26-000001"],
                "filingDate": ["2026-02-01"],
                "reportDate": ["2025-12-31"],
                "primaryDocument": [r"..\evil.htm"],
            }
        }
    }

    with pytest.raises(ValueError, match="safe file name"):
        find_latest_10_k(submissions)


def test_primary_10_k_completeness_accepts_proxy_only_part_iii_reference() -> None:
    result = validate_primary_10_k_html(complete_10_k())

    assert result["status"] == "full"
    assert "Item 8" in result["required_markers"]


def test_primary_10_k_completeness_rejects_separate_annual_report() -> None:
    content = complete_10_k(
        "The Annual Report to Shareholders is incorporated in Parts I, II and IV."
    )

    with pytest.raises(ValueError, match="annual-report attachment"):
        validate_primary_10_k_html(content)


def test_primary_10_k_completeness_rejects_missing_core_section() -> None:
    content = complete_10_k().replace(b"ITEM 8. FINANCIAL STATEMENTS", b"ITEM EIGHT")

    with pytest.raises(ValueError, match="Item 8"):
        validate_primary_10_k_html(content)
