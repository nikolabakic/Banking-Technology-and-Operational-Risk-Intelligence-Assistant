import pytest

from scripts.download import find_latest_10_k, merge_manifest_records


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
