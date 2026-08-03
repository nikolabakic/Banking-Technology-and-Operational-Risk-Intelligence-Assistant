import hashlib

import pytest

from scripts.generate_table_proxies import (
    PROXY_VERSION,
    build_table_proxy,
    validate_table_proxies,
)


@pytest.fixture
def table_chunk() -> dict[str, object]:
    return {
        "chunk_id": "JPM-table-001",
        "ticker": "JPM",
        "report_date": "2025-12-31",
        "sec_item": "Item 8",
        "section_title": "Credit exposure",
        "element_type": "table",
        "table_id": "table-001",
        "table_part_index": 1,
        "table_part_count": 1,
        "table_context": "Description: Credit exposure by category\nUnit: dollars in millions",
        "table_header": "Category | 2025 | 2024",
        "text": "Category | 2025 | 2024\nLoans | 125 | 110\nSecurities | 80 | 75",
    }


def test_build_table_proxy_creates_semantic_text(
    table_chunk: dict[str, object],
) -> None:
    proxy = build_table_proxy(table_chunk)
    proxy_text = str(proxy["proxy_text"])
    expected_id = hashlib.sha256(
        f"{PROXY_VERSION}\0JPM-table-001".encode()
    ).hexdigest()

    assert proxy["proxy_id"] == expected_id
    assert "Bank: JPM" in proxy_text
    assert "Report: 2025 10-K" in proxy_text
    assert "Columns: Category; 2025; 2024" in proxy_text
    assert "Rows: Loans; Securities" in proxy_text
    assert "Units: dollars in millions" in proxy_text


def test_validate_table_proxies_accepts_matching_records(
    table_chunk: dict[str, object],
) -> None:
    proxy = build_table_proxy(table_chunk)

    validate_table_proxies([table_chunk], [proxy])


def test_validate_table_proxies_rejects_missing_proxy(
    table_chunk: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="one proxy per table chunk"):
        validate_table_proxies([table_chunk], [])
