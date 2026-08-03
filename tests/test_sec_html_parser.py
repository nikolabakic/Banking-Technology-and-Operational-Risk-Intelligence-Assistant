from pathlib import Path

from bankscope.parsing.sec_html_parser import normalize_text, parse_filing_html


def test_normalize_text_removes_invisible_unicode() -> None:
    original = "\ufeffcyber\u200bsecurity\u2060 and inter\u00adnational risk"

    assert normalize_text(original) == "cybersecurity and international risk"


def test_normalize_text_preserves_financial_content() -> None:
    original = "U.S. GAAP legal‑entity exposure: $1,234.50"

    assert normalize_text(original) == original


def test_parse_filing_html_filters_hidden_text_and_tracks_sec_item(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "filing.html"
    html_path.write_text(
        """
        <html><body>
          <p style="display: none">Hidden content</p>
          <h1>ITEM 1A. RISK FACTORS</h1>
          <p>Cybersecurity risk.</p>
          <table>
            <tr><th>Exposure</th><th>2025</th></tr>
            <tr><td>Operational risk</td><td>125</td></tr>
          </table>
        </body></html>
        """,
        encoding="utf-8",
    )

    elements = parse_filing_html(html_path)

    assert all("Hidden content" not in str(element["text"]) for element in elements)
    assert elements[0]["element_type"] == "heading"
    assert elements[0]["sec_item"] == "Item 1A"
    assert elements[-1]["element_type"] == "table"
    assert elements[-1]["sec_item"] == "Item 1A"
