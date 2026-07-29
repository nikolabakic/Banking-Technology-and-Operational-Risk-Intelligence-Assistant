import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Tag

CONTENT_TAGS = ("div", "p", "li", "table")

SEC_ITEM_PATTERN = re.compile(
    r"^item\s+(\d{1,2}[a-c]?)"
    r"(?:\s*[.\-:\u2013\u2014]\s*|\s+|$)",
    re.IGNORECASE,
)

HIDDEN_STYLE_PATTERN = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
    re.IGNORECASE,
)


def detect_sec_item(
    elements: list[dict[str, object]],
    start_index: int,
) -> str | None:
    parts: list[str] = []

    for element in elements[start_index : start_index + 3]:
        parts.append(str(element["text"]))

        candidate = normalize_text(" ".join(parts))
        candidate = normalize_text(candidate.replace("|", " "))

        if len(candidate) > 120:
            break

        match = SEC_ITEM_PATTERN.match(candidate)

        if match:
            return f"Item {match.group(1).upper()}"

    return None


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def extract_table_text(table: Tag) -> str:
    rows: list[str] = []

    for row in table.find_all("tr"):
        cells = [
            normalize_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]

        if any(cells):
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def parse_filing_html(path: Path) -> list[dict[str, object]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(path.read_bytes(), "lxml")

    for tag in soup(["script", "style", "noscript", "ix:header"]):
        tag.decompose()

    for tag in soup.find_all(style=HIDDEN_STYLE_PATTERN):
        tag.decompose()

    root = soup.body or soup
    elements: list[dict[str, object]] = []

    for tag in root.find_all(CONTENT_TAGS):
        if tag.find_parent("table") is not None:
            continue

        if tag.name == "table":
            element_type = "table"
            text = extract_table_text(tag)
        else:
            if tag.find(CONTENT_TAGS) is not None:
                continue

            element_type = "list" if tag.name == "li" else "paragraph"
            text = normalize_text(tag.get_text(" ", strip=True))

        if not text:
            continue

        elements.append(
            {
                "order_index": len(elements),
                "element_type": element_type,
                "text": text,
            }
        )

    current_sec_item: str | None = None

    for index, element in enumerate(elements):
        detected_item = detect_sec_item(elements, index)

        if detected_item is not None:
            current_sec_item = detected_item
            element["element_type"] = "heading"

        element["sec_item"] = current_sec_item

    return elements
