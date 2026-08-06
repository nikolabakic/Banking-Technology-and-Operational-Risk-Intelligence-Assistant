import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

MANIFEST_PATH = Path("artifacts/manifests/filings.json")
ITEM_PATTERN = re.compile(
    r"^item\s+\d+[a-c]?(?:\.|\s|$)",
    re.IGNORECASE,
)

warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    filings = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for filing in filings:
        path = Path(filing["local_html_path"])
        soup = BeautifulSoup(path.read_bytes(), "lxml")

        table_count = len(soup.find_all("table"))
        heading_count = len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        texts = [normalize_text(text) for text in soup.stripped_strings]

        item_candidates = sorted(
            {text for text in texts if len(text) <= 100 and ITEM_PATTERN.match(text)}
        )

        size_mb = path.stat().st_size / (1024**2)
        text_chars = sum(len(text) for text in texts)

        print(
            f"{filing['ticker']}: "
            f"{size_mb:.1f} MB, "
            f"text_chars={text_chars:,}, "
            f"tables={table_count}, "
            f"html_headings={heading_count}, "
            f"item_candidates={len(item_candidates)}"
        )
        print(f"  sample: {item_candidates[:5]}")


if __name__ == "__main__":
    main()
