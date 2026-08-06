import json
from pathlib import Path

from bankscope.parsing.sec_html_parser import parse_filing_html

MANIFEST_PATH = Path("artifacts/manifests/filings.json")
OUTPUT_DIR = Path("data/processed/elements")


def main() -> None:
    filings = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filing in filings:
        elements = parse_filing_html(Path(filing["local_html_path"]))

        if not elements:
            raise ValueError(f"{filing['ticker']}: parser nije izdvojio sadržaj")

        records = [
            {
                "ticker": filing["ticker"],
                "cik": filing["cik"],
                "accession_number": filing["accession_number"],
                "filing_date": filing["filing_date"],
                "report_date": filing["report_date"],
                "source_url": filing["source_url"],
                **element,
            }
            for element in elements
        ]

        output_path = OUTPUT_DIR / f"{filing['ticker'].lower()}.jsonl"
        output_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )

        labeled_count = sum(record["sec_item"] is not None for record in records)
        coverage = labeled_count / len(records)

        print(
            f"{filing['ticker']}: "
            f"elements={len(records)}, "
            f"sec_item_coverage={coverage:.1%}, "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
