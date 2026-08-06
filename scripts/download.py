"""Download the latest primary 10-K document for configured banks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from bankscope.config.settings import ApplicationSettings, get_settings
from bankscope.sec.company_registry import BankCompany, load_bank_registry

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "data/filings.json"

Fetcher = Callable[[str, ApplicationSettings], bytes]


def fetch_sec(url: str, settings: ApplicationSettings) -> bytes:
    """Fetch one SEC resource and apply the configured request delay."""

    request = Request(
        url,
        headers={"User-Agent": settings.sec_user_agent, "Accept": "*/*"},
    )

    try:
        with urlopen(request, timeout=settings.sec_timeout_seconds) as response:
            return response.read()
    finally:
        time.sleep(1 / settings.sec_requests_per_second)


def find_latest_10_k(submissions: dict[str, Any]) -> dict[str, str]:
    """Return the latest non-amended 10-K from an SEC submissions response."""

    try:
        recent = submissions["filings"]["recent"]
    except (KeyError, TypeError) as error:
        raise ValueError("SEC submissions response has no filings.recent object") from error

    fields = (
        "form",
        "accessionNumber",
        "filingDate",
        "reportDate",
        "primaryDocument",
    )

    if not isinstance(recent, dict):
        raise ValueError("SEC filings.recent must be an object")

    values: dict[str, list[Any]] = {}

    for field in fields:
        field_values = recent.get(field)

        if not isinstance(field_values, list):
            raise ValueError(f"SEC filings.recent.{field} must be a list")

        values[field] = field_values

    row_count = len(values["form"])

    if any(len(values[field]) != row_count for field in fields):
        raise ValueError("SEC filings.recent arrays have inconsistent lengths")

    indexes = [index for index, form in enumerate(values["form"]) if form == "10-K"]

    if not indexes:
        raise ValueError("No primary 10-K filing was found")

    index = max(indexes, key=lambda item: str(values["filingDate"][item]))

    result = {
        "form": values["form"][index],
        "accession_number": values["accessionNumber"][index],
        "filing_date": values["filingDate"][index],
        "report_date": values["reportDate"][index],
        "primary_document": values["primaryDocument"][index],
    }

    if any(not isinstance(value, str) or not value.strip() for value in result.values()):
        raise ValueError("Latest SEC 10-K row contains an empty or non-text field")

    document_name = result["primary_document"]
    if (
        document_name in {".", ".."}
        or "/" in document_name
        or "\\" in document_name
        or Path(document_name).is_absolute()
    ):
        raise ValueError("SEC primary document is not a safe file name")

    return result


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            output_file.write(content)

        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _relative_to_project(path: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Downloaded filing must stay inside the project: {path}") from error

    return relative_path.as_posix()


def download_latest_10_k(
    bank: BankCompany,
    settings: ApplicationSettings,
    *,
    fetcher: Fetcher = fetch_sec,
) -> tuple[dict[str, str], bool]:
    """Download a bank's latest 10-K and return its manifest record."""

    submissions_url = SUBMISSIONS_URL.format(cik=bank.cik)

    try:
        submissions = json.loads(fetcher(submissions_url, settings))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"SEC returned invalid submissions JSON for {bank.ticker}") from error

    if not isinstance(submissions, dict):
        raise ValueError(f"SEC submissions response for {bank.ticker} must be an object")

    sec_tickers = submissions.get("tickers", [])

    if not isinstance(sec_tickers, list) or bank.ticker not in {
        str(ticker).upper() for ticker in sec_tickers
    }:
        raise ValueError(f"SEC submissions response does not contain ticker {bank.ticker}")

    filing = find_latest_10_k(submissions)
    accession_path = filing["accession_number"].replace("-", "")

    if not accession_path.isdigit():
        raise ValueError(f"Invalid SEC accession number: {filing['accession_number']}")

    source_url = ARCHIVES_URL.format(
        cik=int(bank.cik),
        accession=accession_path,
        document=filing["primary_document"],
    )
    local_path = settings.raw_data_dir / bank.cik / accession_path / filing["primary_document"]
    downloaded = not local_path.exists()

    if downloaded:
        _write_bytes_atomic(local_path, fetcher(source_url, settings))

    record = {
        "ticker": bank.ticker,
        "cik": bank.cik,
        "legal_name": bank.legal_name,
        **filing,
        "source_url": source_url,
        "local_html_path": _relative_to_project(local_path),
    }
    return record, downloaded


def write_manifest(path: Path, records: Sequence[dict[str, str]]) -> None:
    """Atomically write a readable filing manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(records, output_file, ensure_ascii=False, indent=2, allow_nan=False)
            output_file.write("\n")

        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def merge_manifest_records(
    existing: Sequence[dict[str, Any]],
    updated: Sequence[dict[str, str]],
    ticker_order: Sequence[str],
) -> list[dict[str, Any]]:
    """Replace downloaded tickers without dropping the other manifest rows."""

    by_ticker: dict[str, dict[str, Any]] = {}

    enabled = set(ticker_order)
    for record in [*existing, *updated]:
        ticker = str(record.get("ticker") or "").strip().upper()

        if not ticker:
            raise ValueError("Every filing manifest record must contain a ticker")

        if ticker in enabled:
            by_ticker[ticker] = dict(record)

    return [by_ticker[ticker] for ticker in ticker_order if ticker in by_ticker]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, list) or any(not isinstance(record, dict) for record in value):
        raise ValueError(f"Filing manifest must be a JSON array of objects: {path}")

    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", help="Download only this configured ticker.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    registry = load_bank_registry(settings.bank_registry_path)
    enabled_banks = [bank for bank in registry.banks if bank.enabled]
    banks = enabled_banks

    if args.ticker:
        ticker = args.ticker.strip().upper()
        banks = [bank for bank in banks if bank.ticker == ticker]

        if not banks:
            raise ValueError(f"Ticker is not enabled in the bank registry: {ticker}")

    records: list[dict[str, str]] = []

    for bank in banks:
        record, downloaded = download_latest_10_k(bank, settings)
        records.append(record)
        status = "downloaded" if downloaded else "already present"
        print(f"{bank.ticker}: {status} -> {record['local_html_path']}")

    if args.ticker:
        records = merge_manifest_records(
            read_manifest(args.manifest),
            records,
            [bank.ticker for bank in enabled_banks],
        )

    write_manifest(args.manifest, records)
    print(f"Manifest: {args.manifest} ({len(records)} filing(s))")


if __name__ == "__main__":
    main()
