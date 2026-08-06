import json
import time
from typing import Any
from urllib.request import Request, urlopen

from bankscope.config.settings import ApplicationSettings, get_settings
from bankscope.sec.company_registry import BankCompany, load_bank_registry

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"


def fetch_bytes(
    url: str,
    *,
    user_agent: str,
    timeout_seconds: float,
    delay_seconds: float,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "*/*",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    finally:
        time.sleep(delay_seconds)


def find_latest_10_k(submissions: dict[str, Any]) -> dict[str, str]:
    recent = submissions["filings"]["recent"]

    indexes = [index for index, form in enumerate(recent["form"]) if form == "10-K"]

    if not indexes:
        raise ValueError("Nije pronađen 10-K")

    index = max(indexes, key=lambda item: recent["filingDate"][item])

    return {
        "form": recent["form"][index],
        "accession_number": recent["accessionNumber"][index],
        "filing_date": recent["filingDate"][index],
        "report_date": recent["reportDate"][index],
        "primary_document": recent["primaryDocument"][index],
    }


def download_latest_10_k(
    bank: BankCompany,
    settings: ApplicationSettings,
) -> tuple[dict[str, object], bool]:
    delay_seconds = 1 / settings.sec_requests_per_second

    submissions = json.loads(
        fetch_bytes(
            SUBMISSIONS_URL.format(cik=bank.cik),
            user_agent=settings.sec_user_agent,
            timeout_seconds=settings.sec_timeout_seconds,
            delay_seconds=delay_seconds,
        )
    )

    sec_tickers = {ticker.upper() for ticker in submissions.get("tickers", [])}

    if bank.ticker not in sec_tickers:
        raise ValueError(f"SEC odgovor ne sadrži ticker {bank.ticker}")

    filing = find_latest_10_k(submissions)
    accession_path = filing["accession_number"].replace("-", "")

    source_url = ARCHIVES_URL.format(
        cik=int(bank.cik),
        accession=accession_path,
        primary_document=filing["primary_document"],
    )

    local_path = settings.raw_data_dir / bank.cik / accession_path / filing["primary_document"]

    downloaded = not local_path.exists()

    if downloaded:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(
            fetch_bytes(
                source_url,
                user_agent=settings.sec_user_agent,
                timeout_seconds=settings.sec_timeout_seconds,
                delay_seconds=delay_seconds,
            )
        )

    manifest_record: dict[str, object] = {
        "ticker": bank.ticker,
        "cik": bank.cik,
        "legal_name": bank.legal_name,
        **filing,
        "source_url": source_url,
        "local_html_path": local_path.as_posix(),
    }

    return manifest_record, downloaded


def main() -> None:
    settings = get_settings()
    registry = load_bank_registry(settings.bank_registry_path)
    banks = [bank for bank in registry.banks if bank.enabled]

    manifest_records: list[dict[str, object]] = []

    for bank in banks:
        try:
            record, downloaded = download_latest_10_k(bank, settings)
            manifest_records.append(record)

            status = "preuzet" if downloaded else "već postoji"
            print(f"{bank.ticker}: {status} -> {record['local_html_path']}")
        except Exception as error:
            print(f"{bank.ticker}: GREŠKA -> {error}")

    settings.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.manifest_dir / "filings.json"
    manifest_path.write_text(
        json.dumps(manifest_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nUspešno: {len(manifest_records)}/{len(banks)}")
    print(f"Manifest: {manifest_path}")

    if len(manifest_records) != len(banks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
