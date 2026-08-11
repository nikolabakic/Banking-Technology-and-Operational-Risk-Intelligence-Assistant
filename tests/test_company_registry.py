from pathlib import Path

import pytest
from pydantic import ValidationError

from bankscope.sec.company_registry import BankCompany, BankRegistry, load_bank_registry


def test_bank_company_normalizes_ticker() -> None:
    bank = BankCompany(
        ticker="jpm",
        cik="0000019617",
        legal_name="JPMorgan Chase & Co.",
    )

    assert bank.ticker == "JPM"


def test_bank_company_normalizes_and_rejects_duplicate_aliases() -> None:
    with pytest.raises(ValidationError, match="duplirane aliase"):
        BankCompany(
            ticker="USB",
            cik="0000036104",
            legal_name="U.S. Bancorp",
            aliases=("U.S. Bank", "US Bank"),
        )


def test_registry_rejects_duplicate_ticker() -> None:
    banks = (
        BankCompany(ticker="JPM", cik="0000019617", legal_name="JPMorgan"),
        BankCompany(ticker="jpm", cik="0000070858", legal_name="Duplicate"),
    )

    with pytest.raises(ValidationError, match="duplirane tickere"):
        BankRegistry(version=1, banks=banks)


def test_registry_rejects_alias_owned_by_two_banks() -> None:
    banks = (
        BankCompany(
            ticker="JPM",
            cik="0000019617",
            legal_name="JPMorgan Chase & Co.",
            aliases=("Shared Bank",),
        ),
        BankCompany(
            ticker="BAC",
            cik="0000070858",
            legal_name="Bank of America Corporation",
            aliases=("shared-bank",),
        ),
    )

    with pytest.raises(ValidationError, match="Normalizovani alias"):
        BankRegistry(version=1, banks=banks)


def test_load_bank_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "banks.yaml"
    registry_path.write_text(
        """
        version: 1
        banks:
          - ticker: JPM
            cik: "0000019617"
            legal_name: JPMorgan Chase & Co.
            enabled: true
        """,
        encoding="utf-8",
    )

    registry = load_bank_registry(registry_path)

    assert registry.version == 1
    assert registry.banks[0].ticker == "JPM"
