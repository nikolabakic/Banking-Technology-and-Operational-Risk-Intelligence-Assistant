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


def test_registry_rejects_duplicate_ticker() -> None:
    banks = (
        BankCompany(ticker="JPM", cik="0000019617", legal_name="JPMorgan"),
        BankCompany(ticker="jpm", cik="0000070858", legal_name="Duplicate"),
    )

    with pytest.raises(ValidationError, match="duplirane tickere"):
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
