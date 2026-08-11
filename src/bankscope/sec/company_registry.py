import re
import unicodedata
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BankCompany(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    ticker: str = Field(min_length=1, max_length=10)
    cik: str = Field(pattern=r"^\d{10}$")
    legal_name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    enabled: bool = True

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.upper()

        if not ticker[0].isalpha() or not all(
            character.isalnum() or character in ".-" for character in ticker
        ):
            raise ValueError(
                "Ticker mora početi slovom i sadržati samo slova, cifre, tačku ili crticu."
            )

        return ticker

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        aliases = tuple(value.strip() for value in values)
        if any(not value for value in aliases):
            raise ValueError("Alias banke ne sme biti prazan.")
        normalized = [normalize_bank_text(value) for value in aliases]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Banka sadrzi duplirane aliase.")
        return aliases


class BankRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    banks: tuple[BankCompany, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        tickers = [bank.ticker for bank in self.banks]
        ciks = [bank.cik for bank in self.banks]

        if len(tickers) != len(set(tickers)):
            raise ValueError("Registry sadrži duplirane tickere.")

        if len(ciks) != len(set(ciks)):
            raise ValueError("Registry sadrži duplirane CIK vrednosti.")

        if not any(bank.enabled for bank in self.banks):
            raise ValueError("Registry mora sadržati najmanje jednu enabled banku.")

        alias_owners: dict[str, str] = {}
        for bank in self.banks:
            for identifier in (bank.legal_name, bank.ticker, *bank.aliases):
                normalized = normalize_bank_text(identifier)
                owner = alias_owners.get(normalized)
                if owner is not None and owner != bank.ticker:
                    raise ValueError(
                        f"Normalizovani alias {identifier!r} koriste i {owner} i {bank.ticker}."
                    )
                alias_owners[normalized] = bank.ticker

        return self


def normalize_bank_text(value: str) -> str:
    """Normalize bank names and questions for exact phrase matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\b([\w]+)[\u2019']s\b", r"\1", normalized)
    normalized = re.sub(r"[\u2019']", "", normalized)
    normalized = normalized.replace(".", "")
    normalized = re.sub(r"[_\W]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def load_bank_registry(path: str | Path) -> BankRegistry:
    registry_path = Path(path)

    with registry_path.open(encoding="utf-8") as file:
        registry_data = yaml.safe_load(file)

    return BankRegistry.model_validate(registry_data)
