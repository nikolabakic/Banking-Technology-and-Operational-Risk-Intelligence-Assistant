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

        return self


def load_bank_registry(path: str | Path) -> BankRegistry:
    registry_path = Path(path)

    with registry_path.open(encoding="utf-8") as file:
        registry_data = yaml.safe_load(file)

    return BankRegistry.model_validate(registry_data)
