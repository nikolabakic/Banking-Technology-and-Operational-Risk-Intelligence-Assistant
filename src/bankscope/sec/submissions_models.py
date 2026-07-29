from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SecRecentFilings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    form: tuple[str, ...] = ()


class SecFilings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    recent: SecRecentFilings


class SecSubmissions(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    cik: str
    name: str = Field(min_length=1)
    tickers: tuple[str, ...] = ()
    exchanges: tuple[str, ...] = ()
    filings: SecFilings

    @field_validator("cik", mode="before")
    @classmethod
    def normalize_cik(cls, value: Any) -> str:
        cik = str(value).strip()

        if not cik.isdigit() or len(cik) > 10:
            raise ValueError("SEC CIK mora sadržati najviše 10 cifara.")

        return cik.zfill(10)

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(ticker.upper() for ticker in value)

    @property
    def has_10_k(self) -> bool:
        return "10-K" in self.filings.recent.form
