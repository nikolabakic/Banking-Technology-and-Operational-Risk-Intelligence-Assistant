from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from bankscope.sec.company_registry import normalize_bank_text

ResolutionStatus = Literal["resolved", "missing", "multiple"]
ResolutionSource = Literal["question", "session"]


@dataclass(frozen=True)
class BankResolution:
    status: ResolutionStatus
    source: ResolutionSource | None
    ticker: str | None
    detected_tickers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source,
            "ticker": self.ticker,
            "detected_tickers": list(self.detected_tickers),
        }


def _contains_phrase(question: str, phrase: str) -> bool:
    return f" {phrase} " in f" {question} "


def _mentions_c_ticker(question: str) -> bool:
    return bool(re.search(r"(?i)(?:\$c\b|\bticker\s*[:#-]?\s*c\b)", question))


def resolve_bank(
    question: str,
    *,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]] | None = None,
    session_ticker: str | None = None,
) -> BankResolution:
    """Resolve exactly one configured bank without fuzzy matching or model calls."""

    normalized_question = normalize_bank_text(question)
    aliases = bank_aliases or {}
    detected: set[str] = set()

    for raw_ticker, legal_name in bank_names.items():
        ticker = raw_ticker.strip().upper()
        identifiers = {normalize_bank_text(legal_name)}
        identifiers.update(normalize_bank_text(value) for value in aliases.get(ticker, ()))
        if ticker != "C":
            identifiers.add(normalize_bank_text(ticker))
        if any(
            identifier and _contains_phrase(normalized_question, identifier)
            for identifier in identifiers
        ):
            detected.add(ticker)

    if "C" in bank_names and _mentions_c_ticker(question):
        detected.add("C")

    detected_tickers = tuple(sorted(detected))
    if len(detected_tickers) == 1:
        return BankResolution(
            status="resolved",
            source="question",
            ticker=detected_tickers[0],
            detected_tickers=detected_tickers,
        )
    if len(detected_tickers) > 1:
        return BankResolution(
            status="multiple",
            source=None,
            ticker=None,
            detected_tickers=detected_tickers,
        )

    fallback = (session_ticker or "").strip().upper()
    if fallback:
        if fallback not in bank_names:
            raise ValueError(f"Unknown bank ticker: {fallback}.")
        return BankResolution(
            status="resolved",
            source="session",
            ticker=fallback,
            detected_tickers=(),
        )
    return BankResolution(status="missing", source=None, ticker=None, detected_tickers=())
