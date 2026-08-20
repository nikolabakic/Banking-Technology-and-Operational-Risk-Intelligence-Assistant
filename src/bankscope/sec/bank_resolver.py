from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from bankscope.sec.company_registry import bank_identifier_variants, normalize_bank_text

ResolutionStatus = Literal["resolved", "missing", "multiple", "too_many"]
ResolutionSource = Literal["question", "session"]


@dataclass(frozen=True)
class BankResolution:
    status: ResolutionStatus
    source: ResolutionSource | None
    ticker: str | None
    detected_tickers: tuple[str, ...]
    tickers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "source": self.source,
            "ticker": self.ticker,
            "detected_tickers": list(self.detected_tickers),
        }
        if len(self.tickers) > 1 or self.status == "too_many":
            payload["tickers"] = list(self.tickers)
        return payload


def _phrase_positions(question: str, phrase: str) -> tuple[int, ...]:
    """Return safe exact-match positions, including a common omitted-apostrophe possessive.

    ``normalize_bank_text`` already turns ``JP Morgan's`` into ``jp morgan``. Users also
    frequently type ``JP Morgans`` or ``JPMorgans``. Accept that form for multi-token or
    sufficiently specific single-token identifiers so an ordinary verb such as ``chases`` cannot
    accidentally resolve the short ``Chase`` alias.
    """

    padded_question = f" {question} "
    return tuple(
        position
        for candidate in bank_identifier_variants(phrase)
        if (position := padded_question.find(f" {candidate} ")) >= 0
    )


def _mentions_c_ticker(question: str) -> bool:
    return bool(re.search(r"(?i)(?:\$c\b|\bticker\s*[:#-]?\s*c\b)", question))


def resolve_bank(
    question: str,
    *,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]] | None = None,
    session_ticker: str | None = None,
    session_tickers: Sequence[str] = (),
    max_banks: int = 4,
) -> BankResolution:
    """Resolve one comparison scope without fuzzy matching or model calls."""

    if max_banks < 2:
        raise ValueError("max_banks must be at least 2.")

    normalized_question = normalize_bank_text(question)
    aliases = bank_aliases or {}
    detected_positions: dict[str, int] = {}

    for raw_ticker, legal_name in bank_names.items():
        ticker = raw_ticker.strip().upper()
        identifiers = {normalize_bank_text(legal_name)}
        identifiers.update(normalize_bank_text(value) for value in aliases.get(ticker, ()))
        if ticker != "C":
            identifiers.add(normalize_bank_text(ticker))
        positions = [
            position
            for identifier in identifiers
            if identifier
            for position in _phrase_positions(normalized_question, identifier)
        ]
        if positions:
            detected_positions[ticker] = min(position for position in positions if position >= 0)

    if "C" in bank_names and _mentions_c_ticker(question):
        match = re.search(r"(?i)(?:\$c\b|\bticker\s*[:#-]?\s*c\b)", question)
        detected_positions["C"] = match.start() if match else len(question)

    detected_tickers = tuple(
        ticker for ticker, _ in sorted(detected_positions.items(), key=lambda item: item[1])
    )
    if len(detected_tickers) == 1:
        return BankResolution(
            status="resolved",
            source="question",
            ticker=detected_tickers[0],
            detected_tickers=detected_tickers,
            tickers=detected_tickers,
        )
    if 2 <= len(detected_tickers) <= max_banks:
        return BankResolution(
            status="multiple",
            source="question",
            ticker=None,
            detected_tickers=detected_tickers,
            tickers=detected_tickers,
        )
    if len(detected_tickers) > max_banks:
        return BankResolution(
            status="too_many",
            source="question",
            ticker=None,
            detected_tickers=detected_tickers,
            tickers=detected_tickers,
        )

    fallback_values = tuple(
        dict.fromkeys(
            value.strip().upper()
            for value in (session_tickers or ((session_ticker,) if session_ticker else ()))
            if value and value.strip()
        )
    )
    if fallback_values:
        unknown = [ticker for ticker in fallback_values if ticker not in bank_names]
        if unknown:
            raise ValueError(f"Unknown bank ticker(s): {', '.join(unknown)}.")
        if len(fallback_values) > max_banks:
            return BankResolution(
                status="too_many",
                source="session",
                ticker=None,
                detected_tickers=(),
                tickers=fallback_values,
            )
        if len(fallback_values) > 1:
            return BankResolution(
                status="multiple",
                source="session",
                ticker=None,
                detected_tickers=(),
                tickers=fallback_values,
            )
        fallback = fallback_values[0]
        return BankResolution(
            status="resolved",
            source="session",
            ticker=fallback,
            detected_tickers=(),
            tickers=(fallback,),
        )
    return BankResolution(status="missing", source=None, ticker=None, detected_tickers=())
