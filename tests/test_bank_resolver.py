import json
from pathlib import Path

from bankscope.sec.bank_resolver import resolve_bank
from bankscope.sec.company_registry import load_bank_registry

ROOT = Path(__file__).resolve().parents[1]


def registry_maps() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    registry = load_bank_registry(ROOT / "config/banks.yaml")
    enabled = [bank for bank in registry.banks if bank.enabled]
    return (
        {bank.ticker: bank.legal_name for bank in enabled},
        {bank.ticker: bank.aliases for bank in enabled},
    )


def resolve(
    question: str,
    session_ticker: str | None = None,
    session_tickers: tuple[str, ...] = (),
):
    bank_names, bank_aliases = registry_maps()
    return resolve_bank(
        question,
        bank_names=bank_names,
        bank_aliases=bank_aliases,
        session_ticker=session_ticker,
        session_tickers=session_tickers,
    )


def test_resolves_names_aliases_tickers_and_punctuation() -> None:
    cases = {
        "What did JPMorgan report?": "JPM",
        "What did J.P. Morgan Chase report?": "JPM",
        "Summarize the JP Morgans 10-K doc": "JPM",
        "What did BofA report?": "BAC",
        "What did Citi report?": "C",
        "What did ticker C report?": "C",
        "What did $C report?": "C",
        "What did Capital-One report?": "COF",
        "What did State Street Bank report?": "STT",
        "What did PNC's filing report?": "PNC",
        "What did Truist report?": "TFC",
        "What did Goldman Sachs report?": "GS",
        "What did ALLY report?": "ALLY",
        "What did Live Oak Bank report?": "LOB",
    }

    for question, ticker in cases.items():
        resolution = resolve(question)
        assert resolution.status == "resolved"
        assert resolution.source == "question"
        assert resolution.ticker == ticker


def test_c_is_not_resolved_from_an_ordinary_letter_or_substring() -> None:
    assert resolve("Compare capital in scenario C.").status == "missing"
    assert resolve("What generally affects operational risk?").status == "missing"


def test_resolution_priority_and_session_fallback() -> None:
    inherited = resolve("What about its CET1 ratio?", session_ticker="jpm")
    switched = resolve("What did Citi report?", session_ticker="JPM")
    multiple = resolve("Compare Citi and JPMorgan.", session_ticker="TFC")

    assert inherited.as_dict() == {
        "status": "resolved",
        "source": "session",
        "ticker": "JPM",
        "detected_tickers": [],
    }
    assert switched.ticker == "C"
    assert switched.source == "question"
    assert multiple.status == "multiple"
    assert multiple.source == "question"
    assert multiple.ticker is None
    assert multiple.detected_tickers == ("C", "JPM")
    assert multiple.tickers == ("C", "JPM")


def test_resolves_comparison_session_and_rejects_more_than_four_banks() -> None:
    inherited = resolve("What about 2024?", session_tickers=("BAC", "C", "JPM"))
    switched = resolve("What did Capital One report?", session_tickers=("BAC", "C"))
    too_many = resolve("Compare JPMorgan, Bank of America, Citi, Capital One and Goldman Sachs.")

    assert inherited.status == "multiple"
    assert inherited.source == "session"
    assert inherited.tickers == ("BAC", "C", "JPM")
    assert switched.status == "resolved"
    assert switched.tickers == ("COF",)
    assert too_many.status == "too_many"
    assert too_many.tickers == ("JPM", "BAC", "C", "COF", "GS")


def test_omitted_apostrophe_possessives_work_without_matching_short_alias_verbs() -> None:
    spaced = resolve("Compare JP Morgans and Bank of Americas CET1 ratios.")
    compact = resolve("Compare JPMorgans CET1 ratio with Bank of Americas for 2025.")

    assert spaced.status == compact.status == "multiple"
    assert spaced.tickers == compact.tickers == ("JPM", "BAC")
    assert resolve("The strategy chases higher returns.").status == "missing"


def test_frozen_questions_cover_single_multiple_and_missing_resolution() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/evaluation/queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    for row in rows:
        resolution = resolve(row["query"])
        if row.get("ticker"):
            assert resolution.status == "resolved", row["query_id"]
            assert resolution.ticker == row["ticker"], row["query_id"]
        elif row.get("question_type") == "cross_bank_coverage":
            assert resolution.status == "multiple", row["query_id"]
        else:
            assert resolution.status == "missing", row["query_id"]
