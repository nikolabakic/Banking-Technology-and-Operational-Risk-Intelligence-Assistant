from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from statistics import fmean
from typing import Any

STATUS_MAP = {
    "answerable": "supported",
    "ambiguous": "ambiguous",
    "unsupported": "unsupported",
}
NUMBER_PATTERN = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CITATION_MARKER_PATTERN = re.compile(r"\[E\d+\]", re.IGNORECASE)
ENTITY_STOPWORDS = {"and", "co", "company", "corp", "corporation", "inc", "na", "the"}


def expected_answer_status(query_status: str) -> str:
    try:
        return STATUS_MAP[query_status]
    except KeyError as error:
        raise ValueError(f"Unsupported evaluation status: {query_status}.") from error


def _normalized_tokens(value: Any, *, remove_entity_stopwords: bool = False) -> set[str]:
    tokens = set(TOKEN_PATTERN.findall(str(value or "").lower()))
    return tokens - ENTITY_STOPWORDS if remove_entity_stopwords else tokens


def _text_match(expected: Any, answer: str, *, entity: bool = False) -> int:
    expected_tokens = _normalized_tokens(expected, remove_entity_stopwords=entity)
    if not expected_tokens:
        return 1
    answer_tokens = _normalized_tokens(answer, remove_entity_stopwords=entity)
    required = max(1, int(len(expected_tokens) * 0.6 + 0.999999))
    return int(len(expected_tokens & answer_tokens) >= required)


def _numbers(answer: str, expected_unit: str) -> list[Decimal]:
    text = CITATION_MARKER_PATTERN.sub("", answer)
    values: list[Decimal] = []
    for match in NUMBER_PATTERN.finditer(text):
        try:
            value = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            continue
        following = text[match.end() : match.end() + 24].lower()
        if "million" in expected_unit.lower() and "billion" in following:
            value *= Decimal(1000)
        values.append(value)
    return values


def _value_match(expected: Any, expected_unit: str, answer: str) -> int:
    try:
        target = Decimal(str(expected))
    except InvalidOperation as error:
        raise ValueError(f"Invalid expected_value: {expected!r}.") from error
    return int(
        any(abs(value - target) <= Decimal("0.0001") for value in _numbers(answer, expected_unit))
    )


def _unit_match(expected_unit: str, answer: str) -> int:
    expected = expected_unit.strip().lower()
    text = answer.lower()
    if expected == "percent":
        return int("%" in text or "percent" in text)
    if expected == "usd millions":
        return int("million" in text and ("$" in text or "usd" in text or "dollar" in text))
    return _text_match(expected, text)


def _period_match(expected_period: Any, answer: str) -> int:
    expected = str(expected_period or "").strip()
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", expected)
    if years:
        return int(all(year in answer for year in years))
    return int(expected.lower() in answer.lower())


def _audit_accepted_ids(citation_audit: Mapping[str, Any] | None) -> set[str]:
    if not citation_audit:
        return set()
    values = citation_audit.get("accepted_additional_target_chunk_ids", [])
    if not isinstance(values, Sequence) or isinstance(values, str):
        return set()
    return {str(value) for value in values if str(value)}


def _citation_metrics(
    query: Mapping[str, Any],
    answer: Mapping[str, Any],
    expected_status: str,
    citation_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw_citations = answer.get("citations")
    citations = raw_citations if isinstance(raw_citations, Sequence) else []
    cited_ids = [
        str(citation.get("target_chunk_id") or "")
        for citation in citations
        if isinstance(citation, Mapping)
    ]
    cited_ids = [target_id for target_id in cited_ids if target_id]
    cited_set = set(cited_ids)
    relevant_ids = {str(value) for value in query.get("relevant_target_chunk_ids", [])}
    relevant_cited = cited_set & relevant_ids
    precision = len(relevant_cited) / len(cited_set) if cited_set else None

    support_ids = relevant_ids | _audit_accepted_ids(citation_audit)
    supported_cited = cited_set & support_ids
    support_precision = len(supported_cited) / len(cited_set) if cited_set else None

    raw_groups = query.get("required_evidence_groups", [])
    groups = [
        {str(value) for value in group.get("target_chunk_ids", [])}
        for group in raw_groups
        if isinstance(group, Mapping)
    ]
    group_coverage = (
        sum(bool(group & cited_set) for group in groups) / len(groups) if groups else None
    )
    if expected_status == "supported":
        complete = int(group_coverage == 1.0) if groups else int(bool(relevant_cited))
        support_complete = int(group_coverage == 1.0) if groups else int(bool(supported_cited))
    else:
        complete = int(not cited_ids)
        support_complete = complete
    return {
        "citation_count": len(cited_set),
        "relevant_citation_count": len(relevant_cited),
        "citation_precision": precision,
        "citation_relevant_hit": int(bool(relevant_cited)),
        "required_group_count": len(groups),
        "required_group_coverage": group_coverage,
        "citation_complete": complete,
        "citation_support_count": len(supported_cited),
        "citation_support_precision": support_precision,
        "citation_support_hit": int(bool(supported_cited)),
        "citation_support_complete": support_complete,
    }


def _facts(answer: Mapping[str, Any]) -> Mapping[str, Any] | None:
    facts = answer.get("facts")
    return facts if isinstance(facts, Mapping) else None


def evaluate_answer(
    query: Mapping[str, Any],
    answer: Mapping[str, Any],
    citation_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic generation metrics for one frozen evaluation query."""
    expected_status = expected_answer_status(str(query.get("status") or ""))
    actual_status = str(answer.get("status") or "")
    answer_text = str(answer.get("answer") or "")
    facts = _facts(answer)
    structured: dict[str, int] = {}
    if expected_status == "supported":
        value_source = str(facts.get("value_text") or "") if facts else answer_text
        unit_source = str(facts.get("unit") or "") if facts else answer_text
        period_source = str(facts.get("period") or "") if facts else answer_text
        entity_source = str(facts.get("entity") or "") if facts else answer_text
        variant_source = str(facts.get("variant") or "") if facts else answer_text
        if query.get("expected_value") is not None:
            expected_unit = str(query.get("expected_unit") or "")
            structured["value_match"] = _value_match(
                query["expected_value"], expected_unit, value_source
            )
        if query.get("expected_unit"):
            structured["unit_match"] = _unit_match(str(query["expected_unit"]), unit_source)
        if query.get("expected_period"):
            structured["period_match"] = _period_match(query["expected_period"], period_source)
        if query.get("expected_entity"):
            structured["entity_match"] = _text_match(
                query["expected_entity"], entity_source, entity=True
            )
        if query.get("expected_variant"):
            structured["variant_match"] = _text_match(query["expected_variant"], variant_source)

    return {
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status_correct": int(actual_status == expected_status),
        "citations": _citation_metrics(
            query, answer, expected_status, citation_audit=citation_audit
        ),
        "structured": structured,
        "structured_source": "facts" if facts is not None else "answer_text",
    }


def _mean_metric(values: Sequence[Mapping[str, Any]], key: str) -> float | None:
    selected = [float(value[key]) for value in values if value.get(key) is not None]
    return fmean(selected) if selected else None


def summarize_answer_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_rows = [row for row in rows if isinstance(row.get("metrics"), Mapping)]
    summary: dict[str, Any] = {
        "query_count": len(rows),
        "evaluated_count": len(metric_rows),
        "error_count": len(rows) - len(metric_rows),
    }
    if not metric_rows:
        return summary

    summary["status_accuracy"] = fmean(int(row["metrics"]["status_correct"]) for row in metric_rows)
    citation_values = [
        row["metrics"]["citations"] for row in metric_rows if row["metrics"]["citations"]
    ]
    supported_citation_values = [
        row["metrics"]["citations"]
        for row in metric_rows
        if row["metrics"]["expected_status"] == "supported"
    ]
    summary["citation_relevant_hit_rate"] = _mean_metric(
        supported_citation_values, "citation_relevant_hit"
    )
    summary["citation_complete_rate"] = _mean_metric(citation_values, "citation_complete")
    summary["mean_citation_precision"] = _mean_metric(
        supported_citation_values, "citation_precision"
    )
    summary["citation_support_hit_rate"] = _mean_metric(
        supported_citation_values, "citation_support_hit"
    )
    summary["citation_support_complete_rate"] = _mean_metric(
        citation_values, "citation_support_complete"
    )
    summary["mean_citation_support_precision"] = _mean_metric(
        supported_citation_values, "citation_support_precision"
    )

    for key in ("value_match", "unit_match", "period_match", "entity_match", "variant_match"):
        values = [
            int(row["metrics"]["structured"][key])
            for row in metric_rows
            if key in row["metrics"]["structured"]
        ]
        summary[f"{key}_count"] = len(values)
        summary[f"{key}_rate"] = fmean(values) if values else None

    judgements = [row["judge"] for row in metric_rows if isinstance(row.get("judge"), Mapping)]
    summary["semantic_judge_count"] = len(judgements)
    for key in ("correctness", "completeness", "groundedness"):
        values = [int(bool(judgement[key])) for judgement in judgements if key in judgement]
        summary[f"semantic_{key}_rate"] = fmean(values) if values else None

    generation_rows: list[Mapping[str, Any]] = []
    for row in rows:
        answer = row.get("answer")
        error = row.get("error")
        if isinstance(answer, Mapping) and isinstance(answer.get("generation"), Mapping):
            generation_rows.append(answer["generation"])
        elif isinstance(error, Mapping) and isinstance(error.get("generation"), Mapping):
            generation_rows.append(error["generation"])
    summary["generation_request_count"] = sum(
        int(generation.get("request_count") or 0) for generation in generation_rows
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        values = [
            int(generation["usage"][key])
            for generation in generation_rows
            if isinstance(generation.get("usage"), Mapping) and key in generation["usage"]
        ]
        summary[f"generation_{key}"] = sum(values) if values else None
    return summary
