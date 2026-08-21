from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from bankscope.generation.answer_generator import GenerationValidationError
from bankscope.sec.company_registry import bank_identifier_variants, normalize_bank_text

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?%?")
TIER_ONE_PATTERN = re.compile(r"(?i)\btier\s+1\b")
FORM_10K_PATTERN = re.compile(r"(?i)\b10\s*[-\u2010-\u2015 ]?\s*k\b")
FORM_10Q_PATTERN = re.compile(r"(?i)\b10\s*[-\u2010-\u2015 ]?\s*q\b")
FOLLOW_UP_PREFIX = re.compile(
    r"(?i)^\s*(?:and|but|then|also|what\s+about|how\s+about|compare\s+(?:it|that)|"
    r"tell\s+me\s+more|more\s+(?:detail|details|about)|go\s+on|continue|"
    r"a\s+(?:kako|šta|sta|koliko|gde|da\s+li)|ali|onda|takođe|takodje|"
    r"šta\s+je\s+sa|sta\s+je\s+sa|reci\s+(?:mi\s+)?više|reci\s+(?:mi\s+)?vise|"
    r"više\s+o\s+tome|vise\s+o\s+tome|detaljnije|nastavi)\b"
)
FOLLOW_UP_REFERENCE = re.compile(
    r"(?i)\b(?:it|its|they|their|them|this|that|these|those|former|latter|same|"
    r"previous|earlier|above|ona|ono|oni|njega|njen|njegov|njihov|taj|"
    r"ista|isti|isto|prethodn\w*)\b"
)
SUMMARY_PATTERN = re.compile(
    r"(?i)\b(?:summari[sz]e|summary|overview|recap|sažmi|sazmi|rezime|sumariši|"
    r"sumarisi)\b"
)
WHOLE_DOCUMENT_PATTERN = re.compile(r"(?i)\b(?:whole|entire|complete|cela|celokupn\w*)\b")
FOCUSED_SUMMARY_PATTERN = re.compile(
    r"(?i)\b(?:cybersecurity|operational\s+risk|credit\s+risk|liquidity|capital|cet1|"
    r"revenue|net\s+income|loans?|deposits?|segment|legal|regulatory|accounting)\b"
)
FRAGMENT_PATTERN = re.compile(r"(?i)^\s*(?:(?:in|for|za|u)\s+)?(?:19|20)\d{2}\s*[?.!]*\s*$")

SUMMARY_ASPECTS = (
    "business model, operating segments, strategy, and material developments",
    "financial performance, revenue, net income, assets, loans, and deposits",
    "risk factors, operational risk, cybersecurity, legal, and regulatory matters",
    "capital adequacy, CET1, liquidity, funding, and credit quality",
    "management outlook, uncertainties, and critical accounting judgments",
)

FOCUSED_QUERY_VARIANTS = (
    (
        re.compile(r"(?i)\b(?:cyber(?:security)?|sajber)\b"),
        "cybersecurity risk information security cyber attack",
    ),
    (
        re.compile(
            r"(?i)\b(?:third[-\s]?party|trec\w*\s+(?:stran\w*|lic\w*)|"
            r"dobavljac\w*|vendor|outsourc\w*)\b"
        ),
        "third-party risk management vendor service provider outsourcing",
    ),
    (
        re.compile(r"(?i)\b(?:operational\s+risk|operativn\w*\s+rizik\w*)\b"),
        "operational risk framework operational risk management",
    ),
    (
        re.compile(r"(?i)\bcet1\b"),
        "CET1 common equity tier 1 capital ratio",
    ),
)


def needs_contextualization(question: str) -> bool:
    """Return whether the current wording depends on a prior conversational turn."""

    normalized = " ".join(question.split())
    if not normalized:
        return False
    if FOLLOW_UP_PREFIX.search(normalized) or FOLLOW_UP_REFERENCE.search(normalized):
        return True
    if FRAGMENT_PATTERN.fullmatch(normalized):
        return True
    return normalize_bank_text(normalized) in {
        "why",
        "how",
        "when",
        "where",
        "which one",
        "zašto",
        "zasto",
        "kako",
        "kada",
        "gde",
        "koja",
        "ratio",
        "percentage",
        "procenat",
        "iznos",
        "capital amount",
    }


def recent_conversation_history(
    history: Sequence[Mapping[str, str]], *, max_turns: int = 2
) -> list[Mapping[str, str]]:
    """Select recent pairs and strip assistant-authored facts from routing context."""

    if max_turns <= 0:
        raise ValueError("max_turns must be positive.")
    if len(history) % 2:
        raise ValueError("Conversation history must contain complete turn pairs.")
    selected = history[-(max_turns * 2) :]
    compact: list[Mapping[str, str]] = []
    allowed_state_keys = {
        "dialog_act",
        "status",
        "tickers",
        "mode",
        "answer_type",
        "resolved_question",
    }
    for message in selected:
        if message.get("role") != "assistant":
            compact.append(message)
            continue
        content = str(message.get("content") or "")
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = {}
        state = (
            {key: parsed[key] for key in allowed_state_keys if key in parsed}
            if isinstance(parsed, Mapping)
            else {}
        )
        if not state:
            state = {"dialog_act": "answer", "state": "assistant_content_omitted"}
        compact.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    state, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
            }
        )
    return compact


def _numeric_facts(text: str) -> set[str]:
    without_terms = TIER_ONE_PATTERN.sub("tier one", text)
    without_terms = FORM_10K_PATTERN.sub("form ten k", without_terms)
    without_terms = FORM_10Q_PATTERN.sub("form ten q", without_terms)
    return {value.rstrip("%").replace(",", ".") for value in NUMBER_PATTERN.findall(without_terms)}


def remove_untrusted_numeric_facts(
    standalone_question: str,
    *,
    current_question: str,
    allowed_user_context: Sequence[str] = (),
) -> tuple[str, bool]:
    """Remove numeric tokens copied only from assistant-authored history."""

    allowed = _numeric_facts("\n".join((current_question, *allowed_user_context)))
    form_spans = [
        match.span()
        for pattern in (FORM_10K_PATTERN, FORM_10Q_PATTERN)
        for match in pattern.finditer(standalone_question)
    ]

    def replace(match: re.Match[str]) -> str:
        if any(start <= match.start() and match.end() <= end for start, end in form_spans):
            return match.group(0)
        normalized = match.group(0).rstrip("%").replace(",", ".")
        return match.group(0) if normalized in allowed else ""

    sanitized = NUMBER_PATTERN.sub(replace, standalone_question)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = re.sub(r"\s+([,.;:?!])", r"\1", sanitized)
    return sanitized, sanitized != standalone_question


def validate_contextualized_rewrite(
    current_question: str,
    standalone_question: str,
    *,
    allowed_user_context: Sequence[str] = (),
) -> None:
    """Reject periods or numeric facts that did not originate in user-authored text."""

    current_years = set(YEAR_PATTERN.findall(current_question))
    rewritten_years = set(YEAR_PATTERN.findall(standalone_question))
    if current_years and rewritten_years != current_years:
        raise GenerationValidationError(
            "contextualization_changed_period",
            "The standalone question changed an explicit period in the current question.",
            generation={"stage": "contextualizing"},
        )
    allowed_years = current_years | {
        year for text in allowed_user_context for year in YEAR_PATTERN.findall(text)
    }
    if not current_years and rewritten_years - allowed_years:
        raise GenerationValidationError(
            "contextualization_added_period",
            "The standalone question introduced a period absent from user-authored context.",
            generation={"stage": "contextualizing"},
        )

    allowed_numbers = _numeric_facts("\n".join((current_question, *allowed_user_context)))
    rewritten_numbers = _numeric_facts(standalone_question)
    current_numbers = _numeric_facts(current_question)
    if current_numbers - rewritten_numbers:
        raise GenerationValidationError(
            "contextualization_lost_numeric_fact",
            "The standalone question dropped a numeric fact from the current question.",
            generation={"stage": "contextualizing"},
        )
    if rewritten_numbers - allowed_numbers:
        raise GenerationValidationError(
            "contextualization_added_numeric_fact",
            "The standalone question introduced a numeric fact absent from user-authored context.",
            generation={"stage": "contextualizing"},
        )


def is_general_chat_question(question: str) -> bool:
    """Recognize only narrow greetings and product-help requests without an LLM router."""

    normalized = normalize_bank_text(question)
    return normalized in {
        "hi",
        "hello",
        "hey",
        "zdravo",
        "cao",
        "ćao",
        "help",
        "pomoc",
        "pomoć",
        "what can you do",
        "how can you help",
        "šta možeš da uradiš",
        "sta mozes da uradis",
    }


def build_bank_subquestion(
    question: str,
    *,
    ticker: str,
    selected_tickers: Sequence[str],
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]],
) -> str:
    """Create one deterministic bank-only retrieval question from a comparison."""

    cleaned_question = question
    for selected in selected_tickers:
        escaped = re.escape(selected)
        cleaned_question = re.sub(
            rf"(?i)\(\s*(?:ticker\s*:\s*)?{escaped}\s*\)",
            " ",
            cleaned_question,
        )
        cleaned_question = re.sub(
            rf"(?i)\bticker\s*:\s*{escaped}\b|\bticker\s+{escaped}\b",
            " ",
            cleaned_question,
        )
    normalized = f" {normalize_bank_text(cleaned_question)} "
    identifiers: set[str] = set()
    for selected in selected_tickers:
        values = [bank_names.get(selected, selected), *bank_aliases.get(selected, ())]
        if selected != "C":
            values.append(selected)
        for value in values:
            identifiers.update(bank_identifier_variants(value))
    for identifier in sorted(
        identifiers, key=lambda value: (len(value.split()), len(value)), reverse=True
    ):
        normalized = normalized.replace(f" {identifier} ", " ")

    topic = " ".join(normalized.split())
    topic = re.sub(
        r"(?i)\b(?:compare|comparison|versus|vs|against|between|with|sa|uporedi|poredi|"
        r"poređenje|poredjenje|između|izmedju)\b",
        " ",
        topic,
    )
    topic = re.sub(r"\s+", " ", topic).strip(" ,.;:-")
    topic = re.sub(r"(?i)^(?:and|i)\s+|\s+(?:and|i)$", "", topic).strip()
    topic_tokens = topic.split()
    for width in range(min(6, len(topic_tokens) // 2), 0, -1):
        index = 0
        while index + (2 * width) <= len(topic_tokens):
            if topic_tokens[index : index + width] == topic_tokens[
                index + width : index + (2 * width)
            ]:
                del topic_tokens[index + width : index + (2 * width)]
            else:
                index += 1
    topic = " ".join(topic_tokens)
    if not topic:
        topic = "requested filing information"
    bank_name = bank_names.get(ticker, ticker)
    return f"{bank_name} ({ticker}) Form 10-K: {topic}"


def is_document_summary_question(question: str) -> bool:
    normalized = normalize_bank_text(question)
    if not SUMMARY_PATTERN.search(normalized) or not re.search(r"\b10\s*k\b", normalized):
        return False
    if WHOLE_DOCUMENT_PATTERN.search(normalized):
        return True
    return not FOCUSED_SUMMARY_PATTERN.search(normalized)


def build_retrieval_queries(
    question: str,
    *,
    ticker: str,
    bank_name: str,
    original_question: str | None = None,
) -> tuple[str, ...]:
    """Build diverse searches while preserving both original and resolved user intent."""

    if is_document_summary_question(question):
        years = " ".join(dict.fromkeys(YEAR_PATTERN.findall(question)))
        period = f" {years}" if years else ""
        return tuple(
            f"{bank_name} ({ticker}) Form 10-K{period}: {aspect}" for aspect in SUMMARY_ASPECTS
        )

    queries = [question]
    original = str(original_question or "").strip()
    if original:
        queries.append(original)
    concept_text = " ".join((question, original))
    for pattern, canonical_terms in FOCUSED_QUERY_VARIANTS:
        if pattern.search(concept_text):
            queries.append(f"{bank_name} ({ticker}) Form 10-K: {canonical_terms}")
    return tuple(dict.fromkeys(query for query in queries if query.strip()))


def build_focused_recovery_queries(
    question: str,
    *,
    ticker: str,
    bank_name: str,
) -> tuple[str, ...]:
    """Build one bounded second-pass query for a recognized focused filing metric."""

    years = " ".join(dict.fromkeys(YEAR_PATTERN.findall(question)))
    period = f" {years}" if years else ""
    queries = []
    for pattern, canonical_terms in FOCUSED_QUERY_VARIANTS:
        if pattern.search(question):
            queries.append(
                f"{bank_name} ({ticker}) Form 10-K{period}: relevant filing table or section "
                f"{canonical_terms}"
            )
    return tuple(dict.fromkeys(queries))


def focused_evidence_signals(
    question: str,
    evidence: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
    """Assess whether retrieved evidence visibly contains a focused numeric answer candidate."""

    matched_variants = [
        (pattern, canonical_terms)
        for pattern, canonical_terms in FOCUSED_QUERY_VARIANTS
        if pattern.search(question)
    ]
    focused = bool(matched_variants)
    documents = []
    has_table = False
    for item in evidence:
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        documents.append(
            " ".join(
                (
                    str(item.get("evidence") or item.get("document") or ""),
                    str(item.get("section_title") or metadata.get("section_title") or ""),
                    str(item.get("report_date") or metadata.get("report_date") or ""),
                )
            )
        )
        record_type = str(item.get("record_type") or metadata.get("record_type") or "")
        has_table = has_table or record_type.casefold() == "table"
    evidence_text = "\n".join(documents)
    requested_years = set(YEAR_PATTERN.findall(question))
    period = not requested_years or requested_years <= set(YEAR_PATTERN.findall(evidence_text))
    numeric = bool(re.search(r"(?<!\w)\d+(?:[.,]\d+)?\s*%", evidence_text))
    concept = False
    for pattern, canonical_terms in matched_variants:
        canonical_tokens = {
            token
            for token in normalize_bank_text(canonical_terms).split()
            if len(token) >= 4
        }
        evidence_tokens = set(normalize_bank_text(evidence_text).split())
        concept = concept or bool(pattern.search(evidence_text)) or len(
            canonical_tokens & evidence_tokens
        ) >= 2
    return {
        "focused": focused,
        "period": period,
        "numeric": numeric,
        "concept": concept,
        "table": has_table,
        "strong": focused and period and numeric and concept,
    }


def round_robin_evidence(
    groups: Sequence[Sequence[Mapping[str, object]]], *, limit: int
) -> list[dict[str, object]]:
    """Merge query result lists without letting one summary aspect consume the budget."""

    if limit <= 0:
        raise ValueError("limit must be positive.")
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    max_group = max((len(group) for group in groups), default=0)
    for rank in range(max_group):
        for group in groups:
            if rank >= len(group):
                continue
            item = group[rank]
            target = str(item.get("target_chunk_id") or "")
            if not target or target in seen:
                continue
            seen.add(target)
            merged.append(dict(item))
            if len(merged) == limit:
                return merged
    return merged
