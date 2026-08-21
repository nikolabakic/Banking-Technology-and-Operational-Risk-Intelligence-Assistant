import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bankscope.generation.answer_generator import GenerationValidationError
from bankscope.generation.comparison_generator import _normalize_claim_citation_ids
from bankscope.generation.pipeline import (
    SentenceTransformerQueryEncoder,
    SingleBankAnswerPipeline,
)
from bankscope.retrieval.mixed_retriever import BankSearchResult


def test_query_encoder_loads_pinned_model_from_local_cache(monkeypatch) -> None:
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **options):
            captured.update({"name": name, **options})

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    SentenceTransformerQueryEncoder("model-name", "model-revision")
    assert captured == {
        "name": "model-name",
        "revision": "model-revision",
        "local_files_only": True,
    }


def test_comparison_normalizes_redundant_citation_ids_from_inline_markers() -> None:
    normalized, changed = _normalize_claim_citation_ids(
        '{"claims":[{"text":"JPM [E1] and BAC [E2].",'
        '"tickers":["JPM","BAC"],"citation_ids":["E1"]}]}'
    )

    assert changed is True
    assert '"citation_ids":["E1","E2"]' in normalized

    malformed = "not-json"
    assert _normalize_claim_citation_ids(malformed) == (malformed, False)


class MockEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)


class MockRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search_hybrid(self, question, query_vector, **kwargs):
        self.calls.append((question, query_vector, kwargs))
        return [
            {
                "target_chunk_id": "chunk-1",
                "record_type": "text",
                "ticker": kwargs["ticker"],
                "evidence": "The filing states that the CET1 ratio was 14.6%.",
                "metadata": {"report_date": "2025-12-31"},
            }
        ]

    def search_hybrid_by_ticker(
        self, question, query_vector, *, tickers, limit_per_ticker, **kwargs
    ):
        return [
            BankSearchResult(
                ticker=ticker,
                results=self.search_hybrid(
                    question,
                    query_vector,
                    ticker=ticker,
                    limit=limit_per_ticker,
                    **kwargs,
                ),
                latency_ms=0.0,
            )
            for ticker in tickers
        ]


class ContextualizedRetriever(MockRetriever):
    def search_hybrid(self, question, query_vector, **kwargs):
        result = super().search_hybrid(question, query_vector, **kwargs)
        result[0]["metadata"]["report_date"] = "2024-12-31"
        return result


class MockCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_names = {tool["function"]["name"] for tool in (kwargs.get("tools") or [])}
        if "route_conversation" in tool_names:
            payload = json.loads(kwargs["messages"][1]["content"])
            function = SimpleNamespace(
                name="route_conversation",
                arguments=json.dumps(
                    {
                        "action": "filing_research",
                        "confidence": 0.98,
                        "search_question": payload["current_question"],
                        "reason": "The question requires filing evidence.",
                        "response_text": None,
                        "category": None,
                        "missing": None,
                        "citation_ids": [],
                        "presentation_guidance": None,
                    }
                ),
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        arguments = (
            '{"status":"supported","answer_type":"numeric",'
            '"answer":"The ratio was 14.6% [E1].",'
            '"facts":{"entity":"JPMorgan Chase & Co.","metric":"ratio",'
            '"variant":null,"period":"2025","value_text":"14.6%",'
            '"unit":"percent"},"citation_ids":["E1"],'
            '"reason":"Direct support."}'
        )
        function = SimpleNamespace(name="submit_supported_numeric_answer", arguments=arguments)
        message = SimpleNamespace(
            content=None,
            refusal=None,
            tool_calls=[SimpleNamespace(function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )


class ContextualizedCompletions(MockCompletions):
    def create(self, **kwargs):
        if not self.calls:
            self.calls.append(kwargs)
            function = SimpleNamespace(
                name="route_conversation",
                arguments=json.dumps(
                    {
                        "action": "filing_research",
                        "confidence": 0.98,
                        "search_question": ("What was JPMorgan Chase & Co. CET1 ratio in 2024?"),
                        "reason": "Resolve the natural follow-up from recent user context.",
                        "response_text": None,
                        "category": None,
                        "missing": None,
                        "citation_ids": [],
                        "presentation_guidance": None,
                    }
                ),
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        return super().create(**kwargs)


class ComparisonCompletions(MockCompletions):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        prompt = kwargs["messages"][1]["content"]
        if "concise comparison" in system:
            content = (
                '{"claims":[{"text":"JPM reported 14.6% [E1], while BAC reported '
                '14.6% [E2].","tickers":["JPM","BAC"],'
                '"citation_ids":["E1","E2"]}]}'
            )
            function = SimpleNamespace(
                name="submit_comparison_synthesis",
                arguments=content,
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        else:
            entity = (
                "Bank of America Corporation"
                if "Expected ticker: BAC" in prompt
                else "JPMorgan Chase & Co."
            )
            content = (
                '{"status":"supported","answer_type":"numeric",'
                '"answer":"The ratio was 14.6% [E1].",'
                f'"facts":{{"entity":"{entity}","metric":"ratio",'
                '"variant":null,"period":"2025","value_text":"14.6%",'
                '"unit":"percent"},"citation_ids":["E1"],'
                '"reason":"Direct support."}'
            )
        function = SimpleNamespace(name="submit_supported_numeric_answer", arguments=content)
        message = SimpleNamespace(
            content=None,
            refusal=None,
            tool_calls=[SimpleNamespace(function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )


class PartialRetriever(MockRetriever):
    def search_hybrid(self, question, query_vector, **kwargs):
        if kwargs["ticker"] == "BAC":
            self.calls.append((question, query_vector, kwargs))
            return []
        return super().search_hybrid(question, query_vector, **kwargs)


class EmptyRetriever(MockRetriever):
    def search_hybrid(self, question, query_vector, **kwargs):
        self.calls.append((question, query_vector, kwargs))
        return []


class OperationalFrameworkRetriever(MockRetriever):
    def search_hybrid(self, question, query_vector, **kwargs):
        self.calls.append((question, query_vector, kwargs))
        if "operational risk framework operational risk management" not in question.casefold():
            return []
        return [
            {
                "target_chunk_id": "jpm-operational-framework",
                "record_type": "text",
                "ticker": kwargs["ticker"],
                "evidence": (
                    "JPMorgan Chase manages operational risk through a firmwide framework "
                    "with governance, risk identification, assessment, monitoring and controls."
                ),
                "metadata": {"report_date": "2025-12-31"},
            }
        ]


class OperationalFrameworkCompletions(MockCompletions):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        arguments = json.dumps(
            {
                "status": "supported",
                "answer_type": "narrative",
                "answer": (
                    "JPMorgan Chase describes a firmwide operational-risk framework with "
                    "governance, identification, assessment, monitoring and controls [E1]."
                ),
                "facts": None,
                "citation_ids": ["E1"],
                "reason": "The filing evidence directly describes the framework.",
            }
        )
        function = SimpleNamespace(name="submit_supported_narrative_answer", arguments=arguments)
        message = SimpleNamespace(
            content=None,
            refusal=None,
            tool_calls=[SimpleNamespace(function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )


class PartialCompletions(ComparisonCompletions):
    def create(self, **kwargs):
        if "concise comparison" in kwargs["messages"][0]["content"]:
            raise AssertionError("Partial comparisons must not call the synthesis model.")
        return super().create(**kwargs)


class AbstainOnceCompletions(ComparisonCompletions):
    def __init__(self) -> None:
        super().__init__()
        self.abstained = False

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        system = kwargs["messages"][0]["content"]
        if "Expected ticker: JPM" in prompt and not self.abstained:
            self.abstained = True
            self.calls.append(kwargs)
            function = SimpleNamespace(
                name="submit_unsupported_answer",
                arguments=json.dumps(
                    {
                        "status": "unsupported",
                        "answer_type": "narrative",
                        "answer": "Insufficient evidence.",
                        "facts": None,
                        "citation_ids": [],
                        "reason": "The evidence appears insufficient.",
                    }
                ),
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        if "Expected ticker: JPM" in prompt and "bounded evidence recheck" not in system:
            raise AssertionError("The second JPM request must be the bounded evidence recheck.")
        return super().create(**kwargs)


class InvalidCitationOnceCompletions(ComparisonCompletions):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "Expected ticker: JPM" in prompt and not self.failed:
            self.failed = True
            self.calls.append(kwargs)
            function = SimpleNamespace(
                name="submit_supported_narrative_answer",
                arguments=json.dumps(
                    {
                        "status": "supported",
                        "answer_type": "narrative",
                        "answer": "The CET1 ratio was 14.6% [E9].",
                        "facts": None,
                        "citation_ids": ["E9"],
                        "reason": "Direct support.",
                    }
                ),
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        return super().create(**kwargs)


class FocusedRecoveryRetriever(MockRetriever):
    def search_hybrid(self, question, query_vector, **kwargs):
        self.calls.append((question, query_vector, kwargs))
        if kwargs["ticker"] == "BAC" and "relevant filing table or section" not in question:
            return []
        return [
            {
                "target_chunk_id": f"{kwargs['ticker']}-cet1",
                "record_type": "table",
                "ticker": kwargs["ticker"],
                "evidence": "Common Equity Tier 1 (CET1) capital ratio was 14.6%.",
                "metadata": {"report_date": "2025-12-31"},
            }
        ]


class PerBankValidationCompletions(ComparisonCompletions):
    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "Expected ticker: JPM" in prompt:
            self.calls.append(kwargs)
            function = SimpleNamespace(
                name="submit_supported_numeric_answer",
                arguments='{"status":"supported"}',
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        return super().create(**kwargs)


def test_pipeline_reuses_encoder_and_retriever_and_preserves_cli_output() -> None:
    encoder = MockEncoder()
    retriever = MockRetriever()
    completions = MockCompletions()
    close_calls = []
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        close_callback=lambda: close_calls.append(True),
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    first = pipeline.answer("What was the ratio?", ticker="jpm")
    second = pipeline.answer("What was the ratio?", ticker="jpm")
    pipeline.close()
    pipeline.close()

    assert encoder.calls == ["What was the ratio?", "What was the ratio?"]
    assert len(retriever.calls) == 2
    assert close_calls == [True]
    assert first.output["ticker"] == "JPM"
    assert first.output["bank_resolution"] == {
        "status": "resolved",
        "source": "session",
        "ticker": "JPM",
        "detected_tickers": [],
    }
    assert first.output["retrieval"] == {
        "backend": "mixed",
        "mode": "hybrid",
        "evidence_count": 1,
        "queries": ["What was the ratio?"],
    }
    assert first.output["status"] == "supported"
    assert first.output["facts"]["entity"] == "JPMorgan Chase & Co."
    assert first.output["answer"] == ("JPMorgan Chase & Co. — ratio — 2025: 14.6 percent [E1]")
    assert len(completions.calls) == 2
    assert "Expected bank: JPMorgan Chase & Co." in completions.calls[0]["messages"][1]["content"]
    assert second.evidence[0]["target_chunk_id"] == "chunk-1"


def test_simple_jpm_operational_framework_question_recovers_via_concept_query() -> None:
    encoder = MockEncoder()
    retriever = OperationalFrameworkRetriever()
    completions = OperationalFrameworkCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
        bank_aliases={"JPM": ("JPMorgan", "JPMorgan Chase")},
    )

    question = "How does JPMorgan Chase describe its operational risk framework?"
    run = pipeline.answer(question)

    assert run.output["status"] == "supported"
    assert run.output["dialog_act"] == "answer"
    assert run.evidence[0]["target_chunk_id"] == "jpm-operational-framework"
    assert encoder.calls == [
        question,
        "JPMorgan Chase & Co. (JPM) Form 10-K: operational risk framework "
        "operational risk management",
    ]
    assert run.output["retrieval"]["queries"] == encoder.calls


def test_pipeline_resolves_question_bank_before_retrieval() -> None:
    encoder = MockEncoder()
    retriever = MockRetriever()
    completions = MockCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
        bank_aliases={"JPM": ("JPMorgan", "JP Morgan")},
    )

    run = pipeline.answer("What was JPMorgan's ratio?")

    assert run.output["ticker"] == "JPM"
    assert run.output["bank_resolution"]["source"] == "question"
    assert retriever.calls[0][2]["ticker"] == "JPM"
    assert len(completions.calls) == 1


def test_missing_or_too_many_banks_returns_ambiguous_without_pipeline_calls() -> None:
    encoder = MockEncoder()
    retriever = MockRetriever()
    completions = MockCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={
            "JPM": "JPMorgan Chase & Co.",
            "BAC": "Bank of America Corporation",
            "C": "Citigroup Inc.",
            "COF": "Capital One Financial Corporation",
            "GS": "The Goldman Sachs Group, Inc.",
        },
        bank_aliases={
            "JPM": ("JPMorgan",),
            "BAC": ("Bank of America",),
            "C": ("Citi",),
            "COF": ("Capital One",),
            "GS": ("Goldman Sachs",),
        },
    )

    missing = pipeline.answer("What was the CET1 ratio?")
    too_many = pipeline.answer(
        "Compare JPMorgan, Bank of America, Citi, Capital One and Goldman Sachs."
    )

    assert missing.output["status"] == "ambiguous"
    assert missing.output["reason_code"] == "bank_not_identified"
    assert missing.output["bank_resolution"]["status"] == "missing"
    assert too_many.output["status"] == "ambiguous"
    assert too_many.output["reason_code"] == "too_many_banks_identified"
    assert too_many.output["bank_resolution"]["detected_tickers"] == [
        "JPM",
        "BAC",
        "C",
        "COF",
        "GS",
    ]
    assert missing.evidence == too_many.evidence == []
    assert missing.embedding_latency_ms == too_many.embedding_latency_ms == 0
    assert encoder.calls == []
    assert retriever.calls == []
    assert completions.calls == []


def test_comparison_retrieves_independent_bank_subquestions_before_synthesis() -> None:
    encoder = MockEncoder()
    retriever = MockRetriever()
    completions = ComparisonCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )

    run = pipeline.answer("Compare JPMorgans CET1 ratio with Bank of Americas for 2025.")

    assert encoder.calls == [
        "JPMorgan Chase & Co. (JPM) Form 10-K: cet1 ratio for 2025",
        "JPMorgan Chase & Co. (JPM) Form 10-K: CET1 common equity tier 1 capital ratio",
        "Bank of America Corporation (BAC) Form 10-K: cet1 ratio for 2025",
        "Bank of America Corporation (BAC) Form 10-K: CET1 common equity tier 1 capital ratio",
    ]
    assert [call[2]["ticker"] for call in retriever.calls] == ["JPM", "JPM", "BAC", "BAC"]
    assert all("bank of america" not in call[0].casefold() for call in retriever.calls[:2])
    assert all("jpmorgan" not in call[0].casefold() for call in retriever.calls[2:])
    assert len(completions.calls) == 3
    jpm_generation_prompt = completions.calls[0]["messages"][1]["content"].casefold()
    bac_generation_prompt = completions.calls[1]["messages"][1]["content"].casefold()
    assert "bank of america" not in jpm_generation_prompt
    assert "jpmorgan" not in bac_generation_prompt
    assert run.output["mode"] == "comparison"
    assert run.output["tickers"] == ["JPM", "BAC"]
    assert run.output["status"] == "supported"
    assert [citation["label"] for citation in run.output["citations"]] == ["E1", "E2"]
    assert run.output["bank_results"][0]["citations"][0]["ticker"] == "JPM"
    assert run.output["bank_results"][1]["citations"][0]["ticker"] == "BAC"
    assert run.output["generation"]["request_count"] == 3
    assert [item["query"] for item in run.output["retrieval"]["per_bank"]] == [
        encoder.calls[0],
        encoder.calls[2],
    ]
    assert [item["queries"] for item in run.output["retrieval"]["per_bank"]] == [
        encoder.calls[:2],
        encoder.calls[2:],
    ]


def test_comparison_rechecks_one_abstention_when_focused_evidence_is_strong() -> None:
    completions = AbstainOnceCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=MockRetriever(),
        query_encoder=MockEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )

    run = pipeline.answer("Compare JPMorgan and Bank of America CET1 ratios in 2025.")

    assert run.output["status"] == "supported"
    assert run.output["bank_results"][0]["generation"]["bank_generation_retry"] is True
    assert run.output["bank_results"][0]["generation"]["request_count"] == 2
    assert run.output["retrieval"]["per_bank"][0]["generation_retry"] is True
    assert run.output["retrieval"]["per_bank"][1]["generation_retry"] is False
    assert run.output["generation"]["request_count"] == 4
    assert run.output["diagnostics"]["quality_gate"]["checks"]["request_budget"] is True


def test_comparison_rechecks_invalid_citations_when_focused_evidence_is_strong() -> None:
    completions = InvalidCitationOnceCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=MockRetriever(),
        query_encoder=MockEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )

    run = pipeline.answer("Compare JPMorgan and Bank of America CET1 ratios in 2025.")

    assert run.output["status"] == "supported"
    generation = run.output["bank_results"][0]["generation"]
    assert generation["bank_generation_retry"] is True
    assert generation["retry_reason"] == "invalid_citations_with_strong_evidence"
    assert generation["request_count"] == 2
    assert run.output["retrieval"]["per_bank"][0]["generation_retry"] is True


def test_comparison_runs_targeted_retrieval_only_for_evidence_miss() -> None:
    encoder = MockEncoder()
    retriever = FocusedRecoveryRetriever()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=ComparisonCompletions())),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )

    run = pipeline.answer("Compare JPMorgan and Bank of America CET1 ratios in 2025.")

    per_bank = {item["ticker"]: item for item in run.output["retrieval"]["per_bank"]}
    assert run.output["status"] == "supported"
    assert per_bank["JPM"]["recovery_queries"] == []
    assert len(per_bank["BAC"]["recovery_queries"]) == 1
    assert "relevant filing table or section" in per_bank["BAC"]["recovery_queries"][0]
    recovery_calls = [call for call in retriever.calls if "relevant filing" in call[0]]
    assert len(recovery_calls) == 1
    assert recovery_calls[0][2]["ticker"] == "BAC"
    assert any(stage.get("recovery") for stage in run.output["diagnostics"]["stages"])


def test_comparison_returns_partial_or_all_unsupported_without_unvalidated_facts() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=PartialCompletions()))
    kwargs = {
        "query_encoder": MockEncoder(),
        "client": client,
        "generation_model": "generation-model",
        "bank_names": {"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        "bank_aliases": {"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    }
    partial = SingleBankAnswerPipeline(retriever=PartialRetriever(), **kwargs).answer(
        "Compare JPMorgan and Bank of America ratios in 2025."
    )
    assert partial.output["status"] == "partial"
    assert [result["status"] for result in partial.output["bank_results"]] == [
        "supported",
        "unsupported",
    ]
    assert partial.output["answer"].startswith(
        "A complete comparison cannot be made because the supplied evidence is insufficient "
        "for Bank of America Corporation."
    )
    assert "Available supported results:" in partial.output["answer"]
    assert "JPMorgan Chase & Co. (JPM):" in partial.output["answer"]
    assert "JPMorgan Chase & Co. — ratio — 2025: 14.6 percent [E1]" in partial.output["answer"]
    assert partial.output["generation"]["request_count"] == 1
    assert partial.output["generation"]["bank_request_count"] == 1
    assert partial.output["generation"]["synthesis_request_count"] == 0
    assert len(client.chat.completions.calls) == 1

    empty_client = SimpleNamespace(chat=SimpleNamespace(completions=MockCompletions()))
    unsupported = SingleBankAnswerPipeline(
        retriever=EmptyRetriever(), **{**kwargs, "client": empty_client}
    ).answer("Compare JPMorgan and Bank of America ratios in 2025.")
    assert unsupported.output["status"] == "unsupported"
    assert unsupported.output["citations"] == []
    assert unsupported.output["generation"]["request_count"] == 0
    assert empty_client.chat.completions.calls == []


def test_comparison_isolates_invalid_schema_to_one_bank_and_continues() -> None:
    completions = PerBankValidationCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=MockRetriever(),
        query_encoder=MockEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )

    run = pipeline.answer("Compare JPMorgan and Bank of America ratios in 2025.")

    assert run.output["status"] == "partial"
    assert [result["status"] for result in run.output["bank_results"]] == [
        "unsupported",
        "supported",
    ]
    assert run.output["bank_results"][0]["generation"]["error_code"] == "invalid_schema"
    assert run.output["bank_results"][0]["generation"]["request_count"] == 2
    assert run.output["bank_results"][1]["ticker"] == "BAC"
    assert run.output["generation"]["bank_request_count"] == 3


def test_comparison_synthesis_fails_closed_on_invalid_schema() -> None:
    completions = ComparisonCompletions()
    original_create = completions.create

    def invalid_synthesis(**kwargs):
        if "concise comparison" in kwargs["messages"][0]["content"]:
            completions.calls.append(kwargs)
            function = SimpleNamespace(
                name="submit_comparison_synthesis",
                arguments='{"answer":"No citations","citation_ids":[]}',
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        return original_create(**kwargs)

    completions.create = invalid_synthesis
    pipeline = SingleBankAnswerPipeline(
        retriever=MockRetriever(),
        query_encoder=MockEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )
    with pytest.raises(GenerationValidationError) as error:
        pipeline.answer("Compare JPMorgan and Bank of America ratios in 2025.")
    assert error.value.code == "comparison_invalid_schema"


def test_pipeline_contextualizes_follow_up_before_retrieval() -> None:
    encoder = MockEncoder()
    retriever = ContextualizedRetriever()
    completions = ContextualizedCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="AZURE_GPT_51_2025_1113",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )
    history = [
        {"role": "user", "content": "Stale question about Citi governance."},
        {"role": "assistant", "content": "Stale Citi answer [E8]."},
        {"role": "user", "content": "What were JPM's deposits in 2025?"},
        {"role": "assistant", "content": "The deposits were reported [E2]."},
        {"role": "user", "content": "What was JPM's CET1 ratio in 2025?"},
        {"role": "assistant", "content": "It was 14.6% [E1]."},
    ]
    progress = []

    run = pipeline.answer(
        "What about 2024?",
        ticker="JPM",
        conversation_history=history,
        on_progress=lambda stage, _: progress.append(stage),
    )

    standalone = "What was JPMorgan Chase & Co. CET1 ratio in 2024?"
    assert encoder.calls == [
        standalone,
        "What about 2024?",
        "JPMorgan Chase & Co. (JPM) Form 10-K: CET1 common equity tier 1 capital ratio",
    ]
    assert retriever.calls[0][0] == standalone
    assert run.output["question"] == "What about 2024?"
    assert run.output["contextualization"]["applied"] is True
    assert run.output["contextualization"]["history_turns"] == 3
    assert run.output["contextualization"]["available_history_turns"] == 3
    assert run.output["contextualization"]["standalone_question"] == standalone
    assert progress[:2] == ["routing", "contextualizing"]
    assert "Stale question" in completions.calls[0]["messages"][1]["content"]
    assert "14.6" in completions.calls[0]["messages"][1]["content"]
    generation_prompt = completions.calls[-1]["messages"][1]["content"]
    assert "Current user question:\nWhat about 2024?" in generation_prompt
    assert f"Resolved standalone question:\n{standalone}" in generation_prompt


def test_pipeline_includes_history_even_for_a_new_standalone_question() -> None:
    encoder = MockEncoder()
    retriever = MockRetriever()
    completions = MockCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )
    history = [
        {"role": "user", "content": "How does JPMorgan define cybersecurity risk?"},
        {"role": "assistant", "content": "The filing describes it [E1]."},
    ]

    run = pipeline.answer(
        "What was Bank of America's ratio in 2025?",
        ticker="JPM",
        conversation_history=history,
    )

    assert run.output["ticker"] == "BAC"
    assert run.output["contextualization"]["applied"] is False
    assert run.output["contextualization"]["skip_reason"] == ("current_question_is_standalone")
    assert len(completions.calls) == 2
    assert completions.calls[0]["tool_choice"] == "required"
    routing_payload = json.loads(completions.calls[0]["messages"][1]["content"])
    assert routing_payload["conversation_history"] == history


def test_standalone_question_receives_all_bounded_raw_history() -> None:
    completions = MockCompletions()
    pipeline = SingleBankAnswerPipeline(
        retriever=MockRetriever(),
        query_encoder=MockEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )
    history = [
        {"role": role, "content": f"stale-secret-{index}-{role}"}
        for index in range(30)
        for role in ("user", "assistant")
    ]

    run = pipeline.answer(
        "What was Bank of America's ratio in 2025?",
        ticker="JPM",
        conversation_history=history,
    )

    routing_payload = json.loads(completions.calls[0]["messages"][1]["content"])
    assert routing_payload["conversation_history"] == history
    assert run.output["contextualization"]["available_history_turns"] == 30
    assert run.output["contextualization"]["history_turns"] == 30


def test_contextualization_falls_back_when_tool_introduces_bank_outside_thread_scope() -> None:
    completions = ContextualizedCompletions()
    original_create = completions.create

    def injected_bank(**kwargs):
        if not completions.calls:
            completions.calls.append(kwargs)
            function = SimpleNamespace(
                name="route_conversation",
                arguments=json.dumps(
                    {
                        "action": "filing_research",
                        "confidence": 0.98,
                        "search_question": "What was Citi CET1 in 2024?",
                        "reason": "Resolve the follow-up.",
                        "response_text": None,
                        "category": None,
                        "missing": None,
                        "citation_ids": [],
                        "presentation_guidance": None,
                    }
                ),
            )
            message = SimpleNamespace(
                content=None,
                refusal=None,
                tool_calls=[SimpleNamespace(function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )
        return original_create(**kwargs)

    completions.create = injected_bank
    pipeline = SingleBankAnswerPipeline(
        retriever=ContextualizedRetriever(),
        query_encoder=MockEncoder(),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        generation_model="AZURE_GPT_51_2025_1113",
        bank_names={"JPM": "JPMorgan Chase & Co.", "C": "Citigroup Inc."},
        bank_aliases={"JPM": ("JPMorgan",), "C": ("Citi",)},
    )

    run = pipeline.answer(
        "What about 2024?",
        ticker="JPM",
        conversation_history=[
            {"role": "user", "content": "What was JPM CET1 in 2025?"},
            {"role": "assistant", "content": "It was reported [E1]."},
        ],
    )

    assert run.output["ticker"] == "JPM"
    assert run.output["contextualization"]["standalone_question"] == "What about 2024?"
    assert run.output["contextualization"]["fallback"] is True
    assert run.output["contextualization"]["error_code"] == ("contextualization_added_bank_scope")


class SummaryRetriever(MockRetriever):
    def search_hybrid(self, question, query_vector, **kwargs):
        self.calls.append((question, query_vector, kwargs))
        index = len(self.calls)
        return [
            {
                "target_chunk_id": f"summary-{index}",
                "record_type": "text",
                "ticker": kwargs["ticker"],
                "evidence": "The filing states that the ratio was 14.6%.",
                "metadata": {"report_date": "2025-12-31"},
            }
        ]


def test_whole_filing_summary_uses_section_diverse_retrieval() -> None:
    encoder = MockEncoder()
    retriever = SummaryRetriever()
    pipeline = SingleBankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=SimpleNamespace(chat=SimpleNamespace(completions=MockCompletions())),
        generation_model="generation-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
        bank_aliases={"JPM": ("JPMorgan", "JP Morgan")},
    )

    run = pipeline.answer("Summarize the JP Morgans 2025 10-K doc")

    assert run.output["ticker"] == "JPM"
    assert len(encoder.calls) == len(retriever.calls) == 5
    assert len(run.evidence) == 5
    assert all(call[2]["ticker"] == "JPM" for call in retriever.calls)
