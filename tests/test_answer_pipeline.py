import sys
from types import SimpleNamespace

import numpy as np

from bankscope.generation.pipeline import (
    SentenceTransformerQueryEncoder,
    SingleBankAnswerPipeline,
)


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
                "evidence": "The filing states that the ratio was 14.6%.",
                "metadata": {"report_date": "2025-12-31"},
            }
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
        message = SimpleNamespace(
            content=(
                '{"status":"supported","answer_type":"numeric",'
                '"answer":"The ratio was 14.6% [E1].",'
                '"facts":{"entity":"JPMorgan Chase & Co.","metric":"ratio",'
                '"variant":null,"period":"2025","value_text":"14.6%",'
                '"unit":"percent"},"citation_ids":["E1"],'
                '"reason":"Direct support."}'
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class ContextualizedCompletions(MockCompletions):
    def create(self, **kwargs):
        if not self.calls:
            self.calls.append(kwargs)
            message = SimpleNamespace(
                content=(
                    '{"standalone_question":"What was JPMorgan Chase & Co. CET1 ratio in 2024?"}'
                ),
                refusal=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
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
    }
    assert first.output["status"] == "supported"
    assert first.output["facts"]["entity"] == "JPMorgan Chase & Co."
    assert first.output["answer"] == ("JPMorgan Chase & Co. — ratio — 2025: 14.6 percent [E1]")
    assert len(completions.calls) == 2
    assert "Expected bank: JPMorgan Chase & Co." in completions.calls[0]["messages"][1]["content"]
    assert second.evidence[0]["target_chunk_id"] == "chunk-1"


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


def test_missing_or_multiple_bank_returns_ambiguous_without_pipeline_calls() -> None:
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
        },
        bank_aliases={"JPM": ("JPMorgan",), "BAC": ("Bank of America",)},
    )

    missing = pipeline.answer("What was the CET1 ratio?")
    multiple = pipeline.answer("Compare JPMorgan and Bank of America.", ticker="JPM")

    assert missing.output["status"] == "ambiguous"
    assert missing.output["reason_code"] == "bank_not_identified"
    assert missing.output["bank_resolution"]["status"] == "missing"
    assert multiple.output["status"] == "ambiguous"
    assert multiple.output["reason_code"] == "multiple_banks_identified"
    assert multiple.output["bank_resolution"]["detected_tickers"] == ["BAC", "JPM"]
    assert missing.evidence == multiple.evidence == []
    assert missing.embedding_latency_ms == multiple.embedding_latency_ms == 0
    assert encoder.calls == []
    assert retriever.calls == []
    assert completions.calls == []


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
    assert encoder.calls == [standalone]
    assert retriever.calls[0][0] == standalone
    assert run.output["question"] == "What about 2024?"
    assert run.output["contextualization"]["applied"] is True
    assert run.output["contextualization"]["history_turns"] == 1
    assert run.output["contextualization"]["standalone_question"] == standalone
    assert progress[0] == "contextualizing"
    assert "[E1]" not in completions.calls[0]["messages"][1]["content"]
    generation_prompt = completions.calls[-1]["messages"][1]["content"]
    assert "Current user question:\nWhat about 2024?" in generation_prompt
    assert f"Resolved standalone question:\n{standalone}" in generation_prompt
