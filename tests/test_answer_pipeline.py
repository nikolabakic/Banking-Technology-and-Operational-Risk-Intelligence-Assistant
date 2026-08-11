from types import SimpleNamespace

import numpy as np

from bankscope.generation.pipeline import SingleBankAnswerPipeline


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
