from typing import Any

from bankscope.retrieval.reranker import rerank_candidates


class FakeReranker:
    def rank(
        self,
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> list[dict[str, int | float]]:
        assert query == "What is operational risk?"
        assert documents == [
            "Unrelated candidate",
            "Operational risk definition",
        ]
        assert kwargs["top_k"] == 2
        assert kwargs["batch_size"] == 4

        return [
            {
                "corpus_id": 1,
                "score": 9.5,
            },
            {
                "corpus_id": 0,
                "score": 2.0,
            },
        ]


def test_rerank_candidates_uses_model_order() -> None:
    candidates = [
        {
            "target_chunk_id": "first",
            "document": "Unrelated candidate",
            "retrieval_method": "hybrid",
            "rrf_score": 0.03,
        },
        {
            "target_chunk_id": "second",
            "document": "Operational risk definition",
            "retrieval_method": "hybrid",
            "rrf_score": 0.02,
        },
    ]

    results = rerank_candidates(
        FakeReranker(),  # type: ignore[arg-type]
        "What is operational risk?",
        candidates,
        limit=2,
        batch_size=4,
    )

    assert [result["target_chunk_id"] for result in results] == [
        "second",
        "first",
    ]

    assert results[0]["rank"] == 1
    assert results[0]["reranker_score"] == 9.5
    assert results[0]["rrf_score"] == 0.02
    assert candidates[0]["retrieval_method"] == "hybrid"
