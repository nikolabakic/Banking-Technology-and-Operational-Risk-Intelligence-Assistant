from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

import torch  # noqa: F401
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_PROMPT_NAME = "bankscope"
RERANKER_INSTRUCTION = (
    "Retrieve passages and tables from U.S. bank 10-K filings that provide "
    "direct evidence for the question. Prioritize the exact bank, reporting "
    "date, metric, entity, units, and whether a value is actual or a "
    "regulatory requirement."
)


def load_reranker(
    model_name: str = RERANKER_MODEL_NAME,
    *,
    device: str | None = None,
    max_length: int = 1024,
) -> CrossEncoder:
    import torch  # noqa: F811
    from sentence_transformers import CrossEncoder

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs: dict[str, Any] = {}

    if selected_device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16

    return CrossEncoder(
        model_name,
        device=selected_device,
        max_length=max_length,
        prompts={
            RERANKER_PROMPT_NAME: RERANKER_INSTRUCTION,
        },
        default_prompt_name=RERANKER_PROMPT_NAME,
        model_kwargs=model_kwargs,
    )


def rerank_candidates(
    reranker: CrossEncoder,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    limit: int = 10,
    batch_size: int = 4,
    show_progress_bar: bool = False,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not candidates:
        return []

    documents = [str(candidate["document"]) for candidate in candidates]

    ranking = reranker.rank(
        query,
        documents,
        top_k=min(limit, len(documents)),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    reranked_results: list[dict[str, Any]] = []

    for rank, ranking_item in enumerate(
        ranking,
        start=1,
    ):
        candidate_index = int(ranking_item["corpus_id"])

        if not 0 <= candidate_index < len(candidates):
            raise ValueError(f"Reranker returned invalid corpus_id: {candidate_index}.")

        result = dict(candidates[candidate_index])
        result["retrieval_method"] = "reranked"
        result["rank"] = rank
        result["reranker_score"] = float(ranking_item["score"])

        reranked_results.append(result)

    return reranked_results
