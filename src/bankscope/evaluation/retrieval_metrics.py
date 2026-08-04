from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

MetricValue: TypeAlias = float | int | None  # noqa: UP040

DEFAULT_K_VALUES = (1, 3, 5, 10)


def deduplicate_ids(target_chunk_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []

    for target_chunk_id in target_chunk_ids:
        if target_chunk_id in seen:
            continue

        seen.add(target_chunk_id)
        unique_ids.append(target_chunk_id)

    return unique_ids


def evaluate_ranking(
    retrieved_target_chunk_ids: Sequence[str],
    relevant_target_chunk_ids: Sequence[str],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    reciprocal_rank_limit: int = 10,
) -> dict[str, MetricValue]:
    relevant_ids = set(relevant_target_chunk_ids)

    if not relevant_ids:
        raise ValueError("At least one relevant target_chunk_id is required.")

    retrieved_ids = deduplicate_ids(retrieved_target_chunk_ids)

    relevant_ranks = [
        rank
        for rank, target_chunk_id in enumerate(
            retrieved_ids,
            start=1,
        )
        if target_chunk_id in relevant_ids
    ]

    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None

    metrics: dict[str, MetricValue] = {
        "relevant_count": len(relevant_ids),
        "first_relevant_rank": first_relevant_rank,
    }

    for k in k_values:
        relevant_retrieved = sum(rank <= k for rank in relevant_ranks)

        metrics[f"hit_at_{k}"] = int(relevant_retrieved > 0)
        metrics[f"recall_at_{k}"] = relevant_retrieved / len(relevant_ids)

    if first_relevant_rank is not None and first_relevant_rank <= reciprocal_rank_limit:
        reciprocal_rank = 1.0 / first_relevant_rank
    else:
        reciprocal_rank = 0.0

    metrics[f"reciprocal_rank_at_{reciprocal_rank_limit}"] = reciprocal_rank

    return metrics
