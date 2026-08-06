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


def evaluate_evidence_groups(
    retrieved_target_chunk_ids: Sequence[str],
    required_evidence_groups: Sequence[Sequence[str]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, MetricValue]:
    """Measure whether retrieval covers every independently required evidence group."""
    if not required_evidence_groups:
        return {}

    groups = [set(group) for group in required_evidence_groups]
    if any(not group for group in groups):
        raise ValueError("Required evidence groups cannot be empty.")

    retrieved_ids = deduplicate_ids(retrieved_target_chunk_ids)
    metrics: dict[str, MetricValue] = {"required_evidence_group_count": len(groups)}
    for k in k_values:
        top_ids = set(retrieved_ids[:k])
        covered = sum(bool(group & top_ids) for group in groups)
        metrics[f"group_recall_at_{k}"] = covered / len(groups)
        metrics[f"complete_group_hit_at_{k}"] = int(covered == len(groups))
    return metrics
