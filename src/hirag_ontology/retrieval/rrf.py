"""Reciprocal Rank Fusion utilities."""

from __future__ import annotations

from collections import defaultdict


def rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked entity ID lists with Reciprocal Rank Fusion."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)

    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, entity_id in enumerate(ranked_list, start=1):
            if entity_id in seen_in_list:
                continue
            seen_in_list.add(entity_id)
            scores[entity_id] += 1.0 / (k + rank)
            best_rank[entity_id] = min(best_rank.get(entity_id, rank), rank)
            if entity_id not in first_seen:
                first_seen[entity_id] = seen_counter
                seen_counter += 1

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank[item[0]], first_seen[item[0]], item[0]),
    )
