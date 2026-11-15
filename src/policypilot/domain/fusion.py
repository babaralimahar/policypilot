"""Reciprocal Rank Fusion: combine rankings without calibrating their scores."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf(
    rankings: Sequence[Sequence[str]], k: int = RRF_K, limit: int | None = None
) -> list[tuple[str, float]]:
    """score(d) = Σ 1 / (k + rank_i(d)), rank starting at 1. Ties break by id for stability."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:limit] if limit else fused
