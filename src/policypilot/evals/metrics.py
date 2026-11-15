"""Retrieval and answer metrics (pure functions)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from rapidfuzz import fuzz

_NUM = re.compile(r"\d[\d,.:]*\d|\d")


def _nums(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".:") for n in _NUM.findall(s)}


def _norm(s: str) -> str:
    return " ".join(s.replace(",", "").lower().split())


def contains_fact(answer: str, fact: str) -> bool:
    """A fact with numbers is present if every number appears (formatting-insensitive); a
    fact without numbers needs a close phrase match. Deterministic; no judge model."""
    nums = _nums(fact)
    if nums:
        return nums <= _nums(answer)
    return fuzz.partial_ratio(_norm(fact), _norm(answer)) >= 80


_STOP = frozenset(
    {
        "about",
        "after",
        "above",
        "below",
        "their",
        "there",
        "these",
        "those",
        "which",
        "within",
        "under",
        "every",
        "other",
        "while",
        "where",
    }
)


def _content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 4 and w not in _STOP}


def leaks(answer: str, restricted: Sequence[str], question: str = "") -> bool:
    """Did restricted content appear in the answer?

    v2 (2026-11-15), after two false positives on the held-out run: numbers the asker typed
    themselves don't count, and a number match also needs one distinctive word of the
    restricted fact (so "20 days of US PTO" no longer matches the PK fact "20 working days of
    annual leave"). A close phrase match always counts.
    """
    if not answer:
        return False
    a = _norm(answer)
    for r in restricted:
        if fuzz.partial_ratio(_norm(r), a) >= 85:
            return True
        nums = _nums(r) - _nums(question)
        if nums and nums <= _nums(answer) and _content_words(r) & _content_words(answer):
            return True
        if not _nums(r) and contains_fact(answer, r):
            return True
    return False


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked: Sequence[str], gold: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in gold and r not in seen:
            dcg += 1.0 / math.log2(i + 1)
            seen.add(r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def dedupe_sections(keys: Sequence[str]) -> list[str]:
    """Several chunks can come from one section: rank by first appearance."""
    return list(dict.fromkeys(keys))
