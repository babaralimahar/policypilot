"""Citation validation: a citation counts only if it points at a chunk we actually provided."""

from __future__ import annotations

import re
from dataclasses import dataclass

INLINE = re.compile(r"\[c:([^\]\s]+)\]")


@dataclass(frozen=True)
class Checked:
    answer: str
    citations: list[str]
    abstained: bool
    dropped: list[str]


def validate(answer: str, cited: list[str], provided: set[str], abstained: bool) -> Checked:
    """Keep provided citations (deduplicated, in order), drop the rest. An answer with no
    valid evidence is not an answer: it becomes an abstention. Inline [c:id] markers that
    point outside the context are removed from the text."""
    seen: list[str] = []
    for cid in [*cited, *INLINE.findall(answer)]:
        if cid in provided and cid not in seen:
            seen.append(cid)
    dropped = sorted({c for c in [*cited, *INLINE.findall(answer)] if c not in provided})
    if abstained or not seen:
        return Checked("", [], True, dropped)
    text = INLINE.sub(lambda m: m.group(0) if m.group(1) in provided else "", answer).strip()
    return Checked(text, seen, False, dropped)
