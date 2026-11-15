"""Markdown parsing and heading-aware chunking (pure functions)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

import yaml

MAX_TOKENS = 500  # target budget per chunk (about 1.3 x words)
OVERLAP_SENTENCES = 1


@dataclass(frozen=True)
class ParsedDoc:
    slug: str
    title: str
    version: int
    effective_date: date
    acl: list[str]
    superseded: bool
    source_url: str
    sections: list[tuple[str, str]]  # (heading, body)
    content_hash: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ordinal: int
    section_id: str
    heading_path: str
    text: str
    token_count: int
    content_hash: str
    meta: dict[str, str] = field(default_factory=dict)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def est_tokens(text: str) -> int:
    return round(len(text.split()) * 1.3)


def parse_markdown(raw: str) -> ParsedDoc:
    """YAML front matter + `# Title` + `## Section` bodies."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.S)
    if not m:
        raise ValueError("missing front matter")
    meta = yaml.safe_load(m.group(1))
    body = m.group(2)
    for key in ("slug", "title", "acl"):
        if key not in meta:
            raise ValueError(f"front matter needs {key!r}")
    acl = [str(g) for g in meta["acl"]]
    if not acl:
        raise ValueError("acl may not be empty (an unlabelled document is never public by default)")
    sections: list[tuple[str, str]] = []
    for part in re.split(r"^## ", body, flags=re.M)[1:]:
        heading, _, text = part.partition("\n")
        sections.append((heading.strip(), text.strip()))
    eff = meta.get("effective_date", "2026-01-01")
    return ParsedDoc(
        slug=str(meta["slug"]),
        title=str(meta["title"]),
        version=int(meta.get("version", 1)),
        effective_date=eff if isinstance(eff, date) else date.fromisoformat(str(eff)),
        acl=acl,
        superseded=bool(meta.get("superseded", False)),
        source_url=str(meta.get("source_url", "")),
        sections=sections,
        content_hash=sha(raw),
    )


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def chunk_document(doc: ParsedDoc, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """One chunk per section; a long section is split on sentence boundaries with a
    one-sentence overlap. Each chunk carries its heading path, so it stands alone."""
    chunks: list[Chunk] = []
    n = 0
    for heading, body in doc.sections:
        section_id = slugify(heading)
        path = f"{doc.title} > {heading}"
        sents = _sentences(body)
        current: list[str] = []
        pieces: list[str] = []
        for s in sents:
            if current and est_tokens(" ".join([*current, s])) > max_tokens:
                pieces.append(" ".join(current))
                current = current[-OVERLAP_SENTENCES:]
            current.append(s)
        if current:
            pieces.append(" ".join(current))
        for piece in pieces:
            text = piece.strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.slug}@v{doc.version}#{n}",
                    ordinal=n,
                    section_id=section_id,
                    heading_path=path,
                    text=text,
                    token_count=est_tokens(text),
                    content_hash=sha(f"{path}\n{text}"),
                )
            )
            n += 1
    return chunks
