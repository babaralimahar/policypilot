"""Idempotent ingestion: unchanged documents add nothing; a changed section re-embeds only
its own chunk; a newer version supersedes the old one (which is then never retrieved)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from policypilot.adapters.db.models import ChunkRow, DocumentRow
from policypilot.adapters.embeddings import Embedder, EmbeddingError
from policypilot.domain.chunking import chunk_document, parse_markdown

log = structlog.get_logger()


@dataclass
class IngestResult:
    slug: str
    version: int
    status: str  # created | updated | unchanged
    chunks_added: int = 0
    chunks_changed: int = 0
    chunks_removed: int = 0


def embed_text(heading_path: str, body: str) -> str:
    return f"{heading_path}\n{body}"


async def ingest_markdown(session: AsyncSession, raw: str, corpus: str = "main") -> IngestResult:
    """Upsert one document and its chunks (does not embed; see ``embed_missing``). Commits."""
    doc = parse_markdown(raw)
    row = await session.scalar(
        select(DocumentRow).where(
            DocumentRow.corpus == corpus,
            DocumentRow.slug == doc.slug,
            DocumentRow.version == doc.version,
        )
    )
    if row is not None and row.content_hash == doc.content_hash:
        return IngestResult(doc.slug, doc.version, "unchanged")
    status = "updated" if row is not None else "created"
    if row is None:
        row = DocumentRow(
            corpus=corpus,
            slug=doc.slug,
            version=doc.version,
            content_hash=doc.content_hash,
            title=doc.title,
            source_url=doc.source_url,
            acl_groups=doc.acl,
            effective_date=doc.effective_date,
            superseded=doc.superseded,
        )
        session.add(row)
        await session.flush()
    else:
        row.content_hash, row.title, row.source_url = doc.content_hash, doc.title, doc.source_url
        row.acl_groups, row.effective_date, row.superseded = (
            doc.acl,
            doc.effective_date,
            doc.superseded,
        )

    result = IngestResult(doc.slug, doc.version, status)
    existing = {
        c.id: c
        for c in (
            await session.scalars(select(ChunkRow).where(ChunkRow.document_id == row.id))
        ).all()
    }
    keep: set[str] = set()
    for ch in chunk_document(doc):
        pk = f"{corpus}:{ch.chunk_id}"
        keep.add(pk)
        old = existing.get(pk)
        if old is None:
            session.add(
                ChunkRow(
                    id=pk,
                    chunk_id=ch.chunk_id,
                    document_id=row.id,
                    corpus=corpus,
                    slug=doc.slug,
                    section_id=ch.section_id,
                    ordinal=ch.ordinal,
                    heading_path=ch.heading_path,
                    text=ch.text,
                    token_count=ch.token_count,
                    content_hash=ch.content_hash,
                    acl_groups=doc.acl,
                    superseded=doc.superseded,
                    embedding=None,
                )
            )
            result.chunks_added += 1
        else:
            # ACL and superseded flags always follow the document, even if the text is unchanged.
            old.acl_groups, old.superseded = doc.acl, doc.superseded
            if old.content_hash != ch.content_hash:
                old.text, old.heading_path, old.section_id = ch.text, ch.heading_path, ch.section_id
                old.token_count, old.content_hash, old.embedding = (
                    ch.token_count,
                    ch.content_hash,
                    None,
                )
                result.chunks_changed += 1
    for pk, old in existing.items():
        if pk not in keep:
            await session.delete(old)
            result.chunks_removed += 1

    if not doc.superseded:  # a current version supersedes every older one of the same slug
        older = select(DocumentRow.id).where(
            DocumentRow.corpus == corpus,
            DocumentRow.slug == doc.slug,
            DocumentRow.version < doc.version,
        )
        await session.execute(
            update(DocumentRow).where(DocumentRow.id.in_(older)).values(superseded=True)
        )
        await session.execute(
            update(ChunkRow).where(ChunkRow.document_id.in_(older)).values(superseded=True)
        )
    await session.commit()
    return result


async def embed_missing(
    session: AsyncSession, embedder: Embedder, batch: int = 64
) -> tuple[int, str | None]:
    """Embed every chunk without an embedding. Returns (embedded, error). On provider failure
    the chunks stay unembedded and search runs lexical-only (degraded) until the next run."""
    done = 0
    while True:
        rows = (
            await session.execute(
                select(ChunkRow.id, ChunkRow.heading_path, ChunkRow.text)
                .where(ChunkRow.embedding.is_(None))
                .order_by(ChunkRow.id)
                .limit(batch)
            )
        ).all()
        if not rows:
            return done, None
        try:
            vecs = await embedder.embed([embed_text(r.heading_path, r.text) for r in rows])
        except EmbeddingError as e:
            log.warning("embedding_failed", error=str(e), pending=len(rows))
            return done, str(e)
        for r, v in zip(rows, vecs, strict=True):
            await session.execute(
                text("UPDATE chunks SET embedding = CAST(:v AS vector) WHERE id = :id"),
                {"v": "[" + ",".join(f"{x:.7f}" for x in v) + "]", "id": r.id},
            )
        await session.commit()
        done += len(rows)


async def ingest_dir(
    session: AsyncSession, directory: Path, corpus: str = "main"
) -> list[IngestResult]:
    results = []
    files = await asyncio.to_thread(
        lambda: [p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.md"))]
    )
    for raw in files:
        results.append(await ingest_markdown(session, raw, corpus))
    return results
