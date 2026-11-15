"""PolicyPilot tables (shared metadata with the template)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.adapters.db.models import Base

EMBED_DIM = 768


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("corpus", "slug", "version", name="uq_documents_corpus_slug_version"),
    )
    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    corpus: Mapped[str] = mapped_column(String(32), server_default="main")
    slug: Mapped[str] = mapped_column(String(120))
    version: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(300))
    source_url: Mapped[str] = mapped_column(Text, server_default="")
    acl_groups: Mapped[list[str]] = mapped_column(ARRAY(Text))
    effective_date: Mapped[date] = mapped_column(Date)
    superseded: Mapped[bool] = mapped_column(Boolean, server_default="false")
    content_hash: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkRow(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)  # corpus:slug@vN#ordinal
    chunk_id: Mapped[str] = mapped_column(String(160))  # slug@vN#ordinal (what users see)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    corpus: Mapped[str] = mapped_column(String(32))
    slug: Mapped[str] = mapped_column(String(120))
    section_id: Mapped[str] = mapped_column(String(120))
    ordinal: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    acl_groups: Mapped[list[str]] = mapped_column(ARRAY(Text))  # denormalised for filtering
    superseded: Mapped[bool] = mapped_column(Boolean, server_default="false")
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', heading_path || ' ' || text)", persisted=True)
    )
    embedding: Mapped[Any] = mapped_column(Vector(EMBED_DIM), nullable=True)


class QaLogRow(Base):
    __tablename__ = "qa_logs"
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(40), unique=True)
    user_hash: Mapped[str] = mapped_column(String(64))
    groups_hash: Mapped[str] = mapped_column(String(64))
    question_scrubbed: Mapped[str] = mapped_column(Text)
    retrieved_ids: Mapped[list[str]] = mapped_column(ARRAY(Text))
    cited_ids: Mapped[list[str]] = mapped_column(ARRAY(Text))
    answer: Mapped[str] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column(Boolean)
    abstain_reason: Mapped[str | None] = mapped_column(String(40))
    degraded: Mapped[bool] = mapped_column(Boolean)
    blocked: Mapped[bool] = mapped_column(Boolean, server_default="false")
    latency_ms: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("qa_logs.trace_id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(SmallInteger)
    reason: Mapped[str] = mapped_column(Text, server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
