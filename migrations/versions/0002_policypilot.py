"""policypilot: documents, chunks (FTS + pgvector + ACL), qa logs, feedback

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = 768  # gemini-embedding-001 with dimensions=768 (HNSW supports at most 2,000)


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("corpus", sa.String(32), nullable=False, server_default="main"),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("source_url", sa.Text, nullable=False, server_default=""),
        sa.Column("acl_groups", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("effective_date", sa.Date, nullable=False),
        sa.Column("superseded", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("corpus", "slug", "version", name="uq_documents_corpus_slug_version"),
        sa.CheckConstraint("cardinality(acl_groups) > 0", name="ck_documents_acl_nonempty"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(200), primary_key=True),
        sa.Column("chunk_id", sa.String(160), nullable=False),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("corpus", sa.String(32), nullable=False),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("section_id", sa.String(120), nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("heading_path", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("acl_groups", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("superseded", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', heading_path || ' ' || text)", persisted=True),
        ),
        sa.Column("embedding", Vector(DIM)),
        sa.CheckConstraint("cardinality(acl_groups) > 0", name="ck_chunks_acl_nonempty"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("chunks_tsv_gin", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index("chunks_acl_gin", "chunks", ["acl_groups"], postgresql_using="gin")
    op.execute(
        "CREATE INDEX chunks_emb_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.create_table(
        "qa_logs",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("trace_id", sa.String(40), nullable=False, unique=True),
        sa.Column("user_hash", sa.String(64), nullable=False),
        sa.Column("groups_hash", sa.String(64), nullable=False),
        sa.Column("question_scrubbed", sa.Text, nullable=False),
        sa.Column("retrieved_ids", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("cited_ids", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("abstained", sa.Boolean, nullable=False),
        sa.Column("abstain_reason", sa.String(40)),
        sa.Column("degraded", sa.Boolean, nullable=False),
        sa.Column("blocked", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_qa_logs_abstained_created", "qa_logs", ["abstained", "created_at"])
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "trace_id",
            sa.String(40),
            sa.ForeignKey("qa_logs.trace_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.SmallInteger, nullable=False),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
    )


def downgrade() -> None:
    for t in ("feedback", "qa_logs", "chunks", "documents"):
        op.drop_table(t)
