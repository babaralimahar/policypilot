"""PolicyPilot HTTP API."""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text

from app.api.deps import SessionDep
from app.api.errors import DomainError
from policypilot.adapters.db.models import FeedbackRow, QaLogRow
from policypilot.auth import AdminDep, UserDep, issue_token, policy_settings
from policypilot.services.answer import Answerer
from policypilot.services.ingest import embed_missing, ingest_markdown
from policypilot.services.retrieval import retrieve

router = APIRouter(prefix="/v1", tags=["policy"])


def answerer(request: Request) -> Answerer:
    return cast(Answerer, request.app.state.answerer)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(8, ge=1, le=20)
    mode: Literal["lexical", "vector", "hybrid", "hybrid+rerank"] | None = None
    debug: bool = False


@router.post("/ask")
async def ask(body: AskRequest, request: Request, user: UserDep) -> dict[str, Any]:
    if body.debug and not user.is_admin:
        raise DomainError(403, "forbidden", "debug is admin-only")
    res = await answerer(request).ask(body.question, user, mode=body.mode, top_k=body.top_k)
    return res.public(debug=body.debug)


@router.get("/search")
async def search(
    request: Request,
    session: SessionDep,
    user: UserDep,
    q: Annotated[str, Query(min_length=2, max_length=500)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    mode: Literal["lexical", "vector", "hybrid"] = "hybrid",
) -> dict[str, Any]:
    a = answerer(request)
    r = await retrieve(
        session,
        a.embedder,
        q,
        user.groups,
        mode=mode,
        corpora=a.s.corpora,
        candidate_k=max(k, a.s.candidate_k),
        top_k=k,
    )
    return {
        "mode": r.mode,
        "degraded": r.degraded,
        "results": [
            {
                "chunk_id": h.chunk_id,
                "doc_title": h.title,
                "section": h.heading_path,
                "score": round(h.score, 4),
                "snippet": h.text[:200],
            }
            for h in r.hits
        ],
    }


class DocumentIn(BaseModel):
    markdown: str = Field(min_length=10, max_length=500_000)


@router.post("/documents")
async def upsert_document(body: DocumentIn, session: SessionDep, _: AdminDep) -> dict[str, Any]:
    try:
        res = await ingest_markdown(session, body.markdown)
    except ValueError as e:
        raise DomainError(422, "bad-document", "Invalid document", str(e)) from e
    return res.__dict__


@router.post("/reindex")
async def reindex(request: Request, session: SessionDep, _: AdminDep) -> dict[str, Any]:
    n, err = await embed_missing(session, answerer(request).embedder)
    return {"embedded": n, "error": err}


class FeedbackIn(BaseModel):
    trace_id: str = Field(min_length=8, max_length=40)
    rating: Literal[-1, 1]
    reason: str = Field("", max_length=1000)


@router.post("/feedback", status_code=201)
async def feedback(body: FeedbackIn, session: SessionDep, _: UserDep) -> dict[str, str]:
    if await session.scalar(select(QaLogRow.id).where(QaLogRow.trace_id == body.trace_id)) is None:
        raise DomainError(404, "not-found", "Unknown trace_id")
    session.add(FeedbackRow(trace_id=body.trace_id, rating=body.rating, reason=body.reason))
    await session.commit()
    return {"status": "recorded"}


@router.get("/reports/unanswered")
async def unanswered(session: SessionDep, _: AdminDep, limit: int = 20) -> dict[str, Any]:
    """Weekly list for doc owners: questions the assistant could not answer (PII-scrubbed)."""
    rows = (
        await session.execute(
            select(
                QaLogRow.question_scrubbed,
                func.count().label("n"),
                func.max(QaLogRow.created_at).label("last"),
            )
            .where(QaLogRow.abstained, QaLogRow.created_at > func.now() - text("interval '7 days'"))
            .group_by(QaLogRow.question_scrubbed)
            .order_by(func.count().desc())
            .limit(limit)
        )
    ).all()
    return {
        "items": [
            {"question": r.question_scrubbed, "count": r.n, "last_asked": r.last} for r in rows
        ]
    }


class TokenIn(BaseModel):
    persona: str


@router.post("/dev/token")
async def dev_token(body: TokenIn, request: Request) -> dict[str, Any]:
    """Development only: a token for one of the demo personas. Disabled in production."""
    s = policy_settings(request)
    if not s.dev_tokens or s.env == "prod":
        raise DomainError(404, "not-found", "Not found")
    personas: dict[str, list[str]] = request.app.state.personas
    if body.persona not in personas:
        raise DomainError(422, "bad-persona", "Unknown persona", f"one of {sorted(personas)}")
    roles = ["admin"] if "hr" in personas[body.persona] else []
    return {
        "token": issue_token(s, f"demo-{body.persona}", personas[body.persona], roles),
        "groups": personas[body.persona],
        "roles": roles,
    }
