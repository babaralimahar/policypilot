"""Demo page with a persona switcher, to show permission differences live."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from app.api.deps import SessionDep
from app.api.errors import DomainError
from policypilot.adapters.db.models import FeedbackRow, QaLogRow
from policypilot.api.routes import answerer
from policypilot.auth import issue_token, policy_settings, verify_token
from policypilot.services.answer import User

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
router = APIRouter(prefix="/ui", include_in_schema=False)
COOKIE = "pp_token"


def _user(request: Request) -> tuple[User | None, str | None]:
    tok = request.cookies.get(COOKIE)
    if not tok:
        return None, None
    try:
        u = verify_token(policy_settings(request), tok)
    except DomainError:
        return None, None
    return u, u.sub.removeprefix("demo-")


@router.get("")
async def index(request: Request) -> HTMLResponse:
    me, persona = _user(request)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"me": me, "persona": persona, "personas": request.app.state.personas},
    )


@router.post("/persona")
async def persona(request: Request, persona: Annotated[str, Form()]) -> Response:
    s = policy_settings(request)
    personas = request.app.state.personas
    if not s.dev_tokens or s.env == "prod" or persona not in personas:
        return RedirectResponse("/ui", status_code=303)
    roles = ["admin"] if "hr" in personas[persona] else []
    resp = RedirectResponse("/ui", status_code=303)
    resp.set_cookie(
        COOKIE,
        issue_token(s, f"demo-{persona}", personas[persona], roles),
        httponly=True,
        samesite="strict",
        secure=False,  # dev tokens are refused in prod above
        max_age=s.jwt_ttl_s,
    )
    return resp


@router.post("/ask")
async def ask(
    request: Request,
    question: Annotated[str, Form(min_length=3, max_length=1000)],
    mode: Annotated[str, Form()] = "vector",
) -> Response:
    me, _ = _user(request)
    if me is None:
        return HTMLResponse('<p class="note err">Pick a persona first.</p>', status_code=401)
    m = mode if mode in ("lexical", "vector", "hybrid", "hybrid+rerank") else "vector"
    r = await answerer(request).ask(question, me, mode=m)  # type: ignore[arg-type]
    return templates.TemplateResponse(request, "_answer.html", {"r": r})


@router.post("/feedback")
async def feedback(
    request: Request,
    session: SessionDep,
    trace_id: Annotated[str, Form()],
    rating: Annotated[int, Form()],
) -> HTMLResponse:
    me, _ = _user(request)
    if me is None or rating not in (-1, 1):
        return HTMLResponse("", status_code=400)
    if await session.scalar(select(QaLogRow.id).where(QaLogRow.trace_id == trace_id)) is not None:
        session.add(FeedbackRow(trace_id=trace_id, rating=rating, reason="ui"))
        await session.commit()
    return HTMLResponse('<p class="muted">Thanks for the feedback.</p>')
