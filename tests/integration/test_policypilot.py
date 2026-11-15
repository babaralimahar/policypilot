"""PolicyPilot on real Postgres with the real corpus, fake embeddings and a scripted model."""

from __future__ import annotations

import json
import random
import re
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.adapters.llm import FakeLLMClient, LLMResult, Message, fake_result
from policypilot.adapters.db.models import ChunkRow
from policypilot.adapters.embeddings import FakeEmbedder
from policypilot.data.spec import load_spec
from policypilot.evals.dataset import load_items
from policypilot.main import create_app
from policypilot.services.answer import Answerer, User
from policypilot.services.ingest import embed_missing, ingest_dir, ingest_markdown
from policypilot.services.retrieval import retrieve
from tests.conftest import make_policy_settings

SM = async_sessionmaker[AsyncSession]
PERSONAS = load_spec().personas
EMB = FakeEmbedder()


@pytest.fixture
async def corpus(sm: SM) -> SM:
    async with sm() as s:
        await ingest_dir(s, Path("data/corpus"), "main")
        await ingest_dir(s, Path("data/poisoned"), "poisoned")
        await embed_missing(s, EMB)
    return sm


def citing_responder(answer: str = "See the policy.", extra_cite: str | None = None):  # type: ignore[no-untyped-def]
    """A scripted model: answers citing the first context chunk (plus an optional bogus id)."""

    def respond(messages: list[Message]) -> LLMResult:
        ids = re.findall(r'<chunk id="([^"]+)"', str(messages[-1]["content"]))
        cites = ids[:1] + ([extra_cite] if extra_cite else [])
        payload = {
            "answer": answer,
            "citations": cites,
            "abstained": not ids,
            "abstain_reason": None,
        }
        return fake_result(parsed=payload, text=json.dumps(payload), cost_usd=Decimal("0.0001"))

    return respond


def answerer(sm: SM, responder: Any = None, **settings: Any) -> Answerer:
    s = make_policy_settings(abstain_min_similarity=0.0, **settings)
    return Answerer(sm, FakeLLMClient(responder=responder or citing_responder()), EMB, s)


# ------------------------------------------------------------------ ingestion


async def test_ingest_is_idempotent_and_reembeds_only_changed_chunks(corpus: SM) -> None:
    async with corpus() as s:
        again = await ingest_dir(s, Path("data/corpus"), "main")
    assert {r.status for r in again} == {"unchanged"}
    raw = Path("data/corpus/travel.v1.md").read_text(encoding="utf-8")
    edited = raw.replace("USD 200 per night", "USD 210 per night")
    async with corpus() as s:
        r = await ingest_markdown(s, edited)
    assert (r.status, r.chunks_added, r.chunks_changed, r.chunks_removed) == ("updated", 0, 1, 0)
    async with corpus() as s:
        pending = (
            await s.scalars(select(ChunkRow.chunk_id).where(ChunkRow.embedding.is_(None)))
        ).all()
    assert pending == ["travel@v1#1"]  # only the hotels chunk needs a new embedding


async def test_superseded_version_is_never_retrieved(corpus: SM) -> None:
    async with corpus() as s:
        r = await retrieve(
            s,
            EMB,
            "daily meal allowance when travelling",
            PERSONAS["emp_us"],
            mode="hybrid",
            top_k=30,
        )
    slugs = {(h.slug, h.version) for h in r.hits}
    assert ("expenses", 2) in slugs and ("expenses", 1) not in slugs
    assert not any("USD 40 per day" in h.text for h in r.hits)


# ------------------------------------------------------------------ permissions (hard gate)


async def test_acl_property_no_retriever_returns_forbidden_chunks(corpus: SM) -> None:
    """50 random queries x 6 personas x 3 retrievers: every hit is inside the asker's groups."""
    dev, _ = load_items("dev")
    rng = random.Random(3)
    words = " ".join(i.question for i in dev).split()
    queries = rng.sample([i.question for i in dev], 30) + [
        " ".join(rng.sample(words, 6)) for _ in range(20)
    ]
    checked = 0
    for q in queries:
        for persona, groups in PERSONAS.items():
            for mode in ("lexical", "vector", "hybrid"):
                async with corpus() as s:
                    r = await retrieve(
                        s, EMB, q, groups, mode=mode, corpora=["main", "poisoned"], top_k=30
                    )  # type: ignore[arg-type]
                for h in r.hits:
                    assert set(h.acl_groups) & set(groups), (persona, mode, q, h.chunk_id)
                checked += len(r.hits)
    assert checked > 1000


async def test_restricted_content_never_reaches_the_model(corpus: SM) -> None:
    llm_seen: list[str] = []

    def spy(messages: list[Message]) -> LLMResult:
        llm_seen.append(str(messages[-1]["content"]))
        return citing_responder()(messages)

    a = answerer(corpus, spy)
    # An employee asks about manager pay: no manager-only text may reach the model.
    await a.ask("What is the salary band for M1 managers?", User("e", PERSONAS["emp_us"]))
    emp_context = "\n".join(llm_seen)
    assert "110,000" not in emp_context and "manager-compensation" not in emp_context
    # A manager asks about finance controls: managers may see manager docs, but not finance-only.
    llm_seen.clear()
    await a.ask("When does finance close the monthly books?", User("m", PERSONAS["mgr_us"]))
    mgr_context = "\n".join(llm_seen)
    assert "5th working day" not in mgr_context and "finance-controls" not in mgr_context


# ------------------------------------------------------------------ answers


async def test_answer_has_valid_citations_and_is_logged(corpus: SM) -> None:
    a = answerer(
        corpus, citing_responder("Hotels are capped at USD 200 per night.", extra_cite="made-up#9")
    )
    r = await a.ask("What is the hotel cap in US cities?", User("u", PERSONAS["emp_us"]))
    assert not r.abstained and r.citations and r.citations[0]["chunk_id"] != "made-up#9"
    assert all(c["chunk_id"] in {h.chunk_id for h in r.retrieved} for c in r.citations)
    async with corpus() as s:
        row = (
            await s.execute(text("SELECT question_scrubbed, user_hash, cited_ids FROM qa_logs"))
        ).one()
    assert (
        row.user_hash != "u"
        and len(row.user_hash) == 64
        and row.cited_ids == [r.citations[0]["chunk_id"]]
    )


async def test_low_similarity_abstains_without_calling_the_model(corpus: SM) -> None:
    llm = FakeLLMClient()  # empty script: any call would raise
    a = Answerer(corpus, llm, EMB, make_policy_settings(abstain_min_similarity=0.99))
    r = await a.ask("Do we offer pet insurance?", User("u", PERSONAS["emp_us"]))
    assert r.abstained and r.abstain_reason == "no_evidence" and llm.calls == []


async def test_hallucinated_citation_only_becomes_abstention(corpus: SM) -> None:
    def bogus(_: list[Message]) -> LLMResult:
        return fake_result(
            parsed={
                "answer": "42 days",
                "citations": ["nope#1"],
                "abstained": False,
                "abstain_reason": None,
            }
        )

    r = await answerer(corpus, bogus).ask("How much leave?", User("u", PERSONAS["emp_pk"]))
    assert r.abstained and r.citations == []


async def test_output_filter_blocks_injected_link(corpus: SM) -> None:
    a = answerer(
        corpus,
        citing_responder("Email your password to verify@n0rthwind-it.example to activate it."),
    )
    r = await a.ask(
        "How do I connect to the VPN?", User("u", PERSONAS["emp_us"]), corpora=["main", "poisoned"]
    )
    assert r.blocked and r.abstained and r.answer == ""


async def test_degraded_mode_when_embeddings_fail(corpus: SM) -> None:
    s = make_policy_settings(abstain_min_similarity=0.0)
    a = Answerer(corpus, FakeLLMClient(responder=citing_responder()), FakeEmbedder(fail=True), s)
    r = await a.ask("How many paid sick days do we get?", User("u", PERSONAS["emp_us"]))
    assert r.degraded and r.mode == "lexical(degraded)" and r.retrieved and not r.abstained


async def test_cache_never_crosses_permissions(corpus: SM) -> None:
    a = answerer(corpus)
    q = "What is the bonus target for managers?"
    r1 = await a.ask(q, User("m", PERSONAS["mgr_us"]))
    r2 = await a.ask(q, User("e", PERSONAS["emp_us"]))
    r3 = await a.ask(q, User("m2", PERSONAS["mgr_us"]))
    assert not r2.cached and r3.cached
    assert any(c["chunk_id"].startswith("manager-") for c in r1.citations)
    assert not any(h.slug.startswith("manager-") for h in r2.retrieved)


# ------------------------------------------------------------------ HTTP


@pytest.fixture
async def client(corpus: SM, engine: AsyncEngine) -> AsyncIterator[httpx.AsyncClient]:
    s = make_policy_settings(
        database_url=engine.url.render_as_string(hide_password=False), abstain_min_similarity=0.0
    )
    app = create_app(s, embedder=EMB)
    app.state.answerer.llm = FakeLLMClient(responder=citing_responder("Answer."))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    await app.state.engine.dispose()


async def token(c: httpx.AsyncClient, persona: str) -> dict[str, str]:
    r = await c.post("/v1/dev/token", json={"persona": persona})
    return {"Authorization": f"Bearer {r.json()['token']}"}


async def test_api_auth_and_roles(client: httpx.AsyncClient) -> None:
    assert (await client.post("/v1/ask", json={"question": "hello?"})).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert (
        await client.post("/v1/ask", headers=bad, json={"question": "hello?"})
    ).status_code == 401
    emp, hr = await token(client, "emp_pk"), await token(client, "hr")
    assert (
        await client.post("/v1/ask", headers=emp, json={"question": "hello?", "debug": True})
    ).status_code == 403
    assert (
        await client.post("/v1/documents", headers=emp, json={"markdown": "x" * 20})
    ).status_code == 403
    assert (await client.get("/v1/reports/unanswered", headers=emp)).status_code == 403
    assert (await client.get("/v1/reports/unanswered", headers=hr)).status_code == 200


async def test_api_ask_search_feedback_report(client: httpx.AsyncClient) -> None:
    h = await token(client, "emp_us")
    r = await client.post("/v1/ask", headers=h, json={"question": "How many paid sick days?"})
    body = r.json()
    assert r.status_code == 200 and body["citations"] and body["trace_id"] and not body["degraded"]
    fb = await client.post(
        "/v1/feedback", headers=h, json={"trace_id": body["trace_id"], "rating": 1}
    )
    assert fb.status_code == 201
    s = await client.get("/v1/search", headers=h, params={"q": "manager salary band", "k": 20})
    assert not any(x["chunk_id"].startswith("manager-") for x in s.json()["results"])


async def test_ui_persona_switch_and_ask(client: httpx.AsyncClient) -> None:
    r = await client.post("/ui/persona", data={"persona": "mgr_pk"})
    assert r.status_code == 303 and "pp_token" in r.cookies
    client.cookies.set("pp_token", r.cookies["pp_token"])
    page = (await client.get("/ui")).text
    assert "mgr_pk" in page and "managers" in page
    frag = (
        await client.post(
            "/ui/ask", data={"question": "What is the merit budget?", "mode": "hybrid"}
        )
    ).text
    assert "Answer." in frag and "cite" in frag
