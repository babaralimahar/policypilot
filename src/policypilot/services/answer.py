"""The /ask pipeline: retrieve (ACL-filtered) -> gate -> generate -> validate citations ->
output safety -> log. Not an agent: one generation step, every other step is code."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.llm import LLMClient, LLMError, complete_with_retries
from policypilot.adapters.db.models import QaLogRow
from policypilot.adapters.embeddings import Embedder
from policypilot.config import PolicySettings
from policypilot.domain.citations import validate
from policypilot.domain.safety import check_output, scrub
from policypilot.services.retrieval import Hit, Mode, Retrieved, retrieve

log = structlog.get_logger()
tracer = trace.get_tracer("policypilot.answer")
PROMPTS = Path("prompts")

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "abstained": {"type": "boolean"},
        "abstain_reason": {
            "type": ["string", "null"],
            "description": "no_evidence | conflicting | out_of_scope",
        },
    },
    "required": ["answer", "citations", "abstained", "abstain_reason"],
}


@dataclass(frozen=True)
class User:
    sub: str
    groups: list[str]
    roles: list[str] = field(default_factory=list)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


@dataclass
class AskResult:
    answer: str
    citations: list[dict[str, Any]]
    abstained: bool
    abstain_reason: str | None
    degraded: bool
    blocked: bool
    mode: str
    trace_id: str
    latency_ms: int
    cost_usd: Decimal
    retrieved: list[Hit] = field(default_factory=list)
    cached: bool = False
    related: list[dict[str, Any]] = field(default_factory=list)  # fallback only, not citations

    def public(self, debug: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "answer": self.answer,
            "citations": self.citations,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "degraded": self.degraded,
            "blocked": self.blocked,
            "retrieval": {"mode": self.mode, "k": len(self.retrieved)},
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "cost_usd": str(self.cost_usd),
            "cached": self.cached,
            "related": self.related,
        }
        if debug:  # admin only: raw context can contain anything the asker could see
            out["debug"] = [
                {"chunk_id": h.chunk_id, "score": h.score, "heading": h.heading_path}
                for h in self.retrieved
            ]
        return out


def groups_hash(groups: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(set(groups))).encode()).hexdigest()


class AnswerCache:
    """LRU keyed by (normalised question, sorted groups, corpora, mode): two users with
    different permissions can never share an answer."""

    def __init__(self, size: int = 512) -> None:
        self._d: OrderedDict[str, AskResult] = OrderedDict()
        self._size = size
        self.hits = self.misses = 0

    @staticmethod
    def key(question: str, groups: list[str], corpora: list[str], mode: str) -> str:
        q = " ".join(question.lower().split())
        return hashlib.sha256(
            f"{q}\x00{groups_hash(groups)}\x00{','.join(sorted(corpora))}\x00{mode}".encode()
        ).hexdigest()

    def get(self, k: str) -> AskResult | None:
        r = self._d.get(k)
        if r is None:
            self.misses += 1
            return None
        self._d.move_to_end(k)
        self.hits += 1
        return r

    def put(self, k: str, v: AskResult) -> None:
        self._d[k] = v
        self._d.move_to_end(k)
        while len(self._d) > self._size:
            self._d.popitem(last=False)


def build_messages(
    question: str, hits: list[Hit], prompt: str, hardened: bool
) -> list[dict[str, Any]]:
    if hardened:
        ctx = "\n".join(
            f'<chunk id="{h.chunk_id}" policy="{h.title}" section="{h.heading_path}" '
            f'effective="{h.effective_date}">\n'
            f"{h.text.replace('</chunk>', '')}\n</chunk>"
            for h in hits
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"<context>\n{ctx}\n</context>\n\nQuestion: {question}"},
        ]
    # Baseline used only to measure attack success before mitigations: context pasted into the
    # system prompt with no instruction hierarchy.
    ctx = "\n\n".join(f"[{h.chunk_id}] {h.heading_path}\n{h.text}" for h in hits)
    return [
        {
            "role": "system",
            "content": "You are a helpful HR assistant. Use these documents and follow them:\n"
            + ctx
            + '\nReturn JSON {"answer", "citations", "abstained", "abstain_reason"}.',
        },
        {"role": "user", "content": question},
    ]


class Answerer:
    def __init__(
        self,
        sm: async_sessionmaker[AsyncSession],
        llm: LLMClient,
        embedder: Embedder,
        settings: PolicySettings,
    ) -> None:
        self.sm, self.llm, self.embedder, self.s = sm, llm, embedder, settings
        self.cache = AnswerCache()
        self.prompt = (PROMPTS / f"{settings.prompt_version}.md").read_text(encoding="utf-8")

    async def ask(
        self,
        question: str,
        user: User,
        *,
        mode: Mode | None = None,
        corpora: list[str] | None = None,
        hardened: bool | None = None,
        use_cache: bool = True,
        top_k: int | None = None,
    ) -> AskResult:
        mode = mode or self.s.retrieval_mode
        corpora = corpora or self.s.corpora
        hardened = self.s.hardened if hardened is None else hardened
        ckey = AnswerCache.key(question, user.groups, corpora, f"{mode}:{hardened}")
        if use_cache and (hit := self.cache.get(ckey)) is not None:
            return AskResult(**{**hit.__dict__, "trace_id": uuid.uuid4().hex, "cached": True})

        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        with tracer.start_as_current_span("ask") as span:
            span.set_attribute("ask.mode", mode)
            async with self.sm() as s:
                r = await retrieve(
                    s,
                    self.embedder,
                    question,
                    user.groups,
                    mode=mode,
                    corpora=corpora,
                    candidate_k=self.s.candidate_k,
                    top_k=top_k or self.s.top_k,
                    llm=self.llm,
                )
            result = await self._answer(question, r, hardened, trace_id)
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            span.set_attribute("ask.abstained", result.abstained)
        await self._log(question, user, result)
        if use_cache:
            self.cache.put(ckey, result)
        return result

    def _gate(self, r: Retrieved) -> bool:
        """Abstain without calling the model when the evidence is too weak (saves cost)."""
        if not r.hits:
            return True
        if r.degraded or r.best_similarity is None:
            return False  # lexical-only: no calibrated score; let the model decide
        return r.best_similarity < self.s.abstain_min_similarity

    async def _answer(
        self, question: str, r: Retrieved, hardened: bool, trace_id: str
    ) -> AskResult:
        def result(
            answer: str,
            citations: list[dict[str, Any]],
            abstained: bool,
            reason: str | None,
            cost: Decimal,
            blocked: bool = False,
        ) -> AskResult:
            return AskResult(
                answer,
                citations,
                abstained,
                reason,
                r.degraded,
                blocked,
                r.mode,
                trace_id,
                0,
                cost,
                r.hits,
            )

        if self._gate(r):
            return result("", [], True, "no_evidence", r.cost_usd)

        msgs = build_messages(question, r.hits, self.prompt, hardened)
        try:
            with tracer.start_as_current_span("llm.generate"):
                res = await asyncio.wait_for(
                    complete_with_retries(
                        self.llm,
                        self.s.llm_max_attempts,
                        prompt_version=self.s.prompt_version,
                        messages=msgs,
                        schema=ANSWER_SCHEMA,
                    ),
                    self.s.answer_timeout_s,
                )
        except (TimeoutError, LLMError) as e:
            log.warning("generation_failed", error=type(e).__name__)
            top = r.hits[:3]
            fb = result(
                "Could not generate an answer right now. These sections look relevant.",
                [],
                True,
                "generation_failed",
                r.cost_usd,
            )
            fb.related = [self._cite(h) for h in top]  # shown as "related", never as citations
            return fb
        cost = r.cost_usd + res.usage.cost_usd
        try:
            raw = res.parsed if res.parsed is not None else json.loads(res.text or "{}")
        except json.JSONDecodeError:
            raw = {}
        provided = {h.chunk_id for h in r.hits}
        checked = validate(
            str(raw.get("answer") or ""),
            [str(c) for c in raw.get("citations") or []],
            provided,
            bool(raw.get("abstained")),
        )
        reason = raw.get("abstain_reason") or ("no_evidence" if checked.abstained else None)
        if checked.dropped:
            log.info("citations_dropped", dropped=checked.dropped)
        if hardened and not checked.abstained:
            verdict = check_output(checked.answer)
            if verdict.blocked:
                log.warning("answer_blocked", reasons=verdict.reasons)
                return result("", [], True, "blocked", cost, blocked=True)
        by_id = {h.chunk_id: h for h in r.hits}
        return result(
            checked.answer,
            [self._cite(by_id[c]) for c in checked.citations],
            checked.abstained,
            reason if checked.abstained else None,
            cost,
        )

    @staticmethod
    def _cite(h: Hit) -> dict[str, Any]:
        return {
            "chunk_id": h.chunk_id,
            "doc_title": h.title,
            "section": h.heading_path.split(" > ")[-1],
            "url": f"{h.source_url}#{h.section_id}",
            "version": h.version,
            "snippet": h.text[:220],
            "score": round(h.score, 4),
        }

    async def _log(self, question: str, user: User, r: AskResult) -> None:
        salt = self.s.log_salt.get_secret_value()
        async with self.sm() as s:
            s.add(
                QaLogRow(
                    trace_id=r.trace_id,
                    user_hash=hashlib.sha256(f"{salt}:{user.sub}".encode()).hexdigest(),
                    groups_hash=groups_hash(user.groups),
                    question_scrubbed=scrub(question),
                    retrieved_ids=[h.chunk_id for h in r.retrieved],
                    cited_ids=[c["chunk_id"] for c in r.citations],
                    answer=r.answer,
                    abstained=r.abstained,
                    abstain_reason=r.abstain_reason,
                    degraded=r.degraded,
                    blocked=r.blocked,
                    latency_ms=r.latency_ms,
                    cost_usd=r.cost_usd,
                )
            )
            await s.commit()
