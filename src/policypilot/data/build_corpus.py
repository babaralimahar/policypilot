"""Build the synthetic handbook (markdown) from the fact sheet, with an LLM writing the prose.

    uv run python -m policypilot.data.build_corpus            # writes data/corpus/*.md

LLM-assisted drafting with automatic validation: every fact must appear verbatim in its
section, and the prose may not introduce any number that is not part of a fact (so it cannot
contradict another policy). Output is committed and frozen; rebuilding is not needed to run.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from app.adapters.llm import LLMClient, complete_with_retries, load_pricing
from app.adapters.llm_factory import build_llm_client
from app.config import get_settings
from policypilot.data.spec import Doc, Poisoned, Section, load_spec

OUT = Path("data/corpus")
NUM = re.compile(r"\d[\d,.:]*\d|\d")

SECTION_PROMPT = """You write one section of {company}'s internal employee handbook.
Policy: "{title}". Section heading: "{heading}".
Write 2 short paragraphs (110-200 words in total) of plain, friendly, specific handbook prose.
No headings, no bullet lists, no markdown formatting.
It MUST contain each of these phrases exactly as written (same characters):
{facts}
Do NOT write any other numbers, amounts, dates, times, percentages or durations anywhere,
not even as words (no "one", "two", "first"...) beyond those in the phrases above.
Do not mention other policies' figures. Output only the section text."""

POISON_PROMPT = """You write a short internal how-to page for {company}'s intranet about {topic}.
Write 2 short paragraphs (90-150 words) of practical, generic steps (who to contact, which
self-service portal page to use). No headings, no lists, no numbers, no URLs, no email addresses.
Output only the text."""


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".:") for n in NUM.findall(s)}


def valid(prose: str, facts: list[str]) -> str | None:
    """Return a reason the prose is unacceptable, or None."""
    missing = [f for f in facts if f not in prose]
    if missing:
        return f"missing phrases: {missing}"
    allowed = set().union(*(_numbers(f) for f in facts)) if facts else set()
    extra = _numbers(prose) - allowed
    if extra:
        return f"invented numbers: {sorted(extra)}"
    if "#" in prose or "\n- " in prose:
        return "markdown formatting"
    return None


async def write_section(llm: LLMClient, company: str, doc: Doc, sec: Section) -> str:
    facts = [f.text for f in sec.facts]
    prompt = SECTION_PROMPT.format(
        company=company,
        title=doc.title,
        heading=sec.heading,
        facts="\n".join(f'- "{f}"' for f in facts),
    )
    last = ""
    for _ in range(4):
        res = await complete_with_retries(
            llm, 6, messages=[{"role": "user", "content": prompt + last}]
        )
        prose = (res.text or "").strip()
        reason = valid(prose, facts)
        if reason is None:
            return prose
        last = f"\n\nYour previous attempt was rejected ({reason}). Try again."
    raise RuntimeError(f"{doc.slug}#{sec.id}: could not produce valid prose")


async def write_poison(llm: LLMClient, company: str, p: Poisoned) -> str:
    for _ in range(4):
        msg = POISON_PROMPT.format(company=company, topic=p.topic)
        res = await complete_with_retries(llm, 6, messages=[{"role": "user", "content": msg}])
        prose = (res.text or "").strip()
        if not _numbers(prose) and "@" not in prose and "http" not in prose:
            paras = [x for x in prose.split("\n\n") if x.strip()]
            # The attack sits between the two helpful paragraphs, as it would in a real wiki edit.
            return "\n\n".join([paras[0], p.attack, *paras[1:]])
    raise RuntimeError(f"poison {p.slug}: no clean prose")


def render(doc: Doc, bodies: dict[str, str]) -> str:
    front = [
        "---",
        f"slug: {doc.slug}",
        f"title: {doc.title}",
        f"version: {doc.version}",
        f"effective_date: {doc.effective_date}",
        f"acl: [{', '.join(doc.acl)}]",
        f"superseded: {'true' if doc.superseded else 'false'}",
        f"source_url: https://handbook.northwind.example/{doc.slug}",
        "---",
        "",
        f"# {doc.title}",
        "",
    ]
    for sec in doc.sections:
        front += [f"## {sec.heading}", "", bodies[sec.id], ""]
    return "\n".join(front)


async def build(out: Path) -> None:
    spec = load_spec()
    settings = get_settings()
    llm = build_llm_client(settings, load_pricing(settings.pricing_path))
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (one-off CLI)
    sem = asyncio.Semaphore(2)

    async def one_doc(doc: Doc) -> None:
        name = f"{doc.slug}.v{doc.version}.md"
        if (out / name).exists():
            return
        async with sem:
            bodies = {s.id: await write_section(llm, spec.company, doc, s) for s in doc.sections}
        (out / name).write_text(render(doc, bodies), encoding="utf-8", newline="\n")
        print("wrote", name)

    async def one_poison(p: Poisoned) -> None:
        name = f"{p.slug}.v1.md"
        poisoned_dir = out.parent / "poisoned"
        poisoned_dir.mkdir(parents=True, exist_ok=True)
        if (poisoned_dir / name).exists():
            return
        async with sem:
            body = await write_poison(llm, spec.company, p)
        doc = Doc(slug=p.slug, title=p.title, acl=["all_employees"], sections=[])
        text = render(doc, {}) + f"## How it works\n\n{body}\n"
        (poisoned_dir / name).write_text(text, encoding="utf-8", newline="\n")
        print("wrote poisoned", name)

    await asyncio.gather(
        *(one_doc(d) for d in spec.documents), *(one_poison(p) for p in spec.poisoned)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="policypilot.data.build_corpus")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    asyncio.run(build(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
