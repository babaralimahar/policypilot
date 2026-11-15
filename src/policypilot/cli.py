"""Operator commands.

uv run python -m policypilot.cli ingest                 # data/corpus -> DB, then embed
uv run python -m policypilot.cli ingest --poisoned      # also the injection-test corpus
uv run python -m policypilot.cli token --persona emp_pk
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.adapters.db.engine import create_engine, create_sessionmaker
from policypilot.auth import issue_token
from policypilot.config import get_policy_settings
from policypilot.data.spec import load_spec
from policypilot.main import build_embedder
from policypilot.services.ingest import embed_missing, ingest_dir


async def _ingest(poisoned: bool) -> None:
    s = get_policy_settings()
    engine = create_engine(s)
    sm = create_sessionmaker(engine)
    try:
        async with sm() as session:
            for d, corpus in [
                (Path("data/corpus"), "main"),
                *([(Path("data/poisoned"), "poisoned")] if poisoned else []),
            ]:
                results = await ingest_dir(session, d, corpus)
                summary = {
                    k: sum(1 for r in results if r.status == k)
                    for k in ("created", "updated", "unchanged")
                }
                added = sum(r.chunks_added + r.chunks_changed for r in results)
                print(f"{corpus}: {len(results)} docs {summary}, {added} chunks new/changed")
            n, err = await embed_missing(session, build_embedder(s))
            print(f"embedded {n} chunks" + (f" (stopped: {err})" if err else ""))
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="policypilot.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--poisoned", action="store_true")
    tok = sub.add_parser("token")
    tok.add_argument("--persona", required=True)
    args = p.parse_args(argv)
    if args.cmd == "ingest":
        asyncio.run(_ingest(args.poisoned))
        return 0
    personas = load_spec().personas
    groups = personas[args.persona]
    print(
        issue_token(
            get_policy_settings(),
            f"demo-{args.persona}",
            groups,
            ["admin"] if "hr" in groups else [],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
