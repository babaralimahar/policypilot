"""Build and load the frozen evaluation sets from the fact sheet.

    uv run python -m policypilot.evals.dataset      # writes evals/datasets/policy/

Item types: answerable (one fact), multi (two facts in different documents), unanswerable
(not in the handbook), probe (the only relevant section is outside the asker's groups), and
a separate attack set (questions that retrieve a poisoned page).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.evals.dataset import load_dataset, write_sums
from policypilot.data.spec import Spec, load_spec
from policypilot.domain.chunking import slugify

OUT = Path("evals/datasets/policy")
ItemType = Literal["answerable", "multi", "unanswerable", "probe", "attack"]

# Cross-region probes: a regional fact asked by an employee of the other region.
CROSS_REGION = {
    "f1": "emp_us",
    "f10": "emp_us",
    "f53": "emp_us",
    "f54": "emp_us",
    "f57": "emp_us",
    "f71": "emp_us",
    "f6": "emp_pk",
    "f13": "emp_pk",
    "f55": "emp_pk",
    "f59": "emp_pk",
    "f73": "emp_pk",
}


class EvalItem(BaseModel):
    id: str
    type: ItemType
    persona: str
    groups: list[str]
    question: str
    gold_sections: list[str] = Field(default_factory=list)  # slug#section
    expect: list[str] = Field(default_factory=list)  # fact texts a correct answer contains
    restricted: list[str] = Field(default_factory=list)  # fact texts that must never appear
    marker: str | None = None  # attack success marker


def build_items(spec: Spec) -> tuple[list[EvalItem], list[EvalItem]]:
    idx = spec.fact_index()
    P = spec.personas
    items: list[EvalItem] = []

    def gold(fid: str) -> str:
        d, s, _ = idx[fid]
        return f"{d.slug}#{slugify(s.heading)}"

    for d in spec.documents:
        if d.superseded:
            continue
        for s in d.sections:
            for f in s.facts:
                if f.q:
                    persona, q = f.q
                    assert set(P[persona]) & set(d.acl), f"{f.id}: {persona} cannot see {d.slug}"
                    items.append(
                        EvalItem(
                            id=f"a-{f.id}",
                            type="answerable",
                            persona=persona,
                            groups=P[persona],
                            question=q,
                            gold_sections=[gold(f.id)],
                            expect=[f.text],
                        )
                    )
                probes = [f.probe] if f.probe else []
                if f.id in CROSS_REGION and f.q:
                    probes.append((CROSS_REGION[f.id], f.q[1]))
                for n, (persona, q) in enumerate(probes):
                    assert not set(P[persona]) & set(d.acl), (
                        f"{f.id}: probe persona {persona} CAN see {d.slug}"
                    )
                    items.append(
                        EvalItem(
                            id=f"p-{f.id}-{n}",
                            type="probe",
                            persona=persona,
                            groups=P[persona],
                            question=q,
                            gold_sections=[gold(f.id)],
                            restricted=[f.text],
                        )
                    )
    for n, m in enumerate(spec.multi):
        items.append(
            EvalItem(
                id=f"m-{n}",
                type="multi",
                persona=m.persona,
                groups=P[m.persona],
                question=m.q,
                gold_sections=[gold(f) for f in m.facts],
                expect=[idx[f][2].text for f in m.facts],
            )
        )
    for n, (persona, q) in enumerate(spec.unanswerable):
        items.append(
            EvalItem(
                id=f"u-{n}", type="unanswerable", persona=persona, groups=P[persona], question=q
            )
        )
    attacks = [
        EvalItem(
            id=f"x-{p.slug}",
            type="attack",
            persona=p.q[0],
            groups=P[p.q[0]],
            question=p.q[1],
            gold_sections=[f"{p.slug}#how-it-works"],
            marker=p.marker,
        )
        for p in spec.poisoned
    ]
    return items, attacks


def split(
    items: list[EvalItem], test_fraction: float = 1 / 3, seed: int = 7
) -> tuple[list[EvalItem], list[EvalItem]]:
    rng = random.Random(seed)
    dev: list[EvalItem] = []
    test: list[EvalItem] = []
    for t in ("answerable", "multi", "unanswerable", "probe"):
        group = [i for i in items if i.type == t]
        rng.shuffle(group)
        k = round(len(group) * test_fraction)
        test += group[:k]
        dev += group[k:]
    return sorted(dev, key=lambda i: i.id), sorted(test, key=lambda i: i.id)


def _write(path: Path, items: list[EvalItem]) -> None:
    lines = [
        json.dumps({"id": i.id, "input": i.question, "expected": i.model_dump()}, sort_keys=True)
        for i in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build(out: Path = OUT) -> dict[str, Any]:
    items, attacks = build_items(load_spec())
    dev, test = split(items)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "dev.jsonl", dev)
    _write(out / "test.jsonl", test)
    _write(out / "attacks.jsonl", attacks)
    write_sums(out)
    counts = {
        s: {
            t: sum(1 for i in xs if i.type == t)
            for t in ("answerable", "multi", "unanswerable", "probe")
        }
        for s, xs in (("dev", dev), ("test", test))
    }
    counts["attacks"] = {"attack": len(attacks)}
    return counts


def load_items(split_name: str, root: Path = OUT) -> tuple[list[EvalItem], str]:
    ds = load_dataset(root / f"{split_name}.jsonl", require_frozen=True)
    return [EvalItem.model_validate(i.expected) for i in ds.items], ds.sha256


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
