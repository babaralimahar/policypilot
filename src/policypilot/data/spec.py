"""Typed view of data/handbook_spec.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

SPEC_PATH = Path("data/handbook_spec.yaml")


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str
    q: tuple[str, str] | None = None  # (persona, question) asked by someone with access
    probe: tuple[str, str] | None = None  # (persona, question) asked by someone WITHOUT access


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    heading: str
    facts: list[Fact]


class Doc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    title: str
    acl: list[str]
    version: int = 1
    effective_date: str = "2026-01-01"
    superseded: bool = False
    sections: list[Section]


class Multi(BaseModel):
    persona: str
    q: str
    facts: list[str]


class Poisoned(BaseModel):
    slug: str
    title: str
    topic: str
    attack: str
    marker: str
    q: tuple[str, str]


class Spec(BaseModel):
    company: str
    groups: list[str]
    personas: dict[str, list[str]]
    documents: list[Doc]
    multi: list[Multi] = Field(default_factory=list)
    unanswerable: list[tuple[str, str]] = Field(default_factory=list)
    poisoned: list[Poisoned] = Field(default_factory=list)

    def fact_index(self) -> dict[str, tuple[Doc, Section, Fact]]:
        return {f.id: (d, s, f) for d in self.documents for s in d.sections for f in s.facts}


def load_spec(path: Path = SPEC_PATH) -> Spec:
    return Spec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
