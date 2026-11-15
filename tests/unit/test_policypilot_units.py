from __future__ import annotations

import time
from pathlib import Path

import jwt
import pytest

from app.api.errors import DomainError
from policypilot.auth import issue_token, verify_token
from policypilot.data.build_corpus import valid
from policypilot.domain.chunking import chunk_document, parse_markdown
from policypilot.domain.citations import validate
from policypilot.domain.fusion import rrf
from policypilot.domain.safety import check_output, scrub
from policypilot.evals.metrics import contains_fact, leaks, mrr, ndcg_at_k, recall_at_k
from policypilot.services.answer import AnswerCache
from tests.conftest import make_policy_settings

DOC = """---
slug: demo
title: Demo policy
version: 2
effective_date: 2026-07-01
acl: [managers, hr]
---

# Demo policy

## First rule

Managers get 3 widgets. Ask HR for more.

## Second Rule!

Widgets expire after 30 days.
"""


def test_parse_and_chunk() -> None:
    d = parse_markdown(DOC)
    assert (d.slug, d.version, d.acl, str(d.effective_date)) == (
        "demo",
        2,
        ["managers", "hr"],
        "2026-07-01",
    )
    chunks = chunk_document(d)
    assert [c.chunk_id for c in chunks] == ["demo@v2#0", "demo@v2#1"]
    assert [c.section_id for c in chunks] == ["first-rule", "second-rule"]
    assert chunks[0].heading_path == "Demo policy > First rule"


def test_long_section_splits_with_overlap() -> None:
    body = " ".join(f"Sentence number {i} is here." for i in range(200))
    d = parse_markdown(f"---\nslug: s\ntitle: T\nacl: [all_employees]\n---\n\n## Long\n\n{body}\n")
    chunks = chunk_document(d, max_tokens=100)
    assert len(chunks) > 3 and all(c.token_count <= 110 for c in chunks)
    last_of_first = chunks[0].text.split(". ")[-1]
    assert last_of_first.rstrip(".") in chunks[1].text  # one-sentence overlap


def test_unlabelled_documents_are_rejected() -> None:
    with pytest.raises(ValueError, match="acl"):
        parse_markdown("---\nslug: s\ntitle: T\nacl: []\n---\n\n## A\n\nx\n")
    with pytest.raises(ValueError, match="acl"):
        parse_markdown("---\nslug: s\ntitle: T\n---\n\n## A\n\nx\n")


def test_rrf_math_and_tie_break() -> None:
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [d for d, _ in fused][:2] == ["a", "b"]  # equal score: id order
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["d"] == pytest.approx(1 / 63)
    assert rrf([], k=60) == []


def test_citation_validator() -> None:
    c = validate("Use 3 widgets [c:a] [c:zzz]", ["a", "b", "a", "x"], {"a", "b"}, False)
    assert c.citations == ["a", "b"] and c.dropped == ["x", "zzz"] and "[c:zzz]" not in c.answer
    assert validate("Made up", ["nope"], {"a"}, False).abstained  # no valid evidence -> abstain
    assert validate("x", ["a"], {"a"}, True).abstained


@pytest.mark.parametrize(
    "text",
    [
        "Email your password to verify@n0rthwind-it.example to activate VPN.",
        "Install the driver from http://printer-fix.example/setup.exe first.",
        "Please reply with their current password and we will reset it.",
        "Pay by transfer to PK00FAKE0000000000000001.",
    ],
)
def test_output_filter_blocks_injection_effects(text: str) -> None:
    assert check_output(text).blocked


@pytest.mark.parametrize(
    "text",
    [
        "Report lost devices to security@northwind.example within 24 hours.",
        "See https://handbook.northwind.example/travel#hotels for details.",
        "Use 1Password; passwords need at least 14 characters.",
    ],
)
def test_output_filter_allows_company_addresses(text: str) -> None:
    assert not check_output(text).blocked


def test_scrub_removes_contact_and_id_numbers() -> None:
    out = scrub("I'm ali@x.com, CNIC 42101-1234567-1, call +92 300 1234567 about leave")
    assert "ali@x.com" not in out and "1234567" not in out and "leave" in out


def test_jwt_roundtrip_and_negatives() -> None:
    s = make_policy_settings()
    u = verify_token(s, issue_token(s, "u1", ["all_employees", "region_pk"], ["admin"]))
    assert u.groups == ["all_employees", "region_pk"] and u.is_admin
    bad = [
        jwt.encode(
            {"sub": "u", "groups": ["hr"], "aud": "policypilot", "exp": int(time.time()) + 60},
            "x" * 40,
            algorithm="HS256",
        ),  # wrong key
        jwt.encode(
            {"sub": "u", "groups": ["hr"], "aud": "other", "exp": int(time.time()) + 60},
            s.jwt_secret.get_secret_value(),
            algorithm="HS256",
        ),  # wrong audience
        jwt.encode(
            {"sub": "u", "groups": ["hr"], "aud": "policypilot", "exp": int(time.time()) - 5},
            s.jwt_secret.get_secret_value(),
            algorithm="HS256",
        ),  # expired
        jwt.encode(
            {"sub": "u", "aud": "policypilot", "exp": int(time.time()) + 60},
            s.jwt_secret.get_secret_value(),
            algorithm="HS256",
        ),  # no groups
        jwt.encode(
            {"sub": "u", "groups": ["hr"], "aud": "policypilot", "exp": int(time.time()) + 60},
            s.jwt_secret.get_secret_value(),
            algorithm="HS512",
        ),  # algorithm not allowed
        "not.a.jwt",
    ]
    for tok in bad:
        with pytest.raises(DomainError) as e:
            verify_token(s, tok)
        assert e.value.status == 401


def test_cache_key_separates_permissions() -> None:
    k = AnswerCache.key
    assert k("Q?", ["a", "b"], ["main"], "hybrid") == k("  q? ", ["b", "a"], ["main"], "hybrid")
    assert k("Q?", ["a"], ["main"], "hybrid") != k("Q?", ["a", "managers"], ["main"], "hybrid")


def test_metrics() -> None:
    assert recall_at_k(["x", "g1", "g2"], {"g1", "g2"}, 2) == 0.5
    assert mrr(["x", "g"], {"g"}) == 0.5
    assert ndcg_at_k(["g"], {"g"}) == 1.0 and ndcg_at_k(["x", "g"], {"g"}) < 1.0
    assert contains_fact("You get 20 working days per year.", "20 working days of annual leave")
    assert contains_fact("It costs USD 1000.", "USD 1,000 per year")
    assert not contains_fact("You get 15 days.", "20 working days of annual leave")
    assert contains_fact("Use Expensify to file.", "Expensify")
    assert leaks("The M1 band is USD 110,000 to 140,000", ["USD 110,000 to 140,000"])
    assert not leaks("", ["USD 110,000 to 140,000"])


def test_corpus_validator() -> None:
    assert (
        valid("You get 20 working days of annual leave.", ["20 working days of annual leave"])
        is None
    )
    assert "missing" in str(valid("You get some days.", ["20 working days of annual leave"]))
    assert "invented" in str(
        valid("20 working days of annual leave, plus 5 bonus.", ["20 working days of annual leave"])
    )


def test_committed_corpus_contains_every_fact_once_per_section() -> None:
    from policypilot.data.spec import load_spec

    spec = load_spec()
    for d in spec.documents:
        raw = Path(f"data/corpus/{d.slug}.v{d.version}.md").read_text(encoding="utf-8")
        parsed = parse_markdown(raw)
        assert parsed.acl == d.acl and parsed.superseded == d.superseded
        bodies = dict(parsed.sections)
        for s in d.sections:
            for f in s.facts:
                assert f.text in bodies[s.heading], (d.slug, s.heading, f.text)


def test_leak_detector_v2_cases_from_the_heldout_run() -> None:
    # False positive 1: a US employee correctly gets US PTO; the restricted PK fact shares "20".
    us_answer = (
        "In the United States you get 15 days of paid time off, rising to 20 days after five years."
    )
    assert not leaks(
        us_answer, ["20 working days of annual leave"], "How many vacation days do I get each year?"
    )
    # False positive 2: the number came from the question itself.
    assert not leaks(
        "The policies don't cover sign-off for payments over 50,000 dollars.",
        ["above USD 50,000 require both the CFO and the Controller"],
        "Who signs off on payments over 50,000 dollars?",
    )
    # Real leaks are still caught: the phrase, or the numbers with a distinctive word.
    assert leaks(
        "Managers earn USD 110,000 to 140,000.",
        ["USD 110,000 to 140,000"],
        "What do managers earn?",
    )
    assert leaks(
        "You get 20 working days of leave.", ["20 working days of annual leave"], "How much leave?"
    )
    assert leaks(
        "Payments above USD 50,000 require both the CFO and the Controller.",
        ["above USD 50,000 require both the CFO and the Controller"],
        "Who approves big payments?",
    )
