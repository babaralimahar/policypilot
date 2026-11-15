"""Output safety (indirect prompt injection) and PII scrubbing for logs."""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_DOMAINS = ("northwind.example",)

_URL = re.compile(r"\bhttps?://[^\s)>\]]+", re.I)
_DOMAIN_IN_URL = re.compile(r"https?://([^/\s:)>\]]+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@([\w-]+(?:\.[\w-]+)+)\b")
_CREDENTIAL_ASK = re.compile(
    r"(?i)(send|email|reply|share|provide|enter|type)\b[^.]{0,60}"
    r"\b(your|their|the user'?s?)\b[^.]{0,30}"
    r"\b(password|passcode|pin|cnic|passport)"
)
_IBAN_LIKE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")


@dataclass(frozen=True)
class SafetyVerdict:
    blocked: bool
    reasons: list[str]


def _allowed(domain: str) -> bool:
    d = domain.lower().rstrip(".")
    return any(d == a or d.endswith("." + a) for a in ALLOWED_DOMAINS)


def check_output(text: str) -> SafetyVerdict:
    """Block answers that send people to unknown places or ask for secrets. These are
    the observable effects of a successful injection, whatever the injected wording."""
    reasons = []
    for m in _DOMAIN_IN_URL.finditer(text):
        if not _allowed(m.group(1)):
            reasons.append(f"url:{m.group(1)}")
    for m in _EMAIL.finditer(text):
        if not _allowed(m.group(1)):
            reasons.append(f"email:{m.group(1)}")
    if _CREDENTIAL_ASK.search(text):
        reasons.append("credential_request")
    if _IBAN_LIKE.search(text):
        reasons.append("bank_account")
    return SafetyVerdict(bool(reasons), reasons)


_PII = [
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[email]"),
    (re.compile(r"\b\d{5}-\d{7}-\d\b"), "[cnic]"),  # Pakistani national ID
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[number]"),
]


def scrub(question: str) -> str:
    """Questions are logged for the unanswered report, never with contact or ID numbers."""
    out = question
    for pattern, repl in _PII:
        out = pattern.sub(repl, out)
    return out[:500]
