"""PolicyPilot settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr

from app.config import Settings


class PolicySettings(Settings):
    app_name: str = "policypilot"

    # Embeddings (OpenAI-compatible /embeddings). Dimension is fixed by migration 0002.
    embed_provider: Literal["fake", "openai_compat"] = "fake"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    embed_price_per_mtok: float = 0.15  # USD, copied 2026-11-15; usage is estimated (chars/4)

    # Retrieval
    # Chosen by the dev ablation (2026-11-15): vector beat hybrid at R@1 0.94 vs 0.78; see ADR-004.
    retrieval_mode: Literal["lexical", "vector", "hybrid", "hybrid+rerank"] = "vector"
    candidate_k: int = Field(30, ge=5, le=100)  # per retriever, before fusion
    top_k: int = Field(8, ge=1, le=20)  # passed to the model
    # Best vector similarity below this -> abstain without calling the model. Calibrated on dev
    # (2026-11-15): just under the weakest answerable question (0.594); see evals/reports.
    abstain_min_similarity: float = Field(0.59, ge=0, le=1)
    corpora: list[str] = Field(default_factory=lambda: ["main"])

    # Generation
    prompt_version: str = "answer_v2"  # v1 vs v2 on dev: evals/reports
    hardened: bool = True  # instruction hierarchy + output filter (False only for the ASR baseline)
    answer_timeout_s: float = Field(45.0, gt=0)

    # Auth (dev: HS256 tokens from /v1/dev/token; production: OIDC/RS256, see README)
    jwt_secret: SecretStr = SecretStr("dev-only-change-me-dev-only-change-me")
    jwt_audience: str = "policypilot"
    jwt_ttl_s: int = Field(8 * 3600, ge=60)
    dev_tokens: bool = True  # never enable in production
    log_salt: SecretStr = SecretStr("dev-salt")


@lru_cache
def get_policy_settings() -> PolicySettings:
    return PolicySettings()
