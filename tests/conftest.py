from __future__ import annotations

import os
from typing import Any

import pytest
from tenacity import wait_none

import app.adapters.llm as llm_module
from app.config import Settings
from policypilot.config import PolicySettings

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@127.0.0.1:5435/app_test"
)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "env": "test",
        "database_url": TEST_DATABASE_URL,
        "llm_provider": "fake",
        "worker_mode": "off",
        "otel_endpoint": None,
        "_env_file": None,  # tests never read the developer's .env
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry tests fast: tenacity waits are zero in tests."""
    monkeypatch.setattr(llm_module, "_WAIT", wait_none())


def make_policy_settings(**overrides: Any) -> PolicySettings:
    """PolicyPilot settings for tests: fake LLM and embeddings, no worker."""
    values: dict[str, Any] = {
        "env": "test",
        "database_url": TEST_DATABASE_URL,
        "llm_provider": "fake",
        "embed_provider": "fake",
        "worker_mode": "off",
        "_env_file": None,
    }
    values.update(overrides)
    return PolicySettings(**values)
