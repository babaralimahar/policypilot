"""PolicyPilot app. Run with: uvicorn policypilot.main:create_app --factory"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.main import AppContext
from app.main import create_app as create_base_app
from app.worker import Handler
from policypilot import __version__
from policypilot.adapters.embeddings import Embedder, FakeEmbedder, OpenAICompatEmbedder
from policypilot.api.routes import router
from policypilot.config import PolicySettings, get_policy_settings
from policypilot.data.spec import load_spec
from policypilot.services.answer import Answerer
from policypilot.ui.routes import router as ui_router


def build_embedder(s: PolicySettings) -> Embedder:
    if s.embed_provider == "fake":
        return FakeEmbedder(s.embed_dim)
    if s.llm_api_key is None:
        raise ValueError("LLM_API_KEY is required for openai_compat embeddings")
    return OpenAICompatEmbedder(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key.get_secret_value(),
        model=s.embed_model,
        dim=s.embed_dim,
    )


def _no_handlers(_: AppContext) -> Mapping[str, Handler]:
    return {}


def create_app(settings: PolicySettings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_policy_settings()
    app = create_base_app(
        settings,
        routers=[router, ui_router],
        handlers=_no_handlers,
        title="PolicyPilot",
        version=__version__,
    )
    app.state.answerer = Answerer(
        app.state.sessionmaker, app.state.llm, embedder or build_embedder(settings), settings
    )
    app.state.personas = load_spec().personas

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    return app
