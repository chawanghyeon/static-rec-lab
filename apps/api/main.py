"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.routes import recommendations_router
from apps.api.services import build_default_recommendation_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """API 시작 시 model-backed recommendation service를 초기화한다."""
    app.state.recommendation_service = build_default_recommendation_service()
    yield


def create_app() -> FastAPI:
    """FastAPI app을 생성한다."""
    app = FastAPI(
        title="static-rec-lab",
        version="0.1.0",
        description="static_decoding 기반 generative recommendation serving API",
        lifespan=lifespan,
    )
    app.include_router(recommendations_router)
    return app


app = create_app()


def main() -> None:
    """로컬 API 서버를 실행한다."""
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
