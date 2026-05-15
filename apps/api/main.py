"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from apps.api.routes import recommendations_router
from apps.api.services import (
    RecommendationService,
    build_recommendation_service_from_environment,
)


def create_app(recommendation_service: RecommendationService | None = None) -> FastAPI:
    """FastAPI app을 생성한다."""
    app = FastAPI(
        title="static-rec-lab",
        version="0.1.0",
        description="STATIC-style generative recommendation serving API",
    )
    app.state.recommendation_service = (
        build_recommendation_service_from_environment()
        if recommendation_service is None
        else recommendation_service
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
