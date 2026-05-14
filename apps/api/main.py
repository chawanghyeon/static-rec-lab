"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from apps.api.routes import recommendations_router


def create_app() -> FastAPI:
    """FastAPI app을 생성한다."""
    app = FastAPI(
        title="static-rec-lab",
        version="0.1.0",
        description="STATIC-style generative recommendation serving API",
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
