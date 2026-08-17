"""
App factory. Every project in the portfolio uses this same pattern:
build_app() wires config + logging + routes together, so main.py / run.py
stays a two-liner regardless of which project this is.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logger import setup_logging
from app.api.routes import router


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app
