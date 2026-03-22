"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.ingestion import router as ingestion_router
from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A production-style document ingestion pipeline that processes "
            "documents into semantic chunks, generates embeddings, and stores "
            "them in a vector database for retrieval."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingestion_router, prefix="/api/v1")

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok", "version": settings.app_version}

    logger.info("Application %s v%s ready", settings.app_name, settings.app_version)
    return app


app = create_app()
