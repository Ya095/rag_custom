import logging
import logging.config
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from api.v1.routers import router as routers_v1
from exceptions.exceptions import RAGBaseException


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


def add_middlewares(app: FastAPI):
    """Add middleware."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )


def add_exception_handlers(app: FastAPI):
    """Add exception handlers."""

    @app.exception_handler(RAGBaseException)
    async def rag_exception_handler(request: Request, exc: RAGBaseException):
        logger = logging.getLogger(__name__)
        logger.error(f"RAG exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


@asynccontextmanager
async def initialize(app: FastAPI):
    """App initializer."""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("App started")

    yield

    logger.info("App closed")


def create_app() -> FastAPI:
    """Creating fastapi app."""

    app = FastAPI(
        title="Rag custom.",
        lifespan=initialize,
    )

    add_middlewares(app)
    add_exception_handlers(app)
    app.include_router(routers_v1)

    return app
