from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.v1.routers import router as routers_v1


def add_middlewares(app: FastAPI):
    """Add middleware."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
        expose_headers=["Content-Disposition"],
    )


@asynccontextmanager
async def initialize(app: FastAPI):
    """App initializer."""

    print('App started')

    yield

    print('App closed')


def create_app() -> FastAPI:
    """Creating fastapi app."""

    app = FastAPI(
        title='Rag custom.',
        lifespan=initialize,
    )

    add_middlewares(app)
    app.include_router(routers_v1)

    return app
