import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.transcription import service
from src.transcription.router import router


@asynccontextmanager
async def lifespan(app):
    yield
    await service.stop_worker()


app = FastAPI(
    title="Audio Transcription",
    lifespan=lifespan,
    openapi_url="/openapi.json"
    if os.getenv("ENVIRONMENT", "local") in {"local", "staging"}
    else None,
)
app.include_router(router)
