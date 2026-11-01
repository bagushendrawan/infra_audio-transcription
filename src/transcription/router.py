import asyncio
from contextlib import suppress
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request, UploadFile

from src.transcription.schemas import Language, Transcription
from src.transcription.service import transcribe

router = APIRouter()


def get_service():
    return transcribe


@router.post(
    "/v1/transcriptions",
    response_model=Transcription,
    summary="Transcribe audio with anonymous speaker segments",
    tags=["transcription"],
    responses={
        code: {"description": description}
        for code, description in {
            400: "Empty or invalid audio",
            413: "Audio exceeds configured limits",
            422: "Invalid language or missing file",
            503: "Busy or CUDA unavailable",
            504: "Processing timeout",
            500: "Inference failure",
        }.items()
    },
)
async def create_transcription(
    file: UploadFile,
    request: Request,
    service: Annotated[object, Depends(get_service)],
    language: Annotated[Language | None, Form()] = None,
):
    task = asyncio.create_task(service(file, language))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.25)
            if not task.done() and await request.is_disconnected():
                task.cancel()
                break
        return await task
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
