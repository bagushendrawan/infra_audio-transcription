import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from src.transcription.config import Settings

_busy = False
_process = None


async def stop_worker():
    if _process is not None:
        try:
            os.killpg(_process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(_process.wait(), 5)
        except TimeoutError:
            try:
                os.killpg(_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await _process.wait()


async def transcribe(file, language):
    global _busy, _process
    if _busy:
        raise HTTPException(503, "Transcription busy", headers={"Retry-After": "5"})
    _busy = True
    temporary = None
    try:
        settings = Settings()
        temporary = TemporaryDirectory(dir=settings.temp_dir)
        directory = temporary.name
        work = Path(directory)
        audio = work / "input.audio"

        # Disk operations stay outside the event loop.
        def copy_chunk(chunk, mode):
            with audio.open(mode) as target:
                target.write(chunk)

        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_bytes:
                raise HTTPException(413, "Audio exceeds upload limit")
            await run_in_threadpool(copy_chunk, chunk, "wb" if size == len(chunk) else "ab")
        if not size:
            raise HTTPException(400, "Empty audio")
        result = work / "result.json"
        env = {**os.environ, "TMPDIR": directory}
        _process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.transcription.worker",
            str(audio),
            str(result),
            language or "",
            env=env,
            start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(_process.wait(), settings.timeout)
        except TimeoutError as exc:
            raise HTTPException(504, "Transcription timed out") from exc
        if _process.returncode != 0 or not result.exists():
            raise HTTPException(500, "Inference failed; see server logs")
        payload = await run_in_threadpool(lambda: json.loads(result.read_text()))
        if "error" in payload:
            raise HTTPException(payload["status"], payload["error"])
        return payload
    finally:
        await stop_worker()
        if temporary is not None:
            await run_in_threadpool(temporary.cleanup)
        _process = None
        _busy = False
        await file.close()
