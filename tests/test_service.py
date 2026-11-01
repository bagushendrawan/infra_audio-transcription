import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from src.transcription import service


@pytest.mark.parametrize(("data", "limit", "status"), [(b"", "10", 400), (b"123", "2", 413)])
async def test_upload_limits(data, limit, status, monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_MAX_BYTES", limit)
    monkeypatch.setenv("TRANSCRIPTION_TEMP_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as error:
        await service.transcribe(UploadFile(BytesIO(data)), None)
    assert error.value.status_code == status
    assert not list(tmp_path.iterdir())
    assert not service._busy


async def test_timeout_cleans_up_and_releases_capacity(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT", "0.00001")
    monkeypatch.setenv("TRANSCRIPTION_TEMP_DIR", str(tmp_path))
    for _ in range(2):
        with pytest.raises(HTTPException) as error:
            await service.transcribe(UploadFile(BytesIO(b"audio")), None)
        assert error.value.status_code == 504
        assert service._process is None
        assert not service._busy
        assert not list(tmp_path.iterdir())


async def test_busy_and_cancellation(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_TEMP_DIR", str(tmp_path))
    # A blocked upload holds capacity without requiring an inference installation.
    entered = asyncio.Event()

    class SlowUpload:
        async def read(self, size):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            pass

    task = asyncio.create_task(service.transcribe(SlowUpload(), None))
    await entered.wait()
    try:
        with pytest.raises(HTTPException) as error:
            await service.transcribe(UploadFile(BytesIO(b"audio")), None)
        assert error.value.status_code == 503
        assert error.value.headers["Retry-After"] == "5"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not service._busy
    assert not list(tmp_path.iterdir())


async def test_cancel_running_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTION_TEMP_DIR", str(tmp_path))
    task = asyncio.create_task(service.transcribe(UploadFile(BytesIO(b"audio")), None))
    try:
        async with asyncio.timeout(5):
            while service._process is None:
                await asyncio.sleep(0.001)
        process = service._process
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.returncode is not None
        assert service._process is None
        assert not service._busy
        assert not list(tmp_path.iterdir())
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
