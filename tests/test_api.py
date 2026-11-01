import httpx
import pytest

from src.main import app
from src.transcription.router import get_service
from src.transcription.schemas import make_result


@pytest.mark.asyncio
async def test_upload_and_language():
    async def transcribe(file, language):
        assert await file.read() == b"audio"
        assert language == "id"
        return make_result(
            "id", 2.5, [{"start_time": 500, "end_time": 2000, "speaker": 8, "text": " Pagi. "}]
        )

    app.dependency_overrides[get_service] = lambda: transcribe
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/transcriptions", files={"file": ("a.wav", b"audio")}, data={"language": "id"}
            )
            assert response.status_code == 200
            assert response.json()["segments"] == [
                {"start": 0.5, "end": 2.0, "speaker": "SPEAKER_00", "text": "Pagi."}
            ]
            assert (
                await client.post(
                    "/v1/transcriptions",
                    files={"file": ("a.wav", b"a")},
                    data={"language": "invalid"},
                )
            ).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_speaker_ids_and_silence():
    rows = [
        {"start_time": i * 1000, "end_time": (i + 1) * 1000, "speaker": s, "text": "hello"}
        for i, s in enumerate([9, 4, 9])
    ]
    assert [s["speaker"] for s in make_result("en", 3, rows)["segments"]] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert make_result(None, 1, []) == {
        "language": None,
        "duration_seconds": 1,
        "text": "",
        "segments": [],
    }
