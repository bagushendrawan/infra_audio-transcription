# Audio transcription with diarization

Start from this directory with `uv run python main.py` (localhost:9101).
See [shared setup and API smoke tests](../README.md).

One blocking multipart endpoint backed by whisper-diarization. Python 3.12,
`uv`, Linux, an NVIDIA CUDA GPU, FFmpeg, and a C/C++ build toolchain are required.
Install CUDA 12 cuBLAS and cuDNN 9 compatible with the locked PyTorch/CTranslate2
versions. No CPU fallback is provided.

```sh
uv sync --locked --extra inference
cp .env.example .env
uv run --no-sync uvicorn src.main:app --host 127.0.0.1 --port 9101 --workers 1
```

Do not increase Uvicorn workers or run replicas on the same GPU: admission control
is per process. There is no queue, authentication, or persistence. Use a private
network. Interactive API documentation is at `/docs`.

```sh
curl --max-time 3700 http://127.0.0.1:9101/v1/transcriptions \
  -F 'file=@conversation.wav' -F 'language=id'
```

Omit `language` for detection. WAV, MP3, FLAC, M4A, OGG, and WebM are decoded by
PyAV; filenames and MIME types are not trusted. The JSON response contains
`language`, `duration_seconds`, `text`, and `segments`, for example:

```json
{"language":"id","duration_seconds":3.5,"text":"Pagi.","segments":[{"start":0.5,"end":2.0,"speaker":"SPEAKER_00","text":"Pagi."}]}
```

Speaker labels are anonymous and local to a recording, numbered by first
appearance. Times are seconds. Valid silence returns empty text/segments and
null language. Upstream does not reliably handle overlapping speakers; MSDD
uses automatic speaker counts up to eight. Punctuation restoration is skipped
for unsupported languages (including Indonesian).

Settings use the `TRANSCRIPTION_` prefix; see `.env.example`. Defaults are
multilingual `medium`, CUDA FP16, batch size 1, separation off, 100 MiB uploads,
30-minute audio, and a 60-minute inference timeout. Separation optionally uses
Demucs. Every request loads models anew. First use downloads models and can be
slow; Hugging Face/NeMo model caches persist outside the request directory.
Configure client/proxy timeouts to exceed processing time. Configure an ingress
body limit too: FastAPI parses multipart uploads before route-level size checks.

Errors use FastAPI's `detail`: 400 invalid/empty audio, 413 size/duration limit,
422 invalid language, 503 busy or CUDA unavailable, 504 timeout, 500 inference
failure. Worker stderr goes to server logs. Temporary input/output files are
removed after completion, failure, timeout, disconnect, or graceful shutdown;
uncatchable termination can leave OS temporary files.

## Development

```sh
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
```

API tests use dependency overrides; service tests exercise actual process timeout
and upload/cancellation cleanup without downloading ML models. Actual decoding,
model imports, GPU inference, and English/Indonesian speaker quality require
`--extra inference` and a CUDA host; these remain UNVERIFIED in this environment.

## Upstream

Adapted from https://github.com/MahmoudAshraf97/whisper-diarization at
`c6614d717764592c4261f2590c1eb29e87948c3e` (BSD-2-Clause; license in
`src/transcription/vendor/LICENSE`). Vendored helpers are reduced to the language
maps and speaker mapping functions; MSDD code/config are preserved. The worker
replaces CLI parsing, shell invocation, and TXT/SRT export with explicit settings,
argument-array subprocesses, input limits, silence handling, and JSON output.
Git dependencies are pinned by commit in `uv.lock`. Dependencies and downloaded
models retain their own licenses; the chosen ctc-forced-aligner declares
CC-BY-NC-4.0.

## Original resources

- [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) — original pipeline; adaptation and commit details are documented above.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper transcription implementation.
- [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) — speaker diarization models and toolkit.
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) — word alignment.
- [deepmultilingualpunctuation](https://github.com/oliverguhr/deepmultilingualpunctuation) — punctuation restoration.
- [Demucs fork](https://github.com/MahmoudAshraf97/demucs) — optional source separation dependency.
- [indic-numtowords](https://github.com/AI4Bharat/indic-numtowords) — number normalization dependency.
