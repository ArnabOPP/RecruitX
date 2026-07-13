# Recruitix Speech I/O Service

Implements the "Speech in / out" row of the Recruitix BRD's AI/ML model
table: Whisper-class STT + neural TTS, powering the **voice-based HR
round** — spoken questions and transcribed answers. It consumes
[interview-qa](../interview-qa)'s generated question text and feeds
[interview-qa](../interview-qa)'s `/followup` endpoint with the transcribed
candidate answer; it owns neither question generation nor grading.

## Why Groq + edge-tts

Same zero-budget constraint as interview-qa. Two independent choices:

**STT (speech-to-text): Groq's hosted `whisper-large-v3`.** interview-qa
already has a Groq account and API key wired up — this service reuses the
same key, so there's one provider account for the whole voice pipeline,
not two. Unlike the LLM choice, self-hosting Whisper is genuinely viable
too (the small/base models are lightweight enough for CPU inference on
short clips), so `STTClient` is built the same swappable-provider way as
interview-qa's `LLMClient` — a local-Whisper fallback can be registered in
`stt/client.py`'s factory later without touching the endpoint.

**TTS (text-to-speech): `edge-tts`.** An open-source wrapper around the
same neural voices Microsoft Edge's "Read Aloud" feature uses — free, no
API key or account, no published rate limit. The tradeoff: it's
**unofficial** (a reverse-engineered client against a Microsoft service,
not a published/supported API), so it could break if Microsoft changes
something server-side. That risk is exactly why `TTSClient` is a swappable
Protocol from day one — if it ever breaks, Azure Neural TTS (the
officially-supported version of the same voice engine) or Coqui
(self-hosted) is a new class in `tts/client.py`'s factory, not a service
rewrite.

## Architecture

```
GET  /api/v1/speech/synthesize   text -> tts/client.py (EdgeTTSClient)   -> audio/mpeg bytes
POST /api/v1/speech/transcribe   audio -> stt/client.py (GroqWhisperClient) -> text

main.py   FastAPI: POST /api/v1/speech/transcribe
                    POST /api/v1/speech/synthesize
```

Both clients follow the exact resilience pattern interview-qa's
`llm/client.py` established: a `Protocol` interface, a concrete provider
implementation, a `validate()` method for startup credential checks, and
(for STT, since Groq's failure modes apply here too) a `*_with_backoff()`
wrapper that respects `Retry-After` on rate limits and fails fast on
authentication errors instead of wasting retries.

## How it fits the existing pipeline

```
interview-qa generates a question (text)
  -> speech-io /synthesize -> audio the candidate hears
  -> candidate speaks their answer -> audio recorded in the browser
  -> speech-io /transcribe -> text
  -> fed into interview-qa /followup as candidate_answer
```

## Run locally

```bash
pip install -r requirements-dev.txt
cp .env.example .env   # put your Groq key in .env — the same key interview-qa uses works here
uvicorn app.main:app --reload
```

```bash
# Synthesize a question
curl -X POST http://localhost:8000/api/v1/speech/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Tell me about a challenging project you worked on."}' \
  -o question.mp3

# Transcribe an answer
curl -X POST http://localhost:8000/api/v1/speech/transcribe \
  -F "file=@answer.wav"
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/speech/transcribe` | audio file in -> transcribed text out |
| `POST /api/v1/speech/synthesize` | text in -> `audio/mpeg` bytes out |
| `GET /health/live` | liveness — process is up |
| `GET /health/ready` | readiness — both STT and TTS providers confirmed reachable (503 if either fails) |
| `GET /api/v1/capabilities` | provider, model, voice, readiness per-component |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | interactive Swagger UI |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `SPEECH_IO_GROQ_API_KEY` | *(required)* | Groq API key — same one interview-qa uses, free at console.groq.com |
| `SPEECH_IO_STT_MODEL` | `whisper-large-v3` | any Groq-hosted Whisper model |
| `SPEECH_IO_STT_MAX_RETRIES` | `2` | retries on rate limit / transient provider error |
| `SPEECH_IO_VALIDATE_KEY_ON_STARTUP` | `1` | confirms the Groq key actually works at boot, not just that it's set |
| `SPEECH_IO_TTS_DEFAULT_VOICE` | `en-US-AriaNeural` | any edge-tts neural voice name |
| `SPEECH_IO_VALIDATE_TTS_ON_STARTUP` | `1` | makes one real (tiny) synthesis call at boot to confirm edge-tts is reachable — this unofficial API has no lightweight metadata check, so this costs a small real network call every restart |
| `SPEECH_IO_MAX_AUDIO_UPLOAD_BYTES` | `10485760` (10 MB) | caps `/transcribe` audio uploads |
| `SPEECH_IO_MAX_SYNTHESIZE_TEXT_CHARS` | `2000` | caps `/synthesize` input text |
| `SPEECH_IO_MAX_REQUEST_BODY_BYTES` | `512000` | caps non-audio JSON bodies (`/synthesize`) via `Content-Length` before parsing — `/transcribe` is exempt, enforcing its own larger audio-specific limit instead |
| `SPEECH_IO_REQUIRE_API_KEY` | `0` | require an `X-API-Key` header on `/api/v1/speech/*` — **must be turned on before any internet-reachable deploy** |
| `SPEECH_IO_API_KEYS` | empty | comma-separated shared secrets accepted by `X-API-Key` |
| `SPEECH_IO_RATE_LIMIT_STORAGE_URI` | unset | e.g. `redis://host:6379` to share rate limits across replicas |
| `SPEECH_IO_CORS_ALLOW_ORIGINS` | empty | comma-separated allowlist; empty denies all cross-origin |

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

46 tests across seven files:

- `test_stt_client.py` — rate-limit backoff, fail-fast-on-bad-credentials,
  `Retry-After` parsing — mirrors interview-qa's `test_llm_client.py`
  exactly since Groq's failure modes are the same across its APIs.
- `test_tts_client.py` — the fake/error-path surface of `TTSClient`.
- `test_api.py` — HTTP-level tests with both clients mocked at the import
  site `main.py` actually uses them.
- `test_auth.py` — inbound API-key auth across all branches.
- `test_body_size_limit.py` — the request-size middleware, including the
  `exempt_paths` mechanism that lets `/transcribe` carry larger bodies
  than `/synthesize`.
- `test_rate_limit_redis.py` — real `redis:7-alpine` Docker container,
  proving two independent storage connections share hit counts. Skips
  cleanly if Docker isn't available.
- `test_live_speech.py` — **real** calls to edge-tts and the actual Groq
  Whisper API, including a full round-trip test: real text -> real
  synthesized speech -> real transcription, asserting most of the original
  words survive the round trip. The Groq-dependent tests skip cleanly if
  `SPEECH_IO_GROQ_API_KEY` isn't set; the edge-tts tests need no
  credential and always run.

## Deployment

```bash
docker build -t recruitix-speech-io services/speech-io
docker run -p 8000:8000 --env-file .env recruitix-speech-io
```

## Known limitations

- `edge-tts` is an unofficial, reverse-engineered client against a
  Microsoft service — not something to depend on for a real production
  deploy without a fallback plan. It's built as a swappable provider
  specifically so Azure Neural TTS can replace it without a rewrite; that
  swap just hasn't been made yet since it needs an Azure account this
  build doesn't have.
- No audio format conversion — `/transcribe` passes whatever the caller
  uploads straight to Groq, which accepts most common formats
  (flac/mp3/mp4/mpeg/mpga/m4a/ogg/wav/webm) directly, so this has been
  sufficient for browser-recorded audio without needing an `ffmpeg` step.
- Groq's free-tier throughput is the ceiling for STT until a paid tier or
  self-hosted Whisper fallback is needed, same as interview-qa's LLM calls.
- Inbound auth is a single shared secret, not per-user — same posture as
  interview-qa, sufficient until a real API gateway exists.
