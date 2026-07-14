# Recruitix Orchestrator Service

Wires the five independent Recruitix AI/ML services —
[cv-parser](../cv-parser), [interview-qa](../interview-qa),
[speech-io](../speech-io), [answer-grading](../answer-grading), and
[code-eval](../code-eval) — into one coherent interview session: résumé
upload → personal round → HR round → coding round → report. None of the
five services know about each other; this is what turns them into a
product flow.

## What this service actually owns

1. **Session state** — in Redis, reusing the same instance every other
   Recruitix service already depends on for rate limiting. Unlike those
   services (where Redis is optional, with an in-memory fallback), it's
   the entire persistence layer here — there's no fallback mode.
2. **Schema translation** — cv-parser wraps every field in a
   `ConfidentField` (value/confidence/method/source, for its own
   auditability); interview-qa wants plain values. `mapping.py` promotes
   the mapping that was done by hand, repeatedly, during manual
   integration testing before this service existed, into real, tested
   code.
3. **Round progression** — a fixed, configurable sequence: N personal
   questions, M HR questions, then a coding round. Each primary question
   optionally gets exactly one follow-up (`enable_followups`) before
   moving on, exercising both of interview-qa's endpoints coherently
   rather than only ever using one.

## What this service explicitly does not do

- **Author coding problems.** None of the five services generates a
  coding problem with test cases — code-eval only grades a submission
  against test cases someone supplies. The orchestrator accepts a problem
  as input to `/sessions/{id}/code` rather than inventing one.
- **Circuit-break or retry downstream calls.** Each of the five services
  already has its own timeouts and rate limiting. A downstream failure
  surfaces as a clear `502` naming which service failed
  (`DownstreamServiceError`), not a generic retry loop — over-engineering
  for five services this size.
- **Stream audio in real time.** Voice answers are full audio files,
  matching what speech-io already supports (no WebSocket needed).

## Flow

```
POST /api/v1/sessions                     résumé upload -> cv-parser -> mapped context -> first question
POST /api/v1/sessions/{id}/answer         typed answer -> answer-grading -> next question (or follow-up)
POST /api/v1/sessions/{id}/answer/audio   spoken answer -> speech-io transcribe -> (same as above)
POST /api/v1/sessions/{id}/code           source + test cases -> code-eval -> session completed
GET  /api/v1/sessions/{id}/report         full transcript + every score, aggregated
```

## Run locally

Needs all five downstream services and Redis running — see each
service's own README for how to start it, or use
`.claude/launch.json`'s configurations (ports 8000-8003, 8100).

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8004
```

```bash
curl -X POST http://localhost:8004/api/v1/sessions \
  -F "file=@resume.pdf" \
  -F "target_company=Acme"
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/sessions` | résumé upload -> new session + first question |
| `POST /api/v1/sessions/{id}/answer` | typed answer -> score + next question |
| `POST /api/v1/sessions/{id}/answer/audio` | spoken answer -> transcribe, score, next question |
| `POST /api/v1/sessions/{id}/code` | code submission -> code-eval result, completes the session |
| `GET /api/v1/sessions/{id}/report` | full session transcript + aggregate score |
| `GET /health/ready` | readiness — Redis reachable (not gated on all 5 downstream services, see below) |
| `GET /api/v1/capabilities` | configured downstream URLs, default round config |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ORCHESTRATOR_CV_PARSER_BASE_URL` / `_INTERVIEW_QA_BASE_URL` / `_SPEECH_IO_BASE_URL` / `_ANSWER_GRADING_BASE_URL` / `_CODE_EVAL_BASE_URL` | `localhost:8100/8000/8001/8002/8003` | downstream service locations |
| `ORCHESTRATOR_*_API_KEY` (one per downstream service) | empty | shared-secret keys this service presents if that downstream service has its own auth turned on |
| `ORCHESTRATOR_REDIS_URI` | `redis://localhost:6379` | the session store — required, no fallback |
| `ORCHESTRATOR_SESSION_TTL_SECONDS` | `14400` (4h) | how long a session survives in Redis |
| `ORCHESTRATOR_DEFAULT_PERSONAL_QUESTION_COUNT` / `_HR_QUESTION_COUNT` | `3` / `2` | default round sizes, overridable per session |
| `ORCHESTRATOR_DEFAULT_ENABLE_FOLLOWUPS` | `1` | whether each primary question gets one follow-up |
| `ORCHESTRATOR_REQUIRE_API_KEY` | `0` | require an `X-API-Key` header — **this service sits in front of all five others, so leaving it open is strictly worse than leaving any one of them open. Must be turned on before any internet-reachable deploy.** |
| `ORCHESTRATOR_API_KEYS` | empty | comma-separated shared secrets accepted by `X-API-Key` |

## Why readiness only checks Redis

`/health/ready` gates on Redis being reachable — not on all five
downstream services also being up. Gating on every downstream would make
this service's own uptime hostage to any one of theirs going down for
maintenance. A downstream outage instead surfaces per-call, as a `502`
naming exactly which service failed, which is more actionable than a
blanket "orchestrator is down" the moment one of six services blips.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

58 tests across eight files:

- `test_mapping.py` — the cv-parser -> interview-qa schema translation.
- `test_orchestration.py` — round progression, follow-up sequencing, and
  error handling, using mocked downstream clients for speed.
- `test_session_store_live.py` — the real Redis-backed store, against a
  real `redis:7-alpine` container. Skips cleanly if Docker isn't
  available.
- `test_api.py` — HTTP-level tests, mocked clients.
- `test_auth.py` / `test_body_size_limit.py` — same patterns as the other
  five services (the body-size middleware here also needed a
  suffix-matching exemption, not just exact-path, for
  `/sessions/{id}/answer/audio`'s dynamic segment).
- **`test_live_e2e.py`** — the real proof: a full interview session
  through the actual HTTP API, hitting all five real running services and
  a real Redis, nothing mocked. Includes both the text-answer path and a
  full voice round trip (real speech-io synthesis fed back through the
  orchestrator's transcription path). Skips cleanly with a clear reason if
  the full stack isn't up — meant to be run deliberately, not as part of
  the default fast loop. Verified by actually starting all five services
  + Redis and running it for real, not just written and assumed to work.

## Deployment

```bash
docker build -t recruitix-orchestrator services/orchestrator
docker run -p 8004:8000 recruitix-orchestrator
```

## Known limitations

- Round configuration is a fixed sequence (personal → HR → coding), not a
  general-purpose workflow engine — reasonable for the BRD's described
  flow, but not infinitely flexible.
- No retry/circuit-breaking for downstream failures — a transient blip in
  any one service surfaces immediately as a `502`, by design (see above),
  rather than being silently retried.
- Session state has no export/durability beyond Redis's TTL — a
  completed session's report should be copied out (e.g. via
  `GET /report`) before the TTL expires if it needs to persist longer
  than that window.
- Inbound auth is a single shared secret, not per-user — same posture as
  the other five services.
