# Recruitix Interview Q&A Generation Service

Implements the "Interview Q&A generation" row of the Recruitix BRD's AI/ML
model table: an instruction-tuned LLM (GPT-4-class / Llama-3 / Gemini-class
per the BRD — this service uses Llama-3.3-70B via Groq) that generates
CV-grounded personal & HR interview questions and follow-ups. It consumes
[cv-parser](../cv-parser)'s résumé output and feeds Round 2 of the
candidate's mock interview.

**Explicitly out of scope**: grading. Per the BRD's design principle — "LLMs
propose and converse; the deterministic engine decides the score" — this
service only ever produces questions, never a verdict on an answer's
quality. A separate deterministic scoring engine (not yet built) owns that.

## Why Groq

Zero budget for this build. Three real options existed:

- **Groq's free tier** (what's wired up): hosted inference, no
  infrastructure to manage, fast, genuinely free within generous rate
  limits. The pragmatic choice for getting a *reliable, browser-accessible*
  service working today.
- **Self-hosted Ollama on a free-tier VM**: "real" self-hosting, but real
  infrastructure work (provisioning, quantization tuning, uptime) — the
  right move once you need full data-locality or outgrow Groq's free
  limits, not a good place to start.
- **Google Colab + ngrok**: fine for prototyping model quality cheaply, but
  Colab's free tier disconnects on idle/reclaim and has no stable public
  URL — unsuitable as the actual backend a live mock-interview session
  depends on.

The LLM provider is not hardcoded: `app/llm/client.py` defines an
`LLMClient` protocol, and `GroqClient` is one implementation of it. Adding
Ollama or a paid provider later means registering a new class in
`get_llm_client()`'s factory — the question-generation logic in `app/qa/`
never touches a provider SDK directly.

## Architecture

```
GenerateQuestionsRequest / FollowUpRequest
  -> qa/prompts.py     builds a résumé brief + grounding instructions for the LLM
  -> llm/client.py       provider-agnostic LLMClient.generate_json()
  -> qa/generator.py      parses/validates the LLM's JSON, retries on
     qa/followup.py       malformed output, degrades individual bad
                           questions rather than failing the whole request
  -> main.py               FastAPI: POST /api/v1/questions/generate
                                     POST /api/v1/questions/followup
```

`ResumeContext` (in `qa/schemas.py`) is this service's own input contract —
it does not import cv-parser's `ParsedResume` directly. The two are separate
microservices; whatever orchestrates a candidate's session maps cv-parser's
output onto this narrower shape, which keeps each service independently
deployable.

## Run locally

```bash
pip install -r requirements-dev.txt
cp .env.example .env   # then put your real Groq key in .env — get one free at console.groq.com
uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/api/v1/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "resume": {
      "full_name": "Jordan Lee",
      "skills": [{"name": "Python", "evidenced_in_project": true}],
      "projects": [{"title": "TaskTracker", "description": "A task app.", "tech_stack": ["Python", "FastAPI"]}]
    },
    "target_company": "Acme",
    "round": "personal",
    "count": 3
  }'
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | liveness — process is up |
| `GET /health/ready` | readiness — LLM client configured (503 if e.g. no API key) |
| `GET /api/v1/capabilities` | provider, model, readiness |
| `POST /api/v1/questions/generate` | résumé + round + company → grounded questions |
| `POST /api/v1/questions/followup` | question + candidate answer → contextual follow-up |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | interactive Swagger UI |

## What "grounded" actually means here

Every generated question carries a `grounding` object: which résumé item
(`kind` + `reference`) it's anchored to, and the specific `detail` that
justified asking it. The prompt in `qa/prompts.py` explicitly separates
skills the résumé shows being *used* (in a project or role) from skills that
are merely *listed* — the former get concrete "how did you implement X"
questions, the latter get `resume_gap_probe` questions checking depth.
Follow-ups work the same way: the prompt is given the résumé alongside the
candidate's answer and asked to probe anything the answer mentions that
*isn't* on the résumé, framed as curiosity rather than an accusation. This
is the mechanism directly verified in testing — see `tests/test_live_groq.py`.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `INTERVIEW_QA_GROQ_API_KEY` | *(required)* | Groq API key — free at console.groq.com |
| `INTERVIEW_QA_GROQ_MODEL` | `llama-3.3-70b-versatile` | any Groq-hosted model |
| `INTERVIEW_QA_LLM_TEMPERATURE` | `0.6` | generation randomness |
| `INTERVIEW_QA_LLM_MAX_RETRIES` | `2` | retries on malformed JSON / provider error |
| `INTERVIEW_QA_MAX_QUESTIONS_PER_REQUEST` | `10` | cap on `count` |
| `INTERVIEW_QA_RATE_LIMIT_PARSE` \* | `20/minute` | see below |
| `INTERVIEW_QA_RATE_LIMIT_STORAGE_URI` | unset | e.g. `redis://host:6379` to share the limit across replicas |
| `INTERVIEW_QA_CORS_ALLOW_ORIGINS` | empty | comma-separated allowlist; empty denies all cross-origin |
| `INTERVIEW_QA_LOG_LEVEL` / `INTERVIEW_QA_LOG_JSON` | `INFO` / `1` | logging |

\* actual var name is `INTERVIEW_QA_RATE_LIMIT_GENERATE`

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

27 tests across four files:

- `test_generator.py` / `test_followup.py` — unit tests against a fake LLM
  client (scripted responses, including malformed JSON to exercise the
  retry path) — fast, no network, no API costs.
- `test_api.py` — HTTP-level tests with the LLM client mocked at the import
  site each module actually uses it.
- `test_live_groq.py` — **real** calls to the actual Groq API, verifying
  generated questions genuinely reference résumé-specific terms and that
  follow-ups react to answer content. Skips cleanly if
  `INTERVIEW_QA_GROQ_API_KEY` isn't set (e.g. in CI), the same pattern
  cv-parser uses for its Tesseract/Docker-gated tests.

## Deployment

```bash
docker build -t recruitix-interview-qa services/interview-qa
docker run -p 8000:8000 --env-file .env recruitix-interview-qa
```

271MB image (no ML models baked in, unlike cv-parser) — built, run, and
hit with real HTTP requests against the real Groq API as part of verifying
this service, not just reviewed.

## Known limitations

- Groq's free tier has rate limits that aren't under this service's control
  — a real traffic spike would need either a paid Groq tier or the
  self-hosted Ollama path described above.
- No caching of generated questions — every request is a fresh LLM call.
  Fine for the current use case (each candidate's session is unique), but
  worth revisiting if cost/latency becomes a concern at scale.
- The "probes unevidenced skills" behavior is a property of the prompt and
  the model's instruction-following, not a hard guarantee — `test_live_groq.py`
  checks it's *plausible*, not deterministic, since it's calling a real LLM.
- Rate limiting is in-memory by default (single-instance only), same
  Redis-backed opt-in pattern as cv-parser via `INTERVIEW_QA_RATE_LIMIT_STORAGE_URI`.
