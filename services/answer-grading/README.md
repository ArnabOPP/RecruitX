# Recruitix Answer Grading Service

Implements the "Answer grading" row of the Recruitix BRD's AI/ML model
table: a **deterministic engine** — Jaccard / semantic similarity +
keyword weighting + rubric — producing reproducible, auditable scores for
open text and transcripts. This is the counterpart to
[interview-qa](../interview-qa)'s own design principle: *"LLMs propose and
converse; the deterministic engine decides the score."* Nothing in this
service calls an LLM. Given the same question, answer, and rubric, it
always produces the same score — reproducibility is the entire point.

## Why deterministic, not an LLM judge

An LLM grading answers would score differently across runs (no fixed
seed/temperature=0 guarantee across providers) and can't cleanly explain
*why* it gave a score beyond a generated sentence. This service instead
combines three classical techniques into a rubric-based score with a full
audit trail — every number in the response is traceable back to specific
matched/missing keywords and a similarity score, not an LLM's opinion.

## The three techniques

1. **Keyword weighting** — expected terms come from a rubric criterion,
   either supplied by the caller or auto-derived from interview-qa's
   `grounding` object (the résumé evidence a question was anchored to).
2. **Jaccard similarity** — token-set overlap between the expected
   keywords and the candidate's answer. Cheap, exact, zero ML.
3. **Semantic similarity** — a self-hosted sentence-embedding model
   (`all-MiniLM-L6-v2`, same self-hosted-model philosophy as cv-parser's
   spaCy/BERT pipeline) catches paraphrases Jaccard can't: "sped up the
   database" scores highly against "optimized the query" despite sharing
   no literal words. Embeddings are a deterministic function of fixed
   model weights — no sampling — which is what makes this safe to use in
   a "reproducible" engine.

Each rubric criterion's score is `jaccard_weight * jaccard + semantic_weight * semantic`
(configurable, default 0.4/0.6 — semantic weighted higher since it's
strictly more informative when it disagrees with Jaccard). The overall
score is a weight-normalized average across all criteria.

## Rubric resolution

```
caller supplies an explicit rubric?  -> use it verbatim               (rubric_source: "provided")
  no rubric, but grounding supplied? -> derive one criterion from it   (rubric_source: "auto_derived_from_grounding")
  neither supplied?                  -> derive one from the question   (rubric_source: "auto_derived_from_question")
```

When deriving from `grounding`, keywords come from `detail` (the
substantive technical content, e.g. "reducing report generation time by
40% using indexes") when present — not `reference`, which for
kind=experience/education is often just an org or job title ("Software
Engineering Intern @ Flipkart") a good answer has no reason to repeat.
`reference` is only used as the keyword source when there's no `detail` to
draw on (e.g. a bare skill name).

## How it fits the pipeline

```
interview-qa generates a question (with grounding)
  -> candidate answers (typed, or transcribed via speech-io — this
     service doesn't care which)
  -> answer-grading /score -> a score + full audit breakdown
```

Not yet wired together automatically — same manual-handoff situation as
the other three services.

## Run locally

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

No `.env` needed — this service calls no external API.

```bash
curl -X POST http://localhost:8000/api/v1/grading/score \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How did you optimize PostgreSQL queries in your inventory reports project?",
    "candidate_answer": "I noticed the reports were slow due to full table scans, so I added indexes on the WHERE-clause columns, which cut generation time significantly.",
    "grounding": {
      "kind": "experience",
      "reference": "Software Engineering Intern @ Flipkart",
      "detail": "Optimized PostgreSQL queries, reducing report generation time by 40%"
    }
  }'
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/grading/score` | question + candidate answer (+ optional grounding/rubric) -> score with full breakdown |
| `GET /health/live` | liveness — process is up |
| `GET /health/ready` | readiness — embedding model loaded and confirmed working |
| `GET /api/v1/capabilities` | model, readiness, configured weights |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | interactive Swagger UI |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANSWER_GRADING_SEMANTIC_MODEL_NAME` | `all-MiniLM-L6-v2` | any sentence-transformers model |
| `ANSWER_GRADING_VALIDATE_MODEL_ON_STARTUP` | `1` | runs one real inference at boot to confirm the model actually works, and warms it up |
| `ANSWER_GRADING_JACCARD_WEIGHT` / `ANSWER_GRADING_SEMANTIC_WEIGHT` | `0.4` / `0.6` | how each criterion's Jaccard and semantic scores combine |
| `ANSWER_GRADING_MAX_QUESTION_CHARS` | `2000` | caps `question` |
| `ANSWER_GRADING_MAX_ANSWER_CHARS` | `4000` | caps `candidate_answer` |
| `ANSWER_GRADING_MAX_RUBRIC_CRITERIA` | `10` | caps criteria in a caller-supplied rubric |
| `ANSWER_GRADING_MAX_KEYWORDS_PER_CRITERION` | `20` | caps `expected_keywords` per criterion |
| `ANSWER_GRADING_MAX_REQUEST_BODY_BYTES` | `512000` | request body size cap via `Content-Length`, before parsing |
| `ANSWER_GRADING_REQUIRE_API_KEY` | `0` | require an `X-API-Key` header — protects CPU resources (the embedding model), not a paid quota, since there's no external API here. **Must be turned on before any internet-reachable deploy** |
| `ANSWER_GRADING_API_KEYS` | empty | comma-separated shared secrets accepted by `X-API-Key` |
| `ANSWER_GRADING_RATE_LIMIT_STORAGE_URI` | unset | e.g. `redis://host:6379` to share rate limits across replicas |
| `ANSWER_GRADING_CORS_ALLOW_ORIGINS` | empty | comma-separated allowlist; empty denies all cross-origin |

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

56 tests across seven files. Notably, **nothing here needs a skip
condition for missing credentials** — everything is local compute, so
every test (including the ones exercising the real embedding model)
always runs, in CI and locally alike:

- `test_keywords.py` — pure tokenization/Jaccard/keyword-coverage math, no
  model involved.
- `test_scorer.py` — rubric resolution and score combination logic, using
  a fake semantic scorer for speed/determinism.
- `test_semantic.py` — the **real** embedding model: identical text scores
  near 1.0, a genuine paraphrase scores meaningfully higher than an
  unrelated sentence, similarity is symmetric and reproducible.
- `test_api.py` — HTTP-level tests with the scorer mocked for speed, plus
  one full-stack test (`test_full_stack_scoring_with_real_model`) that
  mocks nothing: a real paraphrased good answer through the real HTTP
  endpoint scores higher than a real irrelevant one.
- `test_auth.py` / `test_body_size_limit.py` / `test_rate_limit_redis.py`
  — same patterns as interview-qa/speech-io. The Redis test needs Docker
  and skips cleanly if it's unavailable.

## Deployment

```bash
docker build -t recruitix-answer-grading services/answer-grading
docker run -p 8000:8000 recruitix-answer-grading
```

The embedding model weights are baked into the image at build time (same
pattern as cv-parser's transformer pre-bake) so no network call is needed
on container startup.

## Known limitations

- The auto-derived rubric (when no explicit one is supplied) is a
  reasonable default, not a substitute for real rubric authoring — a
  caller who cares about grading quality should supply an explicit
  `rubric` with criteria and weights tuned to what actually matters for
  that question.
- `all-MiniLM-L6-v2` is small and fast, not the highest-quality embedding
  model available — a larger model would likely improve semantic
  discrimination at the cost of latency/memory; this wasn't the priority
  given the zero-budget, self-hosted constraint.
- Semantic similarity between short phrases (a criterion description +
  keywords vs. a short answer) is inherently noisier than comparing two
  full sentences — scores should be read as relative signals (this answer
  vs. that one) more than absolute ground truth.
- Inbound auth is a single shared secret, not per-user — same posture as
  the other services, sufficient until a real API gateway exists.
