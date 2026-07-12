# Recruitix CV Parsing Service

Implements FR-08 / FR-09 and the "CV parsing NLP pipeline — spaCy NER +
transformer (BERT-class) résumé parser" line item from the Recruitix BRD's
AI & ML Model Architecture section. Extracts structured contact info,
skills, education, experience, projects, and certifications from an
uploaded résumé (PDF / DOCX / TXT), so Round 2 (CV-driven personal
interview) can generate evidence-grounded questions.

## Architecture

```
file bytes
  -> extractor.py     layout-aware text extraction (pdfplumber / python-docx)
                        + magic-byte signature validation
  -> sections.py       header detection + fuzzy section segmentation
  -> ner.py             hybrid NER: spaCy gazetteer (skills/degrees) + statistical
                         NER, ensembled with a BERT-class transformer (dslim/bert-base-NER)
  -> contact/education/experience/projects/certifications/skills.py
                         per-section field extractors, each field carries a
                         confidence score + extraction method
  -> pipeline.py         orchestrator -> ParsedResume (Pydantic schema)
  -> main.py              FastAPI: POST /api/v1/parse
```

Every extracted field is a `ConfidentField { value, confidence, method,
source }` — not just a bare string — so downstream consumers (and a human
reviewer) can see *why* a value was extracted and how much to trust it.
This is the same explainability principle the BRD applies to scoring
("every deduction is logged").

### Why hybrid spaCy + transformer, not either alone

- A pure gazetteer (spaCy `PhraseMatcher`) has perfect precision on known
  vocabulary (skills, degrees) but zero recall on names/orgs it's never
  seen.
- A pure transformer NER model generalizes to unseen names/orgs but has no
  concept of "this is a résumé skill taxonomy" and is comparatively slow.
- Ensembling: gazetteer matches always win (curated vocabulary beats a
  generic guess on domain terms — e.g. spaCy's statistical model tags
  "B.Tech" as ORG; the gazetteer correctly tags it DEGREE and evicts the
  statistical guess). For PERSON/ORG, when spaCy's statistical NER and the
  transformer agree, confidence is boosted; when only one fires, it's used
  at a discount.

The transformer is optional; at startup (not on first request — see
"Production hardening" below) the engine tries to load it, and if model
weights can't be fetched (offline environment) degrades to spaCy-only with
a logged warning rather than failing the whole service.

## Run locally

```bash
pip install -r requirements-dev.txt   # dev deps (test/lint/typecheck) + runtime deps
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload
```

```bash
curl -F "file=@resume.pdf" http://localhost:8000/api/v1/parse
```

For a production deploy, install `requirements.txt` only (no dev/test tooling
in the image) — see `Dockerfile`.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | liveness — process is up |
| `GET /health/ready` | readiness — models are loaded, service can actually parse (503 until then) |
| `GET /health` | back-compat alias for `/health/live` |
| `GET /api/v1/capabilities` | supported file types, model names, whether the transformer loaded |
| `POST /api/v1/parse` | multipart file upload (PDF/DOCX/TXT) → `ParsedResume` JSON |
| `GET /metrics` | Prometheus metrics (request counts/latency histograms) |
| `GET /docs` | interactive Swagger UI |

## Production hardening

This isn't just a script behind FastAPI — the things that tend to bite in
production are handled explicitly:

- **Fast startup, not first-request latency**: spaCy + the transformer load
  once during the app's `lifespan`, before the process accepts traffic.
  `/health/ready` doesn't return 200 until that's done, so a rolling deploy
  or k8s readiness probe won't route traffic to a cold replica.
- **Non-blocking under load**: NER inference is CPU-bound, so `/api/v1/parse`
  offloads it to a threadpool (`run_in_threadpool`) instead of running it
  inline on the async event loop — one slow parse doesn't stall every other
  in-flight request. A configurable timeout (`CV_PARSER_REQUEST_TIMEOUT_SECONDS`,
  default 30s) returns `504` rather than hanging forever on a pathological
  document.
- **Consistent error contract**: every error — validation, unsupported file,
  rate limit, internal exception — returns `{error, detail, request_id}`.
  Internal exception messages/tracebacks are never put in the response body;
  they go to logs only, keyed by the same `request_id` so you can correlate
  a user-reported failure to the exact log line.
- **Structured JSON logs + request correlation**: every log line carries a
  `request_id` (generated per request, echoed back as the `X-Request-ID`
  response header, or reused if the caller supplied one). Résumé contents
  are personal data and are never logged — only metadata (filename, size,
  status, latency).
- **Defense in depth on uploads**: extension is checked, then the file's
  magic bytes are checked against that extension (catches a renamed
  `.exe`/`.jpg` claiming to be a `.pdf`), then size is checked both from the
  `Content-Length` header (fast-fail before reading the body) and the
  actual read bytes.
- **Rate limiting, shareable across replicas**: `/api/v1/parse` is limited
  per-client-IP (default `20/minute`, configurable). By default this uses
  slowapi's in-memory store, which only enforces the limit *per process* —
  fine for one instance, but a client fanning requests across N replicas
  behind a load balancer would get N× the intended budget. Set
  `CV_PARSER_RATE_LIMIT_STORAGE_URI=redis://host:6379` to back it with Redis
  instead, so every replica shares one counter. See "Verifying the
  Redis-backed rate limiter" below — this isn't just documented, it's been
  proven against a real multi-process setup.
- **OCR fallback for scanned/image-only PDFs**: if pdfplumber finds no text
  layer at all (a résumé that was scanned from paper, or exported as a
  flattened image), the extractor rasterizes each page with `pypdfium2` and
  runs Tesseract over it — up to `CV_PARSER_OCR_MAX_PAGES` pages (default
  10), so one huge scanned deck can't hang a request. Like the transformer,
  this is fail-soft: if Tesseract isn't installed, OCR is skipped with a
  logged warning instead of crashing. The production Docker image installs
  `tesseract-ocr` so this actually works in a real deploy, not just local
  dev where it happens to already be on your PATH.
- **Metrics**: Prometheus-format request counts and latency histograms at
  `/metrics`, via `prometheus-fastapi-instrumentator`.
- **CORS is deny-by-default**: `CV_PARSER_CORS_ALLOW_ORIGINS` is empty
  unless you explicitly list allowed origins — no wildcard `*` in production.

## Configuration

All settings live in `app/config.py` (`pydantic-settings`), env-prefixed
`CV_PARSER_`. Wrong types fail fast at startup rather than mid-request.

| Env var | Default | Purpose |
|---|---|---|
| `CV_PARSER_SPACY_MODEL` | `en_core_web_sm` | swap to `en_core_web_trf` for higher-accuracy statistical NER |
| `CV_PARSER_TRANSFORMER_MODEL` | `dslim/bert-base-NER` | any HF token-classification model |
| `CV_PARSER_ENABLE_TRANSFORMER` | `1` | set `0` to force spaCy-only (faster, no model download) |
| `CV_PARSER_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | upload size limit |
| `CV_PARSER_REQUEST_TIMEOUT_SECONDS` | `30` | per-request parse timeout before returning 504 |
| `CV_PARSER_CORS_ALLOW_ORIGINS` | `` (empty) | comma-separated allowed origins; empty = deny all cross-origin |
| `CV_PARSER_RATE_LIMIT_ENABLED` | `1` | set `0` to disable rate limiting (e.g. in tests) |
| `CV_PARSER_RATE_LIMIT_PARSE` | `20/minute` | slowapi rate-limit string for `/parse` |
| `CV_PARSER_RATE_LIMIT_STORAGE_URI` | unset | e.g. `redis://host:6379` — share the rate limit across replicas; unset = in-memory (single instance only) |
| `CV_PARSER_ENABLE_OCR` | `1` | set `0` to skip OCR entirely for scanned/image-only PDFs |
| `CV_PARSER_OCR_DPI` | `200` | resolution used to rasterize pages before OCR — higher improves accuracy at the cost of speed |
| `CV_PARSER_OCR_MAX_PAGES` | `10` | cap on pages OCR'd per document (perf/DoS guard for huge scanned files) |
| `CV_PARSER_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |
| `CV_PARSER_LOG_JSON` | `1` | set `0` for human-readable log lines instead of JSON |
| `CV_PARSER_METRICS_ENABLED` | `1` | set `0` to disable `/metrics` |
| `CV_PARSER_ENVIRONMENT` | `development` | `development`/`staging`/`production` (informational) |

## Tests

```bash
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
pytest tests/ -v
```

44 tests across four files:

- `test_pipeline.py` — contact/skills/education/experience/projects/
  certifications extraction against a realistic fixture résumé.
- `test_extractor.py` — drives the actual PDF and DOCX code paths (built at
  test time with `reportlab`/`python-docx`, not just the `.txt` fixture),
  file-signature-mismatch and unsupported-type error paths, and the OCR
  fallback against a synthetic *scanned* PDF (text rendered to an image,
  embedded with no text layer — genuinely exercises Tesseract, not mocked).
- `test_api.py` — HTTP-level: health/readiness, capabilities, metrics,
  request-ID propagation, `/parse` success and every error status code
  (400/413/415/422), and that error responses never leak a traceback.
- `test_rate_limit_redis.py` — spins up a throwaway `redis:7-alpine`
  container and proves two independent storage connections (standing in for
  two app replicas) share hit counts; skips cleanly if Docker isn't reachable.

By default the transformer is disabled in tests (`conftest.py`) so the suite
runs in ~5s; override with `CV_PARSER_ENABLE_TRANSFORMER=1 pytest tests/` to
also exercise the BERT ensemble path (~15s, needs the model cached/downloadable).
The OCR test skips if Tesseract isn't installed on the host; the Redis test
skips if Docker isn't reachable — both fail-soft the same way their
production code paths do.

### Verifying the Redis-backed rate limiter yourself

The unit test above proves the storage layer shares state; to see it happen
through real HTTP requests across two actual server processes:

```bash
docker run -d --name redis-demo -p 6379:6379 redis:7-alpine

export CV_PARSER_ENABLE_TRANSFORMER=0
export CV_PARSER_RATE_LIMIT_STORAGE_URI=redis://localhost:6379
export CV_PARSER_RATE_LIMIT_PARSE=5/minute
uvicorn app.main:app --port 8041 &
uvicorn app.main:app --port 8042 &

# Alternate requests between the two "replicas" — the 6th request across
# BOTH of them (not the 6th on either one individually) should 429.
for i in 1 2 3 4 5 6 7 8; do
  port=$((8041 + (i % 2)))
  curl -s -o /dev/null -w "request $i -> port $port -> HTTP %{http_code}\n" \
    -F "file=@tests/fixtures/sample_resume.txt" "http://127.0.0.1:${port}/api/v1/parse"
done
```

### Lint / type-check

```bash
python -m ruff check app/
python -m mypy app/
```

Both run in CI (`.github/workflows/cv-parser-ci.yml`) on every push/PR that
touches this service, followed by the test suite and a Docker build. This
workflow has been executed end-to-end locally with [`act`](https://github.com/nektos/act)
(a real Docker-based GitHub Actions runner, not just read/reviewed) — it
caught and fixed a genuine path-doubling bug in the `docker-build` job's
build context (`defaults.run.working-directory` plus a redundant path
segment) that would have silently broken that job on first real push.

## Deployment

Single instance:

```bash
docker build -t recruitix-cv-parser services/cv-parser
docker run -p 8000:8000 recruitix-cv-parser
```

Multiple replicas with a shared rate limiter (nginx load balancer + Redis):

```bash
docker compose up --build
# hits go to nginx on :8080, load-balanced across cv-parser-a and cv-parser-b,
# which share one rate-limit counter via the redis service
```

The image runs as a non-root user, and its `HEALTHCHECK` hits
`/health/ready` (not just liveness) so it isn't marked healthy until models
have actually finished loading. Scale by adding container replicas behind a
load balancer, not by raising `uvicorn --workers` — each worker process
would load its own copy of the transformer model into memory.

The image has been built and run end-to-end in this environment: a real
`docker build`, a real `docker run`, and live `curl` calls against
`/health/ready`, `/api/v1/capabilities` (confirming `transformer_available:
true`), and `/api/v1/parse` inside the running container — not just a
Dockerfile that looks plausible on paper.

## Known limitations / next steps

- Section segmentation assumes reasonably conventional résumé structure;
  highly designed/graphic résumés with no text headers will under-segment.
- `en_core_web_sm` is used by default for speed; swapping to
  `en_core_web_trf` (spaCy's own transformer pipeline) improves PERSON/ORG
  recall further at the cost of latency — worth an A/B on real résumé
  volume before Phase 2.
- Org/company extraction can occasionally trim a legitimate acronym prefix
  (e.g. "IEMA RND Pvt. Ltd." → "RND Pvt. Ltd.") when the statistical NER
  model doesn't recognize the acronym — a known precision/recall tradeoff,
  not yet resolved.
- OCR accuracy depends entirely on scan quality; Tesseract's general English
  model isn't fine-tuned on résumés specifically, so a low-quality scan will
  produce noisier text than a native-text PDF. `CV_PARSER_OCR_DPI` trades
  speed for accuracy if needed.
- Redis-backed rate limiting fails open if Redis is unreachable
  (`in_memory_fallback_enabled=True`) — availability over strict enforcement,
  since a rate limiter outage shouldn't take the whole API down. Worth
  alerting on if it happens.
