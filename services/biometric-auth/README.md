# Recruitix Biometric Auth Service

Implements three rows of the Recruitix BRD's AI/ML model table in one
service, because all three are used together at the same two checkpoints
— enrollment and login — and share the same underlying face-detection
code:

1. **Face detection & landmarks** — MediaPipe FaceLandmarker (468 base +
   10 iris landmarks) + OpenCV, used to locate, validate (exactly one
   clear face), and align a face to a normalized 224x224 crop.
2. **Face recognition** — deep metric learning via ArcFace
   (`insightface`'s `buffalo_l` pack, 512-D embeddings). A face becomes a
   vector; identity is decided by cosine similarity against a stored
   enrollment vector.
3. **Liveness / anti-spoofing** — Eye-Aspect-Ratio blink detection +
   head-pose (solvePnP) heuristics, computed across a short burst of real
   frames, to distinguish a live human from a photo or video replay.

## Core trust principle

Carried over from the rest of Recruitix: **the client never gets to
self-report a result that gates a real decision.** A browser-computed
`isLive: true` or `match: true` boolean is trivially forgeable with a raw
HTTP request that never touched a camera. Every endpoint here receives
actual image frames and computes its own verdict server-side:

- `/enroll` and `/verify` receive real JPEG crops and compute the ArcFace
  embedding and cosine-similarity match themselves.
- `/liveness` receives a real short burst of frames (~15+) and computes
  EAR/head-pose itself — never a client-asserted liveness flag.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/biometric/enroll` | Enroll a candidate: averages the embedding across >= `min_enrollment_images` real photos. |
| POST | `/api/v1/biometric/verify` | Verify a candidate: fresh embedding vs. the stored one, cosine similarity >= `match_similarity_threshold`. |
| DELETE | `/api/v1/biometric/enroll/{candidate_id}` | Remove a candidate's enrollment. |
| POST | `/api/v1/biometric/liveness` | Server-computed blink + head-pose liveness verdict across a real frame burst. |
| GET | `/health/live`, `/health/ready`, `/health` | Standard health/readiness probes. |
| GET | `/api/v1/capabilities` | Reports readiness + active thresholds. |
| GET | `/metrics` | Prometheus metrics. |

All `biometric` endpoints require `X-API-Key` when
`BIOMETRIC_AUTH_REQUIRE_API_KEY=1` — strongly recommended for any
internet-reachable deployment, since an unauthenticated `/verify` would
let anyone probe whether a face matches a given `candidate_id`.

## Storage

A single SQLite file (`BIOMETRIC_AUTH_DB_PATH`, default
`data/biometric.db`) holds `candidate_id -> embedding` mappings. This is
the first *durable* store in Recruitix — every other service is stateless
or Redis-TTL'd ephemeral session state — because an enrolled face
embedding is reference data that must survive indefinitely, not expire
with a session. A single file needs no additional infrastructure; Postgres
is the natural upgrade path if this ever needs concurrent-writer scale.

## Liveness design notes

- A blink is counted as an EAR open -> closed -> open *transition*, not a
  single low-EAR frame, since one low-EAR frame is just as consistent with
  detection noise as with an actual blink.
- Head pose uses `cv2.solvePnP` with `SOLVEPNP_EPNP` against a 6-point
  generic 3D face model whose Y-axis matches image pixel-space convention
  (Y increasing downward). An earlier iteration used a Y-up model + the
  default `SOLVEPNP_ITERATIVE` solver, which converged to a spurious
  ~180-degree-rotated solution for a straightforward frontal face — caught
  and fixed via `tests/test_liveness.py`'s regression tests, which check
  both a real frontal photo and known in-plane rotations.
- `liveness_min_frames` (default 15) is deliberately generous: natural
  blink rate is roughly one every 3-4 seconds, so a short burst has a real
  chance of missing a genuine blink and producing a false rejection.

## Vision model notes

- MediaPipe's newer Tasks API (`mediapipe.tasks.python.vision`) is used,
  not the legacy `mp.solutions.face_mesh` most tutorials reference — the
  pinned mediapipe version no longer exposes `mp.solutions` at all. Needs
  a downloaded `.task` model bundle (baked into the Docker image at build
  time; see Dockerfile).
- ArcFace embeddings are computed from the **original** submitted image,
  not the MediaPipe-aligned crop — insightface's own internal detector
  fails on a tightly-cropped 224x224 alignment output. The MediaPipe
  alignment step is used only for single-face validation and liveness/pose
  math.
- Both models are pre-baked into the Docker image at build time (same
  pattern as cv-parser's transformer and answer-grading's embedding model)
  so no container's first request pays a cold model-download cost.

## Testing

`tests/` exercises the real MediaPipe + ArcFace pipeline throughout — no
mocking of the vision models, since a biometric match only means something
if it's proven against the real models. The one exception is
`test_liveness.py`'s blink-state-machine tests, which mock face detection
with synthetic landmark sequences to isolate the state machine from
detection noise; head-pose correctness is separately verified against a
real photo and known in-plane rotations in the same file.
