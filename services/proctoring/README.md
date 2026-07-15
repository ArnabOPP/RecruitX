# Recruitix Proctoring Service

Implements the "Proctoring" row of the Recruitix BRD's AI/ML model table:
MediaPipe gaze/head-pose + event-driven severity scoring, detecting
integrity events during an interview session and feeding an integrity
summary for the orchestrator's final report.

## Core trust principle

Same as biometric-auth: **the client never gets to self-report an
integrity event.** Every snapshot frame is a real image, re-detected and
re-measured by this service — never a client-asserted `looking_away: true`
boolean, which would be trivially forgeable.

## How it works

1. The client (browser) captures one real snapshot frame every N seconds
   during an interview and POSTs it to `/api/v1/proctoring/{session_id}/snapshot`.
2. This service re-detects the face(s) in that frame with MediaPipe
   FaceLandmarker and computes, server-side:
   - **Face count** — zero (candidate left frame) or more than one
     (possible collusion) are both flagged.
   - **Head pose** — via `cv2.solvePnP` against a 6-point 3D face model
     (the same approach, including the same two verified fixes, as
     biometric-auth's liveness module: a Y-down 3D model point convention
     matching pixel space, and the `SOLVEPNP_EPNP` solver rather than the
     default `SOLVEPNP_ITERATIVE`, which was found to converge to a
     spurious ~180-degree-rotated solution).
   - **Gaze** — iris center position relative to each eye's corners/lids
     (MediaPipe's 478-landmark topology includes iris refinement points),
     independent of head pose: a candidate can hold their head still while
     looking away with just their eyes.
3. A frame is only ever a *candidate* flag. An event is recorded — and
   docked from the session's `integrity_score` (starts at 100, floored at
   0) — only once a flagged condition persists for
   `consecutive_frames_to_flag` (default 3) consecutive frames. A single
   flagged frame is just as consistent with a brief natural glance or a
   momentary detection miss as with genuine inattention; requiring
   persistence is what makes an event mean something. A *sustained*
   condition re-fires a fresh event every `consecutive_frames_to_flag`
   frames rather than counting once for the whole session.
4. `MULTIPLE_FACES` is the exception: it is flagged (and recorded)
   immediately, without the debounce window, since a second person
   appearing in frame at all is itself the signal, however briefly.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/proctoring/{session_id}/snapshot` | Analyze one real frame, update the session's running state, return this frame's flags + current integrity score. |
| GET | `/api/v1/proctoring/{session_id}/summary` | Aggregated integrity summary: frames processed, event counts, integrity score, bounded event timeline. |
| DELETE | `/api/v1/proctoring/{session_id}` | Clear a session's proctoring state. |
| GET | `/health/live`, `/health/ready`, `/health` | Standard health/readiness probes. |
| GET | `/api/v1/capabilities` | Reports readiness + active thresholds. |
| GET | `/metrics` | Prometheus metrics. |

All `proctoring` endpoints require `X-API-Key` when
`PROCTORING_REQUIRE_API_KEY=1` — strongly recommended for any
internet-reachable deployment, since an unauthenticated summary endpoint
would let anyone read another candidate's integrity record.

## Storage

Session state (event counts, integrity score, bounded event log) lives in
Redis, TTL'd (`PROCTORING_SESSION_TTL_SECONDS`, default 4 hours) — same
role Redis plays for the orchestrator's own session store. This is
ephemeral, scoped to one interview sitting, unlike biometric-auth's
durable SQLite-backed enrolled embeddings.

## Testing

`tests/` exercises the real MediaPipe model throughout for detection and
head-pose/gaze correctness (including the same real-photo and
known-in-plane-rotation regression checks as biometric-auth's
liveness tests). The event-classification/severity-scoring state machine
is tested at the unit level against a mocked detector to isolate debounce
logic from detection noise. The session store is tested twice: against an
in-memory fake for fast API-level tests, and against a real, throwaway
Dockerized Redis for the store implementation itself (skips cleanly if
Docker isn't available).
