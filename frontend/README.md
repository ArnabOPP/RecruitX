# Recruitix Frontend

The candidate-facing interview flow: enroll/verify by face, answer
personal/HR round questions by text or voice, solve a coding problem,
and see a final report — all driven by the eight Recruitix backend
services (see `../services`).

## Architecture

Next.js App Router, with all backend calls proxied through server-side
Route Handlers under `src/app/api/` (a BFF layer) rather than called
directly from the browser. This means:

- The browser only ever talks to this Next.js server — **no CORS
  configuration is needed on the orchestrator or biometric-auth
  services at all**, since server-to-server calls aren't subject to
  browser CORS policy.
- Backend base URLs and API keys (`ORCHESTRATOR_BASE_URL`,
  `BIOMETRIC_AUTH_BASE_URL`, `ORCHESTRATOR_API_KEY`,
  `BIOMETRIC_AUTH_API_KEY`) are server-only env vars, never shipped to
  the browser bundle.

No authentication layer exists yet — sessions are identified purely by
the `session_id` the orchestrator returns from `POST /api/v1/sessions`,
matching where the backend is today (no user-account concept anywhere
in the eight services).

## Flow

```
/                        candidate ID + résumé + optional face enroll/verify -> creates a session
/interview/[id]          current question (text or voice answer) + a background proctoring snapshot timer
/interview/[id]/report   final transcript, scores, and proctoring summary
```

The orchestrator has no "get current session state" endpoint — only
`/answer`, `/code`, and the initial `/sessions` response carry the
current question. `src/lib/session-cache.ts` persists that to
`sessionStorage` so the interview page survives a refresh with its
question intact; it's a UI convenience, not a source of truth (the
backend's Redis-held session state is that).

## Camera and microphone

`src/hooks/useWebcam.ts` and `src/hooks/useAudioRecorder.ts` wrap
`getUserMedia`/`MediaRecorder` directly — every biometric verdict
(enrollment match, liveness, proctoring events) is computed server-side
from real captured frames, matching the "the client never gets to
self-report a result that gates a real decision" principle the whole
platform is built around. Nothing here computes or asserts a match/live
verdict client-side.

## Run locally

Needs the full backend stack running (see the repo root's
`docker-compose.yml`, or start each service individually via
`.claude/launch.json`) and a `.env.local` (copy `.env.local.example`):

```bash
npm install
npm run dev
```

## Known limitations

- No account system — a `candidate_id` typed into the landing page is
  the only identity concept, matching the backend.
- Refreshing mid-interview loses the current question (see "Flow"
  above) — the candidate can still view whatever's been recorded via
  the report page.
- The coding round's editor is a plain textarea, not a syntax-highlighted
  editor — a deliberate scope decision to ship a simpler v1 first.
