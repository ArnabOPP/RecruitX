"""The real proof: a full interview session through the orchestrator's
real HTTP API, hitting all seven actual running services (cv-parser,
interview-qa, speech-io, answer-grading, code-eval, biometric-auth,
proctoring) and a real Redis — nothing mocked anywhere in this file. This
is the equivalent of every other service's "live" test file, scaled up to
an integration across the whole platform.

Skips cleanly (with a clear reason) if the full stack isn't running —
this is meant to be run deliberately, with all eight services (plus
Redis) started, not as part of the default fast test loop.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings

_TEST_RESUME_TEXT = b"""Jordan Lee
jordan.lee@email.com

SUMMARY
Computer Science undergraduate specializing in backend systems.

SKILLS
Languages: Python, JavaScript
Frameworks: FastAPI, React
Databases: PostgreSQL

PROJECTS
TaskTracker (Python, FastAPI, PostgreSQL)
2025
Built a full-stack task management app with a REST API backend and
PostgreSQL for persistence. Implemented query optimization using indexes
to reduce report generation time.

EXPERIENCE
Software Engineering Intern at Acme Corp
Jun 2025 - Aug 2025
- Built internal tooling in Python to automate reports.
- Optimized PostgreSQL queries, reducing report generation time by 40%.
"""


def _all_services_reachable() -> bool:
    settings = get_settings()
    urls = [
        f"{settings.cv_parser_base_url}/health/live",
        f"{settings.interview_qa_base_url}/health/live",
        f"{settings.speech_io_base_url}/health/live",
        f"{settings.answer_grading_base_url}/health/live",
        f"{settings.code_eval_base_url}/health/live",
        f"{settings.biometric_auth_base_url}/health/live",
        f"{settings.proctoring_base_url}/health/live",
    ]
    for url in urls:
        try:
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code != 200:
                return False
        except httpx.RequestError:
            return False
    try:
        import redis

        client = redis.from_url(settings.redis_uri)
        client.ping()
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _all_services_reachable(),
    reason="Not all seven downstream services + Redis are reachable — start the full stack to run this test.",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app import session_store
    from app.main import app

    # session_store's Redis client is a module-level singleton, bound to
    # whichever event loop was running when it was first created — correct
    # for a real deployment (one process, one event loop, for the server's
    # whole lifetime) but wrong across separate TestClient instances in one
    # pytest session, each of which gets its own event loop. Reset it so
    # every test's client is bound to *that* test's loop, not a previous
    # (by then closed) one.
    session_store._redis_client = None
    session_store._session_store = None

    with TestClient(app) as c:
        yield c


def test_full_stack_readiness(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200


def test_full_interview_session_end_to_end(client):
    """Uploads a real résumé, answers real generated questions (grading
    against real answer-grading), reaches the coding round, submits real
    code (graded by real code-eval), and confirms the final report
    aggregates everything correctly — end to end, nothing mocked."""

    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.txt", _TEST_RESUME_TEXT, "text/plain")},
        params={
            "target_company": "Acme",
            "personal_question_count": 1,
            "hr_question_count": 0,
            "enable_followups": True,
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    body = create_resp.json()
    session_id = body["session_id"]
    assert body["round"] == "personal"
    assert body["question"] is not None
    first_question = body["question"]["text"]
    assert len(first_question) > 0

    # Answer the primary question with something genuinely on-topic,
    # referencing a real detail from the résumé, so answer-grading has
    # something real to score.
    answer_resp = client.post(
        f"/api/v1/sessions/{session_id}/answer",
        json={"answer_text": "I optimized the PostgreSQL queries by adding indexes, which reduced report generation time significantly."},
    )
    assert answer_resp.status_code == 200, answer_resp.text
    answer_body = answer_resp.json()
    assert 0.0 <= answer_body["score"]["overall_score"] <= 1.0
    assert answer_body["status"] == "in_progress"
    assert answer_body["next_question"] is not None
    assert answer_body["next_question"]["category"] == "followup"

    # Answer the follow-up too -> should advance past the personal round
    # (count=1) straight to awaiting_code (hr_question_count=0).
    followup_answer_resp = client.post(
        f"/api/v1/sessions/{session_id}/answer",
        json={"answer_text": "I used EXPLAIN ANALYZE to find the slow queries first, then added targeted indexes."},
    )
    assert followup_answer_resp.status_code == 200, followup_answer_resp.text
    assert followup_answer_resp.json()["status"] == "awaiting_code"

    # Submit a real, correct O(n) solution and grade it via real code-eval.
    code_resp = client.post(
        f"/api/v1/sessions/{session_id}/code",
        json={
            "language": "python",
            "source_code": "n = int(input())\narr = list(map(int, input().split()))\nprint(sum(arr))",
            "test_cases": [
                {"input": "3\n1 2 3", "expected_output": "6"},
                {"input": "5\n10 20 30 40 50", "expected_output": "150"},
            ],
        },
    )
    assert code_resp.status_code == 200, code_resp.text
    code_body = code_resp.json()
    assert code_body["status"] == "completed"
    assert code_body["result"]["correctness"]["pass_rate"] == 1.0

    # Final report aggregates the whole session.
    report_resp = client.get(f"/api/v1/sessions/{session_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["status"] == "completed"
    assert len(report["history"]) == 3  # primary answer, followup answer, code submission
    assert report["overall_average_score"] is not None


def test_voice_round_trip_through_orchestrator(client):
    """speech-io's synthesize + transcribe, driven through the
    orchestrator's own answer/audio endpoint — proves the voice path
    (not just the text path) works end to end through the real stack."""
    import io

    settings = get_settings()

    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.txt", _TEST_RESUME_TEXT, "text/plain")},
        params={"personal_question_count": 1, "hr_question_count": 0, "enable_followups": False},
    )
    session_id = create_resp.json()["session_id"]

    # Synthesize a real spoken answer via speech-io directly (standing in
    # for what a browser would record), then submit it as audio through
    # the orchestrator's transcription path.
    synth_resp = httpx.post(
        f"{settings.speech_io_base_url}/api/v1/speech/synthesize",
        json={"text": "I used indexes to optimize the database queries."},
        timeout=30.0,
    )
    assert synth_resp.status_code == 200
    audio_bytes = synth_resp.content
    assert len(audio_bytes) > 1000

    answer_resp = client.post(
        f"/api/v1/sessions/{session_id}/answer/audio",
        files={"file": ("answer.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
    )
    assert answer_resp.status_code == 200, answer_resp.text
    body = answer_resp.json()
    assert 0.0 <= body["score"]["overall_score"] <= 1.0

    report = client.get(f"/api/v1/sessions/{session_id}/report").json()
    transcribed_answer = report["history"][0]["answer"]
    assert "index" in transcribed_answer.lower() or "optimiz" in transcribed_answer.lower()


def _real_face_jpeg_bytes() -> bytes:
    """A real, public-domain face photo (skimage's standard test image) —
    a legitimate stand-in for a camera capture in an environment with no
    webcam access. Same fixture image biometric-auth's and proctoring's
    own test suites use."""
    import cv2
    from skimage import data

    rgb = data.astronaut()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def test_full_stack_with_biometric_verification_and_proctoring(client):
    """Enrolls a real candidate against the real biometric-auth service,
    creates a session gated on real face verification, submits real
    proctoring snapshots during the session, and confirms the final
    report's proctoring_summary reflects them — the two newest services,
    proven against the real stack the same way the original five are
    proven above."""
    import uuid

    settings = get_settings()
    face_bytes = _real_face_jpeg_bytes()
    candidate_id = f"live-e2e-{uuid.uuid4().hex[:8]}"

    enroll_resp = httpx.post(
        f"{settings.biometric_auth_base_url}/api/v1/biometric/enroll",
        params={"candidate_id": candidate_id},
        files=[("files", (f"f{i}.jpg", face_bytes, "image/jpeg")) for i in range(3)],
        timeout=30.0,
    )
    assert enroll_resp.status_code == 200, enroll_resp.text
    assert enroll_resp.json()["enrolled"] is True

    create_resp = client.post(
        "/api/v1/sessions",
        files={
            "file": ("resume.txt", _TEST_RESUME_TEXT, "text/plain"),
            "face_files": ("face.jpg", face_bytes, "image/jpeg"),
        },
        params={"candidate_id": candidate_id, "personal_question_count": 0, "hr_question_count": 0},
    )
    assert create_resp.status_code == 200, create_resp.text
    session_id = create_resp.json()["session_id"]
    assert create_resp.json()["round"] == "coding"

    for _ in range(3):
        snapshot_resp = client.post(
            f"/api/v1/sessions/{session_id}/proctoring/snapshot",
            files={"file": ("frame.jpg", face_bytes, "image/jpeg")},
        )
        assert snapshot_resp.status_code == 200, snapshot_resp.text
        assert snapshot_resp.json()["faces_detected"] == 1

    code_resp = client.post(
        f"/api/v1/sessions/{session_id}/code",
        json={
            "language": "python",
            "source_code": "print(int(input()) + int(input()))",
            "test_cases": [{"input": "2\n3", "expected_output": "5"}],
        },
    )
    assert code_resp.status_code == 200, code_resp.text

    report = client.get(f"/api/v1/sessions/{session_id}/report").json()
    proctoring_summary = report["proctoring_summary"]
    assert proctoring_summary is not None
    assert proctoring_summary["frames_processed"] == 3
    assert proctoring_summary["integrity_score"] == 100.0


def test_session_creation_rejects_mismatched_face(client):
    """A candidate_id enrolled with one face, but session creation
    attempted with a submitted image biometric-auth can't match to it
    (here: a blank frame with no detectable face) — real biometric-auth
    rejects the verify call, and the orchestrator must surface that
    clearly rather than silently letting the session through."""
    import uuid

    import numpy as np

    settings = get_settings()
    face_bytes = _real_face_jpeg_bytes()
    candidate_id = f"live-e2e-mismatch-{uuid.uuid4().hex[:8]}"

    enroll_resp = httpx.post(
        f"{settings.biometric_auth_base_url}/api/v1/biometric/enroll",
        params={"candidate_id": candidate_id},
        files=[("files", (f"f{i}.jpg", face_bytes, "image/jpeg")) for i in range(3)],
        timeout=30.0,
    )
    assert enroll_resp.status_code == 200, enroll_resp.text

    import cv2

    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok
    blank_bytes = buf.tobytes()

    create_resp = client.post(
        "/api/v1/sessions",
        files={
            "file": ("resume.txt", _TEST_RESUME_TEXT, "text/plain"),
            "face_files": ("face.jpg", blank_bytes, "image/jpeg"),
        },
        params={"candidate_id": candidate_id, "personal_question_count": 0, "hr_question_count": 0},
    )
    # biometric-auth itself 422s a no-face image before any match/no-match
    # verdict is even possible — the orchestrator must propagate this as a
    # clear downstream error, not a silently-created session.
    assert create_resp.status_code == 502
    assert create_resp.json()["error"] == "downstream_error"
