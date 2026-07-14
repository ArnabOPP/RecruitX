"""The orchestration state machine: drives a session through
personal -> hr -> coding -> completed, calling the five downstream
services and translating between their schemas.

Round progression policy (deliberately simple, not configurable beyond
question counts): each round asks a fixed number of freshly-generated
grounded questions (interview-qa /generate). When enable_followups is on,
each primary question gets exactly one follow-up (interview-qa /followup)
before moving to the next primary question — showing off both endpoints
coherently rather than only ever using one. The coding round is entered
by transitioning to status="awaiting_code" and waits for the caller to
submit code; it does not auto-generate a coding problem (see the
service's README for why: none of the five services authors problems).
"""

from __future__ import annotations

from . import mapping
from .clients import (
    answer_grading_client,
    code_eval_client,
    cv_parser_client,
    interview_qa_client,
    speech_io_client,
)
from .session_store import SessionStore

_TEXT_ROUNDS = ("personal", "hr")


class OrchestrationError(Exception):
    pass


async def create_session(
    store: SessionStore,
    resume_bytes: bytes,
    filename: str,
    target_company: str | None,
    personal_question_count: int,
    hr_question_count: int,
    enable_followups: bool,
) -> dict:
    parsed = await cv_parser_client.parse_resume(resume_bytes, filename)
    resume_context = mapping.map_parsed_resume_to_resume_context(parsed)

    session_data: dict = {
        "status": "in_progress",
        "round": "personal",
        "resume_context": resume_context,
        "target_company": target_company,
        "round_config": {
            "personal_question_count": personal_question_count,
            "hr_question_count": hr_question_count,
            "enable_followups": enable_followups,
        },
        "counts": {"personal_asked": 0, "hr_asked": 0},
        "stage": "primary",
        "current_question": None,
        "history": [],
    }

    if personal_question_count > 0:
        question = await _generate_next_primary_question(session_data)
        session_data["current_question"] = question
        session_data["counts"]["personal_asked"] = 1
    else:
        session_data = await _advance_round_or_next_question(session_data)

    session_id = await store.create(session_data)
    session_data["session_id"] = session_id
    return session_data


async def _generate_next_primary_question(session_data: dict) -> dict:
    resp = await interview_qa_client.generate_questions(
        session_data["resume_context"], session_data["round"], session_data.get("target_company"), 1
    )
    questions = resp.get("questions") or []
    if not questions:
        raise OrchestrationError("interview-qa returned no questions.")
    q = questions[0]
    return {
        "text": q["text"],
        "category": q.get("category"),
        "grounding": q.get("grounding"),
        "difficulty": q.get("difficulty"),
        "round": session_data["round"],
    }


async def _advance_round_or_next_question(session_data: dict) -> dict:
    cfg = session_data["round_config"]
    counts = session_data["counts"]

    if session_data["round"] == "personal":
        if counts["personal_asked"] < cfg["personal_question_count"]:
            session_data["current_question"] = await _generate_next_primary_question(session_data)
            counts["personal_asked"] += 1
            return session_data
        session_data["round"] = "hr"

    if session_data["round"] == "hr":
        if counts["hr_asked"] < cfg["hr_question_count"]:
            session_data["current_question"] = await _generate_next_primary_question(session_data)
            counts["hr_asked"] += 1
            return session_data
        session_data["round"] = "coding"

    if session_data["round"] == "coding":
        session_data["status"] = "awaiting_code"
        session_data["current_question"] = None
        return session_data

    session_data["status"] = "completed"
    session_data["current_question"] = None
    return session_data


async def submit_answer(
    store: SessionStore,
    session_id: str,
    answer_text: str | None,
    answer_audio: bytes | None,
    audio_filename: str | None,
) -> dict:
    session_data = await store.load(session_id)

    if session_data["status"] != "in_progress":
        raise OrchestrationError(f"Session is not accepting answers (status={session_data['status']}).")
    if session_data["round"] not in _TEXT_ROUNDS:
        raise OrchestrationError(f"Session is not in a question-answer round (round={session_data['round']}).")

    if answer_audio is not None:
        transcript = await speech_io_client.transcribe(answer_audio, audio_filename or "answer.wav")
        answer_text = transcript["text"]

    if not answer_text or not answer_text.strip():
        raise OrchestrationError("No answer text (typed or transcribed) was provided.")

    current_question = session_data["current_question"]
    if current_question is None:
        raise OrchestrationError("Session has no pending question to answer.")

    grounding = current_question.get("grounding")
    score_resp = await answer_grading_client.score(current_question["text"], answer_text, grounding)

    session_data["history"].append(
        {
            "round": session_data["round"],
            "stage": session_data["stage"],
            "question": current_question["text"],
            "grounding": grounding,
            "answer": answer_text,
            "score": score_resp,
        }
    )

    enable_followups = session_data["round_config"]["enable_followups"]
    next_question: dict | None

    if session_data["stage"] == "primary" and enable_followups:
        followup_resp = await interview_qa_client.generate_followup(
            session_data["resume_context"],
            current_question["text"],
            answer_text,
            session_data["round"],
            session_data.get("target_company"),
        )
        next_question = {
            "text": followup_resp["follow_up_question"],
            "category": "followup",
            "grounding": grounding,
            "difficulty": None,
            "round": session_data["round"],
        }
        session_data["stage"] = "followup"
        session_data["current_question"] = next_question
    else:
        session_data["stage"] = "primary"
        session_data = await _advance_round_or_next_question(session_data)
        next_question = session_data.get("current_question")

    await store.save(session_id, session_data)

    return {
        "score": score_resp,
        "round": session_data["round"],
        "status": session_data["status"],
        "next_question": next_question,
    }


async def submit_code(
    store: SessionStore,
    session_id: str,
    language: str,
    source_code: str,
    test_cases: list[dict],
    expected_complexity: str | None,
) -> dict:
    session_data = await store.load(session_id)

    if session_data["status"] != "awaiting_code":
        raise OrchestrationError(f"Session is not awaiting a code submission (status={session_data['status']}).")

    result = await code_eval_client.evaluate(language, source_code, test_cases, expected_complexity)

    session_data["history"].append({"round": "coding", "stage": "code", "language": language, "result": result})
    session_data["status"] = "completed"
    session_data["round"] = "completed"
    session_data["current_question"] = None

    await store.save(session_id, session_data)

    return {"result": result, "status": session_data["status"]}


async def get_report(store: SessionStore, session_id: str) -> dict:
    session_data = await store.load(session_id)
    text_scores = [
        h["score"]["overall_score"] for h in session_data["history"] if "score" in h and h["score"] is not None
    ]
    overall_average_score = sum(text_scores) / len(text_scores) if text_scores else None
    return {
        "session_id": session_data["session_id"],
        "status": session_data["status"],
        "round": session_data["round"],
        "history": session_data["history"],
        "overall_average_score": overall_average_score,
    }
