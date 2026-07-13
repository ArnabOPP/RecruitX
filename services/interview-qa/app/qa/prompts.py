"""Prompt construction for CV-grounded question generation and follow-ups.

The core design constraint, straight from the BRD: "LLMs propose and
converse; the deterministic engine decides the score" — this service's only
job is producing well-grounded questions, not judging answers. Every prompt
below pushes the model toward citing *specific* résumé evidence (a named
project, a named skill actually marked as used, a specific bullet) rather
than generic interview-question filler any candidate could be asked.
"""

from __future__ import annotations

from ..config import get_settings
from .schemas import ResumeContext, RoundType

_QUESTION_JSON_SHAPE = """{
  "questions": [
    {
      "text": "the interview question, phrased as it would actually be spoken to the candidate",
      "category": "one of: project_deep_dive, skill_verification, experience_deep_dive, behavioral_star, resume_gap_probe, motivation_fit",
      "grounding": {
        "kind": "one of: skill, project, experience, education, certification",
        "reference": "the exact name from the resume this question is about (e.g. the project title, skill name, or organization)",
        "detail": "a short phrase from the resume that justifies asking this, e.g. the specific bullet or tech stack item"
      },
      "difficulty": "one of: easy, medium, hard"
    }
  ]
}"""

_FOLLOWUP_JSON_SHAPE = """{
  "follow_up_question": "the next question to ask, spoken as it would be to the candidate",
  "rationale": "one sentence explaining why this follow-up, e.g. 'the answer named a technology not on the resume' or 'probing implementation depth on a claimed skill'"
}"""


def _format_resume_context(resume: ResumeContext) -> str:
    """Render the résumé as a compact, natural-language brief for the
    prompt — not a raw JSON dump, which wastes tokens on schema noise the
    model doesn't need and makes it harder for the model to notice what's
    actually evidenced vs. merely listed."""
    lines: list[str] = []

    if resume.full_name:
        lines.append(f"Candidate: {resume.full_name}")
    if resume.summary:
        lines.append(f"Summary: {resume.summary.strip()}")

    if resume.education:
        lines.append("\nEducation:")
        for edu in resume.education:
            parts = [p for p in [edu.degree, edu.field_of_study, edu.institution] if p]
            if parts:
                lines.append(f"- {' in '.join(parts[:2])}{', ' + parts[-1] if len(parts) > 2 else ''}")

    if resume.experience:
        lines.append("\nExperience:")
        for exp in resume.experience:
            header = " @ ".join(p for p in [exp.role_title, exp.organization] if p)
            lines.append(f"- {header or 'Role'}")
            for bullet in exp.description_bullets:
                lines.append(f"  * {bullet}")
            if exp.extracted_skills:
                lines.append(f"  (skills used: {', '.join(exp.extracted_skills)})")

    if resume.projects:
        lines.append("\nProjects:")
        for proj in resume.projects:
            title = proj.title or "Untitled project"
            stack = f" [{', '.join(proj.tech_stack)}]" if proj.tech_stack else ""
            lines.append(f"- {title}{stack}")
            if proj.description:
                lines.append(f"  {proj.description}")

    if resume.skills:
        declared = [s.name for s in resume.skills if not (s.evidenced_in_project or s.evidenced_in_experience)]
        evidenced = [s.name for s in resume.skills if s.evidenced_in_project or s.evidenced_in_experience]
        if evidenced:
            lines.append(f"\nSkills actually used in a project or role (highest-value to probe): {', '.join(evidenced)}")
        if declared:
            lines.append(f"Skills only listed, not shown used anywhere (good for a gap-check question): {', '.join(declared)}")

    if resume.certifications:
        lines.append("\nCertifications: " + ", ".join(c.name for c in resume.certifications))

    text = "\n".join(lines) if lines else "(no résumé data provided)"

    # A caller could submit an arbitrarily large résumé payload (no field
    # has a length cap) — truncating here bounds both the Groq request
    # cost and the risk of blowing past the model's context window,
    # regardless of which field the bulk came from.
    max_chars = get_settings().max_resume_context_chars
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [résumé context truncated]"

    return text


def build_generation_prompt(
    resume: ResumeContext,
    round_type: RoundType,
    target_company: str | None,
    count: int,
) -> tuple[str, str]:
    round_guidance = (
        "This is the PERSONAL round: focus on project_deep_dive, skill_verification, "
        "experience_deep_dive, and resume_gap_probe questions — concrete, technical, "
        "grounded in what the candidate actually built or did."
        if round_type == RoundType.PERSONAL
        else "This is the HR round: focus on behavioral_star and motivation_fit questions "
        "— but still ground them in specific résumé evidence (e.g. 'tell me about a time "
        "you faced a conflict, drawing on your experience at <organization>') rather than "
        "asking something entirely generic."
    )
    company_line = f"The candidate is interviewing for a role at {target_company}." if target_company else ""

    system = (
        "You are an expert technical interviewer for Recruitix, an interview-preparation "
        "platform. Your only job is generating interview questions grounded in a specific "
        "candidate's résumé — you never grade or score answers, and every question you ask "
        "must reference something concrete and specific from the résumé provided (a named "
        "project, a named skill, an actual bullet point), never a generic question a random "
        "candidate could also be asked. Prefer skills marked as evidenced in a project or "
        "role over skills that are merely listed — asking about an unevidenced skill is only "
        "appropriate for a resume_gap_probe question that gently checks depth. "
        "Respond with a single JSON object and nothing else, in exactly this shape:\n"
        f"{_QUESTION_JSON_SHAPE}"
    )
    user = (
        f"{round_guidance}\n{company_line}\n\n"
        f"Generate exactly {count} questions.\n\n"
        f"Résumé:\n{_format_resume_context(resume)}"
    )
    return system, user


def build_followup_prompt(
    resume: ResumeContext,
    original_question: str,
    candidate_answer: str,
    round_type: RoundType,
    target_company: str | None,
) -> tuple[str, str]:
    company_line = f"The candidate is interviewing for a role at {target_company}." if target_company else ""
    round_label = "personal/technical" if round_type == RoundType.PERSONAL else "HR/behavioral"

    system = (
        "You are an expert technical interviewer for Recruitix conducting a live "
        f"{round_label} interview round. You just asked the candidate a question and they "
        "answered — generate ONE natural follow-up question. Prefer probing deeper into "
        "something the answer only touched on lightly, or gently checking consistency if the "
        "answer mentions something not reflected in the résumé (without being accusatory — "
        "frame it as curiosity, e.g. asking them to elaborate). You never grade or score the "
        "answer, only ask the next question. "
        "Respond with a single JSON object and nothing else, in exactly this shape:\n"
        f"{_FOLLOWUP_JSON_SHAPE}"
    )
    user = (
        f"{company_line}\n\n"
        f"Résumé:\n{_format_resume_context(resume)}\n\n"
        f"Original question: {original_question}\n"
        f"Candidate's answer: {candidate_answer}"
    )
    return system, user
