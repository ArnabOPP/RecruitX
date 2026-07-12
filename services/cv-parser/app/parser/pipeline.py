"""End-to-end résumé parsing orchestrator.

file bytes -> ExtractedDocument -> sections -> per-section field extractors
-> ParsedResume, with overall confidence scoring and warnings collection.
"""

from __future__ import annotations

import logging
import time

from .certifications import extract_certifications
from .contact import extract_contact
from .education import extract_education
from .experience import extract_experience, total_experience_years
from .extractor import extract_document
from .projects import extract_projects
from .schemas import ParsedResume, ParseWarning
from .sections import segment
from .skills import extract_skills_from_section, merge_skills

logger = logging.getLogger("cv_parser.pipeline")

_EXPECTED_SECTIONS = {"education", "experience", "skills", "projects"}


def parse_resume(filename: str, file_bytes: bytes) -> ParsedResume:
    start_time = time.perf_counter()
    warnings: list[ParseWarning] = []

    doc = extract_document(filename, file_bytes)

    for w in doc.warnings:
        warnings.append(ParseWarning(code="extraction_warning", message=w))

    sections, header_block = segment(doc.text)
    section_map = {s.name: s for s in sections}
    sections_detected = [s.name for s in sections]

    missing = _EXPECTED_SECTIONS - set(sections_detected)
    for m in missing:
        warnings.append(
            ParseWarning(
                code="section_not_found",
                message=f"Could not detect a '{m}' section in the résumé.",
                section=m,
            )
        )

    contact = extract_contact(header_block, doc.text)
    if not contact.full_name:
        warnings.append(ParseWarning(code="name_not_found", message="Candidate name could not be confidently extracted."))
    if not contact.emails:
        warnings.append(ParseWarning(code="email_not_found", message="No email address found."))

    education = extract_education(section_map["education"].body) if "education" in section_map else []
    experience = extract_experience(section_map["experience"].body) if "experience" in section_map else []
    projects = extract_projects(section_map["projects"].body) if "projects" in section_map else []
    certifications = (
        extract_certifications(section_map["certifications"].body)
        if "certifications" in section_map
        else []
    )

    section_skills = (
        extract_skills_from_section(section_map["skills"].body) if "skills" in section_map else {}
    )
    project_text = "\n".join(p.raw_text for p in projects)
    experience_text = "\n".join(e.raw_text for e in experience)
    skills = merge_skills(section_skills, project_text, experience_text)

    summary = None
    if "summary" in section_map:
        summary = section_map["summary"].body.strip() or None

    exp_years = total_experience_years(experience) if experience else None

    overall_confidence = _compute_overall_confidence(contact, skills, education, experience, warnings)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return ParsedResume(
        contact=contact,
        summary=summary,
        skills=skills,
        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,
        total_experience_years=exp_years,
        sections_detected=sections_detected,
        warnings=warnings,
        overall_confidence=overall_confidence,
        raw_text_length=len(doc.text),
        processing_time_ms=round(elapsed_ms, 1),
    )


def _compute_overall_confidence(contact, skills, education, experience, warnings) -> float:
    signals: list[float] = []
    if contact.full_name:
        signals.append(contact.full_name.confidence)
    if contact.emails:
        signals.append(max(e.confidence for e in contact.emails))
    if skills:
        signals.append(sum(s.confidence for s in skills) / len(skills))
    if education:
        edu_confidences = [
            f.confidence for e in education for f in (e.institution, e.degree) if f
        ]
        if edu_confidences:
            signals.append(sum(edu_confidences) / len(edu_confidences))
    if experience:
        exp_confidences = [
            f.confidence for e in experience for f in (e.role_title, e.organization) if f
        ]
        if exp_confidences:
            signals.append(sum(exp_confidences) / len(exp_confidences))

    if not signals:
        return 0.0

    base = sum(signals) / len(signals)
    penalty = min(0.3, 0.05 * len(warnings))
    return round(max(0.0, base - penalty), 3)
