"""Maps cv-parser's ParsedResume (every field wrapped in a `ConfidentField`
carrying confidence/method/source, for its own auditability) onto
interview-qa's plain ResumeContext shape.

This is exactly the mapping that was done by hand, repeatedly, while
manually testing the cv-parser -> interview-qa pipeline before this
service existed — promoted here into real, tested code instead of a
one-off script re-typed each time.
"""

from __future__ import annotations


def _cf(field: dict | None) -> str | None:
    """Unwrap a ConfidentField {value, confidence, method, source} -> plain value."""
    return field["value"] if field else None


def map_parsed_resume_to_resume_context(parsed: dict) -> dict:
    contact = parsed.get("contact") or {}
    return {
        "full_name": _cf(contact.get("full_name")),
        "summary": parsed.get("summary"),
        "skills": [
            {
                "name": s["name"],
                "evidenced_in_project": s.get("evidenced_in_project", False),
                "evidenced_in_experience": s.get("evidenced_in_experience", False),
            }
            for s in parsed.get("skills", [])
        ],
        "education": [
            {
                "institution": _cf(e.get("institution")),
                "degree": _cf(e.get("degree")),
                "field_of_study": _cf(e.get("field_of_study")),
            }
            for e in parsed.get("education", [])
        ],
        "experience": [
            {
                "role_title": _cf(e.get("role_title")),
                "organization": _cf(e.get("organization")),
                "description_bullets": e.get("description_bullets", []),
                "extracted_skills": e.get("extracted_skills", []),
            }
            for e in parsed.get("experience", [])
        ],
        "projects": [
            {
                "title": _cf(p.get("title")),
                "description": p.get("description"),
                "tech_stack": p.get("tech_stack", []),
            }
            for p in parsed.get("projects", [])
        ],
        "certifications": [
            {"name": c["name"], "issuer": c.get("issuer")} for c in parsed.get("certifications", [])
        ],
    }
