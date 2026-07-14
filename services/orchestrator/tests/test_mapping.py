"""Tests for the cv-parser -> interview-qa schema mapping."""

from __future__ import annotations

from app.mapping import map_parsed_resume_to_resume_context


def test_maps_full_name_from_confident_field():
    parsed = {"contact": {"full_name": {"value": "Jordan Lee", "confidence": 0.9}}}
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["full_name"] == "Jordan Lee"


def test_missing_full_name_maps_to_none():
    parsed = {"contact": {"full_name": None}}
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["full_name"] is None


def test_missing_contact_entirely_does_not_raise():
    result = map_parsed_resume_to_resume_context({})
    assert result["full_name"] is None
    assert result["skills"] == []


def test_maps_skills_with_evidence_flags():
    parsed = {
        "skills": [
            {"name": "Python", "evidenced_in_project": True, "evidenced_in_experience": False},
            {"name": "Kubernetes", "evidenced_in_project": False, "evidenced_in_experience": False},
        ]
    }
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["skills"][0] == {"name": "Python", "evidenced_in_project": True, "evidenced_in_experience": False}
    assert result["skills"][1]["name"] == "Kubernetes"


def test_maps_education_unwrapping_confident_fields():
    parsed = {
        "education": [
            {
                "institution": {"value": "MIT", "confidence": 0.8},
                "degree": {"value": "B.Tech", "confidence": 0.9},
                "field_of_study": None,
            }
        ]
    }
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["education"][0] == {"institution": "MIT", "degree": "B.Tech", "field_of_study": None}


def test_maps_experience_preserving_bullets_and_skills():
    parsed = {
        "experience": [
            {
                "role_title": {"value": "Intern"},
                "organization": {"value": "Acme"},
                "description_bullets": ["Did a thing"],
                "extracted_skills": ["Python"],
            }
        ]
    }
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["experience"][0]["role_title"] == "Intern"
    assert result["experience"][0]["organization"] == "Acme"
    assert result["experience"][0]["description_bullets"] == ["Did a thing"]
    assert result["experience"][0]["extracted_skills"] == ["Python"]


def test_maps_projects():
    parsed = {
        "projects": [
            {"title": {"value": "TaskTracker"}, "description": "A task app.", "tech_stack": ["Python", "FastAPI"]}
        ]
    }
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["projects"][0] == {"title": "TaskTracker", "description": "A task app.", "tech_stack": ["Python", "FastAPI"]}


def test_maps_certifications():
    parsed = {"certifications": [{"name": "AWS Certified", "issuer": "Amazon"}]}
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["certifications"] == [{"name": "AWS Certified", "issuer": "Amazon"}]


def test_maps_summary_directly_no_confident_field_wrapper():
    parsed = {"summary": "A great candidate."}
    result = map_parsed_resume_to_resume_context(parsed)
    assert result["summary"] == "A great candidate."
