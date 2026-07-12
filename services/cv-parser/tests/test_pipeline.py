from pathlib import Path

import pytest

from app.parser.extractor import EmptyDocumentError, UnsupportedFileTypeError
from app.parser.pipeline import parse_resume

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def parsed():
    data = (FIXTURES / "sample_resume.txt").read_bytes()
    return parse_resume("sample_resume.txt", data)


def test_contact_name_extracted(parsed):
    assert parsed.contact.full_name is not None
    assert "Aarav" in parsed.contact.full_name.value


def test_contact_email_extracted(parsed):
    emails = [e.value for e in parsed.contact.emails]
    assert "aarav.sharma@email.com" in emails


def test_contact_phone_extracted(parsed):
    assert len(parsed.contact.phones) >= 1


def test_contact_links_extracted(parsed):
    assert parsed.contact.linkedin is not None
    assert "linkedin.com/in/aaravsharma" in parsed.contact.linkedin.value
    assert parsed.contact.github is not None


def test_skills_extracted(parsed):
    names = {s.name.lower() for s in parsed.skills}
    for expected in ["python", "react", "postgresql", "docker"]:
        assert expected in names, f"expected skill '{expected}' in {names}"


def test_skill_evidenced_in_project(parsed):
    python_skill = next((s for s in parsed.skills if s.name.lower() == "python"), None)
    assert python_skill is not None


def test_education_extracted(parsed):
    assert len(parsed.education) >= 1
    degrees = [e.degree.value for e in parsed.education if e.degree]
    assert any("B.Tech" in d or "BTech" in d for d in degrees)


def test_education_gpa_extracted(parsed):
    gpas = [e.gpa.value for e in parsed.education if e.gpa]
    assert any("8.7" in g for g in gpas)


def test_experience_extracted(parsed):
    assert len(parsed.experience) >= 1
    entry = parsed.experience[0]
    assert entry.organization is not None
    assert len(entry.description_bullets) >= 1


def test_projects_extracted(parsed):
    assert len(parsed.projects) >= 2
    titles = [p.title.value for p in parsed.projects if p.title]
    assert any("Recruitix" in t for t in titles)


def test_project_tech_stack_extracted(parsed):
    project = parsed.projects[0]
    tech_lower = {t.lower() for t in project.tech_stack}
    assert tech_lower & {"react", "fastapi", "postgresql"}


def test_certifications_extracted(parsed):
    assert len(parsed.certifications) >= 1


def test_sections_detected(parsed):
    for section in ["education", "skills", "projects", "experience", "certifications"]:
        assert section in parsed.sections_detected


def test_overall_confidence_reasonable(parsed):
    assert 0.4 <= parsed.overall_confidence <= 1.0


def test_unsupported_file_type_raises():
    with pytest.raises(UnsupportedFileTypeError):
        parse_resume("resume.xyz", b"hello")


def test_empty_document_raises():
    with pytest.raises(EmptyDocumentError):
        parse_resume("empty.txt", b"   \n  \n ")
