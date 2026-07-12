"""Pydantic data contracts for the résumé parsing pipeline.

Every extracted field carries a confidence score and, where possible, the
source span it was derived from, so downstream consumers (and the FR-09/FR-11
CV-grounded interview and evidence-logging requirements in the Recruitix BRD)
can audit *why* a value was extracted, not just *what* was extracted.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    REGEX = "regex"
    SPACY_NER = "spacy_ner"
    TRANSFORMER_NER = "transformer_ner"
    GAZETTEER = "gazetteer"
    RULE_SECTION = "rule_section"
    ENSEMBLE = "ensemble"


class SourceSpan(BaseModel):
    section: str | None = None
    text: str
    start_char: int | None = None
    end_char: int | None = None


class ConfidentField(BaseModel):
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: ExtractionMethod
    source: SourceSpan | None = None


class ContactInfo(BaseModel):
    full_name: ConfidentField | None = None
    emails: list[ConfidentField] = Field(default_factory=list)
    phones: list[ConfidentField] = Field(default_factory=list)
    location: ConfidentField | None = None
    linkedin: ConfidentField | None = None
    github: ConfidentField | None = None
    portfolio_urls: list[ConfidentField] = Field(default_factory=list)


class SkillCategory(str, Enum):
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK_LIBRARY = "framework_library"
    DATABASE = "database"
    CLOUD_DEVOPS = "cloud_devops"
    DATA_ML = "data_ml"
    TOOL_PLATFORM = "tool_platform"
    SOFT_SKILL = "soft_skill"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    OTHER = "other"


class Skill(BaseModel):
    name: str
    normalized_name: str
    category: SkillCategory
    confidence: float = Field(ge=0.0, le=1.0)
    method: ExtractionMethod
    mention_count: int = 1
    evidenced_in_project: bool = False
    evidenced_in_experience: bool = False


class EducationEntry(BaseModel):
    institution: ConfidentField | None = None
    degree: ConfidentField | None = None
    field_of_study: ConfidentField | None = None
    start_date: ConfidentField | None = None
    end_date: ConfidentField | None = None
    gpa: ConfidentField | None = None
    raw_text: str


class ExperienceEntry(BaseModel):
    role_title: ConfidentField | None = None
    organization: ConfidentField | None = None
    location: ConfidentField | None = None
    start_date: ConfidentField | None = None
    end_date: ConfidentField | None = None
    is_current: bool = False
    description_bullets: list[str] = Field(default_factory=list)
    extracted_skills: list[str] = Field(default_factory=list)
    raw_text: str


class ProjectEntry(BaseModel):
    title: ConfidentField | None = None
    description: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    url: str | None = None
    duration: str | None = None
    raw_text: str


class CertificationEntry(BaseModel):
    name: str
    issuer: str | None = None
    date: str | None = None
    raw_text: str


class ParseWarning(BaseModel):
    code: str
    message: str
    section: str | None = None


class ParsedResume(BaseModel):
    schema_version: str = "1.0"
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str | None = None
    skills: list[Skill] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    total_experience_years: float | None = None
    sections_detected: list[str] = Field(default_factory=list)
    warnings: list[ParseWarning] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_text_length: int = 0
    processing_time_ms: float | None = None
