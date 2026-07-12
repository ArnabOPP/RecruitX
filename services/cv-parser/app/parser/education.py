"""Education section extraction: institution, degree, field of study, dates, GPA."""

from __future__ import annotations

import re

from .dates import find_date_range
from .ner import extract_entities
from .schemas import ConfidentField, EducationEntry, ExtractionMethod, SourceSpan
from .sections import group_entries

_GPA_RE = re.compile(
    r"(?:GPA|CGPA|Grade)\s*[:\-]?\s*(\d\.\d{1,2})\s*(?:/\s*(\d{1,2}(?:\.\d)?))?", re.I
)
_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*%")
_FIELD_HINT_RE = re.compile(
    r"\bin\s+([A-Z][A-Za-z&,\s]{3,50}?)(?:\s+from|\s+at|,|\.|\n|$)"
)


def extract_education(section_body: str) -> list[EducationEntry]:
    if not section_body.strip():
        return []

    entries: list[EducationEntry] = []
    for block in group_entries(section_body):
        entries.append(_parse_entry(block))
    return entries


def _parse_entry(block: str) -> EducationEntry:
    entities = extract_entities(block)

    degree_ent = next((e for e in entities if e.label.startswith("DEGREE::")), None)
    org_ent = _pick_institution_entity(entities)

    degree_field = None
    if degree_ent:
        degree_field = ConfidentField(
            value=degree_ent.text,
            confidence=0.95,
            method=ExtractionMethod.GAZETTEER,
            source=SourceSpan(section="education", text=block[:200]),
        )

    # A comma-delimited scan (institution names are near-universally either
    # keyword-flagged — "...School"/"...University" — or the segment right
    # after the degree/field clause) is more reliable here than NER: the
    # small statistical model frequently mistags only a fragment of a
    # multi-word institution name (e.g. "South Point High School" -> just
    # "High School") or, for acronym institutions like "IEM Kolkata", may
    # not fire at all.
    regex_institution = _regex_institution_segment(block.split("\n")[0])

    institution_field = None
    if regex_institution:
        institution_field = ConfidentField(
            value=regex_institution, confidence=0.75, method=ExtractionMethod.RULE_SECTION,
            source=SourceSpan(section="education", text=block[:200]),
        )
    elif org_ent:
        try:
            method = ExtractionMethod(org_ent.method)
        except ValueError:
            method = ExtractionMethod.SPACY_NER
        institution_field = ConfidentField(
            value=org_ent.text,
            confidence=org_ent.confidence,
            method=method,
            source=SourceSpan(section="education", text=block[:200]),
        )
    else:
        institution_field = _fallback_institution(block)

    field_match = _FIELD_HINT_RE.search(block)
    field_of_study = None
    if field_match:
        field_of_study = ConfidentField(
            value=field_match.group(1).strip(),
            confidence=0.6,
            method=ExtractionMethod.REGEX,
        )

    start, end, is_current = find_date_range(block)
    start_field = (
        ConfidentField(value=start, confidence=0.85, method=ExtractionMethod.REGEX)
        if start
        else None
    )
    end_value = end or ("Present" if is_current else None)
    end_field = (
        ConfidentField(
            value=end_value,
            confidence=0.85,
            method=ExtractionMethod.REGEX,
        )
        if end_value
        else None
    )

    gpa_field = None
    gpa_match = _GPA_RE.search(block)
    if gpa_match:
        scale = gpa_match.group(2) or "10"
        gpa_field = ConfidentField(
            value=f"{gpa_match.group(1)}/{scale}", confidence=0.9, method=ExtractionMethod.REGEX
        )
    else:
        pct_match = _PERCENT_RE.search(block)
        if pct_match:
            gpa_field = ConfidentField(
                value=f"{pct_match.group(1)}%", confidence=0.85, method=ExtractionMethod.REGEX
            )

    return EducationEntry(
        institution=institution_field,
        degree=degree_field,
        field_of_study=field_of_study,
        start_date=start_field,
        end_date=end_field,
        gpa=gpa_field,
        raw_text=block,
    )


_INSTITUTION_KEYWORD_RE = re.compile(
    r"\b(Universit(?:y|ies)|Colleges?|Institutes?|Schools?|Academ(?:y|ies)|Polytechnics?|VIT|IIT|NIT)\b",
    re.I,
)


def _pick_institution_entity(entities):
    """When a block yields multiple ORG/GPE candidates (e.g. both the field
    of study and the institution get tagged ORG), prefer the one that looks
    like an institution name, then fall back to the one appearing latest in
    the line — résumés conventionally state "<Degree> in <Field>, <Institution>".
    """
    candidates = [e for e in entities if e.label in ("ORG", "GPE")]
    if not candidates:
        return None
    keyworded = [e for e in candidates if _INSTITUTION_KEYWORD_RE.search(e.text)]
    pool = keyworded or candidates
    return max(pool, key=lambda e: e.start_char)


_TRAILING_DATE_GPA_RE = re.compile(r"\s*\b(19|20)\d{2}\b.*$")


def _split_top_level_commas(text: str) -> list[str]:
    """Comma-split, but ignore commas nested inside parentheses (e.g. the
    "(IoT, Cybersecurity & Blockchain)" field-of-study qualifier shouldn't
    fragment the entry into bogus segments)."""
    segments: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
    segments.append("".join(current))
    return [s.strip() for s in segments if s.strip()]


def _regex_institution_segment(header_line: str) -> str | None:
    segments = _split_top_level_commas(header_line)
    for seg in segments:
        if _INSTITUTION_KEYWORD_RE.search(seg):
            return _TRAILING_DATE_GPA_RE.sub("", seg).strip(" -–|")
    if len(segments) >= 2:
        candidate = segments[-1].split("|")[0]
        candidate = _TRAILING_DATE_GPA_RE.sub("", candidate).strip(" -–|")
        if candidate and not candidate.isdigit():
            return candidate
    return None


def _fallback_institution(block: str) -> ConfidentField | None:
    """When NER misses the org, fall back to the first line that looks like
    a proper-noun-heavy institution name (contains University/College/Institute)."""
    for line in block.split("\n"):
        if _INSTITUTION_KEYWORD_RE.search(line):
            return ConfidentField(
                value=line.strip(), confidence=0.55, method=ExtractionMethod.RULE_SECTION
            )
    return None
