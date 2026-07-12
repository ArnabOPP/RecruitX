"""Work-experience section extraction: role, org, dates, bullets, skills used."""

from __future__ import annotations

import re

from .dates import MONTH_RE, estimate_years_between, find_date_range
from .ner import extract_entities
from .schemas import ConfidentField, ExperienceEntry, ExtractionMethod, SourceSpan
from .sections import group_entries, split_bullets
from .skills import extract_skills_from_gazetteer

_TITLE_ORG_SEP_RE = re.compile(r"\s+(?:at|@|,|\||-|–)\s+")
_ROLE_HINT_RE = re.compile(
    r"\b(Intern|Engineer|Developer|Manager|Analyst|Consultant|Lead|Architect|"
    r"Designer|Scientist|Specialist|Associate|Director|Founder|President|"
    r"Officer|Administrator|Coordinator)\b",
    re.I,
)
_DATE_TAIL_RE = re.compile(rf"\s*\(?(?:{MONTH_RE}\.?\s+)?\b(19|20)\d{{2}}\b.*$", re.I)


def extract_experience(section_body: str) -> list[ExperienceEntry]:
    if not section_body.strip():
        return []
    return [_parse_entry(block) for block in group_entries(section_body)]


def _parse_entry(block: str) -> ExperienceEntry:
    lines = [line for line in block.split("\n") if line.strip()]
    header_line = lines[0] if lines else block[:120]

    role_field, org_field = _parse_role_and_org(header_line, block)

    start, end, is_current = find_date_range(block)
    start_field = (
        ConfidentField(value=start, confidence=0.85, method=ExtractionMethod.REGEX)
        if start
        else None
    )
    end_field = (
        ConfidentField(value=end, confidence=0.85, method=ExtractionMethod.REGEX)
        if end
        else None
    )

    bullets = split_bullets("\n".join(lines[1:])) if len(lines) > 1 else []
    skill_matches = extract_skills_from_gazetteer(block)

    return ExperienceEntry(
        role_title=role_field,
        organization=org_field,
        start_date=start_field,
        end_date=end_field,
        is_current=is_current,
        description_bullets=bullets,
        extracted_skills=[s.name for s in skill_matches.values()],
        raw_text=block,
    )


def _parse_role_and_org(header_line: str, block: str) -> tuple[ConfidentField | None, ConfidentField | None]:
    parts = _TITLE_ORG_SEP_RE.split(header_line, maxsplit=1)
    role_field = None
    org_field = None

    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        second = _DATE_TAIL_RE.sub("", second).strip()
        if _ROLE_HINT_RE.search(first) or not _ROLE_HINT_RE.search(second):
            role_field = ConfidentField(value=first, confidence=0.75, method=ExtractionMethod.RULE_SECTION)
            org_field = ConfidentField(value=second, confidence=0.65, method=ExtractionMethod.RULE_SECTION)
        else:
            role_field = ConfidentField(value=second, confidence=0.65, method=ExtractionMethod.RULE_SECTION)
            org_field = ConfidentField(value=first, confidence=0.75, method=ExtractionMethod.RULE_SECTION)

    # Refine the org guess with NER, but scope the search to the org
    # candidate text itself (not the whole block/header) — running NER over
    # "<Role> - <Org>, <Location> <dates>" tends to greedily tag "<Role> -
    # <Org>" as one ORG span, re-introducing the exact garbling the dash
    # split above was meant to fix.
    ner_scope = org_field.value if org_field else header_line
    entities = extract_entities(ner_scope[:300])
    org_ent = next((e for e in entities if e.label == "ORG"), None)
    if (
        org_ent
        and not _ROLE_HINT_RE.search(org_ent.text)
        and (not org_field or org_ent.confidence > org_field.confidence)
    ):
        org_field = ConfidentField(
            value=org_ent.text,
            confidence=org_ent.confidence,
            method=ExtractionMethod.ENSEMBLE if org_ent.method == "ensemble" else ExtractionMethod.TRANSFORMER_NER,
            source=SourceSpan(section="experience", text=block[:200]),
        )

    if not role_field:
        role_match = _ROLE_HINT_RE.search(header_line)
        if role_match:
            # No clean separator was found — fall back to the clause up to
            # the first comma/dash rather than the entire (possibly
            # date-and-location-laden) header line.
            clause = re.split(r"\s*[,\-–]\s*", header_line, maxsplit=1)[0].strip()
            role_field = ConfidentField(
                value=clause or header_line.strip(), confidence=0.5, method=ExtractionMethod.RULE_SECTION
            )

    return role_field, org_field


def total_experience_years(entries: list[ExperienceEntry]) -> float:
    total = 0.0
    for entry in entries:
        start = entry.start_date.value if entry.start_date else None
        end = entry.end_date.value if entry.end_date else ("Present" if entry.is_current else None)
        total += estimate_years_between(start, end)
    return round(total, 1)
