"""Projects section extraction: title, description, tech stack, links.

This is the highest-value section for Round 2 of the Recruitix BRD (CV-driven
personal interview): the LLM question-generator grounds questions in
"you built X with Y — how did you handle Z?", so title + tech_stack accuracy
here matters more than in any other section.
"""

from __future__ import annotations

import re

from .schemas import ConfidentField, ExtractionMethod, ProjectEntry
from .sections import group_entries
from .skills import extract_skills_from_gazetteer

_URL_RE = re.compile(r"https?://[^\s,)]+|(?:www\.)[^\s,)]+\.[a-z]{2,}[^\s,)]*", re.I)
_TITLE_TECH_SPLIT_RE = re.compile(r"\s+[\|–]\s+|\s+-\s+")
_TECH_PAREN_RE = re.compile(r"\(([^)]+)\)")
_DURATION_RE = re.compile(
    r"\b((?:19|20)\d{2}(?:\s*-\s*(?:(?:19|20)\d{2}|Present))?)\b", re.I
)


def extract_projects(section_body: str) -> list[ProjectEntry]:
    if not section_body.strip():
        return []
    return [_parse_entry(block) for block in group_entries(section_body)]


def _parse_entry(block: str) -> ProjectEntry:
    lines = [line for line in block.split("\n") if line.strip()]
    header_line = lines[0] if lines else block[:120]

    url_match = _URL_RE.search(block)
    url = url_match.group().rstrip(".,;") if url_match else None

    tech_from_paren = []
    paren_match = _TECH_PAREN_RE.search(header_line)
    if paren_match:
        tech_from_paren = [t.strip() for t in re.split(r"[,/]", paren_match.group(1)) if t.strip()]
        header_line = header_line[: paren_match.start()].strip()

    title_part = _TITLE_TECH_SPLIT_RE.split(header_line, maxsplit=1)[0].strip()
    title_part = _URL_RE.sub("", title_part).strip(" -|")

    duration_match = _DURATION_RE.search(header_line) or _DURATION_RE.search(block[:200])
    duration = duration_match.group(1) if duration_match else None

    title_field = None
    if title_part:
        title_field = ConfidentField(
            value=title_part, confidence=0.7, method=ExtractionMethod.RULE_SECTION
        )

    description = "\n".join(lines[1:]).strip() or (header_line if not title_field else "")

    gazetteer_skills = extract_skills_from_gazetteer(block)
    tech_stack = list(dict.fromkeys(tech_from_paren + [s.name for s in gazetteer_skills.values()]))

    return ProjectEntry(
        title=title_field,
        description=description or None,
        tech_stack=tech_stack,
        url=url,
        duration=duration,
        raw_text=block,
    )
