"""Resume section segmentation.

Résumés don't follow a fixed schema, so we detect section boundaries by
scoring each line as a candidate header: short, mostly title-cased/uppercase,
matching (exactly or fuzzily) a known section-name alias, and not itself a
sentence. Everything between two detected headers belongs to the first
header's section. Text before the first header is treated as a "header block"
(name/contact info) plus optional summary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from .dates import date_match_span

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sections_db.json"
with open(_DATA_PATH, encoding="utf-8") as f:
    _SECTION_ALIASES: dict[str, list[str]] = json.load(f)

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _SECTION_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical

_FUZZY_THRESHOLD = 88
_MAX_HEADER_WORDS = 5
_BULLET_PREFIX_RE = re.compile(r"^[•●▪\-\*‣⁃]\s*")


@dataclass
class Section:
    name: str
    header_text: str
    start_line: int
    end_line: int
    body: str = ""
    lines: list[str] = field(default_factory=list)


def _classify_header(line: str) -> str | None:
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped.split()) > _MAX_HEADER_WORDS:
        return None
    if len(stripped) > 40:
        return None
    if stripped.lower() in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[stripped.lower()]

    is_shouty = stripped.isupper() and len(stripped) >= 3
    is_title = stripped.istitle()
    if not (is_shouty or is_title):
        return None

    best_score = 0.0
    best_canonical = None
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        score = fuzz.ratio(stripped.lower(), alias)
        if score > best_score:
            best_score = score
            best_canonical = canonical
    if best_score >= _FUZZY_THRESHOLD:
        return best_canonical
    return None


def segment(text: str) -> tuple[list[Section], str]:
    """Split résumé text into named sections.

    Returns (sections, header_block) where header_block is the raw text
    before the first recognised section (name/contact/summary candidate).
    """
    lines = text.split("\n")
    headers: list[tuple[int, str, str]] = []  # (line_idx, canonical, raw header text)
    for idx, line in enumerate(lines):
        canonical = _classify_header(line)
        if canonical:
            headers.append((idx, canonical, line.strip()))

    if not headers:
        return [], text

    header_block = "\n".join(lines[: headers[0][0]]).strip()

    sections: list[Section] = []
    for i, (start_idx, canonical, header_text) in enumerate(headers):
        end_idx = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        body_lines = lines[start_idx + 1 : end_idx]
        body = "\n".join(body_lines).strip()
        sections.append(
            Section(
                name=canonical,
                header_text=header_text,
                start_line=start_idx,
                end_line=end_idx,
                body=body,
                lines=[bl for bl in body_lines if bl.strip()],
            )
        )
    return sections, header_block


def split_bullets(body: str) -> list[str]:
    """Split a section body into discrete entries/bullets, merging wrapped lines."""
    raw_lines = [ln for ln in body.split("\n") if ln.strip()]
    bullets: list[str] = []
    for line in raw_lines:
        cleaned = _BULLET_PREFIX_RE.sub("", line).strip()
        if _BULLET_PREFIX_RE.match(line) or not bullets:
            bullets.append(cleaned)
        else:
            # Heuristic: a continuation line is lowercase-starting or short,
            # and doesn't look like a new entry header (no date range).
            starts_new_entry = bool(re.search(r"\b(19|20)\d{2}\b", line)) and len(line) < 80
            if starts_new_entry:
                bullets.append(cleaned)
            else:
                bullets[-1] = f"{bullets[-1]} {cleaned}".strip()
    return bullets


def group_entries(body: str) -> list[str]:
    """Group a multi-entry section (e.g. Experience, Education) into per-entry
    text blocks, splitting on blank lines first and falling back to date-range
    heuristics when entries run together without blank-line separation.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    if len(blocks) > 1:
        return blocks

    # No blank-line separation available — split on lines that look like a
    # new entry's title/date line. Two résumé styles both show up in
    # practice: (a) title and date range on separate lines, or (b) an entire
    # entry (title, org, date range, tech stack) crammed onto one physical
    # line. A line is only treated as a new-entry header if it contains a
    # date AND substantial non-date text alongside it — a line that is
    # *just* a date range (style a) is a continuation of the title line
    # directly above it, not a new entry of its own.
    lines = [ln for ln in body.split("\n") if ln.strip()]
    entries: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        is_bullet = bool(_BULLET_PREFIX_RE.match(stripped))
        looks_like_header = False
        if not is_bullet and len(stripped) < 160:
            span = date_match_span(stripped)
            if span:
                remainder = (stripped[: span[0]] + stripped[span[1] :]).strip(" -–,|()")
                looks_like_header = len(remainder) >= 12
        if looks_like_header or not entries:
            entries.append([line])
        else:
            entries[-1].append(line)
    return ["\n".join(e) for e in entries]
