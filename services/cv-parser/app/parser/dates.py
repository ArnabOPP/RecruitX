"""Date-range parsing shared by education and experience extractors."""

from __future__ import annotations

import re

MONTH_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_TOKEN = rf"(?:{MONTH_RE}\.?\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
_RANGE_RE = re.compile(
    rf"({_DATE_TOKEN})\s*(?:-|to|–|—)\s*(Present|Current|Ongoing|{_DATE_TOKEN})",
    re.I,
)
_SINGLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def date_match_span(text: str) -> tuple[int, int] | None:
    """Char span of the first date-like match (range or bare year), or None."""
    match = _RANGE_RE.search(text)
    if match:
        return match.span()
    match = _SINGLE_YEAR_RE.search(text)
    if match:
        return match.span()
    return None


def find_date_range(text: str) -> tuple[str | None, str | None, bool]:
    """Returns (start, end, is_current)."""
    match = _RANGE_RE.search(text)
    if match:
        start, end = match.group(1), match.group(2)
        is_current = end.lower() in {"present", "current", "ongoing"}
        return start, (None if is_current else end), is_current

    years = _SINGLE_YEAR_RE.findall(text)
    if years:
        # findall with a group returns the captured group only; re-search fully.
        full_years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if len(full_years) >= 2:
            return full_years[0], full_years[-1], False
        if full_years:
            return None, full_years[0], False
    return None, None, False


def estimate_years_between(start: str | None, end: str | None) -> float:
    import datetime

    def _year(token: str | None) -> int | None:
        if not token:
            return None
        if token.lower() in {"present", "current", "ongoing"}:
            return datetime.date.today().year
        m = re.search(r"(19|20)\d{2}", token)
        return int(m.group()) if m else None

    y1, y2 = _year(start), _year(end)
    if y1 and y2 and y2 >= y1:
        return float(y2 - y1)
    return 0.0
