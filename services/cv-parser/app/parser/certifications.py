"""Certifications section extraction.

Résumés list certifications two ways: one per line, or as a dense
bullet-separated ("•") run-on list. We split on both. Issuer-vs-name order
also varies ("Coursera - Deep Learning Specialization" vs. "Accenture -
Data Analytics (2025)"), so rather than assume a fixed order we treat the
shorter of the two dash-separated segments as the issuer — company/platform
names are reliably shorter than certification titles.
"""

from __future__ import annotations

import re

from .schemas import CertificationEntry
from .sections import split_bullets

_DATE_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_DASH_SPLIT_RE = re.compile(r"\s+-\s+")


def extract_certifications(section_body: str) -> list[CertificationEntry]:
    if not section_body.strip():
        return []

    # Dense single-line lists use "•" as an inline separator, not a leading
    # bullet marker — split on it before falling back to newline-based bullets.
    if "•" in section_body:
        raw_entries = [p.strip(" •") for p in section_body.split("•") if p.strip(" •")]
    else:
        raw_entries = split_bullets(section_body)

    entries = []
    for raw in raw_entries:
        name = raw
        issuer = None
        date = None

        date_match = _DATE_RE.search(raw)
        if date_match:
            date = date_match.group(1)
            name = raw[: date_match.start()].strip(" -–,|()")

        parts = _DASH_SPLIT_RE.split(name, maxsplit=1)
        if len(parts) == 2:
            first, second = parts[0].strip(), parts[1].strip()
            if len(first.split()) <= len(second.split()):
                issuer, name = first, second
            else:
                issuer, name = second, first

        name = name.strip(" -–,|()")
        if not name:
            continue
        entries.append(CertificationEntry(name=name, issuer=issuer, date=date, raw_text=raw))
    return entries
