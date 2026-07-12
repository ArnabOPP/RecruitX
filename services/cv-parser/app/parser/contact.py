"""Contact-info extraction: name, email, phone, links.

Regex handles the genuinely regular formats (email, phone, URLs) with high
precision. Name extraction is the hard part — it's an unlabeled free-text
token at the very top of the document — so we combine a positional heuristic
(first non-empty line before any contact regex matches) with NER PERSON
entities restricted to the header block, and prefer their agreement.
"""

from __future__ import annotations

import re

from .ner import EntitySpan, extract_entities
from .schemas import ConfidentField, ContactInfo, ExtractionMethod, SourceSpan

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().\-]{7,16}\d")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_/]+", re.I)
_URL_RE = re.compile(r"https?://[^\s,)]+|(?:www\.)[^\s,)]+\.[a-z]{2,}[^\s,)]*", re.I)


def _valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 7 <= len(digits) <= 15


def extract_contact(header_block: str, full_text: str) -> ContactInfo:
    contact = ContactInfo()
    search_text = header_block or full_text[:800]

    emails = list(dict.fromkeys(_EMAIL_RE.findall(search_text) or _EMAIL_RE.findall(full_text)))
    contact.emails = [
        ConfidentField(value=e, confidence=0.99, method=ExtractionMethod.REGEX)
        for e in emails
    ]

    phone_candidates = _PHONE_RE.findall(search_text) or []
    phones = [p.strip() for p in phone_candidates if _valid_phone(p) and len(re.sub(r"\D", "", p)) >= 7]
    seen = set()
    deduped_phones = []
    for p in phones:
        norm = re.sub(r"\D", "", p)
        if norm in seen:
            continue
        seen.add(norm)
        deduped_phones.append(p)
    contact.phones = [
        ConfidentField(value=p, confidence=0.9, method=ExtractionMethod.REGEX)
        for p in deduped_phones[:3]
    ]

    linkedin_match = _LINKEDIN_RE.search(full_text)
    if linkedin_match:
        contact.linkedin = ConfidentField(
            value=_normalize_url(linkedin_match.group()),
            confidence=0.98,
            method=ExtractionMethod.REGEX,
        )

    github_match = _GITHUB_RE.search(full_text)
    if github_match:
        contact.github = ConfidentField(
            value=_normalize_url(github_match.group()),
            confidence=0.98,
            method=ExtractionMethod.REGEX,
        )

    other_urls = [
        _normalize_url(u)
        for u in _URL_RE.findall(full_text)
        if "linkedin.com" not in u.lower() and "github.com" not in u.lower()
    ]
    contact.portfolio_urls = [
        ConfidentField(value=u, confidence=0.85, method=ExtractionMethod.REGEX)
        for u in dict.fromkeys(other_urls)
    ]

    contact.full_name = _extract_name(search_text)
    return contact


def _normalize_url(url: str) -> str:
    url = url.rstrip(".,;")
    if not url.lower().startswith("http"):
        url = "https://" + url
    return url


def _extract_name(header_block: str) -> ConfidentField | None:
    if not header_block.strip():
        return None

    lines = [line.strip() for line in header_block.split("\n") if line.strip()]
    positional_candidate = None
    for line in lines[:3]:
        if _EMAIL_RE.search(line) or _PHONE_RE.search(line) or _URL_RE.search(line):
            continue
        words = line.split()
        if 1 <= len(words) <= 5 and all(w[0].isupper() or not w[0].isalpha() for w in words if w):
            positional_candidate = line
            break

    entities: list[EntitySpan] = extract_entities(header_block[:300])
    person_entities = [e for e in entities if e.label == "PERSON"]

    if person_entities:
        best = max(person_entities, key=lambda e: e.confidence)
        confidence = best.confidence
        method = ExtractionMethod.ENSEMBLE if best.method == "ensemble" else ExtractionMethod.TRANSFORMER_NER
        if positional_candidate and best.text.lower() in positional_candidate.lower():
            confidence = min(0.99, confidence + 0.1)
        return ConfidentField(
            value=best.text.strip(),
            confidence=confidence,
            method=method,
            source=SourceSpan(section="header", text=header_block[:200]),
        )

    if positional_candidate:
        return ConfidentField(
            value=positional_candidate,
            confidence=0.55,
            method=ExtractionMethod.RULE_SECTION,
            source=SourceSpan(section="header", text=header_block[:200]),
        )

    return None
