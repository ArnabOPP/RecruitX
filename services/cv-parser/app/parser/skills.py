"""Skill extraction and cross-referencing.

Skills are gathered from three places: the dedicated Skills section (highest
confidence — the candidate self-declared it), and Projects/Experience bodies
via gazetteer NER (evidence that the skill was actually *used*, not just
listed). A skill mentioned in a project or role description is flagged
`evidenced_in_project` / `evidenced_in_experience`, which is exactly the
signal the BRD's Round-2 CV-grounded interview needs to ask "you used X in
project Y — how did you handle Z?" rather than generic questions.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

from .ner import extract_entities
from .schemas import ExtractionMethod, Skill, SkillCategory

# Threshold picked from measured separation: real OCR misreads of a canonical
# skill name ("fastapl" vs "fastapi") score ~86; unrelated words that just
# happen to share some letters ("git" vs "github", "java" vs "javascript")
# top out around 67. 84 sits safely in that gap. Tokens shorter than this are
# excluded entirely — fuzzy ratios on very short strings ("$3" vs "s3" = 50)
# are too noisy to trust either direction.
_FUZZY_SKILL_THRESHOLD = 84
_FUZZY_MIN_TOKEN_LEN = 4

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_CATEGORY_MAP = {
    "programming_language": SkillCategory.PROGRAMMING_LANGUAGE,
    "framework_library": SkillCategory.FRAMEWORK_LIBRARY,
    "database": SkillCategory.DATABASE,
    "cloud_devops": SkillCategory.CLOUD_DEVOPS,
    "data_ml": SkillCategory.DATA_ML,
    "tool_platform": SkillCategory.TOOL_PLATFORM,
    "soft_skill": SkillCategory.SOFT_SKILL,
    "domain_knowledge": SkillCategory.DOMAIN_KNOWLEDGE,
}

_SPLIT_RE = re.compile(r"[,•|/\n]|(?:\s{2,})|(?:\s-\s)")
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_QUALIFIER_WORDS = {
    "basic", "basics", "intermediate", "advanced", "familiar", "proficient",
    "beginner", "expert", "fundamentals",
}


@lru_cache(maxsize=1)
def _canonical_lookup() -> dict[str, str]:
    """lowercase alias -> canonical display name, for normalizing free-text skills."""
    with open(_DATA_DIR / "skills_db.json", encoding="utf-8") as f:
        db = json.load(f)
    lookup = {}
    for terms in db.values():
        for term in terms:
            lookup[term.lower()] = term
    return lookup


def _normalize(name: str) -> str:
    name = name.strip().strip(".:;")
    lname = name.lower()
    canonical = _canonical_lookup().get(lname)
    if canonical:
        return canonical
    if len(lname) >= _FUZZY_MIN_TOKEN_LEN:
        # Catches OCR-garbled skill tokens ("FastAPl", a lowercase-L misread
        # of "FastAPI") by fuzzy-matching against the curated vocabulary,
        # instead of letting them become their own separate junk skill.
        match = process.extractOne(
            lname,
            _canonical_lookup().keys(),
            scorer=fuzz.ratio,
            score_cutoff=_FUZZY_SKILL_THRESHOLD,
        )
        if match:
            return _canonical_lookup()[match[0]]
    return name


def _category_for(name: str) -> SkillCategory:
    with open(_DATA_DIR / "skills_db.json", encoding="utf-8") as f:
        db = json.load(f)
    lname = name.lower()
    for category, terms in db.items():
        if lname in {t.lower() for t in terms}:
            return _CATEGORY_MAP[category]
    return SkillCategory.OTHER


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace the given char ranges with spaces so a subsequent naive split
    doesn't re-tokenize fragments of an already-matched multi-char skill."""
    chars = list(text)
    for start, end in spans:
        for i in range(start, min(end, len(chars))):
            chars[i] = " "
    return "".join(chars)


def _extract_paren_subtokens(text: str) -> tuple[str, list[str]]:
    """Pull comma-separated content out of "(...)" groups as standalone
    candidate tokens (e.g. "AWS (EC2, S3, IAM)" -> ["EC2", "S3", "IAM"]),
    rather than letting a naive comma-split fragment the outer token into
    "AWS (EC2" / "S3" / "IAM)". A lone qualifier word in parens (e.g.
    "Java(basic)") is dropped rather than emitted as a fake skill.
    """
    subtokens: list[str] = []

    def _replace(match: re.Match) -> str:
        inner = match.group(1).strip()
        if not inner:
            return " "
        if len(inner.split()) == 1 and inner.strip(".").lower() in _QUALIFIER_WORDS:
            return " "
        for part in re.split(r"[,/]", inner):
            part = part.strip()
            if part:
                subtokens.append(part)
        return " "

    return _PAREN_RE.sub(_replace, text), subtokens


def _clean_token(token: str) -> str:
    return token.strip().strip(" .,;:|()-").strip()


def extract_skills_from_section(skills_section_body: str) -> dict[str, Skill]:
    """High-confidence skills explicitly listed by the candidate.

    Two-pass extraction: the curated-vocabulary gazetteer runs first and
    respects real token/word boundaries (so "Firebase Firestore" correctly
    yields two skills, not one unmatched compound string, and "AWS" isn't
    mangled by a comma sitting inside its parenthetical). Whatever text is
    left over after blanking out those matches is then naively split to
    catch genuinely novel terms not in the curated list (e.g. "DSA", "EC2").
    """
    skills: dict[str, Skill] = {}
    if not skills_section_body:
        return skills

    # Strip category labels like "Languages: Python, Java"
    cleaned_lines = []
    for line in skills_section_body.split("\n"):
        line = re.sub(r"^[A-Za-z /&]{2,30}:\s*", "", line)
        cleaned_lines.append(line)
    joined = "\n".join(cleaned_lines)

    gazetteer_entities = [e for e in extract_entities(joined) if e.label.startswith("SKILL::")]
    for ent in gazetteer_entities:
        normalized = _normalize(ent.text)
        key = normalized.lower()
        category_key = ent.label.split("::", 1)[1]
        if key in skills:
            skills[key].mention_count += 1
            continue
        skills[key] = Skill(
            name=normalized,
            normalized_name=key,
            category=_CATEGORY_MAP.get(category_key, SkillCategory.OTHER),
            confidence=0.97,
            method=ExtractionMethod.GAZETTEER,
        )

    remaining = _blank_spans(joined, [(e.start_char, e.end_char) for e in gazetteer_entities])
    remaining, paren_subtokens = _extract_paren_subtokens(remaining)

    raw_tokens = list(_SPLIT_RE.split(remaining)) + paren_subtokens
    for raw_token in raw_tokens:
        token = _clean_token(raw_token)
        if len(token) > 40 or len(token) < 2:
            continue
        normalized = _normalize(token)
        key = normalized.lower()
        if key in skills:
            skills[key].mention_count += 1
            continue
        skills[key] = Skill(
            name=normalized,
            normalized_name=key,
            category=_category_for(normalized),
            confidence=0.88,
            method=ExtractionMethod.RULE_SECTION,
        )
    return skills


def extract_skills_from_gazetteer(text: str) -> dict[str, Skill]:
    """Skills found anywhere via the spaCy PhraseMatcher gazetteer (lower
    confidence than an explicit Skills-section mention, since context matters
    less to a gazetteer hit).
    """
    skills: dict[str, Skill] = {}
    entities = extract_entities(text)
    for ent in entities:
        if not ent.label.startswith("SKILL::"):
            continue
        category_key = ent.label.split("::", 1)[1]
        normalized = _normalize(ent.text)
        key = normalized.lower()
        if key in skills:
            skills[key].mention_count += 1
            continue
        skills[key] = Skill(
            name=normalized,
            normalized_name=key,
            category=_CATEGORY_MAP.get(category_key, SkillCategory.OTHER),
            confidence=0.8,
            method=ExtractionMethod.GAZETTEER,
        )
    return skills


def merge_skills(
    section_skills: dict[str, Skill],
    project_text: str,
    experience_text: str,
) -> list[Skill]:
    merged: dict[str, Skill] = {k: v.model_copy() for k, v in section_skills.items()}

    project_skills = extract_skills_from_gazetteer(project_text) if project_text else {}
    experience_skills = extract_skills_from_gazetteer(experience_text) if experience_text else {}

    for key, skill in project_skills.items():
        if key in merged:
            merged[key].evidenced_in_project = True
            merged[key].mention_count += skill.mention_count
            merged[key].confidence = min(0.99, merged[key].confidence + 0.03)
        else:
            skill.evidenced_in_project = True
            merged[key] = skill

    for key, skill in experience_skills.items():
        if key in merged:
            merged[key].evidenced_in_experience = True
            merged[key].mention_count += skill.mention_count
            merged[key].confidence = min(0.99, merged[key].confidence + 0.03)
        else:
            skill.evidenced_in_experience = True
            merged[key] = skill

    return sorted(
        merged.values(),
        key=lambda s: (s.evidenced_in_project or s.evidenced_in_experience, s.confidence),
        reverse=True,
    )
