"""Regression tests for robustness against OCR-specific noise.

Both bugs here came from a real OCR'd résumé (a phone screenshot run through
Tesseract), not the fixture: the same "•" bullet glyph got read back as
different symbols in different spots of the same image, and a couple of
skill names had a single character misread (capital I as lowercase l).
"""

from __future__ import annotations

from app.parser.projects import extract_projects
from app.parser.sections import split_bullets
from app.parser.skills import _normalize

# --- Bullet-prefix generalization -------------------------------------------


def test_bullet_detection_handles_ocr_misread_glyphs():
    """The same underlying "•" character, misread by OCR as three different
    symbols across three lines, must still be recognized as a bullet each
    time — not just the one canonical "•" character."""
    body = "© First point\n¢ Second point\n» Third point"
    bullets = split_bullets(body)
    assert bullets == ["First point", "Second point", "Third point"]


def test_bullet_detection_still_ignores_ordinary_text():
    # Normal body text starting with a letter must never be treated as a
    # bullet-prefixed line — only lines starting with a lone symbol.
    body = "Experienced backend engineer.\nBuilt several APIs."
    bullets = split_bullets(body)
    assert bullets == ["Experienced backend engineer. Built several APIs."]


def test_bullet_detection_does_not_break_on_mid_word_hyphen():
    # "Real-Time" must not be mistaken for a bullet — the hyphen is mid-word,
    # not a leading bullet character.
    body = "Real-Time Chat Application"
    bullets = split_bullets(body)
    assert bullets == ["Real-Time Chat Application"]


def test_project_split_survives_ocr_misread_bullet():
    """The actual bug: a project's description bullet, OCR'd with a "©"
    instead of "•", was being misread as an entirely new project entry
    (titled after the bullet's own text) instead of a continuation line."""
    block = (
        "Workforce Vision - Employee platform - React, MongoDB February 2026\n"
        "© Built a cloud-based employee management platform for a client.\n"
        "RecruitX - AI Interview Assistant - React November 2025\n"
        "© Built and deployed an AI-proctored interview platform."
    )
    projects = extract_projects(block)
    assert len(projects) == 2
    assert projects[0].title is not None
    assert projects[0].title.value == "Workforce Vision"
    assert projects[1].title is not None
    assert projects[1].title.value == "RecruitX"


def test_project_split_survives_ocr_blank_line_insertion():
    """Tesseract's image_to_string doesn't just misread the bullet glyph —
    it also inserts a blank line between essentially every visually-separated
    line, including between a title and its own description bullet. That
    blank line makes group_entries' fast path (split on blank lines) treat
    each bullet as a new entry before the bullet-prefix check ever runs, so
    the fix has to hold even with those extra blank lines present — this is
    what an actual OCR'd image produces, not just the single-newline version
    of the bug above.
    """
    block = (
        "Workforce Vision - Employee platform - React, MongoDB February 2026\n"
        "\n"
        "© Built a cloud-based employee management platform for a client.\n"
        "\n"
        "RecruitX - AI Interview Assistant - React November 2025\n"
        "\n"
        "© Built and deployed an AI-proctored interview platform."
    )
    projects = extract_projects(block)
    assert len(projects) == 2
    assert projects[0].title is not None
    assert projects[0].title.value == "Workforce Vision"
    assert projects[0].description is not None
    assert "cloud-based employee management" in projects[0].description
    assert projects[1].title is not None
    assert projects[1].title.value == "RecruitX"


# --- Fuzzy canonical skill matching ------------------------------------------


def test_ocr_garbled_skill_name_normalizes_to_canonical():
    # "FastAPl" (lowercase L instead of capital I) is a single-character OCR
    # misread of "FastAPI" — must resolve to the canonical name so it merges
    # with any correctly-OCR'd mentions instead of becoming a duplicate.
    assert _normalize("FastAPl") == "FastAPI"


def test_short_tokens_are_not_fuzzy_matched():
    # Fuzzy ratios on very short strings are too unreliable to trust safely
    # ("$3" vs "S3" scores only 50/100) — these must pass through unchanged
    # rather than risk a wrong "correction".
    assert _normalize("$3") == "$3"


def test_ambiguous_word_is_not_force_merged_into_similar_skill():
    # "Nodes" could legitimately mean something other than "Node.js" (e.g.
    # data-structure nodes) — its fuzzy-ratio similarity to "Node.js" (~83)
    # sits below the safety threshold, so it must be left alone rather than
    # silently rewritten.
    assert _normalize("Nodes") == "Nodes"


def test_unrelated_words_are_never_fuzzy_matched():
    assert _normalize("RecruitX") == "RecruitX"
    assert _normalize("Workforce") == "Workforce"
