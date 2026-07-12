"""Targeted regression tests for experience.py's role/org extraction.

test_pipeline.py covers this indirectly through the shared fixture résumé,
but the acronym-prefix bug below was found against a real user-submitted
résumé, not the fixture — worth a dedicated, minimal repro so it can't
silently regress.
"""

from __future__ import annotations

from app.parser.experience import extract_experience


def test_org_name_keeps_leading_acronym_prefix():
    """Statistical NER tags only the recognizable tail of a company name
    ("RND Pvt. Ltd.") and drops a leading acronym it doesn't recognize
    ("IEMA") — the org field must keep the full name, not the NER-truncated
    version.
    """
    block = "IoT Development Intern - IEMA RND Pvt. Ltd., Kolkata Dec 2025 - March 2026"
    entries = extract_experience(block)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.role_title is not None
    assert entry.role_title.value == "IoT Development Intern"
    assert entry.organization is not None
    # startswith rather than == : with the transformer disabled (the default
    # for the fast test suite — see conftest.py), spaCy-only NER sometimes
    # also swallows the trailing ", Kolkata" into the same span, which is a
    # separate, pre-existing, lower-priority quirk (location leaking into
    # the org field) — not what this test is guarding against. What matters
    # here is that "IEMA" is never dropped.
    assert entry.organization.value.startswith("IEMA RND Pvt. Ltd.")


def test_org_name_without_acronym_prefix_unaffected():
    """Sanity check the fix doesn't regress the common case: when NER
    already captures the full org name from the start, nothing should be
    prepended."""
    block = "Data Engineer Intern - Sinha's GmbH, Switzerland (Remote) Mar 2026 - July 2026"
    entries = extract_experience(block)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.organization is not None
    assert "Sinha's GmbH" in entry.organization.value
    # Must not have grown a spurious prefix from unrelated preceding text.
    assert not entry.organization.value.startswith("Data Engineer")


def test_org_name_prefix_not_stolen_across_clause_boundary():
    """A comma between the preceding clause and the org name is a genuine
    boundary (e.g. a role title ending) — text before it must never be
    glued onto the org name even if it superficially looks acronym-like."""
    block = "Analyst, ABC Corp Jan 2020 - Dec 2021"
    entries = extract_experience(block)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.organization is not None
    assert "Analyst" not in entry.organization.value
