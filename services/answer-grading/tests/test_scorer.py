"""Tests for the scoring engine's combination logic: rubric resolution,
weighting, and aggregation. Uses a FakeSemanticScorer so these tests are
fast and isolated from real embedding-model behavior — the real model's
own correctness is verified separately in test_semantic.py."""

from __future__ import annotations

from app.grading.scorer import resolve_rubric, score_answer, score_criterion
from app.schemas import GroundingEvidence, Rubric, RubricCriterion
from tests.fakes import FakeSemanticScorer


def test_resolve_rubric_prefers_explicit_rubric():
    rubric = Rubric(criteria=[RubricCriterion(description="custom", weight=1.0, expected_keywords=["x"])])
    grounding = GroundingEvidence(kind="skill", reference="Python")
    resolved, source = resolve_rubric("some question", grounding, rubric)
    assert source == "provided"
    assert resolved is rubric


def test_resolve_rubric_from_grounding_prefers_detail_over_reference():
    """detail is the substantive technical content; reference for
    kind=experience/education is often just an org/job title that a good
    answer has no real reason to repeat — keywords should come from detail
    alone when it's present, not be diluted with reference's metadata."""
    grounding = GroundingEvidence(
        kind="experience", reference="Software Engineering Intern @ Flipkart", detail="optimized PostgreSQL queries"
    )
    resolved, source = resolve_rubric("some question", grounding, None)
    assert source == "auto_derived_from_grounding"
    keywords = resolved.criteria[0].expected_keywords
    assert "postgresql" in keywords
    assert "optimized" in keywords
    assert "flipkart" not in keywords
    assert "intern" not in keywords


def test_resolve_rubric_from_grounding_falls_back_to_reference_when_no_detail():
    grounding = GroundingEvidence(kind="skill", reference="Kubernetes", detail=None)
    resolved, source = resolve_rubric("some question", grounding, None)
    assert source == "auto_derived_from_grounding"
    assert "kubernetes" in resolved.criteria[0].expected_keywords


def test_resolve_rubric_falls_back_to_question_when_nothing_else_given():
    resolved, source = resolve_rubric("How did you use PostgreSQL indexing?", None, None)
    assert source == "auto_derived_from_question"
    assert "postgresql" in resolved.criteria[0].expected_keywords
    assert "indexing" in resolved.criteria[0].expected_keywords


def test_score_criterion_combines_jaccard_and_semantic_with_configured_weights(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("ANSWER_GRADING_JACCARD_WEIGHT", "0.5")
    monkeypatch.setenv("ANSWER_GRADING_SEMANTIC_WEIGHT", "0.5")
    get_settings.cache_clear()

    criterion = RubricCriterion(description="uses postgresql", weight=1.0, expected_keywords=["postgresql", "index"])
    fake = FakeSemanticScorer(default=0.8)

    result = score_criterion(criterion, "I used postgresql with an index.", fake)

    # jaccard: keyword_tokens={postgresql,index}, answer_tokens={used,postgresql,index}
    # (stopwords "I","with","an" dropped) -> intersection=2, union=3 -> 2/3
    # score_criterion rounds to 4 decimal places, so the tolerance here
    # must be looser than the rounding step itself, not tighter.
    assert abs(result.jaccard - 2 / 3) < 1e-4
    assert result.semantic == 0.8
    assert abs(result.score - (0.5 * (2 / 3) + 0.5 * 0.8)) < 1e-4
    assert result.matched_keywords == ["postgresql", "index"]
    assert result.missing_keywords == []

    get_settings.cache_clear()


def test_score_criterion_no_expected_keywords_relies_purely_on_semantic():
    criterion = RubricCriterion(description="general relevance", weight=1.0, expected_keywords=[])
    fake = FakeSemanticScorer(default=0.6)

    result = score_criterion(criterion, "any answer at all", fake)

    assert result.jaccard == 0.0  # no keyword tokens -> empty set -> 0 by definition
    assert result.semantic == 0.6
    assert result.matched_keywords == []
    assert result.missing_keywords == []


def test_score_answer_aggregates_multiple_criteria_by_weight():
    rubric = Rubric(
        criteria=[
            RubricCriterion(description="c1", weight=3.0, expected_keywords=["alpha"]),
            RubricCriterion(description="c2", weight=1.0, expected_keywords=["beta"]),
        ]
    )
    fake = FakeSemanticScorer(default=0.0)  # isolate to pure jaccard behavior
    result = score_answer("q", "alpha", None, rubric, fake)

    # c1: answer="alpha" tokens={alpha}, keyword_tokens={alpha} -> jaccard=1.0 -> score=jaccard_weight*1.0
    # c2: answer tokens={alpha}, keyword_tokens={beta} -> jaccard=0.0 -> score=0.0
    # weighted avg = (3*c1_score + 1*0.0) / 4
    from app.config import get_settings

    settings = get_settings()
    expected_c1_score = settings.jaccard_weight * 1.0
    expected_overall = (3 * expected_c1_score + 1 * 0.0) / 4
    assert abs(result.overall_score - expected_overall) < 1e-4
    assert result.rubric_source == "provided"
    assert len(result.criteria_scores) == 2


def test_score_answer_is_fully_reproducible():
    """The core guarantee this whole service exists for: identical input
    must produce byte-identical output, every time."""
    rubric = Rubric(criteria=[RubricCriterion(description="c1", weight=1.0, expected_keywords=["python", "fastapi"])])
    fake = FakeSemanticScorer(default=0.42)

    result_a = score_answer("q", "I used Python and FastAPI to build this.", None, rubric, fake)
    result_b = score_answer("q", "I used Python and FastAPI to build this.", None, rubric, fake)

    assert result_a.model_dump() == result_b.model_dump()


def test_score_answer_empty_rubric_source_relevant_to_question_only():
    result = score_answer("What is your favorite database?", "PostgreSQL, definitely.", None, None, FakeSemanticScorer(default=0.5))
    assert result.rubric_source == "auto_derived_from_question"
    assert result.overall_score > 0
