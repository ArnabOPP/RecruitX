"""The deterministic grading engine: combines Jaccard + keyword weighting
+ semantic similarity into a rubric-based, fully auditable score.

Per the BRD's design principle (established in interview-qa) — "LLMs
propose and converse; the deterministic engine decides the score" — nothing
in this module calls an LLM. Given the same question/answer/rubric, this
always produces the same score: reproducibility is the entire point.
"""

from __future__ import annotations

from ..config import get_settings
from ..schemas import CriterionScore, GroundingEvidence, Rubric, RubricCriterion, ScoreResponse
from .keywords import jaccard_similarity, keyword_coverage, normalize_tokens
from .semantic import SemanticSimilarityScorer


def _derive_rubric_from_grounding(grounding: GroundingEvidence) -> Rubric:
    """interview-qa's generated questions already carry a `grounding`
    object (the résumé evidence the question is anchored to) — when the
    caller doesn't supply an explicit rubric, that evidence is enough to
    build a reasonable default one: a good answer should mention the same
    terms the question was grounded in.

    `detail` (e.g. "reducing report generation time by 40% using indexes")
    is the substantive technical content; `reference` for kind=experience/
    education is often just an org or job title (e.g. "Software Engineering
    Intern @ Flipkart") that a good answer has no real reason to repeat.
    When detail is present, keywords come from it alone rather than
    diluting the set with reference's organizational metadata; reference
    is only used as a fallback when there's no detail to draw on.
    """
    keyword_source = grounding.detail if grounding.detail else grounding.reference
    keywords = sorted(normalize_tokens(keyword_source))
    description = f'Answer should meaningfully address the {grounding.kind} "{grounding.reference}"'
    if grounding.detail:
        description += f" — specifically: {grounding.detail}"
    return Rubric(criteria=[RubricCriterion(description=description, weight=1.0, expected_keywords=keywords)])


def _derive_rubric_from_question(question: str) -> Rubric:
    """Last-resort fallback when there's no rubric and no grounding at
    all — still produces a scorable (if weaker) rubric from the question's
    own content words, so the endpoint never simply refuses to score."""
    keywords = sorted(normalize_tokens(question))
    return Rubric(
        criteria=[
            RubricCriterion(
                description="Answer should be relevant to the question asked",
                weight=1.0,
                expected_keywords=keywords,
            )
        ]
    )


def resolve_rubric(
    question: str, grounding: GroundingEvidence | None, rubric: Rubric | None
) -> tuple[Rubric, str]:
    if rubric is not None:
        return rubric, "provided"
    if grounding is not None:
        return _derive_rubric_from_grounding(grounding), "auto_derived_from_grounding"
    return _derive_rubric_from_question(question), "auto_derived_from_question"


def score_criterion(
    criterion: RubricCriterion, candidate_answer: str, semantic_scorer: SemanticSimilarityScorer
) -> CriterionScore:
    settings = get_settings()

    answer_tokens = normalize_tokens(candidate_answer)
    keyword_tokens: set[str] = set()
    for kw in criterion.expected_keywords:
        keyword_tokens |= normalize_tokens(kw)

    jaccard = jaccard_similarity(keyword_tokens, answer_tokens)
    matched, missing = keyword_coverage(criterion.expected_keywords, candidate_answer)

    reference_text = criterion.description
    if criterion.expected_keywords:
        reference_text += ". Relevant terms: " + ", ".join(criterion.expected_keywords)
    semantic = semantic_scorer.similarity(reference_text, candidate_answer)

    combined = settings.jaccard_weight * jaccard + settings.semantic_weight * semantic

    return CriterionScore(
        description=criterion.description,
        weight=criterion.weight,
        score=round(combined, 4),
        jaccard=round(jaccard, 4),
        semantic=round(semantic, 4),
        matched_keywords=matched,
        missing_keywords=missing,
    )


def _build_explanation(criteria_scores: list[CriterionScore], overall: float) -> str:
    if not criteria_scores:
        return "No rubric criteria to score against."
    parts = []
    for cs in criteria_scores:
        if cs.matched_keywords:
            parts.append(f'"{cs.description}" scored {cs.score:.2f} — mentioned {", ".join(cs.matched_keywords)}')
        elif cs.missing_keywords:
            parts.append(
                f'"{cs.description}" scored {cs.score:.2f} — did not clearly address: {", ".join(cs.missing_keywords)}'
            )
        else:
            parts.append(f'"{cs.description}" scored {cs.score:.2f}')
    return f"Overall score {overall:.2f}. " + "; ".join(parts) + "."


def score_answer(
    question: str,
    candidate_answer: str,
    grounding: GroundingEvidence | None,
    rubric: Rubric | None,
    semantic_scorer: SemanticSimilarityScorer,
) -> ScoreResponse:
    resolved_rubric, rubric_source = resolve_rubric(question, grounding, rubric)

    criteria_scores = [score_criterion(c, candidate_answer, semantic_scorer) for c in resolved_rubric.criteria]

    total_weight = sum(c.weight for c in resolved_rubric.criteria)
    overall = (
        sum(cs.score * c.weight for cs, c in zip(criteria_scores, resolved_rubric.criteria, strict=True)) / total_weight
        if total_weight
        else 0.0
    )

    return ScoreResponse(
        overall_score=round(overall, 4),
        rubric_source=rubric_source,
        criteria_scores=criteria_scores,
        explanation=_build_explanation(criteria_scores, overall),
    )
