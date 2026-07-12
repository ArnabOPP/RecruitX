"""Hybrid NER engine: spaCy (rules + statistical model) ensembled with a
transformer (BERT-class) sequence-tagger.

Design rationale
-----------------
Résumés are entity-dense but low-context (short noun phrases, no full
sentences), which is exactly where generic statistical NER is weakest and
where curated gazetteers (skills, degrees) shine. So we split responsibility:

* spaCy `EntityRuler` + `PhraseMatcher` over curated skill/degree taxonomies
  → near-perfect precision for known technical vocabulary.
* spaCy's statistical NER (en_core_web_sm/lg/trf) → PERSON, ORG, GPE, DATE as
  a first pass, cheap and fast.
* A transformer token-classification model (BERT-class, e.g. dslim/bert-base-NER)
  → a second, independently-trained opinion on PERSON/ORG/LOC, used to raise
  confidence when it agrees with spaCy and to catch entities spaCy misses
  (transformer NER generalises better to unseen names/orgs).

The transformer is optional and lazily loaded: if model weights aren't
available (no network / offline sandbox), the engine degrades gracefully to
spaCy-only with a logged warning rather than failing the whole pipeline.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from ..config import get_settings

logger = logging.getLogger("cv_parser.ner")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _settings():
    return get_settings()


@dataclass
class EntitySpan:
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    method: str


def _load_json(name: str) -> dict:
    with open(_DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _skills_db() -> dict[str, list[str]]:
    return _load_json("skills_db.json")


@lru_cache(maxsize=1)
def _degrees_db() -> dict[str, list[str]]:
    return _load_json("degrees_db.json")


@lru_cache(maxsize=1)
def get_spacy_pipeline() -> spacy.language.Language:
    model_name = _settings().spacy_model
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.warning(
            "spaCy model '%s' not found; falling back to a blank English "
            "pipeline (gazetteer matching still works, statistical NER does not).",
            model_name,
        )
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    _attach_skill_matcher(nlp)
    _attach_degree_matcher(nlp)
    return nlp


def _attach_skill_matcher(nlp: spacy.language.Language) -> None:
    if "skill_matcher" in nlp.pipe_names:
        return
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for category, terms in _skills_db().items():
        patterns = [nlp.make_doc(term) for term in terms]
        matcher.add(f"SKILL::{category}", patterns)

    @spacy.language.Language.component("skill_matcher")
    def skill_matcher_component(doc: Doc) -> Doc:
        doc.ents = _apply_gazetteer_matches(doc, matcher(doc), nlp)
        return doc

    nlp.add_pipe("skill_matcher", last=True)


def _attach_degree_matcher(nlp: spacy.language.Language) -> None:
    if "degree_matcher" in nlp.pipe_names:
        return
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for level, terms in _degrees_db().items():
        patterns = [nlp.make_doc(term) for term in terms]
        matcher.add(f"DEGREE::{level}", patterns)

    @spacy.language.Language.component("degree_matcher")
    def degree_matcher_component(doc: Doc) -> Doc:
        doc.ents = _apply_gazetteer_matches(doc, matcher(doc), nlp)
        return doc

    nlp.add_pipe("degree_matcher", last=True)


def _apply_gazetteer_matches(doc: Doc, matches, nlp: spacy.language.Language) -> list[Span]:
    """Add gazetteer (curated-vocabulary) matches to doc.ents, evicting any
    overlapping *statistical* NER entities — a curated skill/degree match is
    higher precision on résumé text than the base model's generic guess
    (e.g. "B.Tech" mistagged ORG). Overlaps between two gazetteer spans still
    defer to the longer span via `_dedupe_spans`.
    """
    is_gazetteer = lambda ent: ent.label_.startswith(("SKILL::", "DEGREE::"))  # noqa: E731
    kept_ents = [e for e in doc.ents if is_gazetteer(e)]
    statistical_ents = [e for e in doc.ents if not is_gazetteer(e)]

    new_spans = [
        Span(doc, start, end, label=nlp.vocab.strings[match_id])
        for match_id, start, end in matches
    ]

    all_gazetteer = kept_ents + new_spans
    deduped_gazetteer = _dedupe_spans(all_gazetteer)
    gazetteer_ranges = _occupied_ranges(deduped_gazetteer)

    surviving_statistical = [
        e for e in statistical_ents if not _overlaps(e.start, e.end, gazetteer_ranges)
    ]

    try:
        return _dedupe_spans(deduped_gazetteer + surviving_statistical)
    except ValueError:
        return list(doc.ents)


def _occupied_ranges(spans: list[Span]) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in spans]


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < r_end and end > r_start for r_start, r_end in ranges)


def _dedupe_spans(spans: list[Span]) -> list[Span]:
    spans_sorted = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    result: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for span in spans_sorted:
        if _overlaps(span.start, span.end, occupied):
            continue
        result.append(span)
        occupied.append((span.start, span.end))
    return result


class TransformerNer:
    """Thin, fail-soft wrapper around a HF token-classification pipeline."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _settings().transformer_model
        self._pipeline: Callable[..., list[dict]] | None = None
        self._load_attempted = False
        self.available = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not _settings().enable_transformer:
            logger.info("Transformer NER disabled via CV_PARSER_ENABLE_TRANSFORMER=0")
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
            )
            self.available = True
            logger.info("Transformer NER model '%s' loaded.", self.model_name)
        except Exception as exc:  # noqa: BLE001 - must never crash the pipeline
            logger.warning(
                "Transformer NER model '%s' unavailable (%s). "
                "Falling back to spaCy-only NER.",
                self.model_name,
                exc,
            )
            self._pipeline = None
            self.available = False

    def extract(self, text: str) -> list[EntitySpan]:
        self._ensure_loaded()
        if self._pipeline is None or not self.available or not text.strip():
            return []
        try:
            raw = self._pipeline(text[:4000])  # guard against pathological input length
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transformer NER inference failed: %s", exc)
            return []

        spans: list[EntitySpan] = []
        for ent in raw:
            label = ent.get("entity_group", ent.get("entity", ""))
            # Prefer a direct slice of the source text over ent["word"]: the
            # pipeline reconstructs "word" from wordpiece tokens via
            # convert_tokens_to_string, which can inject stray spaces around
            # punctuation (e.g. "Sinha's" -> "Sinha ' s"). start/end are
            # character offsets into the original text and stay accurate
            # regardless of that reconstruction quirk.
            span_text = text[ent["start"] : ent["end"]] or ent["word"]
            spans.append(
                EntitySpan(
                    text=span_text,
                    label=_normalize_transformer_label(label),
                    start_char=ent["start"],
                    end_char=ent["end"],
                    confidence=float(ent.get("score", 0.5)),
                    method="transformer_ner",
                )
            )
        return _merge_adjacent_spans(spans, text)


def _merge_adjacent_spans(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
    """Merge same-label spans that are touching or separated only by a single
    space/wordpiece boundary. Some tokenizer/aggregation-strategy combinations
    (e.g. BERT wordpiece continuations like "A" + "##ara" + "##v Sharma") are
    not fully coalesced by `aggregation_strategy="simple"` across library
    versions, so we coalesce defensively here rather than trust upstream.
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: s.start_char)
    merged: list[EntitySpan] = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        gap = text[prev.end_char : span.start_char]
        if span.label == prev.label and (gap == "" or gap == " "):
            merged[-1] = EntitySpan(
                text=text[prev.start_char : span.end_char],
                label=prev.label,
                start_char=prev.start_char,
                end_char=span.end_char,
                confidence=min(prev.confidence, span.confidence),
                method="transformer_ner",
            )
        else:
            merged.append(span)
    return merged


def _normalize_transformer_label(label: str) -> str:
    mapping = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE", "MISC": "MISC"}
    return mapping.get(label.upper(), label.upper())


@lru_cache(maxsize=1)
def get_transformer_ner() -> TransformerNer:
    return TransformerNer()


def extract_entities(text: str) -> list[EntitySpan]:
    """Run the full hybrid pipeline and return an ensembled, deduped span list.

    Agreement between spaCy and the transformer on PERSON/ORG boosts
    confidence; gazetteer-sourced SKILL/DEGREE spans are always
    high-confidence since they come from exact curated-vocabulary matches.
    """
    nlp = get_spacy_pipeline()
    doc = nlp(text)

    spacy_spans: list[EntitySpan] = []
    for ent in doc.ents:
        if ent.label_.startswith("SKILL::") or ent.label_.startswith("DEGREE::"):
            spacy_spans.append(
                EntitySpan(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.97,
                    method="gazetteer",
                )
            )
        else:
            spacy_spans.append(
                EntitySpan(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.65,
                    method="spacy_ner",
                )
            )

    transformer_spans = get_transformer_ner().extract(text)

    spacy_spans = [_clip_at_line_break(s, text) for s in spacy_spans]
    transformer_spans = [_clip_at_line_break(s, text) for s in transformer_spans]

    return _ensemble(spacy_spans, transformer_spans)


def _clip_at_line_break(span: EntitySpan, text: str) -> EntitySpan:
    """A single entity (name, org, skill...) never legitimately spans a line
    break on a résumé — dense single-column layouts routinely place a name
    directly above a location/contact line with no blank line between them,
    and generic statistical NER sometimes greedily swallows the next line
    (e.g. "DEBANGSHU CHATTERJEE\\nKolkata" tagged as one PERSON). Clip any
    span at the first newline it contains.
    """
    raw = text[span.start_char : span.end_char]
    newline_idx = raw.find("\n")
    if newline_idx == -1:
        return span
    clipped_end = span.start_char + newline_idx
    clipped_text = text[span.start_char : clipped_end].rstrip()
    return EntitySpan(
        text=clipped_text,
        label=span.label,
        start_char=span.start_char,
        end_char=span.start_char + len(clipped_text),
        confidence=span.confidence,
        method=span.method,
    )


def _ensemble(spacy_spans: list[EntitySpan], transformer_spans: list[EntitySpan]) -> list[EntitySpan]:
    gazetteer = [s for s in spacy_spans if s.method == "gazetteer"]
    statistical_spacy = [s for s in spacy_spans if s.method != "gazetteer"]

    merged: list[EntitySpan] = list(gazetteer)
    used_transformer_idx: set[int] = set()

    for s_span in statistical_spacy:
        agreement = None
        for i, t_span in enumerate(transformer_spans):
            if i in used_transformer_idx:
                continue
            if _same_entity_label(s_span.label, t_span.label) and _char_overlap(
                s_span, t_span
            ):
                agreement = t_span
                used_transformer_idx.add(i)
                break
        if agreement:
            merged.append(
                EntitySpan(
                    text=s_span.text,
                    label=s_span.label,
                    start_char=s_span.start_char,
                    end_char=s_span.end_char,
                    confidence=min(0.98, (s_span.confidence + agreement.confidence) / 2 + 0.2),
                    method="ensemble",
                )
            )
        else:
            merged.append(s_span)

    for i, t_span in enumerate(transformer_spans):
        if i not in used_transformer_idx:
            merged.append(
                EntitySpan(
                    text=t_span.text,
                    label=t_span.label,
                    start_char=t_span.start_char,
                    end_char=t_span.end_char,
                    confidence=t_span.confidence * 0.85,
                    method="transformer_ner",
                )
            )

    return merged


def _same_entity_label(spacy_label: str, transformer_label: str) -> bool:
    equivalence = {
        "PERSON": {"PERSON"},
        "ORG": {"ORG"},
        "GPE": {"GPE", "LOC"},
        "LOC": {"GPE", "LOC"},
    }
    return transformer_label in equivalence.get(spacy_label, {spacy_label})


def _char_overlap(a: EntitySpan, b: EntitySpan) -> bool:
    return a.start_char < b.end_char and a.end_char > b.start_char
