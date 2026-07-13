"""Deterministic tokenization, Jaccard similarity, and keyword coverage.

No ML anywhere in this module — everything is pure, reproducible set/string
math, the "Jaccard + keyword weighting" half of the BRD's three techniques.
Same normalized-token input feeds both this module and semantic.py, so the
two scores are directly comparable pieces of the same criterion.
"""

from __future__ import annotations

import re

# A small, hand-picked stopword list — not a full NLTK corpus, since the
# only goal here is dropping function words that would otherwise pad every
# Jaccard union with noise (e.g. "the", "and") without adding a new
# dependency or a data file to ship.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "once", "here", "there", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "is", "am", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "of", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "them",
        "their", "what", "which", "who", "whom", "as", "also", "how",
    }
)

# Keeps tech tokens like "c++", "node.js", "c#" intact instead of splitting
# on the punctuation that's meaningful in that context. "." requires a
# trailing alphanumeric to join (otherwise an ordinary sentence-ending
# period like "a test." would get absorbed into the token); "+"/"#" don't,
# since "c++"/"c#" are conventionally written with nothing after them.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+|[+#][a-z0-9]*)*")


def normalize_tokens(text: str, *, drop_stopwords: bool = True) -> set[str]:
    """Lowercases, strips punctuation, and splits into a set of unique
    tokens. Deterministic — the same input always yields the same output,
    which is the property this entire service depends on."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return set(tokens)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def keyword_coverage(expected_keywords: list[str], answer_text: str) -> tuple[list[str], list[str]]:
    """Per-keyword presence check, for a human-readable audit trail — a
    multi-word keyword like "query plan" counts as matched if every one of
    its significant tokens appears somewhere in the answer, not requiring
    exact phrase adjacency (a candidate might say "the query's execution
    plan" instead of "query plan" verbatim). Returns (matched, missing) in
    the original input order."""
    answer_tokens = normalize_tokens(answer_text)
    matched: list[str] = []
    missing: list[str] = []
    for keyword in expected_keywords:
        keyword_tokens = normalize_tokens(keyword)
        if not keyword_tokens:
            continue
        if keyword_tokens <= answer_tokens:
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing
