"""First-proof evaluation metrics.

Implements the seven metrics named in ``docs/retrieval-modes.md`` and
PACKET-034 §3.2:

1. ``recall_at_k`` (used for Recall@5 and Recall@10)
2. ``mrr`` (mean reciprocal rank — per-question; the harness averages
   across questions)
3. ``citation_precision``
4. ``citation_recall``
5. ``faithfulness_heuristic`` (binary token-overlap; the locked
   first-proof method per the PACKET-043 GO direction; the LLM-judge
   upgrade is documented as a v1+ follow-on)
6. ``exact_match`` (auditable component of answer correctness)
7. ``token_f1`` (the SQuAD-style answer-correctness scalar; the locked
   first-proof method per the PACKET-043 GO direction)

All metrics are pure stdlib — no numpy needed at this layer; the harness
runs them per question and aggregates with simple averaging.

Tokenization is shared across faithfulness, exact_match, and token_f1
so that a normalized answer string round-trips identically through
every metric. The tokenizer normalizes case, strips a small stopword
list, and removes leading/trailing punctuation. It is intentionally
simple — richer tokenization (stemming, broader stopword filtering,
multi-script support) can swap in via the optional ``tokenizer``
parameter on the metric functions without changing return shapes.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# A small, conservative stopword set. The first-proof Q-A set is short
# and concrete; aggressive stopword removal would discard answer-bearing
# words like "in" inside "written in C". Keep this list short.
_STOPWORDS = frozenset({"the", "a", "an", "is", "are", "was", "were", "of"})

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")


def _default_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords.

    The tokenizer is shared across faithfulness, exact_match, and
    token_f1 so the normalization is consistent across the answer-quality
    metric family.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return [t for t in text.split() if t and t not in _STOPWORDS]


# Public alias so callers and tests can reach the default tokenizer.
default_tokenize = _default_tokenize


# ---------------------------------------------------------------------------
# Class A: retrieval-only metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    ranked_passage_ids: list[str],
    expected: Iterable[str],
    k: int,
) -> float:
    """Fraction of expected passages found in the top-``k`` of the ranked list.

    Args:
        ranked_passage_ids: Ranked list of passage_ids, most-relevant first.
        expected: Set of passage_ids that count as relevant for this question.
        k: Top-k cutoff.

    Returns:
        ``|top_k ∩ expected| / |expected|`` as a float in ``[0, 1]``.
        Returns ``0.0`` when ``expected`` is empty (nothing to recall).
    """
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = set(ranked_passage_ids[:k])
    return len(top_k & expected_set) / len(expected_set)


def mrr(
    ranked_passage_ids: list[str],
    expected: Iterable[str],
) -> float:
    """Reciprocal rank of the first expected passage in the ranked list.

    Per-question metric; the harness averages MRR across questions to
    produce the per-mode MRR reported in the headline table.

    Args:
        ranked_passage_ids: Ranked list of passage_ids, most-relevant first.
        expected: Set of passage_ids that count as relevant.

    Returns:
        ``1 / rank`` of the first expected passage (1-indexed), or
        ``0.0`` if no expected passage appears in the list.
    """
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    for rank, pid in enumerate(ranked_passage_ids, start=1):
        if pid in expected_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


def citation_precision(
    citations: list[str],
    expected: Iterable[str],
) -> float:
    """Of cited passages, fraction that are in ``expected``.

    Returns ``0.0`` when the citations list is empty (vacuously: no cited
    passages can be precise).
    """
    if not citations:
        return 0.0
    expected_set = set(expected)
    return sum(1 for c in citations if c in expected_set) / len(citations)


def citation_recall(
    citations: list[str],
    expected: Iterable[str],
) -> float:
    """Of expected passages, fraction that are cited.

    Returns ``0.0`` when ``expected`` is empty.
    """
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    citation_set = set(citations)
    return sum(1 for e in expected_set if e in citation_set) / len(expected_set)


# ---------------------------------------------------------------------------
# Class B: answer-quality metrics (heuristic, no LLM judge)
# ---------------------------------------------------------------------------

DEFAULT_FAITHFULNESS_THRESHOLD = 0.6


def faithfulness_heuristic(
    answer: str,
    cited_passage_texts: Iterable[str],
    *,
    threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
    tokenizer: Callable[[str], list[str]] = _default_tokenize,
) -> int:
    """Binary first-proof faithfulness: token-overlap fraction ≥ threshold.

    Per the PACKET-043 GO direction's locked first-proof method:
    "faithfulness -> heuristic, not LLM judge". Implemented as the
    fraction of answer tokens that appear in the union of cited-passage
    tokens; the binary threshold defaults to ``0.6``.

    Returns ``1`` when the answer's tokens are at least ``threshold``-
    covered by the cited passages, ``0`` otherwise. Empty answers
    return ``0``. Empty citation lists return ``0`` (a non-cited claim
    can't be faithful to its citations).
    """
    answer_tokens = tokenizer(answer)
    if not answer_tokens:
        return 0
    cited_tokens: set[str] = set()
    any_text = False
    for text in cited_passage_texts:
        any_text = True
        cited_tokens.update(tokenizer(text))
    if not any_text:
        return 0
    overlap = sum(1 for tok in answer_tokens if tok in cited_tokens)
    return 1 if overlap / len(answer_tokens) >= threshold else 0


def exact_match(
    prediction: str,
    gold: str,
    *,
    tokenizer: Callable[[str], list[str]] = _default_tokenize,
) -> int:
    """Binary exact match after normalization.

    Returns ``1`` iff ``tokenizer(prediction) == tokenizer(gold)``.
    Reported per-question and per-mode separately from token-F1 as a
    stricter diagnostic; not the headline answer-correctness scalar.
    """
    return 1 if tokenizer(prediction) == tokenizer(gold) else 0


def token_f1(
    prediction: str,
    gold: str,
    *,
    tokenizer: Callable[[str], list[str]] = _default_tokenize,
) -> float:
    """SQuAD-style token-F1 between predicted and gold strings.

    Per the PACKET-043 GO direction's locked first-proof method:
    "answer correctness -> exact-match + token-F1, not graded LLM judge".
    Token-F1 is the scalar reported in the headline ``eval_report.md``
    "Answer correctness" column; exact-match is reported alongside as a
    stricter diagnostic.

    Returns the token-F1 score in ``[0, 1]``. Empty prediction or empty
    gold returns ``0.0`` (consistent with SQuAD convention; an
    empty-on-empty case is treated as a failed answer).
    """
    pred_tokens = tokenizer(prediction)
    gold_tokens = tokenizer(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean. Empty iterable returns ``0.0``."""
    vs = list(values)
    if not vs:
        return 0.0
    return sum(vs) / len(vs)
