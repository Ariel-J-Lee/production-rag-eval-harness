"""Unit tests for src.eval.metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.metrics import (
    DEFAULT_FAITHFULNESS_THRESHOLD,
    citation_precision,
    citation_recall,
    default_tokenize,
    exact_match,
    faithfulness_heuristic,
    mean,
    mrr,
    recall_at_k,
    token_f1,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    assert default_tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_drops_default_stopwords():
    tokens = default_tokenize("the quick brown fox is a fox")
    assert "the" not in tokens
    assert "is" not in tokens
    assert tokens.count("fox") == 2


def test_tokenize_keeps_numerics():
    assert default_tokenize("Released in 2005.") == ["released", "in", "2005"]


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_full_match():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "x", "b", "y"], {"a", "b"}, k=2) == pytest.approx(0.5)
    assert recall_at_k(["a", "x", "b", "y"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_no_match():
    assert recall_at_k(["x", "y"], {"a", "b"}, k=10) == 0.0


def test_recall_at_k_empty_expected_returns_zero():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_recall_at_k_zero_or_negative_k_returns_zero():
    assert recall_at_k(["a", "b"], {"a"}, k=0) == 0.0
    assert recall_at_k(["a", "b"], {"a"}, k=-1) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_first_position():
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_third_position():
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)


def test_mrr_no_match_returns_zero():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_mrr_takes_first_expected_hit():
    """When multiple expected passages appear, MRR uses the first one."""
    assert mrr(["x", "b", "a"], {"a", "b"}) == pytest.approx(0.5)


def test_mrr_empty_expected_returns_zero():
    assert mrr(["a"], set()) == 0.0


# ---------------------------------------------------------------------------
# citation_precision / citation_recall
# ---------------------------------------------------------------------------


def test_citation_precision_full_match():
    assert citation_precision(["a", "b"], {"a", "b"}) == 1.0


def test_citation_precision_partial():
    assert citation_precision(["a", "x"], {"a", "b"}) == pytest.approx(0.5)


def test_citation_precision_empty_citations_returns_zero():
    assert citation_precision([], {"a"}) == 0.0


def test_citation_recall_full_match():
    assert citation_recall(["a", "b"], {"a", "b"}) == 1.0


def test_citation_recall_partial():
    assert citation_recall(["a"], {"a", "b"}) == pytest.approx(0.5)


def test_citation_recall_empty_expected_returns_zero():
    assert citation_recall(["a"], set()) == 0.0


# ---------------------------------------------------------------------------
# faithfulness_heuristic
# ---------------------------------------------------------------------------


def test_faithfulness_full_overlap():
    """Answer fully covered by cited text → faithful."""
    answer = "alpha beta gamma"
    cited = ["the alpha and beta and gamma sequence"]
    assert faithfulness_heuristic(answer, cited) == 1


def test_faithfulness_below_threshold():
    """Answer with only one cited token out of three (33%) is below 60% default."""
    answer = "alpha beta gamma"
    cited = ["alpha unrelated text content"]
    assert faithfulness_heuristic(answer, cited) == 0


def test_faithfulness_at_threshold():
    """Custom threshold makes a previously-failing case pass."""
    answer = "alpha beta gamma"
    cited = ["alpha unrelated"]
    # 1 of 3 = 0.333; below default 0.6
    assert faithfulness_heuristic(answer, cited) == 0
    # threshold=0.3 → above
    assert faithfulness_heuristic(answer, cited, threshold=0.3) == 1


def test_faithfulness_empty_answer_returns_zero():
    assert faithfulness_heuristic("", ["alpha"]) == 0


def test_faithfulness_empty_cited_returns_zero():
    assert faithfulness_heuristic("alpha", []) == 0


def test_faithfulness_default_threshold_is_point_six():
    assert DEFAULT_FAITHFULNESS_THRESHOLD == 0.6


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_after_normalization():
    assert exact_match("Linus Torvalds", "linus torvalds") == 1
    assert exact_match("Linus Torvalds.", "Linus Torvalds") == 1
    assert exact_match("ANSI C", "ANSI C") == 1


def test_exact_match_token_order_matters():
    """Token order matters: exact_match is positional, not bag-of-words."""
    assert exact_match("apple banana", "banana apple") == 0


def test_exact_match_distinct_strings():
    assert exact_match("foo", "bar") == 0


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------


def test_token_f1_perfect_match():
    assert token_f1("Linus Torvalds", "Linus Torvalds") == pytest.approx(1.0)


def test_token_f1_partial_match():
    """F1(2 common, 4 pred, 4 gold) = 2 * 0.5 * 0.5 / 1 = 0.5."""
    assert token_f1(
        "alpha beta xxx yyy",
        "alpha beta zzz www",
    ) == pytest.approx(0.5)


def test_token_f1_no_overlap():
    assert token_f1("alpha", "beta") == 0.0


def test_token_f1_empty_pred_returns_zero():
    assert token_f1("", "alpha") == 0.0


def test_token_f1_empty_gold_returns_zero():
    assert token_f1("alpha", "") == 0.0


def test_token_f1_repeated_tokens_count_with_multiplicity():
    """Counter-based intersection → repeated tokens counted up to min multiplicity."""
    # pred: {alpha: 2}, gold: {alpha: 1} → common = {alpha: 1}
    # P = 1/2 = 0.5; R = 1/1 = 1.0 → F1 = 2 * 0.5 * 1 / 1.5 = 0.667
    assert token_f1("alpha alpha", "alpha") == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def test_mean_simple():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_returns_zero():
    assert mean([]) == 0.0


def test_mean_handles_iterator_input():
    assert mean(x for x in (1.0, 1.0, 1.0)) == pytest.approx(1.0)
