"""Unit tests for src.retrieval.hybrid.

The hybrid retriever's contract is to fuse two ranked lists by
reciprocal rank — it has no opinion on how the upstream retrievers
produce those rankings. The tests exploit that by passing stub upstream
retrievers that return canned :class:`RetrievalHit` lists, so the suite
runs on pure stdlib + pytest with no model download and no
``sentence-transformers`` import.

What these tests cover:

- RRF math: known inputs produce the expected fused score.
- Double-coverage promotion: a passage appearing in both upstream lists
  outranks passages appearing in only one.
- ``candidate_pool`` plumbing: upstreams are queried with
  ``top_k=candidate_pool``.
- ``top_k`` clamping (oversized → fused-set size, zero → empty).
- Empty inputs (empty corpus from both, empty query).
- Output shape: hits are :class:`RetrievalHit` instances with
  ``source_mode == "hybrid"``.
- Determinism: same inputs produce the same ranked output across runs.
- Tie-break: passages with equal RRF scores are ranked by ``passage_id``
  ascending.
- Constructor validation: negative ``k`` and negative ``candidate_pool``
  raise :class:`ValueError`.
- ``from_chunks_file`` integration via a stub encoder, so the dense
  upstream skips the sentence-transformers import path entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow the tests to run without an editable install: include the repo
# root so ``src.retrieval`` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval.hybrid import (
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_K,
    DEFAULT_TOP_K,
    SOURCE_MODE,
    HybridRetriever,
)
from src.retrieval.types import RetrievalHit


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns a fixed list of :class:`RetrievalHit`, recording its query calls.

    The hits are returned in the order given (already-ranked); ``query``
    truncates to ``top_k`` and records the call so tests can assert that
    ``HybridRetriever`` plumbed ``candidate_pool`` correctly.
    """

    def __init__(self, hits: list[RetrievalHit], *, source_mode: str = "dense"):
        self._hits = hits
        self.source_mode = source_mode
        self.calls: list[tuple[str, int]] = []

    def query(self, q: str, top_k: int) -> list[RetrievalHit]:
        self.calls.append((q, top_k))
        return self._hits[:top_k]


def _hit(passage_id: str, score: float, source_mode: str) -> RetrievalHit:
    return RetrievalHit(passage_id=passage_id, score=score, source_mode=source_mode)


# ---------------------------------------------------------------------------
# Output-shape tests
# ---------------------------------------------------------------------------


def test_query_returns_retrieval_hits_with_hybrid_source_mode():
    dense = _StubRetriever(
        [_hit("Q1:0000", 0.9, "dense"), _hit("Q2:0000", 0.8, "dense")],
        source_mode="dense",
    )
    sparse = _StubRetriever(
        [_hit("Q2:0000", 4.0, "sparse"), _hit("Q3:0000", 3.0, "sparse")],
        source_mode="sparse",
    )
    retriever = HybridRetriever(dense, sparse)
    hits = retriever.query("any query", top_k=3)

    assert len(hits) == 3
    for h in hits:
        assert isinstance(h, RetrievalHit)
        assert h.source_mode == SOURCE_MODE == "hybrid"
        assert isinstance(h.passage_id, str)
        assert isinstance(h.score, float)


# ---------------------------------------------------------------------------
# RRF math tests
# ---------------------------------------------------------------------------


def test_rrf_score_for_single_mode_only_match():
    """A passage appearing only in the dense list at rank 1 has fused score 1/(k+1)."""
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse, k=60)
    hits = retriever.query("query", top_k=1)

    assert len(hits) == 1
    assert hits[0].passage_id == "Q1:0000"
    assert hits[0].score == pytest.approx(1.0 / 61.0)


def test_rrf_score_for_double_coverage_at_same_rank():
    """A passage at rank 1 in both lists has fused score 2 * 1/(k+1)."""
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([_hit("Q1:0000", 4.0, "sparse")])
    retriever = HybridRetriever(dense, sparse, k=60)
    hits = retriever.query("query", top_k=1)

    assert hits[0].passage_id == "Q1:0000"
    assert hits[0].score == pytest.approx(2.0 / 61.0)


def test_double_coverage_outranks_single_coverage():
    """Document in both lists at rank 1 should rank above documents in only one list."""
    dense = _StubRetriever([
        _hit("A:0000", 0.9, "dense"),
        _hit("B:0000", 0.8, "dense"),
    ])
    sparse = _StubRetriever([
        _hit("A:0000", 4.0, "sparse"),
        _hit("C:0000", 3.0, "sparse"),
    ])
    retriever = HybridRetriever(dense, sparse, k=60)
    hits = retriever.query("query", top_k=10)

    assert hits[0].passage_id == "A:0000"          # double coverage at rank 1
    # B and C both at rank 2 in their respective single mode → equal score
    # → tie-break by passage_id ascending
    assert hits[1].passage_id == "B:0000"
    assert hits[2].passage_id == "C:0000"


def test_rrf_uses_rank_not_score():
    """Score scale of upstream hits is irrelevant; only rank position matters."""
    # Dense hit at rank 1 has score 0.001; sparse hit at rank 2 has score 9999.
    # If RRF used scores, sparse would dominate; with RRF, dense rank 1 (1/61) beats sparse rank 2 (1/62).
    dense = _StubRetriever([_hit("X:0000", 0.001, "dense")])
    sparse = _StubRetriever([
        _hit("Y:0000", 99999.0, "sparse"),
        _hit("X:0000", 9999.0, "sparse"),
    ])
    retriever = HybridRetriever(dense, sparse, k=60)
    hits = retriever.query("query", top_k=2)

    # X gets rrf = 1/61 (dense rank 1) + 1/62 (sparse rank 2) ≈ 0.03253
    # Y gets rrf = 1/61 (sparse rank 1)                       ≈ 0.01640
    assert hits[0].passage_id == "X:0000"
    assert hits[1].passage_id == "Y:0000"


# ---------------------------------------------------------------------------
# candidate_pool plumbing
# ---------------------------------------------------------------------------


def test_upstreams_are_queried_with_candidate_pool_not_top_k():
    """HybridRetriever must request candidate_pool hits from each upstream, not top_k."""
    dense = _StubRetriever([])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse, k=60, candidate_pool=37)

    retriever.query("a query", top_k=5)

    assert dense.calls == [("a query", 37)]
    assert sparse.calls == [("a query", 37)]


def test_default_candidate_pool_is_one_hundred():
    assert DEFAULT_CANDIDATE_POOL == 100


def test_default_k_is_sixty():
    assert DEFAULT_K == 60


def test_default_top_k_is_ten():
    assert DEFAULT_TOP_K == 10


# ---------------------------------------------------------------------------
# top_k clamping + empties
# ---------------------------------------------------------------------------


def test_top_k_clamps_to_fused_set_size():
    """Requesting more than the fused-set size returns the whole fused set."""
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([_hit("Q2:0000", 4.0, "sparse")])
    retriever = HybridRetriever(dense, sparse)
    hits = retriever.query("query", top_k=999)

    assert len(hits) == 2
    assert {h.passage_id for h in hits} == {"Q1:0000", "Q2:0000"}


def test_top_k_zero_returns_empty():
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse)

    assert retriever.query("query", top_k=0) == []


def test_top_k_negative_returns_empty():
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse)

    assert retriever.query("query", top_k=-3) == []


def test_both_empty_returns_empty():
    dense = _StubRetriever([])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse)

    assert retriever.query("query", top_k=10) == []


def test_only_dense_empty_still_returns_sparse_hits():
    dense = _StubRetriever([])
    sparse = _StubRetriever([_hit("Q1:0000", 4.0, "sparse")])
    retriever = HybridRetriever(dense, sparse)
    hits = retriever.query("query", top_k=10)

    assert [h.passage_id for h in hits] == ["Q1:0000"]


def test_only_sparse_empty_still_returns_dense_hits():
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse)
    hits = retriever.query("query", top_k=10)

    assert [h.passage_id for h in hits] == ["Q1:0000"]


# ---------------------------------------------------------------------------
# Determinism + tie-break
# ---------------------------------------------------------------------------


def test_determinism_two_retrievers_produce_identical_output():
    dense_a = _StubRetriever([_hit("Q1:0000", 0.9, "dense"), _hit("Q2:0000", 0.8, "dense")])
    sparse_a = _StubRetriever([_hit("Q2:0000", 4.0, "sparse"), _hit("Q3:0000", 3.0, "sparse")])
    a = HybridRetriever(dense_a, sparse_a)

    dense_b = _StubRetriever([_hit("Q1:0000", 0.9, "dense"), _hit("Q2:0000", 0.8, "dense")])
    sparse_b = _StubRetriever([_hit("Q2:0000", 4.0, "sparse"), _hit("Q3:0000", 3.0, "sparse")])
    b = HybridRetriever(dense_b, sparse_b)

    assert a.query("query", top_k=5) == b.query("query", top_k=5)


def test_tie_break_is_passage_id_ascending():
    """Two passages with identical fused scores are ordered by passage_id ascending."""
    dense = _StubRetriever([_hit("BBB:0000", 0.9, "dense")])
    sparse = _StubRetriever([_hit("AAA:0000", 4.0, "sparse")])
    # Both at rank 1 in their only mode → identical RRF score.
    retriever = HybridRetriever(dense, sparse)
    hits = retriever.query("query", top_k=2)

    assert [h.passage_id for h in hits] == ["AAA:0000", "BBB:0000"]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_negative_k_raises_value_error():
    dense = _StubRetriever([])
    sparse = _StubRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever(dense, sparse, k=-1)


def test_negative_candidate_pool_raises_value_error():
    dense = _StubRetriever([])
    sparse = _StubRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever(dense, sparse, candidate_pool=-1)


def test_zero_k_is_allowed_but_distorts_scores():
    """k=0 means rank 1 contributes 1/1 = 1.0; allowed but extreme."""
    dense = _StubRetriever([_hit("Q1:0000", 0.9, "dense")])
    sparse = _StubRetriever([])
    retriever = HybridRetriever(dense, sparse, k=0)
    hits = retriever.query("query", top_k=1)

    assert hits[0].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# from_chunks_file integration (uses a stub encoder so the dense upstream
# skips the sentence-transformers import path)
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Minimal stub for the dense upstream so from_chunks_file works without sentence-transformers."""

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}
        self._dim = next(iter(self._mapping.values())).shape[0] if self._mapping else 4

    def encode(self, texts, show_progress_bar: bool = False, **_: object):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self._mapping:
                out[i] = self._mapping[t]
        return out


def test_from_chunks_file_round_trips_jsonl(tmp_path):
    """Build a hybrid retriever from a chunks.jsonl path; query returns hybrid hits."""
    records = [
        {"passage_id": "Q1:0000", "title": "A", "text": "alpha"},
        {"passage_id": "Q2:0000", "title": "B", "text": "beta"},
        {"passage_id": "Q3:0000", "title": "C", "text": "gamma"},
    ]
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    encoder = _StubEncoder({
        "alpha": [1.0, 0.0, 0.0, 0.0],
        "beta":  [0.0, 1.0, 0.0, 0.0],
        "gamma": [0.0, 0.0, 1.0, 0.0],
        "alpha gamma": [1.0, 0.0, 1.0, 0.0],
    })

    retriever = HybridRetriever.from_chunks_file(p, encoder=encoder)
    hits = retriever.query("alpha gamma", top_k=2)

    assert all(h.source_mode == SOURCE_MODE for h in hits)
    # Q1 ("alpha") and Q3 ("gamma") match the query in BOTH dense (stub
    # encoder gives them positive cosine) and sparse (tokenizer matches
    # "alpha" / "gamma" against chunk text), so they get double-coverage
    # RRF and rank above Q2 ("beta") which only appears in dense's
    # candidate pool with zero cosine score.
    returned_ids = [h.passage_id for h in hits]
    assert set(returned_ids) == {"Q1:0000", "Q3:0000"}
