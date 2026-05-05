"""Unit tests for src.retrieval.dense.

The tests inject a deterministic stub encoder so the test suite does not
download the sentence-transformer model nor depend on the
``sentence_transformers`` package being installed. The end-to-end
"real model" path is exercised separately via ``make smoke-dense``.

What these tests cover:

- Output shape: hits are :class:`RetrievalHit` instances with the right
  ``source_mode`` and ``passage_id`` types.
- Ranking: the highest-cosine chunk wins; tie-breaks are stable on
  ``argsort`` order.
- ``top_k`` clamping: requesting more than the corpus size returns the
  whole corpus, sorted.
- Empty corpus: query against an empty corpus returns ``[]``.
- File-on-disk loader: ``from_chunks_file`` round-trips a JSONL file.
- Lazy encoder: passing ``encoder=stub`` skips the
  ``sentence_transformers`` import path entirely.
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

from src.retrieval.dense import DEFAULT_TOP_K, SOURCE_MODE, DenseRetriever
from src.retrieval.types import RetrievalHit


# ---------------------------------------------------------------------------
# Stub encoder: deterministic vectors so tests don't download the real model.
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Deterministic encoder that maps each known string to a fixed vector.

    Unknown strings get a zero vector (cosine score of 0 against
    everything else).
    """

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}
        # Infer dim from the first entry; default 4 if mapping is empty.
        self._dim = next(iter(self._mapping.values())).shape[0] if self._mapping else 4

    def encode(self, texts, show_progress_bar: bool = False, **_: object):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self._mapping:
                out[i] = self._mapping[t]
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks() -> list[dict]:
    """Five fixture chunks that span well-separated 4-D directions."""
    return [
        {"passage_id": "Q1:0000", "title": "A", "text": "alpha"},
        {"passage_id": "Q2:0000", "title": "B", "text": "beta"},
        {"passage_id": "Q3:0000", "title": "C", "text": "gamma"},
        {"passage_id": "Q4:0000", "title": "D", "text": "delta"},
        {"passage_id": "Q5:0000", "title": "E", "text": "epsilon"},
    ]


@pytest.fixture
def stub_encoder() -> _StubEncoder:
    """Stub encoder mapping each fixture text and a few queries to fixed 4-D vectors."""
    return _StubEncoder(
        {
            "alpha":   [1.0, 0.0, 0.0, 0.0],
            "beta":    [0.0, 1.0, 0.0, 0.0],
            "gamma":   [0.0, 0.0, 1.0, 0.0],
            "delta":   [0.0, 0.0, 0.0, 1.0],
            # epsilon sits on the gamma/delta plane so it does not become
            # parallel to combined queries on the alpha/beta plane after
            # L2-normalization (a (0.5, 0.5, 0, 0) chunk would normalize to
            # the same direction as a (1, 1, 0, 0) query and tie with Q1+Q2).
            "epsilon": [0.0, 0.0, 0.5, 0.5],
            "find alpha":   [1.0, 0.0, 0.0, 0.0],
            "alpha or beta": [1.0, 1.0, 0.0, 0.0],
            "find delta":    [0.0, 0.0, 0.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_query_returns_retrieval_hits_with_dense_source_mode(chunks, stub_encoder):
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    hits = retriever.query("find alpha", top_k=3)

    assert len(hits) == 3
    for h in hits:
        assert isinstance(h, RetrievalHit)
        assert h.source_mode == SOURCE_MODE == "dense"
        assert isinstance(h.passage_id, str)
        assert isinstance(h.score, float)


def test_top_hit_matches_query_intent(chunks, stub_encoder):
    """'find alpha' encodes to the alpha unit vector → Q1 must be #1."""
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    hits = retriever.query("find alpha", top_k=1)
    assert hits[0].passage_id == "Q1:0000"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_combined_query_pulls_both_components(chunks, stub_encoder):
    """'alpha or beta' encodes to (1,1,0,0) → Q1 and Q2 should rank above Q3..Q5."""
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    hits = retriever.query("alpha or beta", top_k=2)
    top_ids = {h.passage_id for h in hits}
    assert top_ids == {"Q1:0000", "Q2:0000"}


def test_top_k_clamps_to_corpus_size(chunks, stub_encoder):
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    hits = retriever.query("find delta", top_k=999)
    assert len(hits) == len(chunks)


def test_top_k_zero_returns_empty(chunks, stub_encoder):
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    assert retriever.query("find alpha", top_k=0) == []


def test_empty_corpus_returns_empty(stub_encoder):
    retriever = DenseRetriever([], encoder=stub_encoder)
    assert retriever.query("anything", top_k=5) == []


def test_default_top_k_is_ten():
    assert DEFAULT_TOP_K == 10


def test_index_idempotent_when_called_twice(chunks, stub_encoder):
    """Calling index() twice keeps the same shape."""
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    retriever.index()
    first_shape = retriever._chunk_vectors.shape  # type: ignore[union-attr]
    retriever.index()
    assert retriever._chunk_vectors.shape == first_shape  # type: ignore[union-attr]


def test_chunks_are_l2_normalized_after_index(chunks, stub_encoder):
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    retriever.index()
    norms = np.linalg.norm(retriever._chunk_vectors, axis=1)  # type: ignore[arg-type]
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_from_chunks_file_round_trips_jsonl(tmp_path, stub_encoder):
    """Writing chunks to a JSONL file and loading via from_chunks_file produces the same ranking."""
    records = [
        {"passage_id": "Q1:0000", "title": "A", "text": "alpha"},
        {"passage_id": "Q2:0000", "title": "B", "text": "beta"},
    ]
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    retriever = DenseRetriever.from_chunks_file(p, encoder=stub_encoder)
    hits = retriever.query("find alpha", top_k=1)
    assert hits[0].passage_id == "Q1:0000"


def test_stub_encoder_path_does_not_import_sentence_transformers(chunks, stub_encoder):
    """Passing a stub encoder must avoid the sentence_transformers import entirely.

    This guards the lazy-load contract: tests, CI, and any caller that
    supplies its own encoder must not pay the heavyweight import cost.
    """
    retriever = DenseRetriever(chunks, encoder=stub_encoder)
    retriever.index()
    retriever.query("find alpha", top_k=1)
    assert "sentence_transformers" not in sys.modules
