"""Unit tests for src.retrieval.sparse.

The sparse retriever is pure-stdlib BM25, so the tests run with nothing
beyond pytest in the environment — no model download, no third-party
dependency.

What these tests cover:

- Tokenizer: lowercase + ASCII alphanumeric runs of length >= 2.
- Inverted index correctness: doc lengths, average doc length, postings
  by term, IDF non-negative.
- Query ranking: matching docs win over non-matching; rare-term queries
  prefer the doc containing the rare term; common-term queries prefer
  the shorter doc (length normalization).
- ``top_k`` clamping: requesting more than the matching-doc count
  returns only matching docs.
- Empty corpus, empty query, OOV-only query: all return ``[]``.
- File-on-disk loader: ``from_chunks_file`` round-trips a JSONL file.
- Output shape: hits are :class:`RetrievalHit` instances with
  ``source_mode == "sparse"``.
- Determinism: identical inputs produce identical ranked output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow the tests to run without an editable install: include the repo
# root so ``src.retrieval`` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval.sparse import (
    DEFAULT_B,
    DEFAULT_K1,
    DEFAULT_TOP_K,
    SOURCE_MODE,
    SparseRetriever,
    _tokenize,
)
from src.retrieval.types import RetrievalHit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks() -> list[dict]:
    """Five fixture chunks chosen so simple queries have unambiguous winners."""
    return [
        {
            "passage_id": "Q1:0000",
            "title": "Python",
            "text": "Python is a high-level programming language emphasizing readability.",
        },
        {
            "passage_id": "Q2:0000",
            "title": "PostgreSQL",
            "text": "PostgreSQL is a relational database management system.",
        },
        {
            "passage_id": "Q3:0000",
            "title": "Apache",
            "text": "Apache is a web server released under the Apache License.",
        },
        {
            "passage_id": "Q4:0000",
            "title": "Linux",
            "text": "Linux is an open-source kernel powering operating systems.",
        },
        {
            "passage_id": "Q5:0000",
            "title": "Git",
            "text": "Git is a distributed version control system for tracking source code changes.",
        },
    ]


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_tokenizer_lowercases():
    assert _tokenize("Apache HTTP Server") == ["apache", "http", "server"]


def test_tokenizer_drops_one_char_runs_and_punctuation():
    """Length-1 tokens (a, an's `s`) and punctuation are dropped."""
    tokens = _tokenize("a-1 the QUICK!  brown,fox")
    assert "a" not in tokens
    assert "1" not in tokens
    assert tokens == ["the", "quick", "brown", "fox"]


def test_tokenizer_keeps_alphanumeric_mix():
    assert _tokenize("Python 3.11 release") == ["python", "11", "release"]


# ---------------------------------------------------------------------------
# Index-construction tests
# ---------------------------------------------------------------------------


def test_index_populates_doc_lengths_and_avg(chunks):
    retriever = SparseRetriever(chunks)
    retriever.index()

    assert len(retriever._doc_lengths) == len(chunks)
    assert all(length > 0 for length in retriever._doc_lengths)
    expected_avg = sum(retriever._doc_lengths) / len(chunks)
    assert retriever._avg_doc_length == pytest.approx(expected_avg)


def test_postings_lookup_finds_term(chunks):
    retriever = SparseRetriever(chunks)
    retriever.index()
    # "python" appears only in chunk 0 ("Python is a high-level...")
    assert "python" in retriever._postings
    assert retriever._postings["python"] == [0]


def test_idf_non_negative_for_every_known_term(chunks):
    retriever = SparseRetriever(chunks)
    retriever.index()
    assert all(idf >= 0 for idf in retriever._idf.values())


def test_idf_higher_for_rarer_terms(chunks):
    """Single-doc terms should have higher IDF than multi-doc terms."""
    # "apache" appears twice in one doc → df=1 (rare term)
    # "is" appears in every doc → df=5 (most common, lowest IDF)
    retriever = SparseRetriever(chunks)
    retriever.index()
    assert retriever._idf["apache"] > retriever._idf["is"]


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


def test_query_returns_retrieval_hits_with_sparse_source_mode(chunks):
    retriever = SparseRetriever(chunks)
    hits = retriever.query("Python programming", top_k=3)

    assert hits, "expected at least one hit for an in-vocabulary query"
    for h in hits:
        assert isinstance(h, RetrievalHit)
        assert h.source_mode == SOURCE_MODE == "sparse"
        assert isinstance(h.passage_id, str)
        assert isinstance(h.score, float)
        assert h.score > 0


def test_top_hit_matches_query_intent_python(chunks):
    """'Python programming' must rank Q1 (the Python doc) first."""
    retriever = SparseRetriever(chunks)
    hits = retriever.query("Python programming", top_k=1)
    assert hits[0].passage_id == "Q1:0000"


def test_top_hit_matches_query_intent_apache(chunks):
    """'Apache web server' must rank Q3 (the Apache doc) first."""
    retriever = SparseRetriever(chunks)
    hits = retriever.query("Apache web server", top_k=1)
    assert hits[0].passage_id == "Q3:0000"


def test_top_hit_matches_query_intent_git(chunks):
    """'distributed version control' must rank Q5 (the Git doc) first."""
    retriever = SparseRetriever(chunks)
    hits = retriever.query("distributed version control", top_k=1)
    assert hits[0].passage_id == "Q5:0000"


def test_query_drops_documents_with_zero_score(chunks):
    """A query that matches only one doc should return only that doc, not zero-score padding."""
    retriever = SparseRetriever(chunks)
    hits = retriever.query("postgresql", top_k=10)
    assert len(hits) == 1
    assert hits[0].passage_id == "Q2:0000"


def test_top_k_clamps_to_match_count(chunks):
    """top_k=10 against a 5-doc corpus where 1 doc matches returns 1 hit."""
    retriever = SparseRetriever(chunks)
    hits = retriever.query("postgresql", top_k=10)
    assert len(hits) == 1


def test_top_k_zero_returns_empty(chunks):
    retriever = SparseRetriever(chunks)
    assert retriever.query("python", top_k=0) == []


def test_empty_corpus_returns_empty():
    retriever = SparseRetriever([])
    assert retriever.query("anything", top_k=5) == []


def test_empty_query_returns_empty(chunks):
    retriever = SparseRetriever(chunks)
    assert retriever.query("", top_k=5) == []


def test_oov_only_query_returns_empty(chunks):
    """Query terms that exist nowhere in the corpus should produce no hits."""
    retriever = SparseRetriever(chunks)
    assert retriever.query("zzzz xxxxx yyyyyy", top_k=5) == []


def test_default_top_k_is_ten():
    assert DEFAULT_TOP_K == 10


def test_default_bm25_hyperparameters():
    assert DEFAULT_K1 == 1.5
    assert DEFAULT_B == 0.75


def test_index_idempotent(chunks):
    """Calling index() twice produces identical query results."""
    retriever = SparseRetriever(chunks)
    retriever.index()
    first_hits = retriever.query("python programming", top_k=3)
    retriever.index()
    second_hits = retriever.query("python programming", top_k=3)
    assert first_hits == second_hits


def test_query_is_deterministic(chunks):
    """Two retrievers built from the same chunks rank identically."""
    a = SparseRetriever(chunks)
    b = SparseRetriever(chunks)
    assert a.query("apache web server", top_k=5) == b.query("apache web server", top_k=5)


def test_tie_break_is_doc_id_ascending():
    """When two docs would tie on score, the lower doc_id ranks higher."""
    chunks = [
        {"passage_id": "Q1:0000", "title": "A", "text": "alpha alpha"},
        {"passage_id": "Q2:0000", "title": "B", "text": "alpha alpha"},
    ]
    retriever = SparseRetriever(chunks)
    hits = retriever.query("alpha", top_k=2)
    assert hits[0].passage_id == "Q1:0000"
    assert hits[1].passage_id == "Q2:0000"


def test_from_chunks_file_round_trips_jsonl(tmp_path, chunks):
    """Writing chunks to a JSONL file and loading via from_chunks_file produces the same ranking."""
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in chunks:
            f.write(json.dumps(r) + "\n")

    retriever = SparseRetriever.from_chunks_file(p)
    hits = retriever.query("python programming", top_k=1)
    assert hits[0].passage_id == "Q1:0000"


def test_custom_tokenizer_overrides_default():
    """Passing a custom tokenizer routes around the default re-based one."""
    chunks = [
        {"passage_id": "Q1:0000", "title": "A", "text": "PYTHON RULES"},
    ]
    # Custom tokenizer that does no lowercasing — query case must match exactly.
    retriever = SparseRetriever(chunks, tokenizer=lambda t: t.split())

    # Lowercase query → no match because the doc was indexed with uppercase tokens.
    assert retriever.query("python", top_k=1) == []
    # Uppercase query → matches.
    assert retriever.query("PYTHON", top_k=1)[0].passage_id == "Q1:0000"
