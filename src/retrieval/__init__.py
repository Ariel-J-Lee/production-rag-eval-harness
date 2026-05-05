"""Retrieval modes for the production-rag-eval-harness.

The four retrieval modes documented in ``docs/retrieval-modes.md`` (dense,
sparse, hybrid, graph-aware) all emit results in the shared
``{passage_id, score, source_mode}`` shape so the eval harness can compare
them on the same Q-A set against the same chunked corpus.

Public exports:

- :class:`RetrievalHit` — the shared per-result dataclass
- :func:`load_chunks` — iterator over ``chunks.jsonl`` records produced by
  ``scripts/fetch_corpus.py``
- :class:`DenseRetriever` — local-CPU sentence-transformer + in-process
  cosine implementation of mode 1
- :class:`SparseRetriever` — pure-stdlib BM25 implementation of mode 2

Hybrid and graph-aware retrievers ship in subsequent slices and import
:class:`RetrievalHit` and :func:`load_chunks` from this package.
"""

from src.retrieval._corpus import load_chunks
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.types import RetrievalHit

__all__ = ["DenseRetriever", "RetrievalHit", "SparseRetriever", "load_chunks"]
