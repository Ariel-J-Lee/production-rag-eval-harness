"""Dense retrieval mode (mode 1 of four).

Implements the dense-retrieval commitment from ``docs/retrieval-modes.md``
and ``docs/architecture.md``: local sentence-transformer embeddings +
in-process cosine over chunked passages, no managed vector service.

Default model is ``all-MiniLM-L6-v2`` (384-dimensional sentence
embeddings, ~80 MB on disk after first load, runs on CPU). The default
can be overridden via the constructor's ``model_name`` argument or, for
tests, by passing a stub encoder via ``encoder``.

Index shape: a single in-memory ``numpy.ndarray`` of shape ``(N, D)``
holding L2-normalized chunk vectors. At the first-proof slice cap of 500
articles × ~5 chunks each ≈ 2,500 chunks, the brute-force cosine
``query @ index.T`` is faster than an ANN library's overhead and removes
the dependency surface. ANN can be added without changing the public
interface.

Public entry points:

- :class:`DenseRetriever` — index a chunk stream then call
  :meth:`DenseRetriever.query` for ranked results.
- ``python3 -m src.retrieval.dense --smoke`` — run the in-tree fixture
  smoke and print the top-3 hits per query (target: ``make smoke-dense``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from src.retrieval._corpus import load_chunks
from src.retrieval.types import RetrievalHit

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
SOURCE_MODE = "dense"


class DenseRetriever:
    """Local sentence-transformer + in-process cosine over chunked passages.

    Construction is cheap (chunks are stored, no encoding yet). The
    encoder is loaded lazily on the first call to :meth:`index` or
    :meth:`query` so that test code paths that pass a stub encoder pay no
    sentence-transformers import cost.

    Args:
        chunks: Iterable of chunk dicts with at least ``passage_id`` and
            ``text`` keys. Other keys (``wikidata_id``, ``title``, etc.)
            are ignored by this mode.
        encoder: Optional pre-built encoder with an ``encode(texts) ->
            ndarray`` method. When omitted, ``sentence-transformers`` is
            imported and ``model_name`` is loaded on first use.
        model_name: Sentence-transformer model identifier. Default
            ``"all-MiniLM-L6-v2"``.
    """

    def __init__(
        self,
        chunks: Iterable[dict],
        *,
        encoder: Optional[object] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.chunks: list[dict] = list(chunks)
        self._encoder = encoder
        self._model_name = model_name
        self._chunk_vectors: Optional[np.ndarray] = None

    @property
    def encoder(self):
        """Return the encoder, lazily loading sentence-transformers if needed."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "DenseRetriever needs the 'sentence-transformers' package. "
                    "Install with: pip install -r requirements.txt"
                ) from exc
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def index(self) -> None:
        """Encode every chunk's text and store an L2-normalized matrix.

        Idempotent: calling :meth:`index` more than once re-encodes. In
        the typical flow :meth:`query` calls :meth:`index` automatically
        on first use.
        """
        texts = [c["text"] for c in self.chunks]
        if not texts:
            self._chunk_vectors = np.zeros((0, 0), dtype=np.float32)
            return
        vectors = np.asarray(
            self.encoder.encode(texts, show_progress_bar=False),
            dtype=np.float32,
        )
        self._chunk_vectors = _l2_normalize(vectors)

    def query(self, q: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalHit]:
        """Return up to ``top_k`` ranked hits for the query string.

        Cosine similarity over L2-normalized vectors is computed as a
        single matrix-vector dot product. Results are sorted descending
        by score; ties keep ``np.argsort``'s stable order (chunk index
        ascending).

        Args:
            q: Query text.
            top_k: Maximum number of hits to return. Clamped to the
                corpus size.

        Returns:
            List of :class:`RetrievalHit` of length
            ``min(top_k, len(self.chunks))``. Empty when the corpus is
            empty.
        """
        if not self.chunks:
            return []
        if self._chunk_vectors is None:
            self.index()
        assert self._chunk_vectors is not None  # mypy / runtime invariant

        qv = np.asarray(
            self.encoder.encode([q], show_progress_bar=False),
            dtype=np.float32,
        )[0]
        qv = _l2_normalize(qv[None, :])[0]

        scores = self._chunk_vectors @ qv
        n = len(self.chunks)
        k = max(0, min(int(top_k), n))
        if k == 0:
            return []
        top_idx = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievalHit(
                passage_id=str(self.chunks[i]["passage_id"]),
                score=float(scores[i]),
                source_mode=SOURCE_MODE,
            )
            for i in top_idx
        ]

    @classmethod
    def from_chunks_file(
        cls,
        path: str | Path,
        *,
        encoder: Optional[object] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> "DenseRetriever":
        """Construct a retriever from a ``chunks.jsonl`` path on disk."""
        return cls(
            chunks=load_chunks(path),
            encoder=encoder,
            model_name=model_name,
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of ``matrix``; preserves zero rows as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return matrix / safe


# ---------------------------------------------------------------------------
# Smoke entry point: ``python3 -m src.retrieval.dense --smoke``
# ---------------------------------------------------------------------------

_SMOKE_FIXTURE: Sequence[dict] = (
    {
        "passage_id": "Q28865:0000",
        "title": "Python (programming language)",
        "text": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use "
            "of significant indentation."
        ),
    },
    {
        "passage_id": "Q192490:0000",
        "title": "PostgreSQL",
        "text": (
            "PostgreSQL is a free and open-source relational database "
            "management system emphasizing extensibility and SQL compliance."
        ),
    },
    {
        "passage_id": "Q11354:0000",
        "title": "Apache HTTP Server",
        "text": (
            "The Apache HTTP Server is a free and open-source cross-platform "
            "web server software released under the terms of Apache License 2.0."
        ),
    },
    {
        "passage_id": "Q193321:0000",
        "title": "Linux kernel",
        "text": (
            "The Linux kernel is a free and open-source, monolithic, modular, "
            "multitasking, Unix-like operating system kernel."
        ),
    },
    {
        "passage_id": "Q190909:0000",
        "title": "Git",
        "text": (
            "Git is a distributed version control system that tracks changes "
            "in any set of computer files, usually used for coordinating work "
            "among programmers collaboratively developing source code."
        ),
    },
)

_SMOKE_QUERIES: Sequence[str] = (
    "Which open-source web server uses the Apache License 2.0?",
    "What programming language emphasizes code readability and significant indentation?",
    "Which tool is used to track changes to source code among collaborating programmers?",
)


def _smoke(top_k: int = 3, model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Run the in-tree fixture smoke. Prints top-k hits per query.

    Returns the process exit code (``0`` on success).
    """
    print(
        f"dense-retriever smoke: model={model_name} "
        f"corpus_size={len(_SMOKE_FIXTURE)} top_k={top_k}",
        file=sys.stderr,
    )
    retriever = DenseRetriever(_SMOKE_FIXTURE, model_name=model_name)
    retriever.index()
    for q in _SMOKE_QUERIES:
        print(f"\nquery: {q}")
        hits = retriever.query(q, top_k=top_k)
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  {rank}. passage_id={hit.passage_id:<16}  "
                f"score={hit.score:+.4f}  source_mode={hit.source_mode}"
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ``python3 -m src.retrieval.dense``."""
    parser = argparse.ArgumentParser(
        prog="src.retrieval.dense",
        description="Dense retrieval over chunked OSS-ecosystem corpus.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the in-tree fixture smoke and print top-3 hits per query.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence-transformer model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of hits to print per smoke query (default: 3).",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke(top_k=args.top_k, model_name=args.model)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
