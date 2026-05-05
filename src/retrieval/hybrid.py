"""Hybrid retrieval mode (mode 3 of four).

Reciprocal rank fusion of the dense and sparse single-mode rankings over
the same chunked passages. The implementation matches the v1 commitment
in ``docs/retrieval-modes.md`` ("v1 will fuse modes 1 and 2 by reciprocal
rank") and ``docs/architecture.md`` (hybrid layer = RRF over dense +
sparse, both in-process).

Reciprocal rank fusion (Cormack, Clarke, Büttcher, 2009): each upstream
retriever produces a ranked list; for every passage, the fused score is
the sum across modes of ``1 / (k + rank_in_that_mode)`` where ``rank``
is 1-indexed. A passage that does not appear in a mode's candidate pool
contributes ``0`` from that mode. This is rank-based and not
score-scale-sensitive, which matters because dense cosine and sparse
BM25 produce score scales that are not directly comparable.

Public entry points:

- :class:`HybridRetriever` — constructed with a :class:`DenseRetriever`
  and a :class:`SparseRetriever`; query returns ranked
  :class:`RetrievalHit` results with ``source_mode == "hybrid"``.
- ``python3 -m src.retrieval.hybrid --smoke`` — run the in-tree fixture
  smoke and print the top-3 hits per query (target: ``make smoke-hybrid``).

Locked defaults (from the GO direction for this slice):

- ``k = 60`` (Cormack default)
- ``candidate_pool = 100`` hits pulled from each upstream retriever
- no new dependency added; the fusion is pure stdlib

The hybrid retriever does not redesign the dense or sparse modes. It
holds two already-built retriever instances and combines their ranked
output. The constructor accepts any object whose ``query(q, top_k)``
returns an iterable of :class:`RetrievalHit`, so tests can pass
deterministic stubs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

from src.retrieval._corpus import load_chunks
from src.retrieval.dense import DEFAULT_MODEL_NAME, DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.types import RetrievalHit

DEFAULT_K = 60
DEFAULT_CANDIDATE_POOL = 100
DEFAULT_TOP_K = 10
SOURCE_MODE = "hybrid"


class _RankedRetriever(Protocol):
    """Minimal interface a hybrid upstream retriever must satisfy."""

    def query(self, q: str, top_k: int) -> Iterable[RetrievalHit]:
        ...


class HybridRetriever:
    """Reciprocal rank fusion over dense + sparse rankings.

    The fusion is rank-based: for every passage_id seen in either
    upstream retriever's top-``candidate_pool`` results, the fused score
    is ``Σ_m 1 / (k + rank_m)`` summed over the modes that returned the
    passage. Passages that appear in only one mode's pool still rank,
    but a passage that appears in both modes near the top wins by
    construction.

    Args:
        dense: A retriever that emits dense-mode results
            (``source_mode == "dense"``). Constructor only references
            ``dense.query(q, top_k)``; in tests a stub object with the
            same shape is acceptable.
        sparse: A retriever that emits sparse-mode results
            (``source_mode == "sparse"``).
        k: RRF saturation constant. Default ``60`` (Cormack et al.).
            Larger ``k`` flattens the contribution of top-ranked items.
        candidate_pool: How many hits to pull from each upstream
            retriever per query before fusion. Default ``100``. The
            fused output is then truncated to ``top_k``.
    """

    def __init__(
        self,
        dense: _RankedRetriever,
        sparse: _RankedRetriever,
        *,
        k: int = DEFAULT_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> None:
        if k < 0:
            raise ValueError(f"k must be non-negative; got {k!r}")
        if candidate_pool < 0:
            raise ValueError(
                f"candidate_pool must be non-negative; got {candidate_pool!r}"
            )
        self.dense = dense
        self.sparse = sparse
        self.k = int(k)
        self.candidate_pool = int(candidate_pool)

    def query(self, q: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalHit]:
        """Return up to ``top_k`` fused hits for the query string.

        Steps:

        1. Pull up to ``candidate_pool`` hits from each upstream
           retriever via its ``query()`` method.
        2. Walk both ranked lists, accumulating ``1 / (k + rank)`` per
           passage_id. Rank is 1-indexed.
        3. Sort passages by fused score descending; tie-break by
           ``passage_id`` ascending so output is deterministic.
        4. Emit up to ``top_k`` :class:`RetrievalHit` records with
           ``source_mode == "hybrid"`` and the fused score.

        Args:
            q: Query text.
            top_k: Maximum number of hits to return.

        Returns:
            List of :class:`RetrievalHit` of length at most ``top_k``.
            Empty when both upstream pools are empty.
        """
        if top_k <= 0:
            return []

        dense_hits = list(self.dense.query(q, top_k=self.candidate_pool))
        sparse_hits = list(self.sparse.query(q, top_k=self.candidate_pool))

        scores: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + 1.0 / (self.k + rank)
        for rank, hit in enumerate(sparse_hits, start=1):
            scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + 1.0 / (self.k + rank)

        if not scores:
            return []

        # Sort descending by fused score; tie-break by passage_id ascending
        # for deterministic output across runs.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        n = min(int(top_k), len(ranked))
        return [
            RetrievalHit(
                passage_id=passage_id,
                score=float(score),
                source_mode=SOURCE_MODE,
            )
            for passage_id, score in ranked[:n]
        ]

    @classmethod
    def from_chunks_file(
        cls,
        path: str | Path,
        *,
        k: int = DEFAULT_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        encoder: Optional[object] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> "HybridRetriever":
        """Construct a hybrid retriever from a ``chunks.jsonl`` path.

        Builds a :class:`DenseRetriever` and a :class:`SparseRetriever`
        over the same chunked corpus and wires them into a hybrid
        instance. The two upstream retrievers each load the chunks
        independently; at the first-proof slice cap the duplication is a
        few-MB cost in exchange for keeping the per-mode internals
        unchanged.

        Args:
            path: Filesystem path to ``chunks.jsonl``.
            k: RRF constant. Default ``60``.
            candidate_pool: Per-mode candidate count. Default ``100``.
            encoder: Optional pre-built encoder for the dense retriever
                (skip lazy sentence-transformers load).
            model_name: Sentence-transformer model identifier (default
                ``"all-MiniLM-L6-v2"``).
        """
        chunks = list(load_chunks(path))
        dense = DenseRetriever(chunks, encoder=encoder, model_name=model_name)
        sparse = SparseRetriever(chunks)
        return cls(dense, sparse, k=k, candidate_pool=candidate_pool)


# ---------------------------------------------------------------------------
# Smoke entry point: ``python3 -m src.retrieval.hybrid --smoke``
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
    "Apache License 2.0 web server",
    "high-level programming language emphasizing readability",
    "distributed version control source code",
)


def _smoke(
    top_k: int = 3,
    k: int = DEFAULT_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """Run the in-tree fixture smoke. Prints top-k hits per query.

    The dense upstream loads the real sentence-transformer model on
    first use; ``pip install -r requirements.txt`` is the prerequisite.
    Sparse runs on pure stdlib.
    """
    print(
        f"hybrid-retriever smoke: rrf k={k} candidate_pool={candidate_pool} "
        f"corpus_size={len(_SMOKE_FIXTURE)} top_k={top_k}",
        file=sys.stderr,
    )
    dense = DenseRetriever(_SMOKE_FIXTURE, model_name=model_name)
    sparse = SparseRetriever(_SMOKE_FIXTURE)
    retriever = HybridRetriever(dense, sparse, k=k, candidate_pool=candidate_pool)
    for q in _SMOKE_QUERIES:
        print(f"\nquery: {q}")
        hits = retriever.query(q, top_k=top_k)
        if not hits:
            print("  (no matches)")
            continue
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  {rank}. passage_id={hit.passage_id:<16}  "
                f"score={hit.score:+.5f}  source_mode={hit.source_mode}"
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ``python3 -m src.retrieval.hybrid``."""
    parser = argparse.ArgumentParser(
        prog="src.retrieval.hybrid",
        description="Hybrid (RRF over dense + sparse) retrieval over chunked OSS-ecosystem corpus.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the in-tree fixture smoke and print top-3 hits per query.",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"RRF constant (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL,
        help=f"Per-mode candidate count (default: {DEFAULT_CANDIDATE_POOL}).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME,
        help=f"Sentence-transformer model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of hits to print per smoke query (default: 3).",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke(
            top_k=args.top_k,
            k=args.k,
            candidate_pool=args.candidate_pool,
            model_name=args.model,
        )

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
