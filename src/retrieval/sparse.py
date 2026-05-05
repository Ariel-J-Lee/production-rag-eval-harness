"""Sparse retrieval mode (mode 2 of four).

Implements the sparse-retrieval commitment from ``docs/retrieval-modes.md``
and ``docs/architecture.md``: in-process BM25 (Robertson-Spärck-Jones)
over the same chunked passages the dense retriever indexes, no managed
search service.

Pure standard library: :mod:`re` for tokenization, :class:`collections.Counter`
+ :class:`collections.defaultdict` for the inverted index, :func:`math.log`
for IDF. No third-party dependency added beyond what the dense retriever
already pulls in.

The public interface mirrors :class:`src.retrieval.dense.DenseRetriever`
(constructor, :meth:`index`, :meth:`query`, :meth:`from_chunks_file`) so
the hybrid mode can consume both retrievers through the same shape.

BM25 hyperparameters default to ``k1=1.5`` and ``b=0.75`` — the standard
Robertson values. Tokenization is lowercase + ASCII alphanumeric runs of
length >= 2 (``re.findall(r"[a-z0-9]{2,}", text.lower())``); a more
elaborate tokenizer (stemming, stop-word filtering, multi-script support)
can swap in later without changing the public interface.

Public entry points:

- :class:`SparseRetriever` — index a chunk stream then call
  :meth:`SparseRetriever.query` for ranked results.
- ``python3 -m src.retrieval.sparse --smoke`` — run the in-tree fixture
  smoke and print the top-3 hits per query (target: ``make smoke-sparse``).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.retrieval._corpus import load_chunks
from src.retrieval.types import RetrievalHit

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_TOP_K = 10
SOURCE_MODE = "sparse"

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    """Lowercase + ASCII alphanumeric runs of length >= 2.

    The retrieval-quality tradeoff with this tokenizer is documented in
    the module docstring; a more elaborate tokenizer can be substituted
    by overriding :class:`SparseRetriever`'s ``tokenizer`` argument.
    """
    return _TOKEN_RE.findall(text.lower())


class SparseRetriever:
    """In-process BM25 over chunked passages.

    Construction is cheap (chunks are stored, no tokenization yet). The
    inverted index is built lazily on the first call to :meth:`index` or
    :meth:`query`, so callers that build many retrievers and only query a
    subset pay no upfront cost on the unused ones.

    Args:
        chunks: Iterable of chunk dicts with at least ``passage_id`` and
            ``text`` keys.
        k1: BM25 term-frequency saturation parameter. Default ``1.5``.
        b: BM25 length-normalization parameter. Default ``0.75``.
        tokenizer: Callable mapping a text string to a list of tokens.
            Default :func:`_tokenize` (lowercase ASCII alphanumeric runs).
    """

    def __init__(
        self,
        chunks: Iterable[dict],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        tokenizer=_tokenize,
    ) -> None:
        self.chunks: list[dict] = list(chunks)
        self.k1 = float(k1)
        self.b = float(b)
        self._tokenizer = tokenizer
        # Lazily-populated index state:
        self._doc_term_freqs: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._postings: dict[str, list[int]] = {}
        self._idf: dict[str, float] = {}
        self._indexed = False

    def index(self) -> None:
        """Tokenize every chunk and build the inverted index.

        Idempotent: re-indexing replaces all index state.
        """
        n = len(self.chunks)

        self._doc_term_freqs = [
            Counter(self._tokenizer(c["text"])) for c in self.chunks
        ]
        self._doc_lengths = [sum(tf.values()) for tf in self._doc_term_freqs]
        self._avg_doc_length = (
            sum(self._doc_lengths) / n if n > 0 else 0.0
        )

        postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, tf in enumerate(self._doc_term_freqs):
            for term in tf:
                postings[term].append(doc_id)
        self._postings = dict(postings)

        # BM25+ smoothed IDF:
        #   idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
        # The "+1 inside the log" form keeps IDF non-negative and bounded
        # for terms that appear in every document.
        self._idf = {
            term: math.log((n - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5) + 1.0)
            for term, doc_ids in self._postings.items()
        }

        self._indexed = True

    def query(self, q: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalHit]:
        """Return up to ``top_k`` ranked hits for the query string.

        Score per (query, doc) is the sum over query terms of
        ``IDF(t) * tf(t, doc) * (k1 + 1) / (tf(t, doc) + k1 * (1 - b + b * |doc| / avgdl))``.
        Documents with no matching query term score zero and are dropped
        from the result set rather than returned with score ``0.0``.
        Results are sorted descending by score; ties are broken by doc
        index ascending (insertion order from the chunks iterable).

        Args:
            q: Query text.
            top_k: Maximum number of hits to return.

        Returns:
            List of :class:`RetrievalHit` of length at most ``top_k``.
            Empty when the corpus or query is empty, or when the query
            terms are all out-of-vocabulary.
        """
        if not self.chunks or top_k <= 0:
            return []
        if not self._indexed:
            self.index()

        q_tokens = self._tokenizer(q)
        if not q_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in q_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            doc_ids = self._postings[term]
            for doc_id in doc_ids:
                tf = self._doc_term_freqs[doc_id][term]
                doc_len = self._doc_lengths[doc_id]
                length_norm = 1.0 - self.b + self.b * (
                    doc_len / self._avg_doc_length if self._avg_doc_length > 0 else 0.0
                )
                denom = tf + self.k1 * length_norm
                if denom > 0:
                    scores[doc_id] += idf * tf * (self.k1 + 1.0) / denom

        if not scores:
            return []

        # Sort descending by score; tie-break by doc_id ascending so the
        # ordering is deterministic across runs.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        k = min(int(top_k), len(ranked))
        return [
            RetrievalHit(
                passage_id=str(self.chunks[doc_id]["passage_id"]),
                score=float(score),
                source_mode=SOURCE_MODE,
            )
            for doc_id, score in ranked[:k]
        ]

    @classmethod
    def from_chunks_file(
        cls,
        path: str | Path,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        tokenizer=_tokenize,
    ) -> "SparseRetriever":
        """Construct a retriever from a ``chunks.jsonl`` path on disk."""
        return cls(
            chunks=load_chunks(path),
            k1=k1,
            b=b,
            tokenizer=tokenizer,
        )


# ---------------------------------------------------------------------------
# Smoke entry point: ``python3 -m src.retrieval.sparse --smoke``
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
    "high-level programming language readability indentation",
    "distributed version control source code",
)


def _smoke(top_k: int = 3, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> int:
    """Run the in-tree fixture smoke. Prints top-k hits per query.

    Returns the process exit code (``0`` on success).
    """
    print(
        f"sparse-retriever smoke: bm25 k1={k1} b={b} "
        f"corpus_size={len(_SMOKE_FIXTURE)} top_k={top_k}",
        file=sys.stderr,
    )
    retriever = SparseRetriever(_SMOKE_FIXTURE, k1=k1, b=b)
    retriever.index()
    for q in _SMOKE_QUERIES:
        print(f"\nquery: {q}")
        hits = retriever.query(q, top_k=top_k)
        if not hits:
            print("  (no matches)")
            continue
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  {rank}. passage_id={hit.passage_id:<16}  "
                f"score={hit.score:+.4f}  source_mode={hit.source_mode}"
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ``python3 -m src.retrieval.sparse``."""
    parser = argparse.ArgumentParser(
        prog="src.retrieval.sparse",
        description="Sparse BM25 retrieval over chunked OSS-ecosystem corpus.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the in-tree fixture smoke and print top-3 hits per query.",
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_K1,
                        help=f"BM25 k1 (default: {DEFAULT_K1}).")
    parser.add_argument("--b", type=float, default=DEFAULT_B,
                        help=f"BM25 b (default: {DEFAULT_B}).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of hits to print per smoke query (default: 3).")
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke(top_k=args.top_k, k1=args.k1, b=args.b)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
