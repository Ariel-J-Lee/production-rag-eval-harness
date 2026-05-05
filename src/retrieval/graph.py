"""Graph-aware retrieval mode (mode 4 of four).

Implements the graph-aware retrieval commitment from
``docs/retrieval-modes.md`` and ``docs/architecture.md``: query entity
recognition, Wikidata QID resolution, 1-hop neighbor expansion over the
six relational properties documented in ``data/DATA-SOURCE.md``, and a
two-stage passage boost over the same chunked corpus the dense and
sparse retrievers index.

Pure standard library: :mod:`re` for tokenization and label matching,
:class:`collections.defaultdict` for the entity-and-chunk indices,
:func:`json` (via :func:`src.retrieval._corpus.load_chunks`) for
streaming the chunk file. No third-party dependency added beyond what
the dense retriever already pulls in.

Pipeline:

1. **Entity recognition** — case-folded longest-match-wins scan of the
   query against every English label in ``entities.jsonl``. Multiple
   labels may co-occur; each contributes its QID(s).
2. **Identifier resolution** — labels resolve directly to QIDs through
   the entity index. A single label may map to more than one QID; all
   matching QIDs participate in expansion.
3. **1-hop neighbor expansion** — for every resolved query QID, walk the
   entity's claims for the six relational properties (``P31``, ``P178``,
   ``P277``, ``P275``, ``P306``, ``P407``) and gather neighbor QIDs. No
   multi-hop traversal.
4. **Passage boost** — score each chunk by:

   - **Structural boost** (default ``1.0``): the chunk's ``wikidata_id``
     is one of the resolved query QIDs or one of the 1-hop neighbor
     QIDs.
   - **Mention boost** (default ``0.5``): the chunk's text contains the
     label of a 1-hop neighbor that is not itself a query QID, and the
     chunk's ``wikidata_id`` is not that same neighbor (so a passage
     does not earn a mention boost for naming itself).

   Boosts accumulate: a chunk that satisfies both stages receives the
   sum.

Output shape: :class:`src.retrieval.types.RetrievalHit` with
``source_mode == "graph_aware"``. Cross-mode score comparison is not
meaningful; the eval harness compares modes via per-question metrics,
not raw scores.

The smallest-defensible bar is intentional. No graph database, no
Cypher-style query language, no managed graph service, no multi-hop
traversal. The graph store is the gitignored ``entities.jsonl`` file
materialized by ``scripts/fetch_corpus.py``; traversal happens through
plain-dict adjacency built once at index-load time.

Public entry points:

- :class:`GraphRetriever` — index a chunk stream + an entity stream then
  call :meth:`GraphRetriever.query` for ranked results.
- ``python3 -m src.retrieval.graph --smoke`` — run the in-tree fixture
  smoke and print the top-3 hits per query.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.retrieval._corpus import load_chunks
from src.retrieval.types import RetrievalHit

DEFAULT_TOP_K = 10
DEFAULT_NEIGHBOR_BOOST = 1.0
DEFAULT_LABEL_MENTION_BOOST = 0.5
SOURCE_MODE = "graph_aware"

RELATIONAL_PROPERTIES = ("P31", "P178", "P277", "P275", "P306", "P407")

_WORD_BOUNDARY_RE = re.compile(r"\W+")


def load_entities(path: str | Path) -> "list[dict]":
    """Read ``entities.jsonl`` into a list of entity records.

    Each line is a JSON object with ``wikidata_id``, ``label``, and
    ``claims`` keys per ``data/DATA-SOURCE.md``. Empty and whitespace-only
    lines are skipped. The file is read eagerly because the entity index
    needs all labels to build its longest-match-wins regex.
    """
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _casefold(text: str) -> str:
    """Lowercase + collapse non-word runs to single spaces.

    Used to canonicalize both labels and queries before regex-matching so
    "Apache HTTP Server" and "apache  http  server" hit the same token
    sequence.
    """
    return _WORD_BOUNDARY_RE.sub(" ", text.lower()).strip()


class GraphRetriever:
    """1-hop graph-aware retriever over chunked passages + entity claims.

    Construction is cheap (chunks and entities are stored, no indexing
    yet). Indexing builds:

    - ``_qids_by_label``: case-folded label → list of QIDs (a label can
      collide across entities; all participate in expansion).
    - ``_label_by_qid``: QID → canonical label (used for tier-2 mention
      scoring).
    - ``_claims_by_qid``: QID → list of ``(property, neighbor_qid)``
      pairs limited to :data:`RELATIONAL_PROPERTIES`.
    - ``_chunks_by_wikidata_id``: QID → list of chunk indices whose
      ``wikidata_id`` matches.
    - ``_label_re``: alternation regex of every known label, longest-first
      so the leftmost-first :mod:`re` engine yields longest-match-wins
      behavior.

    Indexing is lazy: :meth:`query` calls :meth:`index` automatically on
    first use. Idempotent: re-indexing replaces all index state.

    Args:
        chunks: Iterable of chunk dicts with at least ``passage_id`` and
            ``text`` keys; ``wikidata_id`` is consulted when present.
        entities: Iterable of entity dicts with ``wikidata_id``,
            ``label``, and ``claims`` keys per ``data/DATA-SOURCE.md``.
        neighbor_boost: Score added to a chunk per matched query or
            neighbor QID via ``wikidata_id``. Default ``1.0``.
        label_mention_boost: Score added to a chunk per matched neighbor
            label found in the chunk text. Default ``0.5``.
    """

    def __init__(
        self,
        chunks: Iterable[dict],
        entities: Iterable[dict],
        *,
        neighbor_boost: float = DEFAULT_NEIGHBOR_BOOST,
        label_mention_boost: float = DEFAULT_LABEL_MENTION_BOOST,
    ) -> None:
        self.chunks: list[dict] = list(chunks)
        self.entities: list[dict] = list(entities)
        self.neighbor_boost = float(neighbor_boost)
        self.label_mention_boost = float(label_mention_boost)
        # Lazily-populated index state:
        self._qids_by_label: dict[str, list[str]] = {}
        self._label_by_qid: dict[str, str] = {}
        self._claims_by_qid: dict[str, list[tuple[str, str]]] = {}
        self._chunks_by_wikidata_id: dict[str, list[int]] = {}
        self._label_re: Optional[re.Pattern] = None
        self._indexed = False

    def index(self) -> None:
        """Build the entity and chunk indices from the stored streams."""
        qids_by_label: dict[str, list[str]] = defaultdict(list)
        label_by_qid: dict[str, str] = {}
        claims_by_qid: dict[str, list[tuple[str, str]]] = {}

        for ent in self.entities:
            qid = ent.get("wikidata_id")
            if not qid:
                continue
            label = ent.get("label", qid)
            label_by_qid[qid] = label
            folded = _casefold(label)
            if folded and qid not in qids_by_label[folded]:
                qids_by_label[folded].append(qid)
            claims: list[tuple[str, str]] = []
            for claim in ent.get("claims", []) or []:
                prop = claim.get("property")
                neighbor = claim.get("value")
                if prop in RELATIONAL_PROPERTIES and isinstance(neighbor, str):
                    claims.append((prop, neighbor))
            claims_by_qid[qid] = claims

        chunks_by_wikidata_id: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.chunks):
            cqid = c.get("wikidata_id")
            if cqid:
                chunks_by_wikidata_id[cqid].append(i)

        # Longest-first alternation so re's leftmost-first behavior
        # selects the longest matching label at each query position.
        labels_longest_first = sorted(qids_by_label.keys(), key=len, reverse=True)
        if labels_longest_first:
            pattern = (
                r"\b("
                + "|".join(re.escape(label) for label in labels_longest_first)
                + r")\b"
            )
            self._label_re = re.compile(pattern)
        else:
            self._label_re = None

        self._qids_by_label = dict(qids_by_label)
        self._label_by_qid = label_by_qid
        self._claims_by_qid = claims_by_qid
        self._chunks_by_wikidata_id = dict(chunks_by_wikidata_id)
        self._indexed = True

    def _resolve_query_qids(self, q: str) -> "list[str]":
        """Recognize entities in the query and return their QIDs.

        Longest-match-wins via the precompiled alternation regex run
        against the case-folded query string. A single match position
        contributes every QID associated with that label. Order is
        preserved by first occurrence so the result is deterministic.
        """
        if self._label_re is None:
            return []
        folded = _casefold(q)
        if not folded:
            return []
        seen: list[str] = []
        seen_set: set[str] = set()
        for m in self._label_re.finditer(folded):
            for qid in self._qids_by_label.get(m.group(1), []):
                if qid not in seen_set:
                    seen.append(qid)
                    seen_set.add(qid)
        return seen

    def query(self, q: str, top_k: int = DEFAULT_TOP_K) -> "list[RetrievalHit]":
        """Return up to ``top_k`` graph-aware ranked hits for the query.

        Score per chunk is the sum of boost values from the two tiers
        described in the module docstring. Chunks scoring zero are
        dropped from the result set rather than returned with score
        ``0.0``. Results are sorted descending by score; ties are broken
        by chunk index ascending so the ordering is deterministic across
        runs.

        Args:
            q: Query text.
            top_k: Maximum number of hits to return.

        Returns:
            List of :class:`RetrievalHit` of length at most ``top_k``.
            Empty when the corpus is empty, the query contains no
            recognizable entity, the query yields no neighbors with
            associated chunks, or ``top_k <= 0``.
        """
        if not self.chunks or top_k <= 0:
            return []
        if not self._indexed:
            self.index()

        query_qids = self._resolve_query_qids(q)
        if not query_qids:
            return []

        # Tier-1 universe: the query QIDs themselves and every 1-hop
        # neighbor reachable through the six relational properties.
        tier1_qids: set[str] = set(query_qids)
        neighbor_only_qids: set[str] = set()
        for qid in query_qids:
            for _prop, neighbor in self._claims_by_qid.get(qid, []):
                if neighbor not in tier1_qids:
                    neighbor_only_qids.add(neighbor)
        tier1_qids |= neighbor_only_qids

        scores: dict[int, float] = defaultdict(float)

        # Structural boost: wikidata_id direct match.
        for qid in tier1_qids:
            for chunk_idx in self._chunks_by_wikidata_id.get(qid, []):
                scores[chunk_idx] += self.neighbor_boost

        # Mention boost: chunk text contains a neighbor's label. Only
        # neighbors that are not themselves query QIDs are considered
        # here. A chunk that *is* the neighbor receives no mention
        # boost for naming itself (self-mention is already represented
        # by the structural boost); chunks whose wikidata_id is a
        # different neighbor may still receive mention boosts for
        # OTHER neighbor labels.
        neighbor_labels: list[tuple[str, str]] = [
            (qid, _casefold(self._label_by_qid[qid]))
            for qid in neighbor_only_qids
            if qid in self._label_by_qid and self._label_by_qid[qid]
        ]
        if neighbor_labels:
            for i, c in enumerate(self.chunks):
                folded_text = _casefold(c.get("text", ""))
                if not folded_text:
                    continue
                chunk_qid = c.get("wikidata_id")
                for neighbor_qid, folded_label in neighbor_labels:
                    if not folded_label:
                        continue
                    if chunk_qid == neighbor_qid:
                        continue
                    # Whole-word check: the case-folded text already has
                    # word-boundary spaces from _casefold, so a leading
                    # and trailing space lets a substring search act as
                    # a whole-word match against the equally-folded
                    # label.
                    if (" " + folded_label + " ") in (" " + folded_text + " "):
                        scores[i] += self.label_mention_boost

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        k = min(int(top_k), len(ranked))
        return [
            RetrievalHit(
                passage_id=str(self.chunks[chunk_idx]["passage_id"]),
                score=float(score),
                source_mode=SOURCE_MODE,
            )
            for chunk_idx, score in ranked[:k]
        ]

    @classmethod
    def from_paths(
        cls,
        chunks_path: str | Path,
        entities_path: str | Path,
        *,
        neighbor_boost: float = DEFAULT_NEIGHBOR_BOOST,
        label_mention_boost: float = DEFAULT_LABEL_MENTION_BOOST,
    ) -> "GraphRetriever":
        """Construct a retriever from on-disk ``chunks.jsonl`` + ``entities.jsonl`` paths."""
        return cls(
            chunks=load_chunks(chunks_path),
            entities=load_entities(entities_path),
            neighbor_boost=neighbor_boost,
            label_mention_boost=label_mention_boost,
        )


# ---------------------------------------------------------------------------
# Smoke entry point: ``python3 -m src.retrieval.graph --smoke``
# ---------------------------------------------------------------------------

_SMOKE_CHUNKS: Sequence[dict] = (
    {
        "passage_id": "Q28865:0000",
        "wikidata_id": "Q28865",
        "title": "Python (programming language)",
        "text": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use "
            "of significant indentation."
        ),
    },
    {
        "passage_id": "Q11354:0000",
        "wikidata_id": "Q11354",
        "title": "Apache HTTP Server",
        "text": (
            "The Apache HTTP Server is a free and open-source cross-platform "
            "web server software released under the terms of Apache License 2.0."
        ),
    },
    {
        "passage_id": "Q616526:0000",
        "wikidata_id": "Q616526",
        "title": "Apache License",
        "text": (
            "The Apache License is a permissive free software license written "
            "by the Apache Software Foundation."
        ),
    },
    {
        "passage_id": "Q489772:0000",
        "wikidata_id": "Q489772",
        "title": "Python Software Foundation License",
        "text": (
            "The Python Software Foundation License is a permissive software "
            "license used for the Python reference implementation."
        ),
    },
    {
        "passage_id": "Q193321:0000",
        "wikidata_id": "Q193321",
        "title": "Linux kernel",
        "text": (
            "The Linux kernel is a free and open-source, monolithic, modular, "
            "multitasking, Unix-like operating system kernel."
        ),
    },
)

_SMOKE_ENTITIES: Sequence[dict] = (
    {
        "wikidata_id": "Q28865",
        "label": "Python",
        "claims": [
            {"property": "P275", "value": "Q489772"},
        ],
    },
    {
        "wikidata_id": "Q11354",
        "label": "Apache HTTP Server",
        "claims": [
            {"property": "P275", "value": "Q616526"},
        ],
    },
    {
        "wikidata_id": "Q616526",
        "label": "Apache License",
        "claims": [],
    },
    {
        "wikidata_id": "Q489772",
        "label": "Python Software Foundation License",
        "claims": [],
    },
    {
        "wikidata_id": "Q193321",
        "label": "Linux kernel",
        "claims": [],
    },
)

_SMOKE_QUERIES: Sequence[str] = (
    "What license does Python use?",
    "What license does Apache HTTP Server use?",
    "Tell me about the Linux kernel.",
)


def _smoke(top_k: int = 3) -> int:
    """Run the in-tree fixture smoke. Prints top-k hits per query.

    Returns the process exit code (``0`` on success).
    """
    print(
        f"graph-retriever smoke: corpus_size={len(_SMOKE_CHUNKS)} "
        f"entities={len(_SMOKE_ENTITIES)} top_k={top_k}",
        file=sys.stderr,
    )
    retriever = GraphRetriever(_SMOKE_CHUNKS, _SMOKE_ENTITIES)
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
    """CLI entrypoint for ``python3 -m src.retrieval.graph``."""
    parser = argparse.ArgumentParser(
        prog="src.retrieval.graph",
        description="Graph-aware 1-hop retrieval over chunked OSS-ecosystem corpus.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the in-tree fixture smoke and print top-3 hits per query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of hits to print per smoke query (default: 3).",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke(top_k=args.top_k)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
