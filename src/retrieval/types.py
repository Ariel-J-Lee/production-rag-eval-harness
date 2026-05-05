"""Shared retrieval-result type used by every retrieval mode.

All four retrieval modes (dense, sparse, hybrid, graph-aware) emit a list
of :class:`RetrievalHit` records. The eval harness consumes the same shape
across modes so comparison rows in the run report are directly aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceMode = Literal["dense", "sparse", "hybrid", "graph_aware"]


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked result from a retrieval mode.

    Attributes:
        passage_id: Stable passage identifier from ``chunks.jsonl``
            (format: ``"<wikidata_id-or-title-stem>:<NNNN>"``). The eval
            harness uses this as the citation target.
        score: Mode-specific scalar score. Cross-mode score comparison is
            not meaningful; comparison happens via per-question metrics
            in the eval harness, not by raw score.
        source_mode: One of ``"dense"``, ``"sparse"``, ``"hybrid"``,
            ``"graph_aware"``. Names match the ``mode`` field in
            ``runs/<run-id>/raw_results.jsonl`` per
            ``docs/retrieval-modes.md``.
    """

    passage_id: str
    score: float
    source_mode: SourceMode
