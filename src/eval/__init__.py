"""Evaluation harness for the production-rag-eval-harness.

Computes the seven first-proof metrics from ``docs/retrieval-modes.md``
across the four retrieval modes (dense, sparse, hybrid, graph-aware) and
emits the canonical run artifacts under ``runs/<run-id>/``.

Public exports:

- :class:`EvalHarness` — orchestrator that loads chunks + Q-A, resolves
  logical expected-passage hints to corpus chunks, runs each retriever
  per question, generates extractive answers + citations, computes
  per-question metrics, aggregates per mode, and writes the run.
- :func:`format_run_id` — stable run identifier
  (``YYYY-MM-DD_<chunks-sha256-prefix>_seed-<int>``).
- per-metric functions in :mod:`src.eval.metrics` for unit-level reuse.
"""

from src.eval.harness import EvalHarness, format_run_id

__all__ = ["EvalHarness", "format_run_id"]
