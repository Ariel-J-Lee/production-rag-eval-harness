"""Evaluation harness for the four-mode comparison.

Loads chunked corpus + Q-A set, resolves logical expected-passage hints
to corpus chunks, runs each retriever per question, generates extractive
answers + citations, computes per-question metrics, aggregates per
mode, and writes the canonical run artifacts under ``runs/<run-id>/``:

- ``runs/<run-id>/eval_report.md`` — headline 4-row × 7-metric table
  plus method notes and limits
- ``runs/<run-id>/raw_results.jsonl`` — one record per (question × mode)
  with the question, retrieved passages, generated answer, citations,
  and per-question metric values
- ``runs/<run-id>/manifest.json`` — corpus snapshot SHA-256, Q-A set
  hash, model version, seed, run timestamp, mode list, citation_count

Locked first-proof choices (from the PACKET-043 GO direction):

- faithfulness  → heuristic token-overlap, not LLM judge
- answer correctness → exact-match + token-F1, not graded LLM judge
- answer generation → extractive (top ``citation_count`` cited passage
  texts joined), not LLM-generated
- citation_count → 3
- run-id format → ``YYYY-MM-DD_<chunks-sha256-prefix>_seed-<int>``

The harness is local-first and laptop-runnable: no hosted LLM calls,
no managed retrieval services. The retrievers it consumes are the
already-merged dense, sparse, hybrid, and graph-aware modes.

Public entry points:

- :class:`EvalHarness` — orchestrator class.
- :func:`format_run_id` — stable run-id formatter.
- ``python3 -m src.eval.harness --chunks <path> --qa <path> ...`` — CLI.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from urllib.parse import unquote

from src.eval.metrics import (
    DEFAULT_FAITHFULNESS_THRESHOLD,
    citation_precision,
    citation_recall,
    default_tokenize,
    exact_match,
    faithfulness_heuristic,
    mean,
    mrr,
    recall_at_k,
    token_f1,
)
from src.retrieval.types import RetrievalHit

DEFAULT_TOP_K = 10
DEFAULT_CITATION_COUNT = 3
DEFAULT_SEED = 0
DEFAULT_MODES = ("dense", "sparse", "hybrid", "graph_aware")
RUN_ID_HASH_PREFIX_LEN = 12


# ---------------------------------------------------------------------------
# Run-id format
# ---------------------------------------------------------------------------


def format_run_id(
    *,
    date_str: Optional[str] = None,
    chunks_sha256: str,
    seed: int = DEFAULT_SEED,
    hash_prefix_len: int = RUN_ID_HASH_PREFIX_LEN,
) -> str:
    """Return ``YYYY-MM-DD_<chunks-sha256[:N]>_seed-<int>`` per the GO direction.

    Args:
        date_str: ISO date for the run (default: today, UTC).
        chunks_sha256: Full hex SHA-256 of the chunks.jsonl file.
        seed: Run seed (integer; default 0).
        hash_prefix_len: Number of hex chars to keep from the SHA-256
            prefix (default 12).

    Returns:
        The run-id string, suitable as a directory name under ``runs/``.
    """
    if date_str is None:
        date_str = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    prefix = chunks_sha256[:hash_prefix_len]
    return f"{date_str}_{prefix}_seed-{seed}"


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file as a list of dicts. Skips empty lines."""
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _wikipedia_slug_to_title(slug: str) -> str:
    """Normalize a Wikipedia article slug to the title form stored in chunks.

    The slug uses underscores and URL-encoding; Wikipedia article titles
    in chunks.jsonl use spaces. This mirrors the inverse of the
    ``_normalize_for_url`` step in ``scripts/fetch_corpus.py``.
    """
    return unquote(slug).replace("_", " ")


# ---------------------------------------------------------------------------
# Resolved expected passages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedExpected:
    """The expected-passage set for one Q-A pair, plus diagnostics."""

    passage_ids: frozenset[str]
    unresolved_logical_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Citation enforcement error
# ---------------------------------------------------------------------------


class CitationContractError(ValueError):
    """Raised when an answer record violates the citations-array contract.

    The contract: every record in ``raw_results.jsonl`` carries a
    ``citations`` field that is a list of non-empty strings (each a
    valid passage_id). Empty list is allowed when the retriever
    returned zero hits for the query (honest reporting of a no-match
    case). Empty list with a non-empty ``retrieved`` field is the
    fail-fast condition the PACKET-043 Validation rule "enforce the
    `citations` array contract on every answer record" forbids.
    """


# ---------------------------------------------------------------------------
# Per-question result record
# ---------------------------------------------------------------------------


@dataclass
class _PerModeResult:
    """The full record for one (question, mode) tuple before serialization."""

    question_id: str
    question: str
    mode: str
    retrieved: list[dict]                 # [{passage_id, score}, ...]
    answer: str
    citations: list[str]
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


@dataclass
class EvalHarness:
    """Orchestrate the four-mode scored run.

    Attributes:
        chunks: Chunked corpus records (dicts as produced by
            ``scripts/fetch_corpus.py``; required keys ``passage_id``,
            ``text``, ``title``, ``wikidata_id``).
        qa_pairs: Q-A pair records (per ``qa/SCHEMA.md``).
        retriever_factory: Mapping ``{mode_name: callable -> retriever}``.
            The callable takes no arguments and returns an object with a
            ``query(q: str, top_k: int) -> list[RetrievalHit]`` method.
            The harness builds each retriever once at the start of the
            run.
        top_k: Top-k cutoff applied to the retriever's ranked list.
            Default 10 (covers Recall@10).
        citation_count: How many top-ranked retrieved passages become
            the answer's citations. Default 3 per the GO direction.
        faithfulness_threshold: Token-overlap threshold for the binary
            faithfulness metric. Default 0.6.
        tokenizer: Tokenizer shared across answer-quality metrics.
            Default :func:`src.eval.metrics.default_tokenize`.
    """

    chunks: list[dict]
    qa_pairs: list[dict]
    retriever_factory: dict[str, Callable[[], object]]
    top_k: int = DEFAULT_TOP_K
    citation_count: int = DEFAULT_CITATION_COUNT
    faithfulness_threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD
    tokenizer: Callable[[str], list[str]] = default_tokenize

    # Lazily built indices
    _by_passage_id: dict[str, dict] = field(init=False, repr=False)
    _by_title: dict[str, list[str]] = field(init=False, repr=False)
    _by_wikidata_id: dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_passage_id = {c["passage_id"]: c for c in self.chunks}
        self._by_title = {}
        self._by_wikidata_id = {}
        for c in self.chunks:
            title = c.get("title")
            if title:
                self._by_title.setdefault(title, []).append(c["passage_id"])
            qid = c.get("wikidata_id")
            if qid:
                self._by_wikidata_id.setdefault(qid, []).append(c["passage_id"])

    # ------------------------------------------------------------------
    # Logical-id resolution
    # ------------------------------------------------------------------

    def resolve_expected(self, qa_pair: dict) -> _ResolvedExpected:
        """Resolve logical expected_passages to corpus chunk passage_ids.

        - ``wikipedia:<slug>`` resolves to every chunk whose ``title``
          matches the slug-decoded title.
        - ``wikidata:<topic_slug>`` is a logical hint pointing through a
          Wikidata property edge; the harness resolves it to every chunk
          whose ``wikidata_id`` matches a QID listed in the question's
          ``public_source_citation`` of kind ``wikidata_entity``. This
          treats the cited entity as the concrete corpus target; the
          per-property graph edge is exercised by the graph-aware
          retriever's own logic, not the resolution step.
        """
        passage_ids: set[str] = set()
        unresolved: list[str] = []

        for logical_id in qa_pair.get("expected_passages", []):
            if logical_id.startswith("wikipedia:"):
                slug = logical_id[len("wikipedia:"):]
                title = _wikipedia_slug_to_title(slug)
                hits = self._by_title.get(title)
                if hits:
                    passage_ids.update(hits)
                else:
                    unresolved.append(logical_id)
            elif logical_id.startswith("wikidata:"):
                # Pull the QIDs from the question's public_source_citation
                resolved_any = False
                for citation in qa_pair.get("public_source_citation", []):
                    if citation.get("kind") == "wikidata_entity":
                        qid = citation.get("qid")
                        if qid:
                            hits = self._by_wikidata_id.get(qid)
                            if hits:
                                passage_ids.update(hits)
                                resolved_any = True
                if not resolved_any:
                    unresolved.append(logical_id)
            else:
                unresolved.append(logical_id)

        return _ResolvedExpected(
            passage_ids=frozenset(passage_ids),
            unresolved_logical_ids=tuple(unresolved),
        )

    # ------------------------------------------------------------------
    # Extractive answer + citations
    # ------------------------------------------------------------------

    def generate_answer(self, hits: Sequence[RetrievalHit]) -> tuple[str, list[str]]:
        """Build an extractive answer and citation list from ranked hits.

        Locked first-proof method: take the top ``citation_count`` hits;
        the citations are exactly those passage_ids; the answer text is
        the joined text of the cited passages (joined by " --- ").
        Fail-fast: returns an empty list if the hit list is empty (the
        caller decides whether that violates the citation contract).
        """
        n = min(int(self.citation_count), len(hits))
        if n == 0:
            return "", []
        cited = list(hits[:n])
        citations = [h.passage_id for h in cited]
        excerpts: list[str] = []
        for hit in cited:
            chunk = self._by_passage_id.get(hit.passage_id)
            if chunk:
                excerpts.append(str(chunk.get("text", "")))
        answer = " --- ".join(excerpts).strip()
        return answer, citations

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------

    def run_one(
        self,
        qa_pair: dict,
        mode: str,
        retriever: object,
    ) -> _PerModeResult:
        """Run one (question, mode) tuple and return its result record."""
        q = qa_pair["question"]
        gold = qa_pair.get("expected_answer", "")
        resolved = self.resolve_expected(qa_pair)
        expected_set = resolved.passage_ids

        hits: list[RetrievalHit] = list(retriever.query(q, top_k=self.top_k))
        retrieved_records = [
            {"passage_id": h.passage_id, "score": float(h.score)} for h in hits
        ]
        ranked_pids = [h.passage_id for h in hits]

        answer, citations = self.generate_answer(hits)
        # Citation-contract enforcement:
        # - retriever returned hits but generate_answer produced no citations
        #   → programmer error; fail fast.
        # - retriever returned zero hits → honest no-match; empty citations OK.
        if hits and not citations:
            raise CitationContractError(
                f"Question {qa_pair.get('id')!r} mode {mode!r}: "
                f"retriever returned {len(hits)} hits but generate_answer "
                f"produced no citations"
            )
        if not all(isinstance(c, str) and c for c in citations):
            raise CitationContractError(
                f"Question {qa_pair.get('id')!r} mode {mode!r}: "
                f"every citation must be a non-empty string"
            )

        cited_texts = [
            str(self._by_passage_id.get(c, {}).get("text", "")) for c in citations
        ]

        per_q_metrics = {
            "recall_at_5": recall_at_k(ranked_pids, expected_set, 5),
            "recall_at_10": recall_at_k(ranked_pids, expected_set, 10),
            "mrr": mrr(ranked_pids, expected_set),
            "citation_precision": citation_precision(citations, expected_set),
            "citation_recall": citation_recall(citations, expected_set),
            "faithfulness": faithfulness_heuristic(
                answer,
                cited_texts,
                threshold=self.faithfulness_threshold,
                tokenizer=self.tokenizer,
            ),
            "answer_correctness": token_f1(answer, gold, tokenizer=self.tokenizer),
            "exact_match": exact_match(answer, gold, tokenizer=self.tokenizer),
        }

        return _PerModeResult(
            question_id=str(qa_pair.get("id", "")),
            question=q,
            mode=mode,
            retrieved=retrieved_records,
            answer=answer,
            citations=citations,
            metrics=per_q_metrics,
        )

    def run(
        self,
        *,
        seed: int = DEFAULT_SEED,
        modes: Sequence[str] = DEFAULT_MODES,
    ) -> tuple[list[_PerModeResult], dict[str, dict[str, float]], dict[str, int]]:
        """Run every mode against every Q-A pair.

        Returns:
            ``(per_question_results, per_mode_aggregates, resolution_diagnostics)``.

            - ``per_question_results``: flat list of :class:`_PerModeResult`
              records, one per (question × mode) tuple.
            - ``per_mode_aggregates``: ``{mode: {metric: mean_value}}``.
            - ``resolution_diagnostics``: ``{
                "questions_with_unresolved_hint": int,
                "total_unresolved_hints": int,
              }`` for the run report's "Notes" section.
        """
        retrievers = {mode: self.retriever_factory[mode]() for mode in modes}

        results: list[_PerModeResult] = []
        unresolved_questions = 0
        total_unresolved_hints = 0
        for qa_pair in self.qa_pairs:
            resolved = self.resolve_expected(qa_pair)
            if resolved.unresolved_logical_ids:
                unresolved_questions += 1
                total_unresolved_hints += len(resolved.unresolved_logical_ids)
            for mode in modes:
                result = self.run_one(qa_pair, mode, retrievers[mode])
                results.append(result)

        per_mode_aggregates: dict[str, dict[str, float]] = {}
        metric_names = (
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "citation_precision",
            "citation_recall",
            "faithfulness",
            "answer_correctness",
            "exact_match",
        )
        for mode in modes:
            mode_results = [r for r in results if r.mode == mode]
            per_mode_aggregates[mode] = {
                m: mean(r.metrics[m] for r in mode_results) for m in metric_names
            }

        diagnostics = {
            "questions_with_unresolved_hint": unresolved_questions,
            "total_unresolved_hints": total_unresolved_hints,
        }
        return results, per_mode_aggregates, diagnostics


# ---------------------------------------------------------------------------
# Run-artifact writers
# ---------------------------------------------------------------------------


_MODE_DISPLAY_NAMES = {
    "dense": "Dense (vector)",
    "sparse": "Sparse (BM25)",
    "hybrid": "Hybrid (RRF)",
    "graph_aware": "Graph-aware",
}


def write_raw_results(
    path: Path,
    results: Sequence[_PerModeResult],
) -> None:
    """Write per-question, per-mode results as JSONL.

    Each line carries: ``question_id``, ``question``, ``mode``,
    ``retrieved``, ``citations``, ``metrics``. Re-validates the citation
    contract on every record before emission.

    The ``answer`` text is intentionally **not** persisted in
    ``raw_results.jsonl``: extractive answers consist of the joined text
    of the cited passages, which carry the Wikipedia CC-BY-SA-4.0
    obligation. Per PACKET-019 §2.5 / PACKET-036 license posture, that
    obligation is isolated to the gitignored ``data/<corpus>/``
    subdirectory; storing the joined passage text inside ``runs/`` would
    cross that isolation boundary unnecessarily. The answer is fully
    reconstructable from ``chunks.jsonl`` + the ``citations`` array,
    and the answer-derived metrics (``faithfulness``,
    ``answer_correctness``, ``exact_match``) are computed at run time
    and stored in the record's ``metrics`` field.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            # Defense in depth: re-check the citation contract before
            # emission. Empty citations are allowed only when retrieval
            # returned no hits; otherwise the field must be a list of
            # non-empty strings.
            if r.retrieved and not r.citations:
                raise CitationContractError(
                    f"Cannot write record with empty citations and non-empty "
                    f"retrieved field: question {r.question_id!r} mode {r.mode!r}"
                )
            if not all(isinstance(c, str) and c for c in r.citations):
                raise CitationContractError(
                    f"Cannot write record with malformed citations: "
                    f"question {r.question_id!r} mode {r.mode!r}"
                )
            payload = {
                "question_id": r.question_id,
                "question": r.question,
                "mode": r.mode,
                "retrieved": r.retrieved,
                "citations": r.citations,
                "metrics": r.metrics,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_eval_report(
    path: Path,
    *,
    run_id: str,
    per_mode_aggregates: dict[str, dict[str, float]],
    qa_count: int,
    chunks_count: int,
    chunks_sha256: str,
    qa_sha256: str,
    embedding_model: str,
    seed: int,
    timestamp_iso: str,
    citation_count: int,
    diagnostics: dict[str, int],
    modes: Sequence[str] = DEFAULT_MODES,
) -> None:
    """Write the headline ``eval_report.md`` per ``runs/README.md``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    headline_metric_keys = (
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
        ("mrr", "MRR"),
        ("citation_precision", "Citation precision"),
        ("citation_recall", "Citation recall"),
        ("faithfulness", "Faithfulness"),
        ("answer_correctness", "Answer correctness"),
    )

    lines: list[str] = []
    lines.append(f"# Retrieval Comparison Run {run_id}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- Corpus: open-source software ecosystem slice "
        f"(Wikidata + Wikipedia), {chunks_count} chunks; "
        f"chunks.jsonl SHA-256 prefix `{chunks_sha256[:RUN_ID_HASH_PREFIX_LEN]}`"
    )
    lines.append(f"- Q-A pairs: {qa_count}; qa_pairs.jsonl SHA-256 prefix `{qa_sha256[:RUN_ID_HASH_PREFIX_LEN]}`")
    lines.append(f"- Embedding model: {embedding_model}")
    lines.append("- LLM: none (extractive answers; no hosted-API call on the run path)")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Timestamp: {timestamp_iso}")
    lines.append(f"- Citation count: {citation_count} (top retrieved passages cited per question)")
    lines.append("")

    lines.append("## Headline Comparison")
    lines.append("")
    header = "| Mode | " + " | ".join(label for _, label in headline_metric_keys) + " |"
    sep = "|---|" + "|".join("---:" for _ in headline_metric_keys) + "|"
    lines.append(header)
    lines.append(sep)
    for mode in modes:
        if mode not in per_mode_aggregates:
            continue
        row = [_MODE_DISPLAY_NAMES.get(mode, mode)]
        for key, _label in headline_metric_keys:
            value = per_mode_aggregates[mode].get(key, 0.0)
            row.append(f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Method Notes")
    lines.append("")
    lines.append(
        "- **Faithfulness** is a binary first-proof heuristic: an answer "
        "is faithful when at least 60% of its tokens appear in the union "
        "of cited-passage tokens (after a small stopword strip and "
        "lowercase normalization). The graded LLM-judge upgrade is a v1+ "
        "follow-on tracked in `ROADMAP.md`."
    )
    lines.append(
        "- **Answer correctness** is SQuAD-style token-F1 against the "
        "Q-A pair's `expected_answer`. Exact-match is reported in "
        "`raw_results.jsonl` (per-question `metrics.exact_match` field) "
        "as a stricter diagnostic; the headline column is token-F1. The "
        "graded LLM-judge upgrade is a v1+ follow-on."
    )
    lines.append(
        "- **Answers** are extractive: each (question, mode) answer is "
        f"the joined text of the top-{citation_count} retrieved "
        "passages. This makes the retrieval signal (Recall@k, MRR) the "
        "load-bearing measurement and removes LLM-generation variance "
        "from the first scored proof. Generated-answer evaluation is a "
        "v1+ follow-on."
    )
    lines.append(
        "- **Citation contract**: every record in "
        "`raw_results.jsonl` carries a `citations` array of passage_ids. "
        "Empty citations are allowed only when the underlying retriever "
        "returned no hits (an honest no-match for the query in that "
        "mode); empty citations with non-empty retrieval is a "
        "fail-fast condition."
    )
    lines.append(
        "- **Answer reconstruction**: `raw_results.jsonl` records do "
        "not store the extractive answer text directly, because the "
        "answer consists of joined Wikipedia passages and the "
        "share-alike isolation rule (PACKET-019 §2.5) keeps Wikipedia "
        "passage text inside the gitignored corpus subdirectory. The "
        "answer is reconstructable from `data/<corpus>/chunks.jsonl` + "
        "the record's `citations` field. The answer-derived metrics "
        "(`faithfulness`, `answer_correctness`, `exact_match`) are "
        "computed at run time and stored in the record's `metrics` "
        "field, so reading the report does not require reconstructing "
        "answers."
    )
    if diagnostics.get("total_unresolved_hints", 0):
        lines.append(
            f"- **Resolution diagnostics**: "
            f"{diagnostics['questions_with_unresolved_hint']} questions had at least one "
            f"`expected_passages` hint that did not resolve to a corpus chunk "
            f"(total unresolved hints: {diagnostics['total_unresolved_hints']}). "
            "Per PACKET-037 coordination note 1, unresolved hints are a "
            "corpus-coverage signal rather than a Q-A defect; the affected "
            "questions still contribute to retrieval and answer-quality "
            "metrics computed against the resolved subset of expected passages."
        )
    lines.append("")

    lines.append("## Per-mode rationale")
    lines.append("")
    lines.append(
        "Per the canonical first-proof report shape, the next slice "
        "(generated-answer evaluation, regression gate) will populate "
        "this section with one example query per mode where that mode's "
        "ranking choice is most distinctive. At first proof the "
        "headline table itself is the load-bearing artifact; per-query "
        "examples live in `raw_results.jsonl`."
    )
    lines.append("")

    lines.append("## Limits")
    lines.append("")
    lines.append(
        f"- Q-A set is {qa_count} pairs; statistical confidence on "
        "retrieval-quality differences is bounded at this sample size."
    )
    lines.append(
        f"- Corpus slice is {chunks_count} chunks; results do not "
        "generalize beyond this slice."
    )
    lines.append(
        "- Faithfulness and answer correctness are first-proof heuristic "
        "implementations; LLM-judge variants are deferred to a v1+ "
        "follow-on."
    )
    lines.append(
        "- This is a **first proof**, not a benchmark. The headline "
        "comparison shows which retrieval mode wins on which metric "
        "category at this corpus and Q-A scale; reading it as a "
        "ranking of retrieval techniques across all corpora is out of "
        "scope."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    run_id: str,
    chunks_path: Path,
    chunks_sha256: str,
    chunks_count: int,
    qa_path: Path,
    qa_sha256: str,
    qa_count: int,
    embedding_model: str,
    seed: int,
    timestamp_iso: str,
    citation_count: int,
    modes: Sequence[str],
    per_mode_aggregates: dict[str, dict[str, float]],
    diagnostics: dict[str, int],
) -> None:
    """Write ``manifest.json`` per ``runs/README.md``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "captured_at": timestamp_iso,
        "seed": seed,
        "modes": list(modes),
        "citation_count": citation_count,
        "embedding_model": embedding_model,
        "corpus": {
            "chunks_path": str(chunks_path),
            "chunks_count": chunks_count,
            "chunks_sha256": chunks_sha256,
        },
        "qa": {
            "path": str(qa_path),
            "count": qa_count,
            "sha256": qa_sha256,
        },
        "method": {
            "faithfulness": "heuristic_token_overlap_binary",
            "faithfulness_threshold": DEFAULT_FAITHFULNESS_THRESHOLD,
            "answer_correctness": "token_f1",
            "answer_generation": "extractive_top_k_passages_joined",
        },
        "regression_gate": {
            "verdict": "not_evaluated",
            "note": "Regression gate is a v1+ follow-on per PACKET-043 forbidden-surface rule "
                    "('No regression-gate or CI expansion beyond the first scored run').",
        },
        "per_mode_aggregates": per_mode_aggregates,
        "diagnostics": diagnostics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_default_retrievers(chunks: list[dict], model_name: str) -> dict[str, Callable[[], object]]:
    """Build the default four-mode retriever factory for the live run.

    Each entry is a no-arg callable that returns a built retriever; the
    harness invokes them once at the start of ``run()`` so the dense
    encoder loads only once.
    """
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.graph import GraphRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.sparse import SparseRetriever

    def _make_dense():
        return DenseRetriever(chunks, model_name=model_name)

    def _make_sparse():
        return SparseRetriever(chunks)

    def _make_hybrid():
        return HybridRetriever(
            DenseRetriever(chunks, model_name=model_name),
            SparseRetriever(chunks),
        )

    def _make_graph():
        # The graph retriever needs an entities.jsonl path; resolve via the
        # standard layout (sibling of chunks.jsonl). Built lazily so this
        # function imports the graph module only when the graph mode runs.
        from src.retrieval.graph import load_entities
        # Caller passes the path through CLI args; here we close over the
        # path established by the chunks_path argument. The CLI rewires
        # this in the main() function below.
        raise RuntimeError(
            "Graph retriever factory not bound; rebind via the CLI's "
            "_resolved_graph_factory before running."
        )

    return {
        "dense": _make_dense,
        "sparse": _make_sparse,
        "hybrid": _make_hybrid,
        "graph_aware": _make_graph,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.eval.harness",
        description="Run the four-mode evaluation and write runs/<run-id>/.",
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--entities", default=None,
                        help="Path to entities.jsonl (default: sibling of --chunks)")
    parser.add_argument("--qa", required=True, help="Path to qa_pairs.jsonl")
    parser.add_argument("--runs-dir", default="runs", help="Output directory (default: runs)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--citation-count", type=int, default=DEFAULT_CITATION_COUNT)
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model name for the dense retriever")
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES),
                        help="Subset of modes to run (default: all four)")
    args = parser.parse_args(argv)

    chunks_path = Path(args.chunks)
    qa_path = Path(args.qa)
    runs_dir = Path(args.runs_dir)

    if args.entities is None:
        entities_path = chunks_path.parent / "entities.jsonl"
    else:
        entities_path = Path(args.entities)

    print(f"[eval] loading chunks from {chunks_path}", file=sys.stderr)
    chunks = _load_jsonl(chunks_path)
    print(f"[eval] loaded {len(chunks)} chunks", file=sys.stderr)

    print(f"[eval] loading qa from {qa_path}", file=sys.stderr)
    qa_pairs = _load_jsonl(qa_path)
    print(f"[eval] loaded {len(qa_pairs)} qa pairs", file=sys.stderr)

    chunks_sha256 = _sha256_file(chunks_path)
    qa_sha256 = _sha256_file(qa_path)
    run_id = format_run_id(chunks_sha256=chunks_sha256, seed=args.seed)
    print(f"[eval] run_id = {run_id}", file=sys.stderr)

    # Build the retriever factory with entities_path bound for graph mode.
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.graph import GraphRetriever, load_entities
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.sparse import SparseRetriever

    def _make_dense():
        return DenseRetriever(chunks, model_name=args.model)

    def _make_sparse():
        return SparseRetriever(chunks)

    def _make_hybrid():
        return HybridRetriever(
            DenseRetriever(chunks, model_name=args.model),
            SparseRetriever(chunks),
        )

    def _make_graph():
        entities = load_entities(entities_path)
        return GraphRetriever(chunks, entities)

    factory: dict[str, Callable[[], object]] = {
        "dense": _make_dense,
        "sparse": _make_sparse,
        "hybrid": _make_hybrid,
        "graph_aware": _make_graph,
    }

    harness = EvalHarness(
        chunks=chunks,
        qa_pairs=qa_pairs,
        retriever_factory=factory,
        top_k=args.top_k,
        citation_count=args.citation_count,
    )

    print(f"[eval] running modes: {args.modes}", file=sys.stderr)
    t0 = time.perf_counter()
    results, aggregates, diagnostics = harness.run(seed=args.seed, modes=args.modes)
    elapsed = time.perf_counter() - t0
    print(f"[eval] {len(results)} (question × mode) records in {elapsed:.1f}s", file=sys.stderr)

    timestamp_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_dir = runs_dir / run_id
    write_raw_results(run_dir / "raw_results.jsonl", results)
    write_eval_report(
        run_dir / "eval_report.md",
        run_id=run_id,
        per_mode_aggregates=aggregates,
        qa_count=len(qa_pairs),
        chunks_count=len(chunks),
        chunks_sha256=chunks_sha256,
        qa_sha256=qa_sha256,
        embedding_model=args.model,
        seed=args.seed,
        timestamp_iso=timestamp_iso,
        citation_count=args.citation_count,
        diagnostics=diagnostics,
        modes=args.modes,
    )
    write_manifest(
        run_dir / "manifest.json",
        run_id=run_id,
        chunks_path=chunks_path,
        chunks_sha256=chunks_sha256,
        chunks_count=len(chunks),
        qa_path=qa_path,
        qa_sha256=qa_sha256,
        qa_count=len(qa_pairs),
        embedding_model=args.model,
        seed=args.seed,
        timestamp_iso=timestamp_iso,
        citation_count=args.citation_count,
        modes=args.modes,
        per_mode_aggregates=aggregates,
        diagnostics=diagnostics,
    )
    print(f"[eval] wrote {run_dir}/", file=sys.stderr)
    print(f"[eval]   eval_report.md")
    print(f"[eval]   raw_results.jsonl")
    print(f"[eval]   manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
