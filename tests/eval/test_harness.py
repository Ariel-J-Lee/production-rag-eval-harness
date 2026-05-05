"""Integration tests for src.eval.harness.

Uses small fixture chunks + Q-A files plus stub retrievers so the test
runs on stdlib + pytest. The full real-corpus run is exercised by
``make eval`` against the materialized corpus, not by these unit tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.harness import (
    DEFAULT_CITATION_COUNT,
    DEFAULT_MODES,
    DEFAULT_TOP_K,
    CitationContractError,
    EvalHarness,
    _load_jsonl,
    format_run_id,
    write_eval_report,
    write_manifest,
    write_raw_results,
)
from src.retrieval.types import RetrievalHit


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns a fixed list of RetrievalHit for any query (truncated to top_k)."""

    def __init__(self, hits: list[RetrievalHit]):
        self._hits = hits

    def query(self, q: str, top_k: int) -> list[RetrievalHit]:
        return list(self._hits[:top_k])


def _hit(passage_id: str, score: float = 1.0, source_mode: str = "dense") -> RetrievalHit:
    return RetrievalHit(passage_id=passage_id, score=score, source_mode=source_mode)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks() -> list[dict]:
    return _load_jsonl(_FIXTURE_DIR / "chunks.jsonl")


@pytest.fixture
def qa_pairs() -> list[dict]:
    return _load_jsonl(_FIXTURE_DIR / "qa_pairs.jsonl")


@pytest.fixture
def harness(chunks, qa_pairs) -> EvalHarness:
    """Harness with stub retrievers that return Q1 first."""
    factory = {
        "dense": lambda: _StubRetriever([
            _hit("Q1:0000", 0.95, "dense"),
            _hit("Q2:0000", 0.85, "dense"),
            _hit("Q3:0000", 0.75, "dense"),
        ]),
        "sparse": lambda: _StubRetriever([
            _hit("Q2:0000", 4.0, "sparse"),
            _hit("Q1:0000", 3.0, "sparse"),
            _hit("Q3:0000", 2.0, "sparse"),
        ]),
        "hybrid": lambda: _StubRetriever([
            _hit("Q1:0000", 0.5, "hybrid"),
            _hit("Q2:0000", 0.4, "hybrid"),
            _hit("Q3:0000", 0.3, "hybrid"),
        ]),
        "graph_aware": lambda: _StubRetriever([
            _hit("Q2:0000", 1.0, "graph_aware"),
            _hit("Q1:0000", 0.8, "graph_aware"),
            _hit("Q3:0000", 0.6, "graph_aware"),
        ]),
    }
    return EvalHarness(
        chunks=chunks,
        qa_pairs=qa_pairs,
        retriever_factory=factory,
    )


# ---------------------------------------------------------------------------
# Run-id format
# ---------------------------------------------------------------------------


def test_run_id_format():
    rid = format_run_id(
        date_str="2026-05-05",
        chunks_sha256="abcdef0123456789" + "0" * 48,
        seed=42,
    )
    assert rid == "2026-05-05_abcdef012345_seed-42"


def test_run_id_default_seed_zero():
    rid = format_run_id(date_str="2026-05-05", chunks_sha256="0" * 64)
    assert rid.endswith("_seed-0")


# ---------------------------------------------------------------------------
# Logical-passage resolution
# ---------------------------------------------------------------------------


def test_resolve_wikipedia_slug(harness, qa_pairs):
    """wikipedia:Alpha → chunk Q1:0000 (title='Alpha')."""
    resolved = harness.resolve_expected(qa_pairs[0])
    assert "Q1:0000" in resolved.passage_ids
    assert not resolved.unresolved_logical_ids


def test_resolve_wikidata_hint(harness, qa_pairs):
    """wikidata:beta_after_alpha → chunk Q2:0000 (the cited wikidata_entity is Q2)."""
    resolved = harness.resolve_expected(qa_pairs[1])
    assert "Q2:0000" in resolved.passage_ids


def test_resolve_multi_step_pulls_both_articles(harness, qa_pairs):
    """multi_step expected_passages = [wikipedia:Gamma, wikipedia:Delta]."""
    resolved = harness.resolve_expected(qa_pairs[2])
    assert "Q3:0000" in resolved.passage_ids
    assert "Q4:0000" in resolved.passage_ids


def test_resolve_unknown_slug_recorded_as_unresolved(harness):
    qa = {
        "id": "fake",
        "question": "?",
        "expected_answer": "x",
        "expected_passages": ["wikipedia:NonExistentArticle"],
        "public_source_citation": [],
    }
    resolved = harness.resolve_expected(qa)
    assert resolved.passage_ids == frozenset()
    assert resolved.unresolved_logical_ids == ("wikipedia:NonExistentArticle",)


# ---------------------------------------------------------------------------
# Extractive answer + citation contract
# ---------------------------------------------------------------------------


def test_generate_answer_returns_top_citation_count_passage_texts(harness):
    hits = [
        _hit("Q1:0000", 1.0),
        _hit("Q2:0000", 0.9),
        _hit("Q3:0000", 0.8),
        _hit("Q4:0000", 0.7),
        _hit("Q5:0000", 0.6),
    ]
    answer, citations = harness.generate_answer(hits)
    assert citations == ["Q1:0000", "Q2:0000", "Q3:0000"]
    # The answer joins three excerpts with " --- "
    assert "Alpha is the first letter" in answer
    assert "Beta is the second letter" in answer
    assert "Gamma is the third letter" in answer


def test_generate_answer_handles_fewer_hits_than_citation_count(harness):
    hits = [_hit("Q1:0000")]
    answer, citations = harness.generate_answer(hits)
    assert citations == ["Q1:0000"]
    assert "Alpha" in answer


def test_generate_answer_empty_hits_returns_empty():
    """No hits → no citations (caller decides whether to raise)."""
    h = EvalHarness(
        chunks=[{"passage_id": "Q1:0000", "title": "A", "text": "alpha"}],
        qa_pairs=[],
        retriever_factory={},
    )
    assert h.generate_answer([]) == ("", [])


def test_run_one_zero_hits_emits_empty_citations_record(qa_pairs, chunks):
    """A retriever returning zero hits is honest no-match: empty citations + zero metrics, no raise.

    The citation contract forbids empty citations only when the retriever
    DID return hits (a programmer-error case where the answer-generation
    step dropped them); a retriever that genuinely has no signal for the
    query (graph mode on a query with no recognized entities, for example)
    correctly emits a record with empty citations and zero retrieval metrics.
    """
    factory = {"dense": lambda: _StubRetriever([])}
    h = EvalHarness(
        chunks=chunks,
        qa_pairs=qa_pairs,
        retriever_factory=factory,
    )
    retriever = factory["dense"]()
    result = h.run_one(qa_pairs[0], "dense", retriever)
    assert result.retrieved == []
    assert result.citations == []
    assert result.answer == ""
    assert result.metrics["recall_at_5"] == 0.0
    assert result.metrics["mrr"] == 0.0
    assert result.metrics["faithfulness"] == 0


def test_run_one_raises_when_hits_present_but_citations_empty(qa_pairs, chunks):
    """The contract still forbids empty citations when retrieval returned hits."""

    class _PassageMissingFromCorpus:
        """Returns hits for passage_ids that don't exist in chunks; would
        normally still produce citations from those passage_ids — but if
        we monkeypatch generate_answer to return [] anyway, that's the
        programmer-error case the contract catches.
        """
        def query(self, q, top_k):
            return [_hit("Q1:0000")]

    h = EvalHarness(
        chunks=chunks,
        qa_pairs=qa_pairs,
        retriever_factory={"dense": lambda: _PassageMissingFromCorpus()},
    )
    # Force the programmer-error case: monkeypatch generate_answer to
    # drop citations even though hits were returned.
    h.generate_answer = lambda hits: ("", [])
    retriever = h.retriever_factory["dense"]()
    with pytest.raises(CitationContractError):
        h.run_one(qa_pairs[0], "dense", retriever)


# ---------------------------------------------------------------------------
# Run pipeline + per-mode aggregation
# ---------------------------------------------------------------------------


def test_run_returns_records_for_every_question_mode_pair(harness, qa_pairs):
    results, aggregates, diagnostics = harness.run(modes=DEFAULT_MODES)
    assert len(results) == len(qa_pairs) * len(DEFAULT_MODES)
    assert set(r.mode for r in results) == set(DEFAULT_MODES)
    assert all(r.citations for r in results)


def test_run_aggregates_have_all_metric_keys(harness):
    _results, aggregates, _diagnostics = harness.run(modes=("dense",))
    for metric in (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "citation_precision",
        "citation_recall",
        "faithfulness",
        "answer_correctness",
        "exact_match",
    ):
        assert metric in aggregates["dense"]


def test_run_diagnostics_records_unresolved(chunks):
    """A QA pair pointing at an unknown article logs in the diagnostics."""
    bad_qa = [{
        "id": "fake",
        "question": "what?",
        "expected_answer": "x",
        "expected_passages": ["wikipedia:NoSuchArticle"],
        "public_source_citation": [],
    }]
    factory = {"dense": lambda: _StubRetriever([_hit("Q1:0000")])}
    h = EvalHarness(chunks=chunks, qa_pairs=bad_qa, retriever_factory=factory)
    _r, _a, diagnostics = h.run(modes=("dense",))
    assert diagnostics["questions_with_unresolved_hint"] == 1
    assert diagnostics["total_unresolved_hints"] == 1


def test_run_subset_of_modes(harness):
    """Running only a subset of modes still produces records for the chosen modes."""
    results, aggregates, _diagnostics = harness.run(modes=("dense", "sparse"))
    assert set(r.mode for r in results) == {"dense", "sparse"}
    assert set(aggregates.keys()) == {"dense", "sparse"}


# ---------------------------------------------------------------------------
# Run-artifact writers
# ---------------------------------------------------------------------------


def test_write_raw_results_emits_jsonl_with_required_fields(harness, tmp_path):
    results, _aggregates, _diagnostics = harness.run(modes=("dense",))
    out = tmp_path / "raw_results.jsonl"
    write_raw_results(out, results)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(results)
    for line in lines:
        record = json.loads(line)
        # Required fields per the share-alike-aware schema:
        # question_id / question / mode / retrieved / citations / metrics.
        # The "answer" text is intentionally not persisted (PACKET-019
        # §2.5 share-alike isolation); it is reconstructable from
        # chunks.jsonl + citations and the answer-derived metrics live
        # under metrics.{faithfulness, answer_correctness, exact_match}.
        assert set(record.keys()) >= {
            "question_id", "question", "mode", "retrieved",
            "citations", "metrics",
        }
        assert "answer" not in record
        assert record["citations"]
        assert record["mode"] == "dense"
        assert "faithfulness" in record["metrics"]
        assert "answer_correctness" in record["metrics"]
        assert "exact_match" in record["metrics"]


def test_write_raw_results_emits_empty_citations_when_retrieval_empty(tmp_path):
    """Defense in depth: empty citations are allowed when retrieved is also empty."""
    from src.eval.harness import _PerModeResult
    record = _PerModeResult(
        question_id="q",
        question="?",
        mode="graph_aware",
        retrieved=[],
        answer="",
        citations=[],
        metrics={"recall_at_5": 0.0},
    )
    out = tmp_path / "raw.jsonl"
    write_raw_results(out, [record])
    payload = json.loads(out.read_text(encoding="utf-8").strip())
    assert payload["citations"] == []
    assert payload["retrieved"] == []


def test_write_raw_results_raises_when_hits_present_but_citations_empty(tmp_path):
    """The contract still forbids empty citations when retrieved is non-empty."""
    from src.eval.harness import _PerModeResult
    bad = _PerModeResult(
        question_id="q",
        question="?",
        mode="dense",
        retrieved=[{"passage_id": "Q1:0000", "score": 0.5}],
        answer="some answer",
        citations=[],
        metrics={},
    )
    with pytest.raises(CitationContractError):
        write_raw_results(tmp_path / "raw.jsonl", [bad])


def test_write_raw_results_raises_on_malformed_citation(tmp_path):
    """Citations must be non-empty strings."""
    from src.eval.harness import _PerModeResult
    bad = _PerModeResult(
        question_id="q",
        question="?",
        mode="dense",
        retrieved=[{"passage_id": "Q1:0000", "score": 0.5}],
        answer="some answer",
        citations=["", "Q1:0000"],
        metrics={},
    )
    with pytest.raises(CitationContractError):
        write_raw_results(tmp_path / "raw.jsonl", [bad])


def test_write_eval_report_writes_headline_table(harness, tmp_path):
    _results, aggregates, diagnostics = harness.run(modes=DEFAULT_MODES)
    out = tmp_path / "eval_report.md"
    write_eval_report(
        out,
        run_id="2026-05-05_abc123def456_seed-0",
        per_mode_aggregates=aggregates,
        qa_count=3,
        chunks_count=5,
        chunks_sha256="abc123def456" + "0" * 52,
        qa_sha256="ffeeddccbbaa" + "0" * 52,
        embedding_model="all-MiniLM-L6-v2",
        seed=0,
        timestamp_iso="2026-05-05T12:00:00Z",
        citation_count=DEFAULT_CITATION_COUNT,
        diagnostics=diagnostics,
    )
    text = out.read_text(encoding="utf-8")
    assert "Retrieval Comparison Run 2026-05-05_abc123def456_seed-0" in text
    assert "Recall@5" in text
    assert "Recall@10" in text
    assert "MRR" in text
    assert "Citation precision" in text
    assert "Citation recall" in text
    assert "Faithfulness" in text
    assert "Answer correctness" in text
    assert "Dense (vector)" in text
    assert "Sparse (BM25)" in text
    assert "Hybrid (RRF)" in text
    assert "Graph-aware" in text
    assert "Limits" in text


def test_write_manifest_contains_pinned_method_choices(harness, tmp_path):
    _results, aggregates, diagnostics = harness.run(modes=("dense",))
    out = tmp_path / "manifest.json"
    write_manifest(
        out,
        run_id="rid",
        chunks_path=Path("data/oss-ecosystem/chunks.jsonl"),
        chunks_sha256="0" * 64,
        chunks_count=5,
        qa_path=Path("qa/qa_pairs.jsonl"),
        qa_sha256="1" * 64,
        qa_count=3,
        embedding_model="all-MiniLM-L6-v2",
        seed=0,
        timestamp_iso="2026-05-05T12:00:00Z",
        citation_count=3,
        modes=("dense",),
        per_mode_aggregates=aggregates,
        diagnostics=diagnostics,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == "rid"
    assert payload["citation_count"] == 3
    assert payload["embedding_model"] == "all-MiniLM-L6-v2"
    assert payload["method"]["faithfulness"] == "heuristic_token_overlap_binary"
    assert payload["method"]["answer_correctness"] == "token_f1"
    assert payload["method"]["answer_generation"] == "extractive_top_k_passages_joined"
    assert payload["regression_gate"]["verdict"] == "not_evaluated"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_citation_count_is_three():
    assert DEFAULT_CITATION_COUNT == 3


def test_default_top_k_is_ten():
    assert DEFAULT_TOP_K == 10


def test_default_modes_is_four_modes():
    assert DEFAULT_MODES == ("dense", "sparse", "hybrid", "graph_aware")
