"""Unit tests for src.retrieval.graph.

The graph-aware retriever is pure-stdlib 1-hop expansion + label
matching, so the tests run with nothing beyond pytest in the
environment — no model download, no third-party dependency.

What these tests cover:

- Entity index: case-folded label → QID(s); claims by QID restricted to
  the six relational properties; chunks indexed by ``wikidata_id``.
- Longest-match-wins entity recognition: a query mentioning
  "Apache HTTP Server" resolves to the Apache QID, not "Apache" alone
  when both labels exist.
- Structural boost: a chunk whose ``wikidata_id`` is a query QID or
  1-hop neighbor receives the neighbor boost.
- Mention boost: a chunk whose text contains a neighbor's label
  receives the smaller label-mention boost.
- Boost accumulation: a chunk satisfying both stages receives the sum.
- Empty corpus, empty query, OOV-only query: all return ``[]``.
- ``top_k`` clamping and ``top_k=0``: behave like the dense and sparse
  retrievers.
- File-on-disk loader: ``from_paths`` round-trips a chunks.jsonl +
  entities.jsonl pair.
- Output shape: hits are :class:`RetrievalHit` instances with
  ``source_mode == "graph_aware"``.
- Determinism: identical inputs produce identical ranked output.
- Tie-break: chunk index ascending when scores match.
- Property filter: only the six relational properties contribute to
  1-hop expansion; out-of-set properties are ignored.
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

from src.retrieval.graph import (
    DEFAULT_LABEL_MENTION_BOOST,
    DEFAULT_NEIGHBOR_BOOST,
    DEFAULT_TOP_K,
    RELATIONAL_PROPERTIES,
    SOURCE_MODE,
    GraphRetriever,
    _casefold,
    load_entities,
)
from src.retrieval.types import RetrievalHit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks() -> list[dict]:
    """Chunks chosen so 1-hop expansion has unambiguous winners.

    Each chunk's ``wikidata_id`` matches the entity index below, except
    Q12345 which is a control: an entity that exists only as text and
    has no entity record, so it should never be reachable.
    """
    return [
        {
            "passage_id": "Q28865:0000",
            "wikidata_id": "Q28865",
            "title": "Python (programming language)",
            "text": "Python is a high-level programming language emphasizing readability.",
        },
        {
            "passage_id": "Q11354:0000",
            "wikidata_id": "Q11354",
            "title": "Apache HTTP Server",
            "text": "The Apache HTTP Server is a cross-platform web server.",
        },
        {
            "passage_id": "Q616526:0000",
            "wikidata_id": "Q616526",
            "title": "Apache License",
            "text": "The Apache License is a permissive free software license.",
        },
        {
            "passage_id": "Q489772:0000",
            "wikidata_id": "Q489772",
            "title": "Python Software Foundation License",
            "text": "The Python Software Foundation License is permissive.",
        },
        {
            "passage_id": "Q193321:0000",
            "wikidata_id": "Q193321",
            "title": "Linux kernel",
            "text": "The Linux kernel is a free and open-source operating system kernel.",
        },
    ]


@pytest.fixture
def entities() -> list[dict]:
    """Entities matching the chunks above plus claim edges for tier-1 expansion.

    - Python (Q28865) → P275 license = Q489772 (Python Software Foundation License)
    - Apache HTTP Server (Q11354) → P275 license = Q616526 (Apache License)
    - The two licenses and Linux kernel are leaves with no claims.
    """
    return [
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
    ]


# ---------------------------------------------------------------------------
# Casefold + load_entities
# ---------------------------------------------------------------------------


def test_casefold_collapses_punctuation_and_lowercases():
    assert _casefold("Apache HTTP Server!") == "apache http server"
    assert _casefold("  multiple   spaces  ") == "multiple spaces"


def test_casefold_empty_string():
    assert _casefold("") == ""
    assert _casefold("   ") == ""


def test_load_entities_round_trips_jsonl(tmp_path, entities):
    p = tmp_path / "entities.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for e in entities:
            f.write(json.dumps(e) + "\n")
        f.write("\n")  # trailing blank line should be skipped

    loaded = load_entities(p)
    assert loaded == entities


# ---------------------------------------------------------------------------
# Index-construction tests
# ---------------------------------------------------------------------------


def test_index_populates_qids_by_label(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()

    assert retriever._qids_by_label["python"] == ["Q28865"]
    assert retriever._qids_by_label["apache http server"] == ["Q11354"]
    assert retriever._qids_by_label["linux kernel"] == ["Q193321"]


def test_index_populates_label_by_qid(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()

    assert retriever._label_by_qid["Q28865"] == "Python"
    assert retriever._label_by_qid["Q11354"] == "Apache HTTP Server"


def test_index_claims_restricted_to_relational_properties(chunks):
    """Claims with properties outside the six-property set must be dropped."""
    entities = [
        {
            "wikidata_id": "Q1",
            "label": "Test",
            "claims": [
                {"property": "P275", "value": "Q2"},  # in set
                {"property": "P999", "value": "Q3"},  # out of set, must drop
            ],
        }
    ]
    retriever = GraphRetriever(chunks, entities)
    retriever.index()

    assert retriever._claims_by_qid["Q1"] == [("P275", "Q2")]


def test_relational_properties_match_data_source_md():
    """The property allow-list must match the corpus attestation."""
    assert RELATIONAL_PROPERTIES == ("P31", "P178", "P277", "P275", "P306", "P407")


def test_index_chunks_by_wikidata_id(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()

    assert retriever._chunks_by_wikidata_id["Q28865"] == [0]
    assert retriever._chunks_by_wikidata_id["Q616526"] == [2]


def test_index_idempotent(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    first_state = dict(retriever._qids_by_label)
    retriever.index()
    assert retriever._qids_by_label == first_state


def test_index_handles_missing_wikidata_id(entities):
    """Chunks without a wikidata_id are eligible for the mention boost only."""
    chunks = [
        {"passage_id": "T1:0000", "title": "x", "text": "Apache License is permissive."},
    ]
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    assert retriever._chunks_by_wikidata_id == {}


# ---------------------------------------------------------------------------
# Entity recognition (longest-match-wins)
# ---------------------------------------------------------------------------


def test_query_resolves_simple_entity(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    assert retriever._resolve_query_qids("python is awesome") == ["Q28865"]


def test_query_resolves_multi_word_entity(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    assert retriever._resolve_query_qids(
        "what license does the Apache HTTP Server use"
    ) == ["Q11354"]


def test_query_resolves_multiple_entities(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    qids = retriever._resolve_query_qids("compare Python and the Linux kernel")
    assert set(qids) == {"Q28865", "Q193321"}


def test_query_returns_empty_for_oov(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    assert retriever._resolve_query_qids("zzzz xxxxx yyyyyy") == []


def test_query_resolves_case_insensitive(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    retriever.index()
    assert retriever._resolve_query_qids("PYTHON") == ["Q28865"]
    assert retriever._resolve_query_qids("apache http server") == ["Q11354"]


def test_longest_match_wins_when_labels_overlap():
    """When 'Apache' and 'Apache HTTP Server' both label distinct entities,
    a query mentioning the longer label resolves to the longer entity only.
    """
    entities = [
        {"wikidata_id": "Q1", "label": "Apache", "claims": []},
        {"wikidata_id": "Q2", "label": "Apache HTTP Server", "claims": []},
    ]
    chunks = [
        {"passage_id": "Q1:0000", "wikidata_id": "Q1", "title": "x", "text": "x"},
        {"passage_id": "Q2:0000", "wikidata_id": "Q2", "title": "x", "text": "x"},
    ]
    retriever = GraphRetriever(chunks, entities)
    retriever.index()

    assert retriever._resolve_query_qids("the Apache HTTP Server") == ["Q2"]
    assert retriever._resolve_query_qids("Apache alone") == ["Q1"]


# ---------------------------------------------------------------------------
# Query / boost tests
# ---------------------------------------------------------------------------


def test_query_returns_retrieval_hits_with_graph_source_mode(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=5)
    assert hits, "expected at least one hit for an in-vocabulary query"
    for h in hits:
        assert isinstance(h, RetrievalHit)
        assert h.source_mode == SOURCE_MODE == "graph_aware"
        assert isinstance(h.passage_id, str)
        assert isinstance(h.score, float)
        assert h.score > 0


def test_query_qid_chunk_gets_structural_boost(chunks, entities):
    """Querying for Python should rank the Python chunk first (structural wikidata_id match)."""
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=5)
    assert hits[0].passage_id == "Q28865:0000"
    assert hits[0].score == pytest.approx(DEFAULT_NEIGHBOR_BOOST)


def test_neighbor_qid_chunk_also_gets_structural_boost(chunks, entities):
    """Querying Python expands to its license neighbor; that license chunk gets the structural boost too."""
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=5)
    passage_ids = [h.passage_id for h in hits]
    assert "Q489772:0000" in passage_ids
    license_hit = next(h for h in hits if h.passage_id == "Q489772:0000")
    assert license_hit.score >= DEFAULT_NEIGHBOR_BOOST


def test_neighbor_label_mention_gets_mention_boost():
    """A chunk that mentions a neighbor's label but does not have its wikidata_id
    receives the mention boost only.
    """
    entities = [
        {
            "wikidata_id": "Q28865",
            "label": "Python",
            "claims": [{"property": "P275", "value": "Q489772"}],
        },
        {
            "wikidata_id": "Q489772",
            "label": "Python Software Foundation License",
            "claims": [],
        },
    ]
    chunks = [
        # Python article — structural wikidata_id match.
        {
            "passage_id": "Q28865:0000",
            "wikidata_id": "Q28865",
            "title": "Python",
            "text": "Python is a programming language.",
        },
        # Random chunk that *mentions* the neighbor label by string but
        # carries an unrelated wikidata_id.
        {
            "passage_id": "QX:0000",
            "wikidata_id": "QX",
            "title": "Random",
            "text": "It is licensed under the Python Software Foundation License.",
        },
    ]
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=5)

    by_id = {h.passage_id: h for h in hits}
    assert by_id["Q28865:0000"].score == pytest.approx(DEFAULT_NEIGHBOR_BOOST)
    assert by_id["QX:0000"].score == pytest.approx(DEFAULT_LABEL_MENTION_BOOST)
    # Structural boost wins over mention boost.
    assert hits[0].passage_id == "Q28865:0000"


def test_chunk_satisfying_both_stages_gets_sum():
    """A chunk whose wikidata_id matches AND whose text mentions another neighbor's
    label receives both boosts (sum).
    """
    entities = [
        {
            "wikidata_id": "Q28865",
            "label": "Python",
            "claims": [
                {"property": "P275", "value": "Q489772"},  # license neighbor
                {"property": "P178", "value": "Q123"},      # developer neighbor
            ],
        },
        {"wikidata_id": "Q489772", "label": "Python Software Foundation License", "claims": []},
        {"wikidata_id": "Q123", "label": "Guido van Rossum", "claims": []},
    ]
    chunks = [
        # Python license chunk — structural (its wikidata_id is a
        # neighbor) AND mentions another neighbor's label (Guido van
        # Rossum).
        {
            "passage_id": "Q489772:0000",
            "wikidata_id": "Q489772",
            "title": "Python Software Foundation License",
            "text": "Originally written for the Python interpreter by Guido van Rossum.",
        },
    ]
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=5)
    assert hits[0].passage_id == "Q489772:0000"
    assert hits[0].score == pytest.approx(
        DEFAULT_NEIGHBOR_BOOST + DEFAULT_LABEL_MENTION_BOOST
    )


def test_query_drops_zero_score_chunks(chunks, entities):
    """Querying for Python should not return the Linux kernel chunk."""
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=10)
    assert "Q193321:0000" not in [h.passage_id for h in hits]


def test_top_k_clamps(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=1)
    assert len(hits) == 1


def test_top_k_zero_returns_empty(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    assert retriever.query("Python", top_k=0) == []


def test_empty_corpus_returns_empty(entities):
    retriever = GraphRetriever([], entities)
    assert retriever.query("Python", top_k=5) == []


def test_empty_entities_returns_empty(chunks):
    retriever = GraphRetriever(chunks, [])
    assert retriever.query("Python", top_k=5) == []


def test_empty_query_returns_empty(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    assert retriever.query("", top_k=5) == []


def test_oov_only_query_returns_empty(chunks, entities):
    retriever = GraphRetriever(chunks, entities)
    assert retriever.query("zzzz xxxxx yyyyyy", top_k=5) == []


def test_default_top_k_is_ten():
    assert DEFAULT_TOP_K == 10


def test_default_boosts():
    assert DEFAULT_NEIGHBOR_BOOST == 1.0
    assert DEFAULT_LABEL_MENTION_BOOST == 0.5


def test_query_is_deterministic(chunks, entities):
    """Two retrievers built from the same inputs rank identically."""
    a = GraphRetriever(chunks, entities)
    b = GraphRetriever(chunks, entities)
    assert a.query("Python", top_k=5) == b.query("Python", top_k=5)


def test_tie_break_is_chunk_index_ascending():
    """When two chunks tie on score, the lower chunk index ranks higher."""
    entities = [
        {"wikidata_id": "Q1", "label": "Topic", "claims": []},
    ]
    chunks = [
        {"passage_id": "Q1:0000", "wikidata_id": "Q1", "title": "A", "text": "..."},
        {"passage_id": "Q1:0001", "wikidata_id": "Q1", "title": "A", "text": "..."},
    ]
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("topic", top_k=2)
    assert [h.passage_id for h in hits] == ["Q1:0000", "Q1:0001"]
    assert hits[0].score == hits[1].score


def test_custom_boosts_propagate(chunks, entities):
    """Constructor boost overrides take effect on returned hit scores."""
    retriever = GraphRetriever(
        chunks,
        entities,
        neighbor_boost=2.0,
        label_mention_boost=0.25,
    )
    hits = retriever.query("Python", top_k=5)
    by_id = {h.passage_id: h.score for h in hits}
    # Q28865 is a query QID → structural boost only
    assert by_id["Q28865:0000"] == pytest.approx(2.0)


def test_from_paths_round_trips_jsonl(tmp_path, chunks, entities):
    """Writing both files to disk and loading via from_paths produces identical ranking."""
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    entities_path = tmp_path / "entities.jsonl"
    with entities_path.open("w", encoding="utf-8") as f:
        for e in entities:
            f.write(json.dumps(e) + "\n")

    on_disk = GraphRetriever.from_paths(chunks_path, entities_path)
    in_memory = GraphRetriever(chunks, entities)
    assert on_disk.query("Python", top_k=5) == in_memory.query("Python", top_k=5)


# ---------------------------------------------------------------------------
# Output-shape contract — graph-aware mode in the four-mode union
# ---------------------------------------------------------------------------


def test_source_mode_is_graph_aware(chunks, entities):
    """The shared-shape contract names this mode 'graph_aware'."""
    retriever = GraphRetriever(chunks, entities)
    hits = retriever.query("Python", top_k=1)
    assert hits[0].source_mode == "graph_aware"
