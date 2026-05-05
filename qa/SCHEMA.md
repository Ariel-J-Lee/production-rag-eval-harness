# Q-A Pair Schema

`qa_pairs.jsonl` is a newline-delimited JSON file. Each line is one Q-A pair.

## Required Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier. Convention: `q<NN>-<type>-<topic-slug>`. Used by the eval harness to key per-question results. |
| `type` | enum | One of `lookup`, `relational`, `multi_step`. See "Question Types" below. |
| `graph_favored` | boolean | `true` when the question's answer is most reliably reached through 1-hop graph traversal of a Wikidata property (e.g., `P275` license, `P178` developer) rather than through article-text retrieval alone. The graph-aware retrieval mode is expected to outperform vector and BM25 alone on these questions. |
| `question` | string | The question, in plain English. |
| `expected_answer` | string | A concise canonical answer used for answer-correctness scoring. May include qualifiers (e.g., "ANSI C" rather than just "C") when the source canonically uses a qualified form. |
| `expected_passages` | array of strings | Logical public-source identifiers for passages that contain the answer. Format: `wikipedia:<article_slug>` or `wikidata:<topic_slug>` (the `wikidata:` form is a logical hint that the answer comes through a Wikidata property edge; the eval harness resolves both forms to corpus chunks at run time). |
| `public_source_citation` | array of objects | Public sources used to construct and verify the pair. See "Citation Object Shape" below. |
| `construction_notes` | string | One-sentence rationale: why this pair, what it tests, and (when applicable) why it is graph-favored. |

## Question Types

- `lookup` — Single entity, surface fact. Expected to be answerable from the article intro of a single Wikipedia page. All four retrieval modes should perform reasonably.
- `relational` — Requires reading a property of a single entity (e.g., the license under which Python is released, the primary developer of PostgreSQL). Some relational questions are graph-favored when the property value is a discrete entity reachable via a Wikidata edge.
- `multi_step` — Requires combining facts from two or more distinct articles or entities. Tests citation completeness across multiple supporting passages.

Distribution in this set: 7 lookup, 7 relational, 6 multi-step (20 pairs total). At least 2 are graph-favored (5 in this set: `q08`, `q09`, `q13`, `q17`, `q20`).

## Citation Object Shape

Each entry in `public_source_citation` is one of:

```json
{
  "kind": "wikipedia_article",
  "url": "https://en.wikipedia.org/wiki/<slug>",
  "retrieved": "YYYY-MM-DD"
}
```

```json
{
  "kind": "wikidata_entity",
  "qid": "Q<digits>",
  "url": "https://www.wikidata.org/wiki/Q<digits>",
  "retrieved": "YYYY-MM-DD",
  "property": "P<digits>"
}
```

The `property` field on `wikidata_entity` citations names the specific Wikidata property the answer comes through. This is the load-bearing field for graph-aware retrieval evaluation.

## Provenance Discipline

- All 20 pairs are hand-curated against public sources cited in the `public_source_citation` field.
- No LLM ideation tools were used to generate the question text or the expected answers.
- All facts are verified against the cited Wikipedia article or Wikidata entity at the recorded `retrieved` date.
- See `qa_construction_notes.md` for per-question rationale and the construction process.

## Reproducibility Note

When the corpus snapshot lands (via the corpus-fetch script in a sibling implementation lane), the eval harness will:

1. Resolve each `expected_passages[i]` of the form `wikipedia:<slug>` to the corresponding corpus chunk(s).
2. Resolve each `wikidata:<topic_slug>` hint to the corresponding 1-hop Wikidata neighborhood, if present in the corpus.
3. Compute Recall@5, Recall@10, MRR, citation precision/recall, faithfulness, and answer correctness against the resolved targets.

`expected_passages` are intentionally logical (article slugs and entity hints) rather than corpus chunk IDs, because the corpus snapshot may be fetched after this Q-A set lands. The eval harness performs the join at run time.
