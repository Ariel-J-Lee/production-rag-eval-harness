# Q-A Construction Notes

This file documents the construction process for the 20-pair Q-A evaluation set in `qa_pairs.jsonl`. The schema is documented in `SCHEMA.md`.

## Construction Discipline

- All 20 pairs are hand-curated. No LLM ideation tools were used.
- Every pair is verified against a public source: a Wikipedia article (CC-BY-SA-4.0) or a Wikidata entity (CC0). Citations are recorded inline on each pair in `qa_pairs.jsonl`.
- All cited public sources were retrieved on 2026-05-05.
- Facts that change frequently (e.g., recent license changes, version numbers) were avoided in favor of stable historical facts (creators, original release years, primary implementation languages).
- Where a project has had license changes (e.g., Redis), the question was framed to ask for an unambiguous fact (primary implementation language) rather than the contested one.

## Topical Scope

Pairs draw from well-known open-source software projects, languages, tools, and editor ecosystems:

- Programming languages: Python, Ruby
- Operating systems and kernels: Linux, Linux kernel, FreeBSD
- Databases: PostgreSQL, MySQL, Redis
- Web servers: Apache HTTP Server, Nginx
- Version control and hosting: Git, GitHub
- Editors: Vim, Emacs
- Tools and runtimes: Docker, GNU Compiler Collection (GCC)

This topic distribution is intended to overlap with any reasonable open-source-software seed list a parallel corpus-construction lane would pick.

## Pair Distribution

| Type | Count | Pair IDs |
|---|---:|---|
| Lookup | 7 | q01, q02, q03, q04, q05, q06, q07 |
| Relational | 7 | q08, q09, q10, q11, q12, q13, q14 |
| Multi-step | 6 | q15, q16, q17, q18, q19, q20 |
| **Total** | **20** | |

Of the 20 pairs, **5 are graph-favored**: `q08`, `q09`, `q13`, `q17`, `q20`. This exceeds the minimum of 2 graph-favored questions required by the first-runnable-proof plan.

## Graph-Favored Pair Rationale

Graph-favored questions are constructed so that 1-hop Wikidata neighbor expansion is expected to outperform vector retrieval alone. The pattern across the five graph-favored pairs:

- `q08` — Python license. Wikidata property `P275` on entity Q28865 (Python) yields the license entity directly. Wikipedia article body mentions open-source licensing in general terms; the specific license name (Python Software Foundation License) is most reliably reached through the Wikidata edge.
- `q09` — PostgreSQL developer. Wikidata property `P178` on entity Q192490 (PostgreSQL) yields PostgreSQL Global Development Group. The article does mention the group, but the developer-as-property edge is exactly what graph-aware retrieval should boost.
- `q13` — Apache HTTP Server license. Same `P275` pattern as q08, applied to a different entity (Q11354). Tests consistency of graph-aware mode across multiple license-property questions.
- `q17` — Apache HTTP Server and Nginx licenses. Multi-step graph-favored: requires `P275` on two distinct entities. Tests graph-aware mode at multi-entity scale.
- `q20` — Python and Ruby developers. Multi-step graph-favored: requires `P178` on two language entities, with the secondary case (Ruby) testing graph-aware mode's behavior on entities with looser developer-property structure (Ruby's maintenance is community-driven rather than singular-developer).

The five graph-favored pairs are designed so that an honest four-mode comparison run should show graph-aware outperforming vector and BM25 alone on these questions, particularly on citation precision (since the cited passage should be the entity-property neighbor rather than a generic article passage).

## Type-By-Type Construction Process

### Lookup pairs (q01–q07)

Each lookup pair asks a surface fact answered in the intro of a single Wikipedia article. Construction process:

1. Pick a well-known OSS project from the topical scope above.
2. Identify a single fact stated in the article's first paragraph or first infobox row (creator, primary language, release year, file extension).
3. Frame the question minimally; the answer is a single token or short phrase.
4. Cite the Wikipedia article URL and retrieval date.

Lookup pairs serve as a baseline: dense and sparse retrieval should both perform reasonably here. Failure on lookup pairs indicates a retrieval pipeline problem rather than a question-shape problem.

### Relational pairs (q08–q14)

Relational pairs ask for a specific property of a single entity. Some are graph-favored (when the property answer is most reliably a Wikidata edge), some are not (when the article body covers the property prominently).

- Graph-favored relational pairs: `q08` (Python license), `q09` (PostgreSQL developer), `q13` (Apache HTTP Server license).
- Not graph-favored: `q10` (Docker programming language; article body mentions Go), `q11` (FreeBSD family; article intro names BSD lineage), `q12` (Git creator; article intro), `q14` (Vim developer; article intro).

The split tests whether graph-aware retrieval correctly distinguishes "needs the graph edge" from "article-text already sufficient" — the graph-aware mode should not over-boost on questions where vector or sparse retrieval is already strong.

### Multi-step pairs (q15–q20)

Multi-step pairs require combining facts from two or more articles or entities. Two of the six are graph-favored (`q17`, `q20`) and four are not.

Construction process:

1. Pick two related entities (e.g., Git and GitHub; PostgreSQL and MySQL; Vim and Emacs).
2. Frame a question that requires a fact from each.
3. Compose the expected answer to capture both facts plus the relationship.
4. Cite both articles (and any Wikidata entities for graph-favored cases).

Multi-step pairs stress-test citation completeness: a system that retrieves only one of the two articles should fail on citation recall even if the answer text is partially correct.

## Graph-Favored Coverage Summary

The five graph-favored pairs cover two Wikidata properties:

- `P275` (license): q08, q13, q17 (q17 is multi-step requiring P275 on two entities)
- `P178` (developer): q09, q20 (q20 is multi-step)

The harness's first scored run is expected to show graph-aware retrieval winning on at least one of these five questions where vector retrieval alone misses the specific property neighbor. The "smallest defensible" bar from the first-runnable-proof plan is satisfied by any one such win; the five-question margin gives the implementation packets enough headroom for this bar to be reached even if some questions resolve through article text alone.

## Public-Source Citation Strategy

Every pair carries:

- At least one Wikipedia article URL (the canonical source for the article-text passage).
- For graph-favored pairs: at least one Wikidata entity citation with a `qid`, `url`, and the specific `property` (`P275`, `P178`) that the question turns on.

The Wikidata entity QIDs cited are the well-known top-level entity identifiers for the named projects (Python, PostgreSQL, Apache HTTP Server). For Wikidata entities representing the property values themselves (e.g., the specific license entity), no QID is committed in this Q-A set; the corpus-fetch lane will resolve property neighbors to their Wikidata QIDs at fetch time.

Retrieval date is recorded as `2026-05-05` for every citation.

## What This Q-A Set Does Not Do

- It does not implement retrieval. Dense, sparse, hybrid, and graph-aware retrievers ship in sibling implementation lanes.
- It does not run the eval. The eval harness is a separate sibling lane.
- It does not commit corpus blobs. The corpus-fetch script and snapshot land in their own lane.
- It does not specify the corpus chunk shape. `expected_passages` are logical (article slugs and entity hints); the eval harness resolves these to chunks at run time.
- It does not include questions whose ground-truth answer depends on facts that have changed during 2024–2026 (recent license changes, recent fork events, recent CEO changes). Stable historical facts only.

## Limits

- 20 pairs is a smallness-bound first proof. Statistical confidence on retrieval-quality differences across the four modes is bounded; the run is a first proof, not a benchmark.
- All pairs are English-language, drawn from English Wikipedia and Wikidata's primary English labels. No non-English coverage.
- The OSS topical scope is deliberate; the set does not generalize beyond well-known open-source software projects.

## Adjacent Lanes

This Q-A set is one piece of a larger first-runnable-proof effort. The corpus-fetch lane (running in parallel) builds the snapshot the eval will run against. The four retrieval-mode implementations and the eval harness ship in their own subsequent lanes. The captured first scored run will land at `runs/<run-id>/{eval_report.md, raw_results.jsonl, manifest.json}` per the schema named in those lanes.
