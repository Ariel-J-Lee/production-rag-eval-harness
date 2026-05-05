# v1 Retrieval Modes (Commitment, Not Claim)

This document describes the v1 retrieval-mode commitments and the comparison shape the v1 evaluation will produce. None of the modes are implemented at shell-creation time. Every cell in the tables below is a v1 commitment.

## Four-Mode Comparison

| # | Mode | What v1 will do | What v1 will demonstrate |
|---|---|---|---|
| 1 | Dense (vector) | v1 will run local embeddings + cosine ANN over chunked passages | semantic-match baseline |
| 2 | Sparse (BM25) | v1 will run term-weighted retrieval over the same chunked corpus | exact-term and rare-name retrieval baseline |
| 3 | Hybrid | v1 will fuse modes 1 and 2 by reciprocal rank | the standard production-grade RAG retrieval pattern |
| 4 | Graph-aware | v1 will resolve query entities to public-graph identifiers, expand 1-hop neighbors, and boost passages that mention any 1-hop neighbor | retrieval that uses structured relations a vector index alone cannot see |

## Smallest Defensible Graph-Aware Demonstration

The graph-aware mode commits to all of the following at v1, but no more:

- read the public-graph snapshot from a serialized JSON or JSONL file (no graph-database dependency)
- run NER on the user query and resolve detected entities to public-graph identifiers
- fetch direct (1-hop) neighbors of those identifiers
- boost passages from the dense + sparse pool that mention any 1-hop neighbor
- emit the same `{passage_id, score, source_mode}` shape as the other modes for evaluation comparability

Multi-hop expansion, Cypher-style queries, and a Neo4j / Memgraph backend are explicit v1 non-goals.

## v1 Comparison Output

v1 will ship a single evaluation table with rows = retrieval mode (1–4) and columns = metrics. The metrics v1 will report are:

| Metric | What it measures |
|---|---|
| Recall@5 | Retrieval-set quality at top 5 |
| Recall@10 | Retrieval-set quality at top 10 |
| MRR | Mean reciprocal rank — retrieval ordering quality |
| Citation precision | Of cited passages, fraction that actually support the answer |
| Citation recall | Of supporting passages in gold, fraction cited by the system |
| Faithfulness | Whether the answer stays within the cited evidence (binary or scalar; method documented at v1) |
| Answer correctness | Match against Q-A gold (exact-match plus a graded judge; method documented at v1) |

The eval table is the headline artifact. v1 will commit it at `runs/<run-id>/eval_report.md`; subsequent runs will commit additional `runs/<run-id>/` directories.

## v1 Q-A Set Constraints

The v1 Q-A evaluation set will satisfy:

- approximately 50–100 Q-A pairs across the chosen corpus slice
- a documented mix that includes lookup questions (single entity, surface fact), relational questions (require following at least one public-graph property), and multi-step questions (multiple passages required for a grounded answer)
- public provenance: hand-curated by the implementer with sources cited, OR LLM-synthesized and human-validated; either way, the construction process is documented
- public-safe: no Q-A pair reconstructed from a private example
- pairs committed to the repository at v1; running the eval is deterministic given the corpus snapshot and Q-A set

## What This Document Does Not Specify

- the specific public-graph used for graph-aware mode (the open-source software ecosystem slice is the primary direction)
- the specific NER tool
- the specific embedding model
- the regression-gate tolerance number
- the Q-A construction method (hand-curated vs LLM-synthesized + validated)

These remain implementation-packet decisions and are tracked in `ROADMAP.md`.
