# v1 Architecture (Commitment, Not Claim)

This document describes the v1 architectural commitments. Every paragraph speaks in declarative-of-architecture or future-tense voice. No present-perfect or past-tense claim is made about retrieval quality, eval results, or benchmarks at shell-creation time.

## v1 Architectural Commitments

The v1 release will ship a local-first retrieval evaluation harness with four retrieval modes compared on a single public corpus. The four modes are dense (vector), sparse (BM25), hybrid (reciprocal rank fusion of dense + sparse), and graph-aware (1-hop neighbor expansion over a public-graph snapshot serialized as JSON).

The graph-aware mode commits to the smallest defensible demonstration: query entity recognition, public-graph identifier resolution, 1-hop neighbor fetch, and a passage-retrieval boost when a retrieved passage mentions any 1-hop neighbor of a query entity. Multi-hop traversal, graph-database backends, and managed-service stacks are explicit non-goals for v1.

## v1 Demo Stack Composition

The v1 demo stack is locked to deviate from any private composition. Each layer is a deliberate choice; alternatives that would mirror a private stack are out of v1 scope.

| Layer | v1 commitment | Out of v1 scope |
|---|---|---|
| Orchestration | Lightweight Python with per-lab `make` targets and a small CLI entry point | Managed agent service |
| Embeddings / vector | Local provider abstraction (e.g., a small public sentence-transformer + an in-process vector index) | Pinecone |
| Sparse / keyword | In-process BM25 (e.g., `rank_bm25`) | OpenSearch managed service |
| Graph | Serialized JSON snapshot of the public-graph slice + NetworkX in-process traversal | Neptune |
| Tabular metadata | SQLite | Managed Postgres |
| LLM | Provider-abstracted client; local-first execution; hosted-API integration is opt-in only | Bedrock-only path |

This composition is the **deliberate-deviation set** for v1. Any drift toward the out-of-scope column is rejected at PR review.

Cloud-side composition (managed services, IaC, deploy patterns) is the subject of a separate canonical repository in the broader portfolio. This repository defers all cloud-side content there.

## v1 Component Sketch

```
                                 +--------------------+
                                 | public corpus      |
                                 | (DATA-SOURCE.md)   |
                                 +---------+----------+
                                           |
                              fetch + chunk |
                                           v
                  +---------+   +-------------+   +----------+
                  | dense   |   | sparse      |   | graph    |
                  | index   |   | BM25 index  |   | (JSON)   |
                  +----+----+   +------+------+   +----+-----+
                       \              |                /
                        \             |               /
                         v            v              v
                          +-----------------------------+
                          | retrieval modes:            |
                          |   dense | sparse | hybrid   |
                          |       | graph-aware         |
                          +---------------+-------------+
                                          |
                                          v
                          +-----------------------------+
                          | answer + citation generator |
                          | (provider-abstracted LLM)   |
                          +---------------+-------------+
                                          |
                                          v
                          +-----------------------------+
                          | eval harness:               |
                          | recall@k, MRR, citation     |
                          | precision/recall, faithful, |
                          | answer correctness          |
                          +---------------+-------------+
                                          |
                                          v
                          +-----------------------------+
                          | runs/<id>/eval_report.md    |
                          | runs/<id>/raw_results.jsonl |
                          | runs/<id>/manifest.json     |
                          +-----------------------------+
```

This component sketch describes the v1 commitment. None of the components are implemented at shell creation time.

## What This Document Does Not Specify

Implementation details that the v1 implementation packet will pick:

- specific embedding model
- specific vector-index library version
- specific NER tool / model
- specific LLM client library
- specific public corpus selection (the open-source software ecosystem slice is the primary direction; PubMed and arXiv are documented fallbacks)
- specific Q-A construction method
- specific regression-gate tolerance number

These open decisions are tracked in `ROADMAP.md`.
