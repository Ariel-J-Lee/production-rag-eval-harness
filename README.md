# production-rag-eval-harness

A reproducible local harness that compares dense, sparse, hybrid, and graph-aware retrieval over a public corpus, with citation-grounded answers and regression-gated evaluations — designed to run on a laptop and audit, not a hosted demo.

## Status

This repository is currently a scaffold. The retrieval implementation, evaluation runs, and benchmark numbers land in a subsequent release; see [`ROADMAP.md`](./ROADMAP.md). The Reproduce section below describes the path the future release will satisfy, not a runnable command today.

## What the next release will demonstrate

Four retrieval modes compared head-to-head on a single public corpus:

| # | Mode | What it does |
|---|---|---|
| 1 | Dense | local embeddings + cosine ANN over chunked passages |
| 2 | Sparse | term-weighted (BM25) retrieval over the same corpus |
| 3 | Hybrid | reciprocal rank fusion of dense and sparse |
| 4 | Graph-aware | resolves query entities to public-graph identifiers, expands one-hop neighbors, and boosts passages mentioning a neighbor |

Multi-hop graph traversal, graph-database backends (Neptune, Neo4j cluster), and managed-service stacks (Pinecone, OpenSearch) are out of scope. See [`docs/architecture.md`](./docs/architecture.md) for the demo-stack composition.

## Reproduce

The eventual path: clone, install pinned dependencies, `make fetch` to materialize the public corpus, `make eval` to run the four-mode comparison and emit the captured evaluation report. The target is a reviewer with no prior context completing the path on a 16 GB laptop in under thirty minutes.

This path is not yet runnable.

## License

Repository code is licensed under [Apache License 2.0](./LICENSE).

The corpus, when added, will carry its own license at `data/LICENSE.<corpus>`. If the corpus includes Wikipedia content, the CC-BY-SA-4.0 share-alike obligation will be isolated to the `data/` subdirectory.

## Adjacent repositories

- [`agent-runtime-observability`](https://github.com/Ariel-J-Lee/agent-runtime-observability) — governed agent runtime with traces, retries, and policy gates.
- [`aws-bedrock-iac-reference`](https://github.com/Ariel-J-Lee/aws-bedrock-iac-reference) — Bedrock-anchored AWS reference architecture as IaC.

Cross-references are descriptive only; this repository does not import or deploy them.
