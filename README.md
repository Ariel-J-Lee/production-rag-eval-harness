# production-rag-eval-harness

## Status

**Shell-only.** v1 release in progress; see `ROADMAP.md` for the eleven hard gates that must pass before a v1 tag lands. No retrieval implementation, eval results, or benchmark numbers exist yet.

This repository is the first commit of an engineering retrieval-evaluation harness whose v1 architectural commitments are described below. Every retrieval, eval, citation, and reproducibility statement on this page describes a v1 commitment, not a present-tense claim. The Reproduce path is not yet runnable.

## v1 Commitment (not yet demonstrated)

> A reproducible local harness that explicitly compares dense, sparse, hybrid, and graph-aware retrieval against a single public corpus, with citation-grounded answers, regression-gated evaluations, and an evidence trail a reviewer can rerun on a laptop.

The above is the v1 thesis the harness will satisfy at first release. It is not a current claim of working software.

## v1 Retrieval Modes (will demonstrate)

| # | Mode | What it will do (v1 commitment) | What it will demonstrate (v1 commitment) |
|---|---|---|---|
| 1 | Dense (vector) | v1 will run local embeddings + cosine ANN over chunked passages | v1 will demonstrate a semantic-match baseline |
| 2 | Sparse (BM25) | v1 will run term-weighted retrieval over the same chunked corpus | v1 will demonstrate an exact-term and rare-name retrieval baseline |
| 3 | Hybrid | v1 will fuse modes 1 and 2 by reciprocal rank | v1 will demonstrate the standard production-grade pattern |
| 4 | Graph-aware | v1 will resolve query entities to public-graph identifiers, expand 1-hop neighbors, and boost passages that mention any neighbor | v1 will demonstrate retrieval that uses structured relations a vector index cannot see (smallest defensible demonstration; no multi-hop, no graph database) |

Multi-hop graph traversal, graph-database backends (Neptune, Neo4j cluster), and managed-service stacks (Pinecone, OpenSearch) are out of scope for v1. See `docs/architecture.md` for the v1 demo-stack composition.

## Reproduce

**Reproduce path: not yet runnable.** Tracks at `ROADMAP.md` gate G1 (Tier-4 evidence reached).

The eventual v1 path will be: clone, install pinned dependencies, `make fetch` to materialize the public corpus, `make eval` to run the four-mode comparison and emit the captured evaluation report. The v1 commitment is a reviewer with no prior context can complete the path on a 16 GB laptop in under 30 minutes wall-clock.

## Evidence Tier

This repository sits at **Tier 1 — Static Trace** at shell creation. The v1 first release targets **Tier 4 — Real Command Path** per `docs/evidence-tier.md`. v1 explicitly does not claim Tier 5 (live runtime / observed against deployed system) or Tier 6 (customer-visible proof). The repo is a runnable harness when v1 lands; it is not, and will not be, a service.

Every retrieval-quality, eval-result, or benchmark claim is held until its captured-evidence path ships in `runs/<run-id>/`. The shell ships zero captured runs.

## License

This repository's code is licensed under **Apache License 2.0** (`LICENSE`).

The corpus that will land at v1 will carry its own license attached at `data/LICENSE.<corpus>`. If the corpus is or includes Wikipedia content, the CC-BY-SA-4.0 share-alike obligation will be isolated to the `data/` subdirectory; the rest of the repository remains under Apache 2.0.

## Cross-Repository Note

This repository is one of four canonical public repos in the broader portfolio. Out of scope here:

- agent runtime, traces, and tool-use observability — separate repository
- AWS / Bedrock / IaC reference — separate repository
- engineering-organization GenAI enablement field kit — separate repository

Cross-references are soft (named in narrative, not hard-imported).
