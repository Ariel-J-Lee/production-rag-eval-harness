# production-rag-eval-harness

A reproducible local harness that compares dense, sparse, hybrid, and graph-aware retrieval over a public corpus, with citation-grounded answers — designed to run on a laptop and audit, not a hosted demo.

## Status

Reproducible local harness for retrieval evaluation. The first scored run is committed under [`runs/2026-05-05_6d8256d1fe5c_seed-0/`](./runs/2026-05-05_6d8256d1fe5c_seed-0/). The four retrieval modes below run end-to-end against a public corpus and a hand-curated Q-A set; the Reproduce section is a working command path.

## Retrieval modes

Four retrieval modes compared on a single public corpus:

| # | Mode | What it does |
|---|---|---|
| 1 | Dense | local sentence-transformer embeddings (all-MiniLM-L6-v2) + in-process cosine over chunked passages |
| 2 | Sparse | in-process term-weighted (BM25) retrieval over the same chunks |
| 3 | Hybrid | reciprocal rank fusion of dense and sparse |
| 4 | Graph-aware | resolves query entities to Wikidata Q-IDs, expands one-hop neighbors, and boosts passages that mention a neighbor |

All four indices run in-process; there is no hosted search service and no graph-database dependency. The graph-aware mode reads from a serialized Wikidata neighborhood snapshot, not a graph database. Multi-hop graph traversal, graph-database backends (Neptune, Neo4j cluster), and managed-service stacks (Pinecone, OpenSearch) are out of scope. See [`docs/architecture.md`](./docs/architecture.md) for the demo-stack composition.

## First scored run

The committed first run (`2026-05-05_6d8256d1fe5c_seed-0`) compares the four retrieval modes on a 1,659-chunk public corpus (an open-source software ecosystem slice from Wikipedia and Wikidata) and a 20-pair hand-curated Q-A set. Headline outcomes:

- Dense and Hybrid lead on top-1 ranking (MRR 0.95 each).
- Hybrid edges Dense on Recall@10 (0.28 vs 0.27).
- Citation precision is highest for Dense (0.88) and lowest for Graph-aware (0.43).
- Graph-aware underperforms on this Q-A set; see [Limits](#limits) and the [run report](./runs/2026-05-05_6d8256d1fe5c_seed-0/eval_report.md) for the per-metric table and method notes.

The full four-row × seven-metric comparison and the citation contract live in [`runs/2026-05-05_6d8256d1fe5c_seed-0/eval_report.md`](./runs/2026-05-05_6d8256d1fe5c_seed-0/eval_report.md). Per-question results are in `raw_results.jsonl`; the deterministic-run manifest is in `manifest.json`.

## Reproduce

1. `git clone https://github.com/Ariel-J-Lee/production-rag-eval-harness && cd production-rag-eval-harness`
2. `pip install -r requirements.txt`
3. `make fetch` — fetch the public corpus snapshot (Wikipedia + Wikidata OSS slice).
4. `make eval` — run the four-mode comparison and write `runs/<run-id>/{eval_report.md, raw_results.jsonl, manifest.json}`.

Designed to run on a laptop. The committed run has its corpus and Q-A snapshot SHA-256 hashes pinned in `manifest.json`; a re-fetch against the same Wikipedia and Wikidata state reproduces the same run id and the same numbers.

## Limits

This is a first scored proof, not a benchmark.

- **Corpus**: 1,659 chunks from a hand-seeded open-source software ecosystem slice. Results do not generalize beyond this slice.
- **Q-A set**: 20 hand-curated pairs (lookup, relational, multi-step). Statistical confidence on cross-mode differences is bounded at this sample size.
- **Faithfulness and answer correctness** are first-proof heuristic implementations: faithfulness is a binary token-overlap threshold against cited-passage tokens; answer correctness is SQuAD-style token-F1 against the gold answer. Graded LLM-judge variants are deferred.
- **Answers are extractive**: every answer is the joined text of the top-3 retrieved passages, so the retrieval signal (Recall@k, MRR, citation metrics) is the load-bearing measurement. Generated-answer evaluation is deferred.
- **Graph-aware mode underperforms** on this Q-A set because the entity-recognition step misses several queries; recovery is a follow-on tuning concern, not a contract issue. See the run report's method notes for the detail.
- **Reproducibility** depends on the Wikipedia and Wikidata public-API state at fetch time; the pinned snapshot hashes in `manifest.json` define the reproducible boundary.
- No production deployment claim. This is a local harness, not a hosted service.

## License

Repository code is licensed under [Apache License 2.0](./LICENSE).

The corpus, when fetched, carries its own license at `data/LICENSE.<corpus>`. Wikipedia content is CC-BY-SA-4.0; the share-alike obligation is isolated to the `data/` subdirectory.

## Adjacent repositories

- [`agent-runtime-observability`](https://github.com/Ariel-J-Lee/agent-runtime-observability) — governed agent runtime with traces, retries, and policy gates.
- [`aws-bedrock-iac-reference`](https://github.com/Ariel-J-Lee/aws-bedrock-iac-reference) — Bedrock-anchored AWS reference architecture as IaC.

Cross-references are descriptive only; this repository does not import or deploy them.
