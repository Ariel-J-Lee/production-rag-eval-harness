# ROADMAP

This file enumerates the hard gates between today (shell-only) and v1 ship-readiness. v1 ships when every G1–G11 gate is satisfied and PM/QA records merge approval for the v1 release packet.

## v1 Hard Gates

| Gate | Description | Status | Verification |
|---|---|---|---|
| G1 | Tier-4 evidence reached: `make eval` runs end-to-end from a clean checkout. `runs/<id>/eval_report.md`, `raw_results.jsonl`, `manifest.json` are committed. | pending | runnable command + committed run artifacts |
| G2 | All four retrieval modes implemented (dense, sparse, hybrid, graph-aware) with numerically distinct comparison rows. | pending | eval-report comparison table |
| G3 | Graph-aware demonstration meets the "smallest defensible" bar: NER → public-graph identifier resolution → 1-hop neighbor expansion → passage boost. No graph database dependency. | pending | inspectable in `src/retrieval/graph.py` + `runs/<id>/eval_report.md` |
| G4 | Public-safe corpus. `data/DATA-SOURCE.md` complete with source class, license, generation provenance, PII check, customer-derivation check, reconstruction check. | pending | attestation file with all required fields |
| G5 | Reproducibility on a 16 GB laptop in under 30 minutes wall-clock from a clean checkout, after `make fetch`. | pending | README Reproduce section, runnable |
| G6 | Hard-banned regex set returns zero hits across the entire repo tree. | pending | grep run; results recorded in PR review |
| G7 | Citation grounding: every answer in `qa/qa_pairs.jsonl` carries cited passage IDs; uncited answers fail eval, regardless of textual correctness. | pending | inspectable in `src/eval/` + run artifacts |
| G8 | Regression gate: `make regression` re-runs eval and fails on a stated tolerance drop in {recall@10, MRR, citation precision, faithfulness} against the committed baseline manifest. | pending | runnable command + commit baseline |
| G9 | License posture intact: repo code Apache 2.0; corpus license isolated to `data/` if share-alike. | pending | `LICENSE` + `data/LICENSE.<corpus>` |
| G10 | No private-stack composition (no Pinecone + OpenSearch + Neptune + Bedrock combination matching any private architecture). | pending | architecture-doc review + grep |
| G11 | All claims in README and `docs/` reach their honest evidence tier; no Tier-5/6 outcome claim. | pending | PR review against the publication-gate sidecar |

## Open Decisions (resolved before or during the implementation packet)

| ID | Topic | Default direction | Decision route |
|---|---|---|---|
| D1 | v1 corpus selection | open-source software ecosystem slice (Wikidata + Wikipedia, with CC-BY-SA isolated to `data/`) | implementation packet |
| D2 | Embedding model | local provider (e.g., a small public sentence-transformer) | implementation packet |
| D3 | Vector index library | in-process FAISS or Chroma local | implementation packet |
| D4 | NER tool for graph-aware mode | small public NER library or model | implementation packet |
| D5 | LLM for answer/citation generation | local-first; hosted-API integration is opt-in only | implementation packet |
| D6 | Regression-gate tolerance | 10% relative drop on any of the four headline metrics | implementation packet may tune with stated rationale |
| D7 | Citation payload format | structured `citations` array on every answer; schema defined in `src/eval/` | implementation packet |
| D8 | Hosted-API opt-in mechanism | env var or CLI flag (not config-file default) | implementation packet |
| D9 | Q-A construction method | hand-curated with public provenance, OR LLM-synthesized + human-validated; either acceptable | implementation packet |
| D10 | First-class CI surface | smoke + regression + license + link-rot + composition-check at v1 | implementation packet |

## Explicit Non-Goals For v1

- production deployment (Tier 5 or 6 claims)
- multi-tenant isolation
- streaming / latency / throughput benchmarks
- cloud or managed-service deployment
- agent runtime concerns (tool use, retries, agent traces)
- workshop / enablement content
- full graph backend (no Neo4j cluster, no Apache AGE, no Neptune)
- multi-hop graph queries beyond 1-hop neighbor expansion
- custom embedding models or fine-tuning
- hosted-API as the default execution path
- internationalization or non-English corpora

## v2 Commitments

None. v2 scope is out of v1 contract. The roadmap names the eleven v1 gates and stops here.
