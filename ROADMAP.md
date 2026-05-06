# ROADMAP

> **Status note — 2026-05-06.** This roadmap predates the first committed scored run. The repository now has a first proof under `runs/2026-05-05_6d8256d1fe5c_seed-0/` with dense, sparse, hybrid, and graph-aware retrieval rows. G1, G2, G3, G7, and G11 are partially satisfied by that first proof. G4, G5, and G9 are partially supported by the artifacts that landed alongside it. G6, G8, and G10 remain pending. v1 still ships when every G1–G11 gate is fully satisfied and a v1 release is approved.

This file enumerates the hard gates between the committed first scored run and v1 ship-readiness.

## v1 Hard Gates

| Gate | Description | Status | Verification |
|---|---|---|---|
| G1 | Tier-4 evidence reached: `make eval` runs end-to-end from a clean checkout. `runs/<id>/eval_report.md`, `raw_results.jsonl`, `manifest.json` are committed. | partial | First scored run committed at `runs/2026-05-05_6d8256d1fe5c_seed-0/`. Clean-checkout reproduction depends on Wikipedia / Wikidata API state at fetch time; an unconditional Tier-4 reproduction guarantee is not yet in place. |
| G2 | All four retrieval modes implemented (dense, sparse, hybrid, graph-aware) with numerically distinct comparison rows. | partial | Four retrieval modes are implemented and produce four numerically distinct rows in the committed `eval_report.md`. Sample size (20 Q-A pairs) bounds the claim. |
| G3 | Graph-aware demonstration meets the "smallest defensible" bar: NER → public-graph identifier resolution → 1-hop neighbor expansion → passage boost. No graph database dependency. | partial | Graph-aware mode is implemented per the named pipeline and runs against a serialized Wikidata neighborhood snapshot with no graph-database dependency. The first run shows graph-aware underperforming on the 20-pair Q-A set; recovery is a follow-on tuning concern. |
| G4 | Public-safe corpus. `data/DATA-SOURCE.md` complete with source class, license, generation provenance, PII check, customer-derivation check, reconstruction check. | partial | `data/DATA-SOURCE.md` exists with source class, license, and generation provenance for the OSS-ecosystem slice. PII / customer-derivation / reconstruction-check fields are documented; full third-party attestation is not in place. |
| G5 | Reproducibility on a 16 GB laptop in under 30 minutes wall-clock from a clean checkout, after `make fetch`. | partial | The Reproduce section is runnable end-to-end. Wall-clock laptop runtime has not been independently measured; the 30-minute budget is a design target, not a verified number. |
| G6 | Hard-banned regex set returns zero hits across the entire repo tree. | pending | Per-PR grep enforcement exists; a standing tree-wide check is not yet a CI gate. |
| G7 | Citation grounding: every answer in `qa/qa_pairs.jsonl` carries cited passage IDs; uncited answers fail eval, regardless of textual correctness. | partial | Citation contract is implemented and enforced in `src/eval/`. The first run carries the contract end-to-end; broader-corpus generalization is a follow-on. |
| G8 | Regression gate: `make regression` re-runs eval and fails on a stated tolerance drop in {recall@10, MRR, citation precision, faithfulness} against the committed baseline manifest. | pending | `make regression` is not implemented. The first run's `manifest.json` records `regression_gate.verdict = not_evaluated`. The regression gate is on the roadmap. |
| G9 | License posture intact: repo code Apache 2.0; corpus license isolated to `data/` if share-alike. | partial | Repo code is Apache 2.0; CC-BY-SA-4.0 share-alike isolation pattern is in place at `data/LICENSE.<corpus>`. Formal license-policy review is a follow-on. |
| G10 | No private-stack composition (no Pinecone + OpenSearch + Neptune + Bedrock combination matching any private architecture). | pending | The deliberate-deviation composition is in place in code; an architecture-doc review formal sign-off is not yet recorded. |
| G11 | All claims in README and `docs/` reach their honest evidence tier; no Tier-5 / Tier-6 outcome claim. | partial | The repository README and `runs/README.md` were aligned to present-tense first-proof framing. `docs/architecture.md` has not yet been audited against the same standard. |

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
