# Retrieval Comparison Run 2026-05-05_6d8256d1fe5c_seed-0

## Setup

- Corpus: open-source software ecosystem slice (Wikidata + Wikipedia), 1659 chunks; chunks.jsonl SHA-256 prefix `6d8256d1fe5c`
- Q-A pairs: 20; qa_pairs.jsonl SHA-256 prefix `5568e875569d`
- Embedding model: all-MiniLM-L6-v2
- LLM: none (extractive answers; no hosted-API call on the run path)
- Seed: 0
- Timestamp: 2026-05-05T23:47:21Z
- Citation count: 3 (top retrieved passages cited per question)

## Headline Comparison

| Mode | Recall@5 | Recall@10 | MRR | Citation precision | Citation recall | Faithfulness | Answer correctness |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (vector) | 0.160 | 0.273 | 0.950 | 0.883 | 0.106 | 1.000 | 0.027 |
| Sparse (BM25) | 0.118 | 0.203 | 0.817 | 0.617 | 0.072 | 1.000 | 0.027 |
| Hybrid (RRF) | 0.156 | 0.283 | 0.950 | 0.850 | 0.101 | 1.000 | 0.026 |
| Graph-aware | 0.073 | 0.108 | 0.500 | 0.433 | 0.046 | 0.500 | 0.011 |

## Method Notes

- **Faithfulness** is a binary first-proof heuristic: an answer is faithful when at least 60% of its tokens appear in the union of cited-passage tokens (after a small stopword strip and lowercase normalization). The graded LLM-judge upgrade is a v1+ follow-on tracked in `ROADMAP.md`.
- **Answer correctness** is SQuAD-style token-F1 against the Q-A pair's `expected_answer`. Exact-match is reported in `raw_results.jsonl` (per-question `metrics.exact_match` field) as a stricter diagnostic; the headline column is token-F1. The graded LLM-judge upgrade is a v1+ follow-on.
- **Answers** are extractive: each (question, mode) answer is the joined text of the top-3 retrieved passages. This makes the retrieval signal (Recall@k, MRR) the load-bearing measurement and removes LLM-generation variance from the first scored proof. Generated-answer evaluation is a v1+ follow-on.
- **Citation contract**: every record in `raw_results.jsonl` carries a `citations` array of passage_ids. Empty citations are allowed only when the underlying retriever returned no hits (an honest no-match for the query in that mode); empty citations with non-empty retrieval is a fail-fast condition.
- **Answer reconstruction**: `raw_results.jsonl` records do not store the extractive answer text directly, because the answer consists of joined Wikipedia passages and the share-alike isolation rule (PACKET-019 §2.5) keeps Wikipedia passage text inside the gitignored corpus subdirectory. The answer is reconstructable from `data/<corpus>/chunks.jsonl` + the record's `citations` field. The answer-derived metrics (`faithfulness`, `answer_correctness`, `exact_match`) are computed at run time and stored in the record's `metrics` field, so reading the report does not require reconstructing answers.
- **Resolution diagnostics**: 3 questions had at least one `expected_passages` hint that did not resolve to a corpus chunk (total unresolved hints: 3). Per PACKET-037 coordination note 1, unresolved hints are a corpus-coverage signal rather than a Q-A defect; the affected questions still contribute to retrieval and answer-quality metrics computed against the resolved subset of expected passages.

## Per-mode rationale

Per the canonical first-proof report shape, the next slice (generated-answer evaluation, regression gate) will populate this section with one example query per mode where that mode's ranking choice is most distinctive. At first proof the headline table itself is the load-bearing artifact; per-query examples live in `raw_results.jsonl`.

## Limits

- Q-A set is 20 pairs; statistical confidence on retrieval-quality differences is bounded at this sample size.
- Corpus slice is 1659 chunks; results do not generalize beyond this slice.
- Faithfulness and answer correctness are first-proof heuristic implementations; LLM-judge variants are deferred to a v1+ follow-on.
- This is a **first proof**, not a benchmark. The headline comparison shows which retrieval mode wins on which metric category at this corpus and Q-A scale; reading it as a ranking of retrieval techniques across all corpora is out of scope.
