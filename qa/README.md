# Q-A Evaluation Set

This directory will hold the v1 evaluation Q-A set. The set is not yet committed; the v1 implementation packet authors `qa_pairs.jsonl` and `qa_construction_notes.md`.

## v1 Constraints

- approximately 50–100 Q-A pairs
- a documented mix that includes lookup questions, relational questions (require following at least one public-graph property), and multi-step questions (multiple passages required for a grounded answer)
- public provenance: hand-curated with sources cited, OR LLM-synthesized + human-validated; method documented
- public-safe: no Q-A pair reconstructed from a private example
- deterministic given the corpus snapshot and seed

## Forbidden

- Q-A pairs reconstructed from a private example
- Q-A pairs that reference customer/employer/private-system context
- Q-A pairs whose distribution was copied from a real private dataset

## Format Hint (non-binding for shell)

Suggested JSONL shape (the v1 implementation packet may extend):

```json
{"id": "<id>", "question": "<text>", "expected_passages": ["<passage_id>"], "expected_answer": "<text>"}
```

The implementation packet will pin the schema and validator at v1.
