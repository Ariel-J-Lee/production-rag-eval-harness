# Captured Eval Runs

This directory will hold captured outputs from the v1 evaluation. Each run will live at `runs/<run-id>/` and contain the artifacts the eval emitted. The shell ships zero runs; the v1 release ships at least one canonical baseline run when the implementation lands.

## Required Files Per Run (v1 commitment)

- `runs/<run-id>/eval_report.md` — comparison table (4 rows × 7 metric columns) plus method notes
- `runs/<run-id>/raw_results.jsonl` — per-question results across all four retrieval modes
- `runs/<run-id>/manifest.json` — corpus snapshot id, Q-A set hash, model versions, seed, run timestamp, regression-gate verdict

## Anti-Claim Rule

Until at least one canonical run is committed under this directory, the repository's headline retrieval-quality and citation-quality claims are held at Tier 1 Static Trace. Any claim made elsewhere in the repository without a captured run here is not yet substantiated.

The shell explicitly ships no runs and explicitly makes no run-based claim.

## Top-Level Convention

When v1 ships, `eval_report.md` at the repo root (or a symlink) will point at the latest canonical run for casual readers. The shell does not ship the symlink because no run exists.
