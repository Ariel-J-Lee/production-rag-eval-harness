# Captured Eval Runs

This directory holds captured outputs from the eval harness. Each run lives at `runs/<run-id>/` and contains the artifacts the eval emitted. The first scored run is committed at [`2026-05-05_6d8256d1fe5c_seed-0/`](./2026-05-05_6d8256d1fe5c_seed-0/).

## Required files per run

- `runs/<run-id>/eval_report.md` — comparison table (4 rows × 7 metric columns) plus method notes
- `runs/<run-id>/raw_results.jsonl` — per-question results across all four retrieval modes
- `runs/<run-id>/manifest.json` — corpus snapshot id, Q-A set hash, model versions, seed, run timestamp, regression-gate verdict (`not_evaluated` at first proof)

## Numeric claims

The README's headline retrieval-quality and citation-quality numbers come from a committed run under this directory. Numbers anywhere else in the repo without a corresponding captured run here are not substantiated by a measurement.
