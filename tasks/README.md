# Implementation Task Lanes

This directory holds task-lane definitions for downstream implementation packets that will land each portion of v1. Tasks are scoped (one task = one packet's worth of work) and each task names its `ROADMAP.md` acceptance gate.

## Initial Task List (Tier 1 plan only; tasks are authored in their own implementation packets)

| Task ID | Description | Gates served |
|---|---|---|
| T-CORPUS | Corpus selection + ingestion + attestation | G3, G4 |
| T-DENSE | Dense retriever implementation | part of G2 |
| T-SPARSE | Sparse retriever implementation | part of G2 |
| T-HYBRID | RRF hybrid implementation | part of G2 |
| T-GRAPH | Graph-aware implementation per the smallest-defensible bar | G3 |
| T-EVAL | Eval harness + metrics | G2, G7 |
| T-QA | Q-A evaluation set construction | part of G7 |
| T-REGRESSION | Regression gate + baseline manifest | G8 |
| T-CI | Public CI surface (smoke + regression + license + composition-check) | G6, G8 |
| T-DOCS | README authoring + Reproduce-section walk | G5, G11 |

## Operational Rule

Each task is an upstream implementation-packet booking, not a directory of code. The implementation packets live in the parent operating hub; this directory's task list is the public-side reference of what is still ahead.

## Anti-Pattern Callout

Tasks are not silently dropped. If a task is removed from the v1 plan, the removal lands as a `ROADMAP.md` amendment with a stated rationale, not as a quiet deletion from this list.
