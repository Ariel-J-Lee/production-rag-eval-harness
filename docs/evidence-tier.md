# Evidence Tier Posture

## v1 Target

The v1 first release targets **Tier 4 — Real Command Path**. The harness command will run end-to-end against a committed corpus snapshot; the evaluation report will be committed; raw output will be captured under `runs/<run-id>/`. Evidence at every Tier 4 claim is the captured artifact, not a paraphrased summary.

## What v1 Will NOT Claim

v1 explicitly does NOT claim:

- **Tier 5 (live runtime / observed against deployed system).** No "running in production at X" claim. No "deployed and serving traffic" framing.
- **Tier 6 (customer-visible proof).** No "real users have validated" claim. No "reduced incident rate by Y%" framing.

The repository is a runnable harness when v1 lands; it is not, and will not be, a service.

## Per-Claim Tier-Routing Rule

Every claim made anywhere in the repository declares its evidence tier honestly. Any claim that would require Tier 4+ evidence appears only when its captured-evidence path ships in the repository. Until then, the claim is held.

| Tier | Means | Allowed in v1 |
|---|---|---|
| 1 | Static trace; source read and reasoned about | yes (architectural commitments) |
| 2 | Build / syntax checked | yes (when implementation lands) |
| 3 | Targeted unit / integration tests pass | yes (when tests land) |
| 4 | Real command path executed; output captured | yes (this is the v1 target) |
| 5 | Live runtime observed against deployed system | no |
| 6 | Customer-visible proof | no |

## Shell-State Declaration

At shell-creation time, every retrieval, eval, citation-quality, and benchmark claim is held. The shell sits at **Tier 1 — Static Trace** for all content. The architectural commitments in `docs/architecture.md` and the v1 commitments in `ROADMAP.md` are commitments, not claims.

The shell does not include `runs/<run-id>/` content because no run has occurred. The shell does not include `qa/qa_pairs.jsonl` content because no Q-A set has been constructed. The shell does not include any retrieval implementation because no retrieval has been built.

Every README and `docs/` paragraph in this shell uses commitment-tense framing. Any reader who interprets the shell as a present-tense claim of working software is reading against the explicit framing of this document.
