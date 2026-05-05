# Corpus Data Source Attestation

This document attests the provenance and license posture of the corpus fetched by `scripts/fetch_corpus.py`. The fetched corpus itself is **not committed to this repository**; only this attestation, the two corpus license files in `data/`, the seed-list-bearing fetch script, and `data/README.md` live in version control. The `data/<corpus>/` subdirectory is gitignored at the repo root via `data/*/` with an allowlist for the metadata files.

## Corpus identity

- **Name** — open-source software ecosystem slice (Wikidata + Wikipedia).
- **Origin** — public-licensed.
  - Wikipedia article text under **CC-BY-SA-4.0** (passage source for dense + sparse retrieval).
  - Wikidata entity facts under **CC0 1.0 Universal** (graph slice for graph-aware retrieval).
- **Slice composition** — approximately 50 seed entities (well-known open-source software projects, programming languages, tooling ecosystems, and contributing organizations) plus their 1-hop Wikidata neighborhoods, hard-capped at 500 entities per the first-proof slice cap.
- **Properties used for relational questions** — `P31` (instance of), `P178` (developer), `P277` (programmed in), `P275` (license), `P306` (operating system), `P407` (language of work).

## Source attestation

| Field | Value |
|---|---|
| Wikipedia API | `https://en.wikipedia.org/w/api.php` (`action=query`, `prop=extracts\|pageprops`) |
| Wikidata API | `https://www.wikidata.org/w/api.php` (`action=wbgetentities`, `props=labels\|claims`) |
| Wikipedia license | CC-BY-SA-4.0 ([`data/LICENSE.wikipedia`](./LICENSE.wikipedia)) |
| Wikidata license | CC0 1.0 Universal ([`data/LICENSE.wikidata`](./LICENSE.wikidata)) |
| Authentication | none (both APIs are public read-only endpoints) |
| Fetch script | [`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py) |
| Default output directory | `data/oss-ecosystem/` (gitignored) |
| Snapshot identifier | written into `data/<out-dir>/manifest.json` per fetch run with a `captured_at` timestamp and per-file byte count / SHA-256 |

## Required-fields contract

| Field | Value |
|---|---|
| `name` | open-source software ecosystem slice (Wikidata + Wikipedia) |
| `origin` | permissive-license (CC0 + CC-BY-SA-4.0) |
| `source_url` | Wikipedia API + Wikidata API (above) |
| `license` | mixed: CC0 (Wikidata facts) + CC-BY-SA-4.0 (Wikipedia text) |
| `license_file` | `data/LICENSE.wikidata`, `data/LICENSE.wikipedia` |
| `generation_script` | `scripts/fetch_corpus.py` (deterministic given the seed list and upstream snapshot date) |
| `generation_seed` | the inline `SEED_LIST` constant in `scripts/fetch_corpus.py`; the script is deterministic given that list and the upstream API state |
| `generator_parameter_rationale` | not synthetic — the corpus is fetched from public sources, not generated parametrically |
| `snapshot_id` | written per fetch run into `data/<out-dir>/manifest.json` (`captured_at` + per-file SHA-256) |
| `pii_check` | Wikidata + Wikipedia content is public; the slice contains only public-figure / public-organization information about open-source software entities. No PII reconstruction from private sources. |
| `customer_derivation_check` | no private source touched any step of corpus selection or fetch. |
| `reconstruction_check` | the seed list is hand-curated from public knowledge of the open-source software ecosystem. No private corpus, private example, or private parameter source was used. |
| `size_bytes_committed` | 0 corpus bytes committed to the repository. The corpus lives at the gitignored output directory; only this attestation, the two license files, and the fetch script (with the seed list inline) are in version control. |

## Share-alike isolation (Wikipedia CC-BY-SA-4.0)

The CC-BY-SA-4.0 share-alike obligation that comes with Wikipedia article text is isolated to the corpus subdirectory:

- The fetched Wikipedia article text lives at `data/<out-dir>/articles.jsonl` (gitignored). When that file is materialized by `make fetch`, it inherits CC-BY-SA-4.0.
- The repository code (root `LICENSE` = Apache 2.0) is not pulled into share-alike scope because the Wikipedia content does not live in the source tree.
- The license file at `data/LICENSE.wikipedia` documents the obligation for any reviewer who clones and runs `make fetch` and is the canonical share-alike notice for the materialized corpus.

This pattern follows the documented isolation rule from the project's publication gate. If the legal-policy review of the isolation pattern returns a tighter requirement, the implementation pivots to a documented fallback (PubMed/MEDLINE + Wikidata biomedical, then arXiv + OpenAlex citation graph, then a synthetic enterprise corpus) before publication of any scored run.

## What is NOT committed at this slice

- No corpus blobs (no fetched Wikipedia article text, no fetched Wikidata entity facts, no fetched 1-hop neighborhoods, no manifest from a real fetch run).
- No Q-A set (lands in a future T-QA packet).
- No retriever code (lands in future T-DENSE / T-SPARSE / T-HYBRID / T-GRAPH packets).
- No eval harness or scored runs (lands in a future T-EVAL packet).

## Reproducing the fetch

```sh
make fetch
# or
python scripts/fetch_corpus.py --out-dir data/oss-ecosystem
```

The script is deterministic given the seed list inline in `scripts/fetch_corpus.py` and the upstream Wikipedia / Wikidata snapshot at fetch time. Two reviewers who run `make fetch` against the same upstream snapshot get the same corpus bytes (verified by the per-file `sha256` written to `data/<out-dir>/manifest.json`).

The fetch is paced courteously to public APIs (default 1 request/second) and uses only the Python standard library (no external dependencies are added by this slice).
