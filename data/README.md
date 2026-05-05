# Corpus

This directory will hold corpus attestation and (when v1 is built out) the public-domain or permissively-licensed corpus the harness uses. Corpus blobs are not committed; the v1 implementation will materialize them via a deterministic loader script. This README pins the corpus structure before the corpus is selected.

## Allowed Corpus Sources (whitelist for v1)

In priority order:

1. **Synthetic** — generated for the demo, no input from any private source. Generation script committed; deterministic with a documented RNG seed; parameter source must be public.
2. **Public-domain** — clearly public (CC0, US Government work, expired copyright). Origin URL recorded.
3. **Permissively-licensed public datasets** — license recorded with SPDX identifier (CC-BY-4.0, MIT, Apache-2.0, or equivalent). License compatibility with the repo license verified.
4. **Vendor-reference data marked public** — only if vendor terms allow redistribution.

## Forbidden Corpus Sources

- Any private repository, private storage, private collaboration tool.
- Customer documents, customer data exports, tenant-specific data of any kind.
- Datasets derived from a real customer scenario by any transformation.
- Datasets reconstructed from memory of a real architecture or real example.
- Any corpus shaped to mirror a private domain dataset.

## v1 Corpus Direction

The primary v1 direction is a coherent **Wikidata + Wikipedia slice scoped to the open-source software ecosystem** (projects, programming languages, contributors, dependencies, licenses, organizations). This direction is documented but not yet selected; final selection lands in the v1 implementation packet.

Documented fallbacks (any of these is whitelist-allowed):

1. PubMed/MEDLINE abstracts (public-domain) + Wikidata biomedical slice (CC0)
2. arXiv abstracts (CC-BY-4.0) + OpenAlex citation graph (CC0)
3. Synthetic enterprise corpus generated from a parametric script committed to the repo

## Required Attestation

Every committed corpus carries `data/DATA-SOURCE.md` with the full required-fields contract. Absence of `data/DATA-SOURCE.md` blocks publication of any captured run output.

## Reverse-Identifiable Hardness

Customer-derived, customer-shaped, or reconstructed-from-private datasets are forbidden in any form, even if names are removed and units are reshaped. The corpus's domain, schema, distributions, and scenario layout must be public-derived from the start.
