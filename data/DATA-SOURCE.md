# Corpus Data Source Attestation (placeholder — not yet a real attestation)

This file's required-fields contract is the canonical attestation shape. The shell commits this file with all fields marked TBD so the file's existence is not mistaken for a satisfied attestation. The v1 implementation packet fills the fields when the corpus is fetched.

```
- name: TBD
- origin: TBD (synthetic | public-domain | permissive-license | vendor-reference)
- source_url: TBD
- license: TBD
- license_file: TBD (data/LICENSE.<corpus> path)
- generation_script: TBD (if synthetic)
- generation_seed: TBD (if synthetic)
- generator_parameter_rationale: TBD (if synthetic; documents public source for each parameter choice)
- snapshot_id: TBD (stable identifier; matches runs/<run-id>/manifest.json)
- pii_check: TBD (date and reviewer)
- customer_derivation_check: TBD (one-line statement: no private source touched any step of provenance)
- reconstruction_check: TBD (one-line statement: no private example was used as parameter source)
- size_bytes_committed: TBD (after .gitignore for blobs)
```

The v1 release blocks until every field is filled.
