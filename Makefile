# production-rag-eval-harness — Makefile
#
# Targets land per the v1 implementation slices documented in ROADMAP.md.
# Implemented targets run real work; not-yet-implemented targets print
# "not yet implemented" and exit non-zero.

.PHONY: smoke smoke-dense smoke-sparse smoke-hybrid eval regression fetch composition-check help

help:
	@echo "production-rag-eval-harness"
	@echo ""
	@echo "Available targets:"
	@echo "  fetch              — corpus fetch via scripts/fetch_corpus.py (Wikidata + Wikipedia OSS slice; tracks ROADMAP G4)"
	@echo "  smoke-dense        — dense retrieval smoke against an in-tree fixture (tracks ROADMAP G2; needs 'pip install -r requirements.txt')"
	@echo "  smoke-sparse       — sparse BM25 retrieval smoke against an in-tree fixture (tracks ROADMAP G2; pure stdlib)"
	@echo "  smoke-hybrid       — hybrid (RRF over dense + sparse) retrieval smoke against an in-tree fixture (tracks ROADMAP G2; needs 'pip install -r requirements.txt')"
	@echo "  smoke              — aggregated smoke tests per retrieval mode — not yet implemented (tracks ROADMAP G2)"
	@echo "  eval               — full evaluation across all four modes — not yet implemented (tracks ROADMAP G1, G2, G7)"
	@echo "  regression         — regression gate against committed baseline — not yet implemented (tracks ROADMAP G8)"
	@echo "  composition-check  — verify B7 deviation set against private composition — not yet implemented (tracks ROADMAP G10)"
	@echo ""
	@echo "Status: dense (mode 1) and sparse (mode 2) retrieval are runnable; hybrid and graph-aware modes follow. See ROADMAP.md for the eleven hard gates."

smoke:
	@echo "make smoke: not yet implemented; tracks at ROADMAP.md gate G2."
	@exit 2

smoke-dense:
	@python3 -m src.retrieval.dense --smoke

smoke-sparse:
	@python3 -m src.retrieval.sparse --smoke

smoke-hybrid:
	@python3 -m src.retrieval.hybrid --smoke

eval:
	@echo "make eval: not yet implemented; tracks at ROADMAP.md gates G1, G2, G7."
	@exit 2

regression:
	@echo "make regression: not yet implemented; tracks at ROADMAP.md gate G8."
	@exit 2

fetch:
	@python3 scripts/fetch_corpus.py --out-dir data/oss-ecosystem

composition-check:
	@echo "make composition-check: not yet implemented; tracks at ROADMAP.md gate G10."
	@exit 2
