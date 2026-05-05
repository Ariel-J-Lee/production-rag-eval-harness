# production-rag-eval-harness — shell-only Makefile
#
# Each target prints "not yet implemented" and exits non-zero. The v1
# implementation packets land the actual targets per ROADMAP.md.

.PHONY: smoke eval regression fetch composition-check help

help:
	@echo "production-rag-eval-harness — shell-only"
	@echo ""
	@echo "Available targets (all not yet implemented):"
	@echo "  smoke              — smoke tests per retrieval mode (tracks ROADMAP G2)"
	@echo "  eval               — full evaluation across all four modes (tracks ROADMAP G1, G2, G7)"
	@echo "  regression         — regression gate against committed baseline (tracks ROADMAP G8)"
	@echo "  fetch              — corpus fetch via scripts/fetch_corpus.py (tracks ROADMAP G4)"
	@echo "  composition-check  — verify B7 deviation set against private composition (tracks ROADMAP G10)"
	@echo ""
	@echo "Status: shell-only. See ROADMAP.md for the eleven hard gates."

smoke:
	@echo "make smoke: not yet implemented; tracks at ROADMAP.md gate G2."
	@exit 2

eval:
	@echo "make eval: not yet implemented; tracks at ROADMAP.md gates G1, G2, G7."
	@exit 2

regression:
	@echo "make regression: not yet implemented; tracks at ROADMAP.md gate G8."
	@exit 2

fetch:
	@echo "make fetch: not yet implemented; tracks at ROADMAP.md gate G4."
	@exit 2

composition-check:
	@echo "make composition-check: not yet implemented; tracks at ROADMAP.md gate G10."
	@exit 2
