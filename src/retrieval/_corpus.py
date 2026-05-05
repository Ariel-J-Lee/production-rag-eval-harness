"""Shared chunked-corpus loader.

Every retrieval mode reads the same chunked-passage stream produced by
``scripts/fetch_corpus.py``. The chunked output lives at
``data/<corpus>/chunks.jsonl`` (gitignored at the repo root via
``data/*/`` with an allowlist for metadata files; the corpus is
materialized locally by ``make fetch``).

Each line of ``chunks.jsonl`` is a JSON object with the schema documented
in ``data/DATA-SOURCE.md``::

    {
      "passage_id":   str,           # stable: "<wikidata_id-or-title-stem>:<NNNN>"
      "wikidata_id":  str | None,
      "title":        str,
      "chunk_index":  int,
      "char_count":   int,
      "text":         str,
    }

The loader is pure-stdlib and streams lazily so retrievers can build their
indices in a single pass without holding the whole file in memory before
indexing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_chunks(path: str | Path) -> Iterator[dict]:
    """Yield chunk records from a ``chunks.jsonl`` file.

    Empty and whitespace-only lines are skipped. Each non-empty line is
    parsed as JSON; malformed lines raise :class:`json.JSONDecodeError`
    with the offending line number in the exception message context (the
    caller's traceback).

    Args:
        path: Filesystem path to the ``chunks.jsonl`` file.

    Yields:
        One ``dict`` per chunk, with the keys documented above.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
