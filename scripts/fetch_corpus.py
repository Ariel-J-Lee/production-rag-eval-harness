#!/usr/bin/env python3
"""Deterministic corpus fetch for the first runnable proof of
production-rag-eval-harness.

Fetches a Wikidata + Wikipedia slice scoped to the open-source software
ecosystem (~50 seed entities plus their 1-hop Wikidata neighborhoods,
hard-capped at 500 articles per the first-proof slice cap).

Sources
-------
- Wikipedia content under CC-BY-SA-4.0 — see ``data/LICENSE.wikipedia``.
- Wikidata content under CC0 1.0 Universal — see ``data/LICENSE.wikidata``.

Both are accessed via public APIs without authentication. The fetched
corpus is written under ``--out-dir`` (default ``data/oss-ecosystem``)
which is gitignored at the repo root via the ``data/*/`` pattern with
allowlist for ``data/README.md``, ``data/DATA-SOURCE.md``, and
``data/LICENSE.*``. Only this script, the seed list (inline below), and
the four attestation/license files live in version control.

Usage
-----
::

    make fetch
    # or
    python scripts/fetch_corpus.py --out-dir data/oss-ecosystem [--limit N] [--delay S]

The script uses the Python standard library only. It paces requests
(``--delay``, default 1.0s) to be courteous to the public APIs. A
manifest with byte counts and SHA-256 digests is written under the
output directory at the end of each fetch run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

USER_AGENT = (
    "production-rag-eval-harness corpus-fetch "
    "(+https://github.com/Ariel-J-Lee/production-rag-eval-harness)"
)

# Wikidata properties used for relational questions per the first-proof
# corpus commitment: instance-of, developer, programmed-in, license,
# operating-system, language-of-work.
RELATIONAL_PROPERTIES = ["P31", "P178", "P277", "P275", "P306", "P407"]

# Seed list — well-known open-source software projects, programming
# languages, tooling ecosystems, and contributing organizations. Each
# entry is the title of the canonical English Wikipedia page; the
# fetcher resolves each title to its Wikidata Q-ID via pageprops.
SEED_LIST = [
    # Operating systems and kernels
    "Linux kernel", "FreeBSD", "OpenBSD", "NetBSD", "Debian", "Ubuntu",
    "Fedora Linux", "Arch Linux",
    # Programming languages
    "Python (programming language)", "JavaScript", "TypeScript",
    "Rust (programming language)", "Go (programming language)",
    "Java (programming language)", "C (programming language)",
    "C++", "Ruby (programming language)",
    # Web servers / proxies / databases
    "Nginx", "Apache HTTP Server", "PostgreSQL", "MySQL", "SQLite",
    "Redis", "MongoDB",
    # Distributed data
    "Apache Kafka", "Apache Spark", "Apache Hadoop",
    # Source control + collaboration
    "Git", "GitHub", "GitLab",
    # Containers + orchestration
    "Docker (software)", "Kubernetes",
    # Machine learning / numerics
    "TensorFlow", "PyTorch", "scikit-learn", "NumPy", "Pandas (software)",
    # Browsers + rendering
    "Mozilla Firefox", "Chromium (web browser)", "WebKit",
    # Editors / IDEs
    "Vim (text editor)", "GNU Emacs", "Visual Studio Code",
    # Web frameworks
    "Django (web framework)", "Ruby on Rails", "Express.js",
    "React (JavaScript library)", "Vue.js", "Angular (web framework)",
    "Node.js",
    # Foundations + governance
    "Linux Foundation", "Apache Software Foundation",
    "Free Software Foundation",
    # Licenses
    "Apache License", "MIT License",
    "GNU General Public License",
    # Reference projects + crypto + Wiki
    "OpenSSL", "GnuPG", "Wikipedia", "MediaWiki", "Wikidata",
]


def http_get_json(url: str, params: dict | None = None, timeout: float = 30.0) -> dict:
    """HTTP GET with stdlib, JSON response, polite User-Agent."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_wikipedia_article(title: str) -> dict | None:
    """Fetch a Wikipedia article extract plus its Wikidata Q-ID.

    Returns a dict with ``title``, ``wikidata_id``, ``extract``, and
    ``url``, or ``None`` if no page exists for the title.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|pageprops",
        "explaintext": "1",
        "redirects": "1",
    }
    data = http_get_json(WIKIPEDIA_API, params)
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        resolved_title = page.get("title", title)
        return {
            "title": resolved_title,
            "wikidata_id": page.get("pageprops", {}).get("wikibase_item"),
            "extract": page.get("extract", ""),
            "url": (
                "https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(resolved_title.replace(" ", "_"))
            ),
        }
    return None


def fetch_wikidata_entity(qid: str) -> dict | None:
    """Fetch a Wikidata entity's English label and the relevant claim
    targets for the relational properties of interest.
    """
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": qid,
        "props": "labels|claims",
        "languages": "en",
    }
    data = http_get_json(WIKIDATA_API, params)
    entity = data.get("entities", {}).get(qid)
    if not entity:
        return None
    label = entity.get("labels", {}).get("en", {}).get("value", qid)
    claims: list[dict] = []
    for prop in RELATIONAL_PROPERTIES:
        for claim in entity.get("claims", {}).get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            value = mainsnak.get("datavalue", {}).get("value", {})
            if isinstance(value, dict) and value.get("id"):
                claims.append({"property": prop, "value": value["id"]})
    return {"wikidata_id": qid, "label": label, "claims": claims}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the first-proof corpus slice for production-rag-eval-harness",
    )
    parser.add_argument(
        "--out-dir",
        default="data/oss-ecosystem",
        help="Output directory for fetched corpus (gitignored)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Hard cap on total entities/articles fetched (first-proof slice cap is 500)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API requests (politeness pacing)",
    )
    parser.add_argument(
        "--seeds-only",
        action="store_true",
        help="Skip 1-hop neighbor expansion; fetch only the seed entities",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    articles_path = out / "articles.jsonl"
    entities_path = out / "entities.jsonl"
    manifest_path = out / "manifest.json"

    print(
        f"Fetching {len(SEED_LIST)} seed entities into {out}",
        file=sys.stderr,
    )
    print(
        f"Hard cap: {args.limit} entities. Delay between requests: {args.delay}s.",
        file=sys.stderr,
    )

    seed_qids: set[str] = set()
    article_count = 0

    # Step 1: Wikipedia articles + Wikidata Q-ID resolution per seed.
    with articles_path.open("w", encoding="utf-8") as af:
        for seed in SEED_LIST:
            if article_count >= args.limit:
                print(
                    f"Limit {args.limit} reached during seed fetch",
                    file=sys.stderr,
                )
                break
            time.sleep(args.delay)
            article = fetch_wikipedia_article(seed)
            if not article:
                print(f"  [skip] no Wikipedia page for: {seed}", file=sys.stderr)
                continue
            af.write(json.dumps(article, ensure_ascii=False) + "\n")
            article_count += 1
            qid = article.get("wikidata_id")
            if qid:
                seed_qids.add(qid)
            print(
                f"  [{article_count:3d}] {seed} -> {qid}",
                file=sys.stderr,
            )

    # Step 2: Wikidata entity claims for each seed Q-ID, and 1-hop
    # neighbor entities (unless --seeds-only).
    neighbor_qids: set[str] = set()
    entity_count = 0
    with entities_path.open("w", encoding="utf-8") as ef:
        for qid in sorted(seed_qids):
            time.sleep(args.delay)
            entity = fetch_wikidata_entity(qid)
            if not entity:
                continue
            ef.write(json.dumps(entity, ensure_ascii=False) + "\n")
            entity_count += 1
            for claim in entity["claims"]:
                neighbor_qids.add(claim["value"])

        if not args.seeds_only:
            new_neighbors = neighbor_qids - seed_qids
            for qid in sorted(new_neighbors):
                if entity_count >= args.limit:
                    print(
                        "  [limit] cap reached during 1-hop neighbor fetch",
                        file=sys.stderr,
                    )
                    break
                time.sleep(args.delay)
                entity = fetch_wikidata_entity(qid)
                if not entity:
                    continue
                ef.write(json.dumps(entity, ensure_ascii=False) + "\n")
                entity_count += 1

    # Step 3: Manifest with captured-at, byte counts, SHA-256.
    manifest = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_list_count": len(SEED_LIST),
        "seed_articles_fetched": article_count,
        "seed_qids_resolved": len(seed_qids),
        "total_entities_fetched": entity_count,
        "limit": args.limit,
        "seeds_only": bool(args.seeds_only),
        "files": {},
    }
    for f in (articles_path, entities_path):
        if f.exists():
            content = f.read_bytes()
            manifest["files"][f.name] = {
                "bytes": len(content),
                "lines": len(content.splitlines()),
                "sha256": sha256_of(f),
            }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote manifest: {manifest_path}", file=sys.stderr)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
