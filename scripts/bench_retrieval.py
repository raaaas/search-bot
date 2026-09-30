#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Measure retrieval quality on a labeled set, with lane ablations.

Reports doc-level metrics: a query counts as a hit when ANY chunk of the gold
paper reaches the cutoff, so landing on a sibling chunk of the right paper is
not punished.

Usage:
  .venv/bin/python scripts/bench_retrieval.py                     # hybrid, bm25, vec
  .venv/bin/python scripts/bench_retrieval.py --lanes hybrid vec
  .venv/bin/python scripts/bench_retrieval.py --recency 0.2 --json /tmp/run.json
"""
import argparse
import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sqlite_vec  # noqa: E402
from searchbot import config, retriever  # noqa: E402

LANES = {"hybrid": ("vec", "fts"), "bm25": ("fts",), "vec": ("vec",)}
CUTS = (1, 5, 8, 20)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--db", default=str(config.DB_PATH))
ap.add_argument("--eval", default=str(config.ROOT / "tests" / "eval" / "known_item.json"))
ap.add_argument("--lanes", nargs="+", default=["hybrid", "bm25", "vec"],
                choices=sorted(LANES))
ap.add_argument("--slug", help="restrict the pool to one folder (default: whole corpus)")
ap.add_argument("--limit", type=int, help="only this many queries (debug)")
ap.add_argument("--depth", type=int, default=max(CUTS), help="how deep to measure")
ap.add_argument("--recency", type=float, help="also run every lane with this recency weight")
ap.add_argument("--citations", type=float, help="also run every lane with this citation weight")
ap.add_argument("--json", help="write the full result here")
a = ap.parse_args()

c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
c.enable_load_extension(True)
sqlite_vec.load(c)
c.enable_load_extension(False)

ev = json.loads(Path(a.eval).read_text())
queries = ev["queries"]
if a.limit:
    queries = queries[:a.limit]

# Gold is resolved by pmid/doi first so the set survives a re-index; the title
# is the last resort for papers that lost both identifiers.
by_pmid, by_doi, by_title = {}, {}, {}
for r in c.execute("SELECT id, title, doi, pmid FROM docs"):
    if r["pmid"]:
        by_pmid[str(r["pmid"]).strip()] = r["id"]
    if r["doi"]:
        by_doi[r["doi"].strip().lower()] = r["id"]
    by_title[re.sub(r"[^a-z0-9]", "", (r["title"] or "").lower())] = r["id"]

# hits are chunks; gold is a paper — the metric is "first chunk of the right paper"
chunk_doc = {r["id"]: r["doc_id"] for r in c.execute("SELECT id, doc_id FROM chunks")}


def gold_id(g):
    key = re.sub(r"[^a-z0-9]", "", (g.get("title") or "").lower())
    return (by_pmid.get(str(g.get("pmid") or "").strip())
            or by_doi.get((g.get("doi") or "").strip().lower())
            or by_title.get(key)
            or g.get("doc_id"))


def first_gold_rank(hits, doc_id):
    for i, h in enumerate(hits, 1):
        if chunk_doc.get(h["id"]) == doc_id:
            return i
    return None


def dcg(rank):
    return 1.0 / math.log2(1.0 + rank) if rank else 0.0


def summarize(pairs):
    """pairs = [(family, gold_rank or None)]"""
    n = len(pairs) or 1
    found = [r for _, r in pairs if r]
    out = {"n": len(pairs),
           "no_hit": len(pairs) - len(found),
           "mrr": sum(1.0 / r for r in found) / n,
           "ndcg@8": sum(dcg(min(r, 8) if r else 0) if r else 0 for _, r in pairs) / n,
           "median_rank": sorted(found)[len(found) // 2] if found else None}
    for k in CUTS:
        out[f"recall@{k}"] = sum(1 for r in found if r <= k) / n
    return out


configs = [(name, {"lanes": LANES[name]}) for name in a.lanes]
for weight, key in ((a.recency, "recency"), (a.citations, "citations")):
    if weight is not None:
        configs += [(f"{name}+{key}", {"lanes": LANES[name], key: weight})
                    for name in a.lanes]

unresolved = [q for q in queries if gold_id(q["gold"]) is None]
if unresolved:
    print(f"note: skipping {len(unresolved)} queries whose gold paper is not in this db")
queries = [q for q in queries if gold_id(q["gold"]) is not None]

print(f"pool: {c.execute('SELECT COUNT(*) FROM docs').fetchone()[0]} docs / "
      f"{c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks"
      f"{' slug=' + a.slug if a.slug else ' (whole corpus)'} | "
      f"{len(queries)} queries | embed={config.EMBED_URL}")

results = {}
for name, opts in configs:
    t0 = time.time()
    pairs = []
    for q in queries:
        gid = gold_id(q["gold"])
        hits = retriever.retrieve(c, q["text"], slug=a.slug, k=a.depth, **opts)
        pairs.append((q["family"], first_gold_rank(hits, gid)))
    results[name] = {"overall": summarize(pairs),
                     "by_family": {fam: summarize([p for p in pairs if p[0] == fam])
                                   for fam in sorted({f for f, _ in pairs})},
                     "seconds": round(time.time() - t0, 1)}
    print(f"  {name:<20} {results[name]['seconds']}s")

HDR = f"\n{'config':<22}{'n':>5}{'R@1':>8}{'R@5':>8}{'R@8':>8}{'R@20':>8}{'MRR':>8}{'nDCG@8':>9}{'miss':>6}"


def table(rows, title):
    print(f"\n{title}")
    print(HDR)
    for name, s in rows:
        print(f"{name:<22}{s['n']:>5}{s['recall@1']:>8.3f}{s['recall@5']:>8.3f}"
              f"{s['recall@8']:>8.3f}{s['recall@20']:>8.3f}{s['mrr']:>8.3f}"
              f"{s['ndcg@8']:>9.3f}{s['no_hit']:>6}")


table([(n, r["overall"]) for n, r in results.items()], "overall (any chunk of the gold paper)")
fams = sorted({f for r in results.values() for f in r["by_family"]})
for fam in fams:
    table([(n, r["by_family"][fam]) for n, r in results.items()],
          f"family: {fam}"
          + (" — known-item, favours lexical matching" if fam == "title" else
             " — verbatim prose, the easy case" if fam == "sentence" else
             " — short typed keyword string" if fam == "keywords" else ""))

if a.json:
    Path(a.json).write_text(json.dumps({
        "pool": ev.get("generated_from"), "embed_url": config.EMBED_URL, "slug": a.slug,
        "queries": len(queries), "results": results}, indent=1))
    print(f"\nwrote {a.json}")
