#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Build a labeled retrieval eval set from an indexed corpus.

Three query families, all with gold by construction (no judgment calls):

  title    the paper's own title — known-item search, what a "find me this
           paper" request looks like
  sentence a mid-document sentence with no title overlap — verbatim prose, so it
           is the easy case; a lexical lane should ace it
  keywords 3-4 of the title's rarest terms, stripped of everything generic —
           what someone actually types into a search box: six words, no prose

Gold is recorded as pmid/doi (plus the doc id of this particular database) so
the set survives a re-index.

Usage:
  .venv/bin/python scripts/build_eval_set.py --out tests/eval/known_item.json
  .venv/bin/python scripts/build_eval_set.py --docs 120 --seed 7
"""
import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import config  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--db", default=str(config.DB_PATH))
ap.add_argument("--out", default=str(config.ROOT / "tests" / "eval" / "known_item.json"))
ap.add_argument("--docs", type=int, default=120, help="papers to draw queries from")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--min-sentence", type=int, default=70)
ap.add_argument("--max-sentence", type=int, default=240)
a = ap.parse_args()

c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
c.row_factory = sqlite3.Row


def clean(t):
    t = re.sub(r"<[^>]+>", "", t or "")
    return re.sub(r"\s+", " ", t).strip()


def sentences(text):
    parts = re.split(r"(?<=[.!?]) (?=[A-Z0-9])", re.sub(r"\s+", " ", text))
    return [p.strip() for p in parts]


# document frequency over the whole corpus: what makes a term discriminating
# is how often it appears *elsewhere*, not how it reads
from collections import Counter  # noqa: E402
from searchbot.retriever import _STOP  # noqa: E402

df = Counter()
for row in c.execute("SELECT d.title AS t, ch.text AS x FROM chunks ch "
                     "JOIN docs d ON d.id = ch.doc_id"):
    df.update(set(re.findall(r"[a-z0-9]+",
                             ((row["t"] or "") + " " + (row["x"] or "")).lower())))


docs = [dict(r) for r in c.execute(
    "SELECT id, slug, title, year, doi, pmid FROM docs "
    "WHERE COALESCE(title,'') != '' ORDER BY id")]
random.seed(a.seed)
random.shuffle(docs)

queries, seen, used = [], set(), set()
for d in docs:
    if len(used) >= a.docs:
        break
    start = len(queries)
    title = clean(d["title"])
    n_chunks = c.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (d["id"],)).fetchone()[0]
    gold = {"doc_id": d["id"], "slug": d["slug"], "title": title,
            "pmid": d["pmid"] or "", "doi": (d["doi"] or "").lower()}
    if len(title.split()) >= 4 and title.lower() not in seen:
        seen.add(title.lower())
        queries.append({"family": "title", "text": title, "gold": gold,
                        "gold_chunks": n_chunks})
        terms = [w for w in re.findall(r"[a-z0-9]+", title.lower())
                 if len(w) > 3 and w not in _STOP and not w.isdigit()]
        rare = set(sorted(terms, key=lambda w: df.get(w, 0))[:4])
        kw = " ".join(w for w in terms if w in rare)
        if len(rare) >= 3 and kw.lower() not in seen:
            seen.add(kw.lower())
            queries.append({"family": "keywords", "text": kw, "gold": gold,
                            "gold_chunks": n_chunks})
    if n_chunks >= 3:                 # need a real middle-of-paper sentence
        twords = set(re.sub(r"[^a-z0-9 ]", "", title.lower()).split())
        mid = c.execute("SELECT text FROM chunks WHERE doc_id=? AND ordinal IN "
                        "(SELECT ordinal FROM chunks WHERE doc_id=? ORDER BY ordinal "
                        "LIMIT 3 OFFSET 1)", (d["id"], d["id"])).fetchall()
        for row in mid:
            cands = [s for s in sentences(row["text"])
                     if a.min_sentence <= len(s) <= a.max_sentence
                     and len(s.split()) >= 10
                     and not any(w in twords for w in s.lower().split()
                                 if len(w) > 5)][:1]
            if cands and cands[0].lower() not in seen:
                seen.add(cands[0].lower())
                queries.append({"family": "sentence", "text": cands[0], "gold": gold,
                                "gold_chunks": n_chunks})
                break
    if len(queries) > start:
        used.add(d["id"])

out = {"generated_from": {"db": str(a.db), "corpus_docs": len(docs), "papers": len(used),
                          "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]},
       "seed": a.seed, "queries": queries}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
fams = {}
for q in queries:
    fams[q["family"]] = fams.get(q["family"], 0) + 1
print(f"{len(queries)} queries from {len(set(q['gold']['doc_id'] for q in queries))} papers "
      f"-> {a.out}")
print("   by family:", fams)
print("   example:", queries[0]["family"], "|", queries[0]["text"][:90])
