#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Score the engine on nfcorpus: public questions, human graded qrels.

Index the 3,633-document corpus into a throwaway database (the real corpus is
untouched), then run every question through retrieve() and score with BEIR's
metrics: nDCG@10, MRR@10, Recall@100. Documents are chunked, embedded and stored
through the same indexer functions the PDF path uses, so this measures the
retrieval engine, not a hand-tuned variant of it.

The index is per-embedder: switch SEARCHBOT_EMBED_URL/EMBED_MODEL and use a
different --tag to build a comparable one.

Usage:
  .venv/bin/python scripts/bench_nfcorpus.py --tag gemma
  SEARCHBOT_EMBED_MODEL=nomic-embed-text SEARCHBOT_EMBED_URL=... \
      .venv/bin/python scripts/bench_nfcorpus.py --tag nomic
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import config, db, indexer, llm, retriever  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=str(config.DATA_DIR / "bench" / "nfcorpus"))
ap.add_argument("--tag", default="gemma", help="name for this index (per embedder)")
ap.add_argument("--reindex", action="store_true", help="rebuild even if the index exists")
ap.add_argument("--lanes", nargs="+", default=["hybrid", "bm25", "vec", "bm25ref"],
                choices=["hybrid", "bm25", "vec", "bm25ref"])
ap.add_argument("--limit", type=int, help="score only this many queries (debug)")
ap.add_argument("--json", help="write results here")
a = ap.parse_args()

SLUG = "nfcorpus"
config.DB_PATH = config.DATA_DIR / "bench" / f"nfcorpus-{a.tag}.db"
config.RECALL_K = 100        # BEIR scores down to 100; the app default is 30 per lane

rows = lambda name: [json.loads(l) for l in open(Path(a.data) / f"{name}.jsonl")]
corpus = rows("corpus")
queries = rows("queries")
qrels = {}
for r in rows("qrels_test"):
    qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
qs = [(q["_id"], q["text"]) for q in queries if q["_id"] in qrels]
if a.limit:
    qs = qs[:a.limit]

c = db.connect()
db.init(c, llm.embed_dim())
have = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (SLUG,)).fetchone()["n"]
if have and not a.reindex:
    print(f"reusing index {config.DB_PATH.name} ({have} docs) — pass --reindex to rebuild")
else:
    c.execute("DELETE FROM docs WHERE slug=?", (SLUG,))
    c.execute("DELETE FROM chunks WHERE slug=?", (SLUG,))
    c.execute("DELETE FROM chunks_fts WHERE slug=?", (SLUG,))
    c.execute("DELETE FROM chunks_vec WHERE slug=?", (SLUG,))
    c.commit()
    id_map = {}                      # nfcorpus id -> rowid space of this db
    t0 = time.time()
    for d in corpus:
        text = ((d["title"] + ". ") if d.get("title") else "") + d["text"]
        chunks = indexer.chunk_text(text)
        if not chunks:
            continue
        c.execute("INSERT INTO docs(slug,kind,source_file,title,n_chunks) VALUES(?,?,?,?,?)",
                  (SLUG, "beir", d["_id"], d.get("title", ""), len(chunks)))
        doc_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        id_map[d["_id"]] = doc_id
        for ordinal, ch in enumerate(chunks):
            c.execute("INSERT INTO chunks(slug,doc_id,ordinal,text) VALUES(?,?,?,?)",
                      (SLUG, doc_id, ordinal, ch))
            cid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            c.execute("INSERT INTO chunks_fts(rowid,text,slug,doc_id,chunk_id) VALUES(?,?,?,?,?)",
                      (cid, ch, SLUG, doc_id, cid))
        vecs = llm.embed(chunks)
        first = c.execute("SELECT id FROM chunks WHERE doc_id=? ORDER BY ordinal",
                          (doc_id,)).fetchall()
        c.executemany("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                      [(r["id"], SLUG, db.ser(v)) for r, v in zip(first, vecs)])
        if len(id_map) % 500 == 0:
            print(f"   {len(id_map)}/{len(corpus)} docs, {time.time()-t0:.0f}s")
    c.commit()
    print(f"indexed {len(id_map)} docs -> {config.DB_PATH.name} in {time.time()-t0:.0f}s")

chunk2doc = {r["id"]: r["source_file"] for r in c.execute(
    "SELECT ch.id AS id, d.source_file AS source_file FROM chunks ch "
    "JOIN docs d ON d.id=ch.doc_id WHERE ch.slug=?", (SLUG,))}

LANES = {"hybrid": ("vec", "fts"), "bm25": ("fts",), "vec": ("vec",)}


def idcg(rels, k=10):
    gains = sorted(rels.values(), reverse=True)[:k]
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains)) or 1.0


def evaluate(rank_fn):
    """rank_fn(qid, text) -> ranked corpus ids, best first (document level)."""
    ndcg = mrr = rec = hit = 0.0
    n = 0
    for qid, text in qs:
        rels = qrels[qid]
        ranked, seen = [], set()
        for doc in rank_fn(qid, text):
            if doc and doc not in seen:
                seen.add(doc)
                ranked.append(doc)
        top = ranked[:100]
        gains = [rels.get(d, 0) for d in top]
        ndcg += sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10])) / idcg(rels)
        first = next((i for i, d in enumerate(top, 1) if rels.get(d)), None)
        mrr += 1.0 / first if first and first <= 10 else 0.0
        rec += sum(1 for d in top if rels.get(d)) / len({d for d, s in rels.items() if s > 0})
        hit += 1 if first else 0
        n += 1
    return {"n": n, "nDCG@10": round(ndcg / n, 4), "MRR@10": round(mrr / n, 4),
            "Recall@100": round(rec / n, 4), "HitRate@100": round(hit / n, 4)}


def lane_ranker(lanes):
    def run(qid, text):
        return (chunk2doc.get(h["id"])
                for h in retriever.retrieve(c, text, slug=SLUG, k=100, lanes=lanes))
    return run


# Reference BM25 (Robertson et al., k1=1.2 b=0.75) at document level — this is
# what "BM25" means in the IR literature and on leaderboards, so the engine's own
# lexical lane is compared against it rather than against a remembered number.
# One deliberate difference: no stemming, because there is no stemmer in the
# standard library and pulling one in would not be reproducible elsewhere.
def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


dl, postings = [], {}
for idx, d in enumerate(corpus):
    tf = Counter(tokenize((d.get("title") or "") + " " + d["text"]))
    dl.append(sum(tf.values()) or 1)
    for t, f in tf.items():
        postings.setdefault(t, []).append((idx, f))
AVGDL = sum(dl) / len(dl)
N_DOCS = len(corpus)


def bm25_reference(qid, text, top=100):
    k1, b = 1.2, 0.75
    scores = {}
    for t in set(tokenize(text)):
        chain = postings.get(t)
        if not chain:
            continue
        idf = math.log(1 + (N_DOCS - len(chain) + 0.5) / (len(chain) + 0.5))
        for i, f in chain:
            scores[i] = scores.get(i, 0) + \
                idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / AVGDL))
    return [corpus[i]["_id"] for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]


systems = {}
for name in a.lanes:
    systems[name] = bm25_reference if name == "bm25ref" else lane_ranker(LANES[name])

results = {}
for name, ranker in systems.items():
    t0 = time.time()
    results[name] = evaluate(ranker)
    results[name]["seconds"] = round(time.time() - t0, 1)
    print(f"  {name:<9} scored in {results[name]['seconds']}s")

print(f"\nnfcorpus test | {len(qs)} questions | {len(corpus)} docs | "
      f"{c.execute('SELECT COUNT(*) FROM chunks WHERE slug=?', (SLUG,)).fetchone()[0]} chunks")
print(f"embed={config.EMBED_URL} model={config.EMBED_MODEL or 'default'} tag={a.tag}\n")
print(f"{'system':<12}{'nDCG@10':>9}{'MRR@10':>9}{'Recall@100':>12}{'HitRate@100':>13}{'s':>7}")
for name, r in results.items():
    print(f"{name:<12}{r['nDCG@10']:>9}{r['MRR@10']:>9}{r['Recall@100']:>12}"
          f"{r['HitRate@100']:>13}{r['seconds']:>7}")

if a.json:
    Path(a.json).write_text(json.dumps({"dataset": "nfcorpus", "tag": a.tag,
                                        "embed_url": config.EMBED_URL,
                                        "embed_model": config.EMBED_MODEL,
                                        "queries": len(qs), "results": results}, indent=1))
    print(f"\nwrote {a.json}")
