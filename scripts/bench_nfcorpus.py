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
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchbot import config, db, llm, retriever  # noqa: E402
from bench_index import index_beir_corpus, verify_embedder  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=str(config.DATA_DIR / "bench" / "nfcorpus"))
ap.add_argument("--tag", default="gemma", help="name for this index (per embedder)")
ap.add_argument("--reindex", action="store_true", help="rebuild even if the index exists")
ap.add_argument("--force", action="store_true",
                help="score an index the current embedder did not build")
ap.add_argument("--systems", nargs="+", dest="lanes",
                default=["hybrid", "bm25", "vec", "bm25okapi", "bm25beir",
                         "bm25title", "fts5porter", "tfidf", "random"],
                choices=["hybrid", "bm25", "vec", "bm25okapi", "bm25beir",
                         "bm25title", "fts5porter", "tfidf", "random"],
                help="engine lanes and baselines to score")
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
if a.reindex:
    for t in ("docs", "chunks", "chunks_fts", "chunks_vec"):
        c.execute(f"DELETE FROM {t} WHERE slug=?", (SLUG,))
    c.commit()
# Unconditional: index_beir_corpus skips whatever is already stored and commits
# as it goes, so a build interrupted by a server error finishes on the next run
# instead of being silently reused as though it were complete.
before = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (SLUG,)).fetchone()["n"]
index_beir_corpus(c, corpus, SLUG, log=(lambda *a: None) if before else print)

# Refuse to measure an index the configured embedder did not build. The dense
# lane returns a confident, well-formed, meaningless ranking in that case, and
# two indexes in this project's history had numbers written down against the
# wrong one before probe_embedder caught it.
if {"vec", "hybrid"} & set(a.lanes):
    cos = verify_embedder(c, SLUG)
    if cos < 0.99:
        print(f"\nSTOP: the configured embedder ({config.EMBED_URL}, "
              f"{config.EMBED_MODEL or 'server default'}) reproduces this index's "
              f"stored vectors at cosine {cos:.4f}. It was built by a different "
              f"model, so the dense lane is noise. Point SEARCHBOT_EMBED_* at the "
              f"model that built --tag {a.tag}, use another --tag, or pass --force "
              f"to score it anyway (bm25-only runs are unaffected).")
        if not a.force:
            sys.exit(1)
    print(f"index embedder verified: cosine {cos:.4f}")

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


# Baselines. All of them are document-level over the *whole* nfcorpus text
# (title + body, no chunking) — the unit BEIR/Anserini score nfcorpus with — so
# the engine's chunked, deduped lanes are being compared against the strongest
# form of each method, not a strawman built on smaller units.
def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def build_index(field):
    """Inverted postings + lengths for one text field of the corpus."""
    dl, postings, doc_tf = [], {}, []
    for d in corpus:
        tf = Counter(tokenize(field(d)))
        dl.append(sum(tf.values()) or 1)
        doc_tf.append(tf)
        for t, f in tf.items():
            postings.setdefault(t, []).append((len(dl) - 1, f))
    return postings, dl, doc_tf


DOC_IDS = [d["_id"] for d in corpus]
POSTINGS, DL, DOC_TF = build_index(
    lambda d: (d.get("title") or "") + " " + d["text"])
AVGDL = sum(DL) / len(DL)
N_DOCS = len(corpus)
TITLE_POST, TITLE_DL, _ = build_index(lambda d: d.get("title") or "")
TITLE_AVGDL = sum(TITLE_DL) / len(TITLE_DL) or 1.0

# Robertson/Sparck Jones weighting (Okapi BM25), and the TF-IDF cosine that
# preceded it. Both are textbook, both here in ~15 lines, so the numbers are
# computed on this machine rather than remembered off a leaderboard.
def okapi(postings, dl, avgdl, k1=1.2, b=0.75, top=100):
    def run(qid, text):
        scores = {}
        for t in set(tokenize(text)):
            chain = postings.get(t)
            if not chain:
                continue
            idf = math.log(1 + (N_DOCS - len(chain) + 0.5) / (len(chain) + 0.5))
            for i, f in chain:
                scores[i] = scores.get(i, 0) + \
                    idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avgdl))
        return [DOC_IDS[i] for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]
    return run


IDF = {t: math.log(1 + N_DOCS / len(ch)) for t, ch in POSTINGS.items()}
NORMS = [math.sqrt(sum((1 + math.log(f)) ** 2 * IDF[t] ** 2
                       for t, f in tf.items())) or 1.0 for tf in DOC_TF]


def tfidf(qid, text, top=100):
    """TF-IDF cosine similarity (Salton & Buckley): log-log weighting, L2 norms."""
    q = Counter(tokenize(text))
    qn = math.sqrt(sum((1 + math.log(f)) ** 2 * IDF.get(t, 0) ** 2
                       for t, f in q.items())) or 1.0
    scores = {}
    for t, qf in q.items():
        if t not in IDF:
            continue
        qw = (1 + math.log(qf)) * IDF[t]
        for i, f in POSTINGS[t]:
            scores[i] = scores.get(i, 0) + qw * (1 + math.log(f))
    ranked = ((i, s / (NORMS[i] * qn)) for i, s in scores.items())
    return [DOC_IDS[i] for i, _ in sorted(ranked, key=lambda x: -x[1])[:top]]


# SQLite's own FTS5 ranking over the whole corpus, with porter stemming — a
# third-party implementation of the same public method, so agreement with the
# hand-written Okapi is a check on both.
mem = sqlite3.connect(":memory:")
mem.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(text, tokenize='porter unicode61')")
mem.executemany("INSERT INTO docs_fts(rowid, text) VALUES(?,?)",
                [(i, (d.get("title") or "") + " " + d["text"])
                 for i, d in enumerate(corpus)])


def fts5_porter(qid, text, top=100):
    terms = " OR ".join('"%s"' % t for t in retriever._keywords(text))
    if not terms:
        return []
    return [DOC_IDS[r[0]] for r in mem.execute(
        "SELECT rowid FROM docs_fts WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?",
        (terms, top))]


def shuffled(qid, text, top=100):
    """Seeded random order — the floor. A metric that scores this well is broken."""
    pool = list(DOC_IDS)
    random.Random(qid).shuffle(pool)
    return pool[:top]


BASELINES = {
    "bm25okapi": okapi(POSTINGS, DL, AVGDL),                # k1=1.2, b=0.75
    "bm25beir": okapi(POSTINGS, DL, AVGDL, k1=0.9, b=0.4),  # BEIR's nfcorpus config
    "bm25title": okapi(TITLE_POST, TITLE_DL, TITLE_AVGDL),   # titles only
    "tfidf": tfidf,
    "fts5porter": fts5_porter,
    "random": shuffled,
}

systems = {}
for name in a.lanes:
    systems[name] = BASELINES[name] if name in BASELINES else lane_ranker(LANES[name])

results = {}
for name, ranker in systems.items():
    t0 = time.time()
    results[name] = evaluate(ranker)
    results[name]["seconds"] = round(time.time() - t0, 1)
    print(f"  {name:<9} scored in {results[name]['seconds']}s")

print(f"\nnfcorpus test | {len(qs)} questions | {len(corpus)} docs | "
      f"{c.execute('SELECT COUNT(*) FROM chunks WHERE slug=?', (SLUG,)).fetchone()[0]} chunks")
print(f"embed={config.EMBED_URL} model={config.EMBED_MODEL or 'default'} tag={a.tag}")
# The query instruction is part of the query vector, so it belongs in the record
# of a run: with embeddinggemma's default prefix on nomic the dense lane drops
# from 0.3464 to 0.2528 and nothing else about the run changes.
qi = config.QUERY_INSTRUCT.strip().replace("\n", " ") or "(none)"
print(f"query_instruct={qi[:70]}")
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
