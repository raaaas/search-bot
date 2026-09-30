#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Report whether the configured embedder is the one that built this index.

Two embedders can share an output dimension and produce unrelated vectors —
nomic-embed-text and embeddinggemma-300M are both 768. Swapping one in does not
error, it silently returns noise, because sqlite-vec only checks the dimension.

This measures the mismatch directly: re-embed text that is already stored and
compare it to the vector in the table. Mean cosine ~1.00 means the configured
embedder is the one that built the index; anything near 0 means it is a
different model and every vector search is returning noise. Exits 0 on a match,
1 on a mismatch, 2 on a dimension change (which does fail loudly).

Usage:
  .venv/bin/python scripts/probe_embedder.py
  SEARCHBOT_EMBED_MODEL=ollama/bge-m3:latest .venv/bin/python scripts/probe_embedder.py --db data/bench/nfcorpus-nomic.db
"""
import argparse
import math
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import config, db, llm  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--db", default=str(config.DB_PATH))
ap.add_argument("--chunks", type=int, default=8, help="stored chunks to re-embed")
a = ap.parse_args()

config.DB_PATH = Path(a.db)
c = db.connect()
row = c.execute("SELECT sql FROM sqlite_master WHERE name='chunks_vec'").fetchone()
if row is None:
    sys.exit(f"{a.db}: no chunks_vec table — is this a search-bot database?")
table_dim = row["sql"].split("float[")[1].split("]")[0]

stored = c.execute("SELECT chunk_id, embedding FROM chunks_vec "
                   "ORDER BY chunk_id LIMIT 2000").fetchall()
texts = {r["id"]: r["text"] for r in c.execute(
    "SELECT id, text FROM chunks ORDER BY id LIMIT 2000")}
random.seed(0)
sample = random.sample([r for r in stored if r["chunk_id"] in texts],
                       min(a.chunks, len(texts)))
if not sample:
    sys.exit("index has no chunks to probe")

vectors = llm.embed([texts[r["chunk_id"]][:1500] for r in sample])
if len(vectors[0]) != int(table_dim):
    print(f"embedder returns {len(vectors[0])}-d, index is {table_dim}-d: "
          "these cannot be mixed. Delete the database and re-index.")
    sys.exit(2)


def cos(x, y):
    nx = math.sqrt(sum(t * t for t in x)) or 1.0
    ny = math.sqrt(sum(t * t for t in y)) or 1.0
    return sum(p * q for p, q in zip(x, y)) / (nx * ny)


def unpack(blob):
    return struct.unpack(f"{len(blob) // 4}f", blob)


scores = [cos(unpack(r["embedding"]), v) for r, v in zip(sample, vectors)]

print(f"db        {a.db}")
print(f"embed     {config.EMBED_URL} model={config.EMBED_MODEL or 'server default'} "
      f"query_instruct={'yes' if config.QUERY_INSTRUCT else 'no'}")
print(f"dim       {table_dim}, {len(sample)} chunks probed")
for cid, s in zip([r["chunk_id"] for r in sample], scores):
    print(f"  chunk {cid:<7} cosine {s:.4f}")
mean = sum(scores) / len(scores)
verdict = ("MATCH — the configured embedder is the one that built this index"
           if mean > 0.999 else
           "MISMATCH — retrieval will run and return noise. Re-index with this "
           "embedder, or delete the database before switching.")
print(f"\nmean cosine {mean:.4f}  ->  {verdict}")
sys.exit(0 if mean > 0.999 else 1)
