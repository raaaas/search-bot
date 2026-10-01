#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Indexing shared by the public-benchmark scripts.

Both benchmarks measure the engine over someone else's collection, and both must
build it the same way or their numbers are not comparable to each other. This is
the one place that turns BEIR documents into docs + chunks + fts + vec rows, and
it goes through the same indexer.chunk_text and llm.embed the PDF path uses.
"""
import time

import numpy as np

from searchbot import db, indexer, llm


def index_beir_corpus(c, docs, slug, log=print, every=500):
    """Store BEIR-shaped docs ({_id, title, text}) under `slug`.

    Returns {corpus _id: docs.rowid}. The title is prefixed to the body because
    that is how these collections are judged — which means only the first chunk
    of a long document carries it, and a title-only query can therefore miss the
    later chunks. That is a property of chunking, not a bug to hide here.

    Documents already stored under the slug are skipped and the tables are
    committed every `every` documents, because building an index means a few
    thousand embedding calls and one 500 from the embed server should cost a
    re-run of the remaining documents, not the whole build.
    """
    done = {r["source_file"]: r["id"] for r in c.execute(
        "SELECT id, source_file FROM docs WHERE slug=?", (slug,))}
    id_map, t0, since = dict(done), time.time(), 0
    for d in docs:
        if d["_id"] in id_map:
            continue
        text = ((d["title"] + ". ") if d.get("title") else "") + d["text"]
        chunks = indexer.chunk_text(text)
        if not chunks:
            continue
        c.execute("INSERT INTO docs(slug,kind,source_file,title,n_chunks) VALUES(?,?,?,?,?)",
                  (slug, "beir", d["_id"], d.get("title", ""), len(chunks)))
        doc_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        id_map[d["_id"]] = doc_id
        for ordinal, ch in enumerate(chunks):
            c.execute("INSERT INTO chunks(slug,doc_id,ordinal,text) VALUES(?,?,?,?)",
                      (slug, doc_id, ordinal, ch))
            cid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            c.execute("INSERT INTO chunks_fts(rowid,text,slug,doc_id,chunk_id) VALUES(?,?,?,?,?)",
                      (cid, ch, slug, doc_id, cid))
        vecs = llm.embed(chunks)
        rows = c.execute("SELECT id FROM chunks WHERE doc_id=? ORDER BY ordinal",
                         (doc_id,)).fetchall()
        c.executemany("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                      [(r["id"], slug, db.ser(v)) for r, v in zip(rows, vecs)])
        since += 1
        if since >= every:
            c.commit()
            since = 0
        if log and len(id_map) % every == 0:
            log(f"   {len(id_map)}/{len(docs)} docs, {time.time() - t0:.0f}s")
    c.commit()
    if log:
        log(f"indexed {len(id_map)} docs ({len(docs) - len(done)} new) -> "
            f"{c.execute('PRAGMA database_list').fetchone()[2]} in {time.time() - t0:.0f}s")
    return id_map


def verify_embedder(c, slug, n=4):
    """Cosine between stored vectors and the same text re-embedded right now.

    Two 768-dimension models are indistinguishable to SQLite: an index built by
    one embedder and queried with another returns a confident, well-formed,
    meaningless ranking. Returns 1.0 when the index has no vectors to compare.
    """
    rows = c.execute("SELECT ch.id, ch.text, v.embedding FROM chunks ch "
                     "JOIN chunks_vec v ON v.chunk_id=ch.id AND v.slug=ch.slug "
                     "WHERE ch.slug=? LIMIT ?", (slug, n)).fetchall()
    if not rows:
        return 1.0
    fresh = llm.embed([r["text"] for r in rows])
    out = []
    for f, r in zip(fresh, rows):
        s = np.frombuffer(r["embedding"], dtype=np.float32)
        out.append(float(np.dot(f / np.linalg.norm(f), s / np.linalg.norm(s))))
    return float(np.mean(out))
