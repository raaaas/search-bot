# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Shared helpers for the test suite: a temp corpus and a deterministic embedder.

No test here touches a real model server, the network, or the user's data
directory — llm.embed is replaced by a token-hash vectoriser so the vector lane
is reproducible without llama.cpp.
"""
import hashlib
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from searchbot import config, db, llm, retriever  # noqa: E402

DIM = 32
_VOCAB = ("ephedrine hypotension anesthesia spinal ephedra metformin glucose "
          "berberine blood pressure review case study patients").split()


def embed_fake(texts, is_query=False, batch_size=8):
    """Deterministic bag-of-tokens vector; shared vocabulary makes similarity sane."""
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        low = t.lower()
        for i, w in enumerate(_VOCAB):
            if w in low:
                v[i] += 1.0
        if not v.any():
            v[0] = 1.0
        v = v / np.linalg.norm(v)
        out.append(v)
    return out


class TempCase(unittest.TestCase):
    """Base: isolated data dir, fake embedder, config snapshot restored after."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="searchbot-test-")
        self._saved = {k: getattr(config, k) for k in
                       ("DATA_DIR", "DB_PATH", "SEARCH_DIR", "RECENCY_WEIGHT",
                        "CITATION_WEIGHT", "RECENCY_HALF_LIFE_YEARS",
                        "OPENALEX_BATCH", "API_KEY", "CHAT_MODEL")}
        config.DATA_DIR = pathlib.Path(self._tmp)
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.SEARCH_DIR = config.DATA_DIR / "search"
        config.RECENCY_WEIGHT = 0.0
        config.CITATION_WEIGHT = 0.0
        config.RECENCY_HALF_LIFE_YEARS = 10.0
        config.OPENALEX_BATCH = 40
        config.API_KEY = ""
        config.CHAT_MODEL = "test-model"
        self._embed = llm.embed
        llm.embed = embed_fake
        llm._embed_dim = DIM
        self.c = db.connect()
        db.init(self.c, DIM)

    def tearDown(self):
        self.c.close()
        llm.embed = self._embed
        llm._embed_dim = None
        for k, v in self._saved.items():
            setattr(config, k, v)

    def add_doc(self, slug, title, year, citations=None, text="ephedrine hypotension anesthesia",
                doi="", pmid=""):
        """One doc with one chunk, vector inserted through the same fake embedder."""
        cur = self.c.execute(
            "INSERT INTO docs(slug,kind,source_file,title,year,citations,doi,pmid) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (slug, "pdf", f"{title}.pdf", title, year, citations, doi, pmid))
        doc_id = cur.lastrowid
        ch = self.c.execute("INSERT INTO chunks(slug,doc_id,ordinal,text) VALUES(?,?,?,?)",
                            (slug, doc_id, 0, text))
        cid = ch.lastrowid
        self.c.execute("INSERT INTO chunks_fts(rowid,text,slug,doc_id,chunk_id) VALUES(?,?,?,?,?)",
                       (cid, text, slug, doc_id, cid))
        self.c.execute("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                       (cid, slug, db.ser(embed_fake([text])[0])))
        self.c.commit()
        return cid

    # --- retrieval convenience (shared by every subclass) ---
    QUERY = "ephedrine hypotension"

    def ids(self, **kw):
        return [h["chunk_id"] for h in retriever.retrieve(self.c, self.QUERY, **kw)]

    def scores(self, **kw):
        return {h["chunk_id"]: h["score"]
                for h in retriever.retrieve(self.c, self.QUERY, **kw)}

    def hits(self, **kw):
        return retriever.retrieve(self.c, self.QUERY, **kw)
