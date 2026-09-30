# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Hybrid retrieval: vec KNN + FTS5 BM25, fused with Reciprocal Rank Fusion."""
import math
import re
from . import config, db, llm


RANK_KEYS = ("year_after", "year_before", "recency", "citations")


def rank_opts(src) -> dict:
    """Extract only the known ranking keys from a request body / tool args dict.

    Unknown keys are dropped so a stray field cannot become a TypeError deep
    inside the agent loop, and absent keys stay absent so config defaults win.
    """
    if not src:
        return {}
    out = {}
    for k in RANK_KEYS:
        v = src.get(k)
        if v is None or v == "":
            continue
        out[k] = float(v) if k in ("recency", "citations") else int(v)
    return out


def retrieve(c, query: str, slug: str = None, k: int = None,
             year_after: int = None, year_before: int = None,
             recency: float = None, citations: float = None,
             lanes=("vec", "fts")):
    """Hybrid retrieval with optional metadata signals.

    year_after / year_before restrict candidates to a publication-year range;
    a doc with no recoverable year is excluded when either bound is set.
    recency and citations multiply the RRF score by (1 + w*signal), signal in
    [0,1]; they default to config (0.0 = pure RRF, i.e. the historical
    behaviour). Multiplicative is deliberate: adjacent RRF ranks differ by
    ~1e-4, so an additive bonus of any usable size rewrites the ranking.

    lanes can be narrowed to ("vec",) or ("fts",) — for ablation measurements
    of one lane, and for a corpus whose vector index is unusable.
    """
    k = k or config.FINAL_K
    recall = config.RECALL_K
    w_rec = config.RECENCY_WEIGHT if recency is None else recency
    w_cit = config.CITATION_WEIGHT if citations is None else citations
    qvec = llm.embed([query], is_query=True)[0] if "vec" in lanes else None

    # --- vector lane ---
    vec_hits = {}
    if "vec" in lanes:
        if slug:
            rows = c.execute(
                "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? AND slug = ? ORDER BY distance",
                (db.ser(qvec), recall, slug)).fetchall()
        else:
            rows = c.execute(
                "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (db.ser(qvec), recall)).fetchall()
        for rank, r in enumerate(rows):
            vec_hits[r["chunk_id"]] = rank

    # --- BM25 lane ---
    fts_hits = {}
    terms = " OR ".join('"%s"' % t for t in _keywords(query)) if "fts" in lanes else ""
    if terms:
        try:
            if slug:
                frows = c.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? AND slug = ? ORDER BY s LIMIT ?",
                    (terms, slug, recall)).fetchall()
            else:
                frows = c.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                    (terms, recall)).fetchall()
            for rank, r in enumerate(frows):
                fts_hits[r["chunk_id"]] = rank
        except Exception:
            pass

    # --- RRF fusion ---
    scores = {}
    for cid, rank in vec_hits.items():
        scores[cid] = scores.get(cid, 0) + 1.0 / (60 + rank)
    for cid, rank in fts_hits.items():
        scores[cid] = scores.get(cid, 0) + 1.0 / (60 + rank)

    # --- metadata signals + year bounds ---
    if (w_rec or w_cit or year_after or year_before) and scores:
        meta = _doc_meta(c, scores, year_after=year_after, year_before=year_before)
        scores = meta["scores"]
        if w_rec or w_cit:
            ceiling = meta["cit_ceiling"]
            for cid in list(scores):
                factor = 1.0
                if w_rec:
                    factor += w_rec * meta["recency"].get(cid, 0.0)
                if w_cit:
                    n = meta["citations"].get(cid)
                    if n:
                        factor += w_cit * (math.log1p(n) / ceiling)
                scores[cid] *= factor

    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    out = []
    for cid, score in top:
        row = c.execute(
            "SELECT ch.id, ch.text, d.title, d.year, d.journal, d.doi, d.pmcid, d.pmid, d.source_file, d.slug, "
            "d.citations "
            "FROM chunks ch JOIN docs d ON d.id=ch.doc_id WHERE ch.id=?", (cid,)).fetchone()
        if row:
            out.append({"chunk_id": cid, "score": round(score, 5),
                        "in_vec": cid in vec_hits, "in_fts": cid in fts_hits, **dict(row)})
    return out


def _now_year() -> int:
    import datetime
    return datetime.datetime.now().year


def _doc_meta(c, scores, year_after=None, year_before=None) -> dict:
    """Year/citation metadata for the candidate chunks, applying year bounds.

    Recency is a half-life decay on the age of the publication year; citations
    are log-scaled against the busiest candidate so the weight means the same
    thing whatever the corpus size. Docs with no usable year are dropped when a
    bound is requested rather than silently passing as "recent enough".
    """
    ids = list(scores)
    marks = ",".join("?" * len(ids))
    rows = c.execute(
        f"SELECT ch.id AS cid, d.year AS year, d.citations AS citations "
        f"FROM chunks ch JOIN docs d ON d.id=ch.doc_id WHERE ch.id IN ({marks})",
        ids).fetchall()
    now_year = _now_year()
    half_life = config.RECENCY_HALF_LIFE_YEARS or 10.0
    citations_by_cid, recency_by_cid, kept = {}, {}, {}
    max_log = 0.0
    for r in rows:
        cid = r["cid"]
        year = None
        m = re.search(r"\d{4}", r["year"] or "")
        if m:
            year = int(m.group())
        if year_after and (year is None or year < year_after):
            continue
        if year_before and (year is None or year > year_before):
            continue
        kept[cid] = scores[cid]
        if year:
            age = max(now_year - year, 0)
            recency_by_cid[cid] = 0.5 ** (age / half_life)
        n = r["citations"]
        if isinstance(n, int) and n > 0:
            citations_by_cid[cid] = n
            max_log = max(max_log, math.log1p(n))
    return {"scores": kept, "recency": recency_by_cid, "citations": citations_by_cid,
            "cit_ceiling": max_log or 1.0}


_STOP = set("what how why does do is are the of a an in on for and or to with by from was were be been being can could should would may might this that these those it its as at into about which who whom whose when where while after before between during under over more most other some such no nor not only own same than too very s t just now there here they them their".split())


def _keywords(query: str):
    words = [w.strip(".,?!;:'\"()[]") for w in query.lower().split()]
    return [w for w in words if len(w) > 2 and w not in _STOP][:12]
