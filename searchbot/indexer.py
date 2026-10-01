# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Corpus indexing: extract text from PDFs/XML in search/{slug}/, chunk, embed, store."""
import csv
import hashlib
import re
from pathlib import Path
from pypdf import PdfReader
from lxml import etree
from . import config, db, llm


def extract_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        parts = []
        for p in reader.pages[:60]:  # cap pathological docs
            parts.append(p.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


def extract_xml(path: Path) -> str:
    try:
        tree = etree.parse(str(path))
        for root in tree.iter("abstract", "sec", "p", "title", "body"):
            pass  # cheap parse warmup
        text = "".join(tree.getroot().itertext())
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def meta_from_csv(slug_dir: Path) -> dict:
    """filename -> metadata dict, from fulltext_index.csv / *_metadata.csv if present."""
    out = {}
    for csvf in slug_dir.glob("*.csv"):
        try:
            with open(csvf, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    fn = (row.get("file") or "").strip()
                    if fn:
                        out[Path(fn).name] = row
                        out[Path(fn).stem] = row
        except Exception:
            continue
    return out


def title_from_pdf_name(path: Path) -> str:
    # 2023_PMC10464275_Hemodynamic impact of ephedrine ....pdf
    m = re.match(r"^(\d{4})_((PMC|PMID)?\d+)?_?(.*)\.pdf$", path.name)
    if m:
        year, pmcid, _, rest = m.groups()
        return rest.replace("_", " ").strip() or path.stem
    return path.stem


def chunk_text(text: str):
    """Split on paragraph/sentence boundaries into ~CHUNK_CHARS pieces with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    # hard-split any monster sentence (XML tables / no punctuation) into safe pieces
    safe = []
    for s in sentences:
        while len(s) > config.CHUNK_CHARS:
            safe.append(s[:config.CHUNK_CHARS])
            s = s[config.CHUNK_CHARS:]
        if s:
            safe.append(s)
    sentences = safe
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > config.CHUNK_CHARS and cur:
            chunks.append(cur.strip())
            cur = cur[-config.CHUNK_OVERLAP:] + " " + s if len(cur) > config.CHUNK_OVERLAP else s
        else:
            cur = (cur + " " + s).strip()
    if len(cur.strip()) >= 80:
        chunks.append(cur.strip())
    return chunks[:config.MAX_CHUNKS_PER_DOC]


def stamp_embedder(c, dim: int) -> None:
    """Record which embedder the stored vectors came from, so a later model swap
    can be reported as a mismatch instead of quietly corrupting retrieval."""
    db.set_setting(c, "indexed_embedder", {"model": llm.embed_model(), "dim": int(dim)})


def reembed(c, slug: str = None, job_id=None, batch_size: int = 8, log=print) -> dict:
    """Re-embed every stored chunk with the ACTIVE embedder.

    index_search will not revisit a document it has already seen, so switching
    embedders needs this path instead: the chunk text is the source of truth and
    only the vectors are rebuilt.
    """
    dim = llm.embed_dim()
    if slug:
        rows = c.execute("SELECT id, slug, text FROM chunks WHERE slug=? ORDER BY id",
                         (slug,)).fetchall()
    else:
        rows = c.execute("SELECT id, slug, text FROM chunks ORDER BY id").fetchall()
    db.reset_vec_table(c, dim)
    stats = {"chunks": 0, "docs": 0, "dim": dim, "model": llm.embed_model()}
    for i in range(0, len(rows), batch_size):
        part = rows[i:i + batch_size]
        vecs = llm.embed([r["text"] for r in part])
        c.executemany("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                      [(r["id"], r["slug"], db.ser(v)) for r, v in zip(part, vecs)])
        stats["chunks"] += len(part)
        c.commit()
        if job_id:
            db.job_update(c, job_id, "running", f"re-embedded {stats['chunks']}/{len(rows)}")
        log(f"re-embedded {stats['chunks']}/{len(rows)} chunks")
    stamp_embedder(c, dim)
    stats["docs"] = c.execute(
        "SELECT COUNT(DISTINCT doc_id) n FROM chunks" + (" WHERE slug=?" if slug else ""),
        ((slug,) if slug else ())).fetchone()["n"]
    return stats


def index_search(c, slug: str, job_id=None, log=print) -> dict:
    """(Re)index search/{slug}. Skips chunks already stored (content-hash per doc file)."""
    slug_dir = config.SEARCH_DIR / slug
    db.upsert_search(c, slug, path=str(slug_dir))
    files = sorted(list(slug_dir.rglob("*.pdf")) + list(slug_dir.rglob("*.xml")))
    metas = meta_from_csv(slug_dir)
    stats = {"files": 0, "new_docs": 0, "chunks": 0, "errors": 0}
    dim = llm.embed_dim()
    db.init(c, dim)

    for fpath in files:
        stats["files"] += 1
        rel = str(fpath.relative_to(slug_dir))
        existing = c.execute("SELECT id FROM docs WHERE slug=? AND source_file=?",
                             (slug, rel)).fetchone()
        if existing:
            continue
        meta = metas.get(fpath.name) or metas.get(fpath.stem) or {}
        text = extract_pdf(fpath) if fpath.suffix == ".pdf" else extract_xml(fpath)
        if len(text.strip()) < 100:
            stats["errors"] += 1
            continue
        chunks = chunk_text(text)
        if not chunks:
            stats["errors"] += 1
            continue
        title = (meta.get("title") or title_from_pdf_name(fpath))
        title = re.sub(r"&lt;i&gt;|&lt;/i&gt;|&lt;b&gt;|&lt;/b&gt;|&lt;[^>]+&gt;", "", title)
        c.execute("INSERT INTO docs(slug,kind,source_file,title,authors,year,journal,doi,pmcid,pmid,url,license,n_chunks) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (slug, "pdf" if fpath.suffix == ".pdf" else "xml", rel,
                   title.strip(), meta.get("authors", ""), meta.get("year", ""),
                   meta.get("journal", ""), meta.get("doi", ""), meta.get("pmcid", ""),
                   meta.get("pmid", ""), meta.get("link", ""), meta.get("license", ""), len(chunks)))
        doc_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        stats["new_docs"] += 1

        chunk_ids = []
        for ordinal, ch in enumerate(chunks):
            c.execute("INSERT INTO chunks(slug,doc_id,ordinal,text) VALUES(?,?,?,?)",
                      (slug, doc_id, ordinal, ch))
            chunk_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            chunk_ids.append(chunk_id)
            c.execute("INSERT INTO chunks_fts(rowid,text,slug,doc_id,chunk_id) VALUES(?,?,?,?,?)",
                      (chunk_id, ch, slug, doc_id, chunk_id))
        vecs = llm.embed(chunks)
        c.executemany("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                      [(cid, slug, db.ser(v)) for cid, v in zip(chunk_ids, vecs)])
        stats["chunks"] += len(chunks)
        stamp_embedder(c, len(vecs[0]) if vecs else dim)
        c.commit()
        if job_id:
            db.job_update(c, job_id, "running", f"{rel} ({len(chunks)} chunks)")
        log(f"indexed {rel}: {len(chunks)} chunks")
    c.execute("UPDATE searches SET title=COALESCE(NULLIF(title,''),?) WHERE slug=?", (slug, slug))
    c.commit()
    return stats
