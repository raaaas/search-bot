# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""SQLite schema: corpus (docs+chunks+fts5+vec0) and memory (sessions/turns/facts)."""
import json
import re
import sqlite3
import struct
from . import config

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS searches(
  slug TEXT PRIMARY KEY, title TEXT, path TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY, slug TEXT, kind TEXT, source_file TEXT,
  title TEXT, authors TEXT, year TEXT, journal TEXT, doi TEXT, pmcid TEXT, pmid TEXT,
  url TEXT, license TEXT, n_chunks INTEGER,
  citations INTEGER, citations_updated TEXT, citations_source TEXT);
CREATE INDEX IF NOT EXISTS ix_docs_slug ON docs(slug);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, slug TEXT, doc_id INTEGER, ordinal INTEGER, text TEXT);
CREATE INDEX IF NOT EXISTS ix_chunks_slug ON chunks(slug);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, slug UNINDEXED, doc_id UNINDEXED, chunk_id UNINDEXED, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, slug TEXT, summary TEXT DEFAULT '', summary_upto INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY, session_id TEXT, slug TEXT, role TEXT,
  text TEXT, refs_json TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS ix_turns_sess ON turns(session_id, id);
CREATE TABLE IF NOT EXISTS facts(
  id INTEGER PRIMARY KEY, slug TEXT, session_id TEXT, kind TEXT DEFAULT 'note',
  text TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS acquired(
  slug TEXT, md5 TEXT, title TEXT, path TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (slug, md5));
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, slug TEXT, kind TEXT, state TEXT, info TEXT,
  updated_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT (datetime('now')));
"""

_vec_schema = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
  chunk_id INTEGER PRIMARY KEY,
  slug TEXT PARTITION KEY,
  embedding float[{dim}]
);
"""


def connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(config.DB_PATH), timeout=60, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=60000")
    try:
        c.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(c)
        c.enable_load_extension(False)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"sqlite-vec load failed: {e}")
    return c


"""Columns added after the first release: added by ALTER on existing databases,
since CREATE TABLE IF NOT EXISTS leaves an already-present table untouched."""
_ADDED_COLUMNS = {
    "docs": [("citations", "INTEGER"), ("citations_updated", "TEXT"),
             ("citations_source", "TEXT")],
}


def migrate(c: sqlite3.Connection) -> None:
    for table, cols in _ADDED_COLUMNS.items():
        have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        for name, sqltype in cols:
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")
    c.commit()


def init(c: sqlite3.Connection, dim: int) -> None:
    c.executescript(_SCHEMA)
    c.executescript(_vec_schema.format(dim=dim))
    migrate(c)
    c.commit()


def reset_vec_table(c: sqlite3.Connection, dim: int) -> None:
    """Recreate the vector table for a different embedding width.

    Only vectors are thrown away: chunk text and the FTS index are untouched, so
    the corpus is recovered by re-embedding rather than by re-reading the PDFs.
    """
    c.execute("DROP TABLE IF EXISTS chunks_vec")
    c.executescript(_vec_schema.format(dim=dim))
    c.commit()


def vec_dim(c) -> int:
    """The vector width this database was created with, or 0 if it has none."""
    row = c.execute("SELECT sql FROM sqlite_master WHERE name='chunks_vec'").fetchone()
    if not row or not row["sql"]:
        return 0
    m = re.search(r"embedding\s+float\[(\d+)\]", str(row["sql"]))
    return int(m.group(1)) if m else 0


def ser(v) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def upsert_search(c, slug: str, title: str = "", path: str = ""):
    c.execute("INSERT INTO searches(slug,title,path) VALUES(?,?,?) "
              "ON CONFLICT(slug) DO UPDATE SET title=excluded.title", (slug, title or slug, path))
    c.commit()


def job_set(c, slug, kind, state, info=""):
    c.execute("INSERT INTO jobs(slug,kind,state,info,updated_at) VALUES(?,?,?,?,datetime('now'))",
              (slug, kind, state, info))
    c.commit()
    return c.execute("SELECT last_insert_rowid()").fetchone()[0]


def job_update(c, job_id, state, info=""):
    c.execute("UPDATE jobs SET state=?, info=?, updated_at=datetime('now') WHERE id=?",
              (state, info, job_id))
    c.commit()


def set_setting(c, key, value) -> None:
    c.execute("INSERT INTO settings(key,value,updated_at) VALUES(?,?,datetime('now')) "
              "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
              (key, json.dumps(value)))
    c.commit()


def get_setting(c, key, default=None):
    row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def all_settings(c) -> dict:
    out = {}
    for row in c.execute("SELECT key,value FROM settings"):
        try:
            out[row["key"]] = json.loads(row["value"])
        except Exception:
            continue
    return out


def drop_corpus(c, slug: str) -> dict:
    """Forget a corpus partition: its docs, chunks, index rows and search entry.

    Leaves the files under search/{slug}/ and the chat memory that referenced
    them alone — deleting a partition from the UI must not silently destroy the
    PDFs it was built from or the answers already grounded in it.
    """
    ids = [r["id"] for r in c.execute("SELECT id FROM chunks WHERE slug=?", (slug,))]
    for cid in ids:
        c.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
        c.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (cid,))
    n_docs = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (slug,)).fetchone()["n"]
    c.execute("DELETE FROM chunks WHERE slug=?", (slug,))
    c.execute("DELETE FROM docs WHERE slug=?", (slug,))
    c.execute("DELETE FROM acquired WHERE slug=?", (slug,))
    c.execute("DELETE FROM searches WHERE slug=?", (slug,))
    c.commit()
    return {"slug": slug, "docs": n_docs, "chunks": len(ids)}
