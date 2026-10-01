# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Local web server: tiny HTML chat UI + JSON API for the search-bot RAG engine.

Endpoints:
  GET    /                     - chat UI
  GET    /api/searches         - list search folders
  POST   /api/searches         - {slug} create+index (background job)
  DELETE /api/searches/{slug}  - forget that corpus (files and chat memory stay)
  GET    /api/status?slug=     - index jobs
  GET    /api/stats            - corpus, embedder and memory counts for the UI
  GET    /api/settings         - effective settings + models the servers offer
  POST   /api/settings         - save settings (model ids, topk, temperature…)
  POST   /api/reindex          - re-embed stored chunks with the active embedder
  POST   /api/ask              - {slug, question, session_id} -> answer+evidence
  GET    /api/trace?id=&since= - live agent events for a run
  POST   /api/libgen/search    - {query, slug?} catalogue search (LibGen/Anna's Archive/...)
  POST   /api/libgen/get       - {slug, id} bring a record into the corpus + index
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

from . import config, db, indexer, libgen, pipeline, llm, retriever, trace as trace_mod

WEB = config.ROOT / "web"
_conn = None
_conn_lock = threading.Lock()

# Settings the UI may change. They are defaults for /api/ask; a field in the
# request body still wins, so a per-call override from the MCP or a script keeps
# working. Model ids are the exception: they are applied to config directly.
DEFAULT_SETTINGS = {
    "chat_model": "",            # "" = auto-detect from the chat server
    "embed_model": "",           # "" = whatever single model the server serves
    "topk": config.FINAL_K,
    "temperature": 0.2,
    "acquire": True,
    "recency": config.RECENCY_WEIGHT,
}
_MODEL_TTL = 30.0
_model_cache = {}


def conn():
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = db.connect()
            db.init(_conn, llm.embed_dim())
        return _conn


def _slugify(s):
    s = re.sub(r"[^a-z0-9_-]+", "-", s.lower()).strip("-")
    return s[:48] or "untitled"


def _human(nbytes) -> str:
    n = float(nbytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _model_ids(url) -> list:
    """Model ids a server reports, cached so a stopped server costs one short
    timeout per TTL instead of a stall on every stats call.

    OpenAI-compatible servers disagree on the envelope: llama.cpp answers
    {"models":[...]}, vLLM and the OpenAI spec answer {"data":[...]}.
    """
    now = time.time()
    hit = _model_cache.get(url)
    if hit and now - hit[0] < _MODEL_TTL:
        return hit[1]
    ids = []
    try:
        data = requests.get(url.rstrip("/") + "/models", headers=llm.auth(), timeout=2.0).json()
        rows = data.get("data") or data.get("models") or []
        ids = [m.get("id") or m.get("name") for m in rows
               if isinstance(m, dict) and (m.get("id") or m.get("name"))]
    except Exception:
        ids = []
    _model_cache[url] = (now, ids)
    return ids


def effective_settings(c) -> dict:
    stored = db.all_settings(c)
    out = dict(DEFAULT_SETTINGS)
    out.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    return out


def apply_settings(c, s: dict) -> None:
    """Push the model ids into config so llm.py picks them up, then drop the
    cached names/dimension — a new embedder has a new vector width."""
    config.CHAT_MODEL = s["chat_model"]
    config.EMBED_MODEL = s["embed_model"]
    llm.forget_models()


def embedder_state(c) -> dict:
    """Compare the embedder the vectors were made with against the active one.

    The model name is only a mismatch when *both* sides are known: a bare
    llama.cpp embedding server reports nothing, and guessing there would flag a
    model swap that has not happened. The width is compared against the vector
    table itself, not just the stamp, because that is what refuses a new insert —
    and databases indexed before stamping existed have no stamp.
    """
    indexed = db.get_setting(c, "indexed_embedder") or {}
    table_dim = db.vec_dim(c)
    active_model = llm.embed_model()
    try:
        active_dim = llm.embed_dim()
    except Exception:
        active_dim = None
    indexed_dim = indexed.get("dim") or table_dim
    names = {n for n in (active_model, indexed.get("model")) if n and n != "unknown"}
    return {
        "indexed_model": indexed.get("model"),
        "indexed_dim": indexed_dim or None,
        "active_model": active_model,
        "active_dim": active_dim,
        "dim_mismatch": bool(active_dim and indexed_dim and active_dim != indexed_dim),
        "model_mismatch": len(names) == 2 and active_model != indexed.get("model"),
        "chunks_missing_vectors": c.execute(
            # NOT EXISTS is a correlated probe, and a vec0 virtual table has no
            # index to answer it with: 5.6s per call over 20k chunks. Materialised
            # as NOT IN the same question costs 20ms. The IS NOT NULL keeps a NULL
            # chunk_id from turning the whole comparison unknown, which would
            # silently report zero missing vectors.
            "SELECT COUNT(*) n FROM chunks WHERE id NOT IN "
            "(SELECT chunk_id FROM chunks_vec WHERE chunk_id IS NOT NULL)").fetchone()["n"],
    }


def corpus_stats(c) -> dict:
    dim = db.vec_dim(c)
    folders = []
    for r in c.execute("SELECT slug,title,path FROM searches ORDER BY slug"):
        n_docs = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (r["slug"],)).fetchone()["n"]
        n_ch = c.execute("SELECT COUNT(*) n FROM chunks WHERE slug=?", (r["slug"],)).fetchone()["n"]
        text_bytes = c.execute("SELECT COALESCE(SUM(LENGTH(text)),0) b FROM chunks WHERE slug=?",
                               (r["slug"],)).fetchone()["b"]
        folders.append({"slug": r["slug"], "title": r["title"], "path": r["path"],
                        "docs": n_docs, "chunks": n_ch,
                        "bytes": int(text_bytes) + n_ch * dim * 4})
    total_docs = sum(f["docs"] for f in folders)
    total_chunks = sum(f["chunks"] for f in folders)
    total_bytes = sum(f["bytes"] for f in folders)
    memory = {k: c.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"]
              for k, t in (("sessions", "sessions"), ("turns", "turns"), ("facts", "facts"))}
    jobs = [dict(r) for r in c.execute(
        "SELECT id,slug,kind,state,info,updated_at FROM jobs ORDER BY id DESC LIMIT 8")]
    return {
        "folders": folders,
        "total_folders": len(folders),
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "total_bytes": total_bytes,
        "formatted_size": _human(total_bytes),
        "vector_dimension": dim,
        "embedder": embedder_state(c),
        "memory": memory,
        "jobs": jobs,
        "settings": effective_settings(c),
        "engine": {
            "chunk_chars": config.CHUNK_CHARS,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "recall_k": config.RECALL_K,
            "final_k": config.FINAL_K,
            "query_instruction": bool(config.QUERY_INSTRUCT),
            "citation_weight": config.CITATION_WEIGHT,
            "recency_half_life_years": config.RECENCY_HALF_LIFE_YEARS,
        },
        "servers": {"chat_url": config.CHAT_URL, "embed_url": config.EMBED_URL},
        "models": {**model_options(c), "active_chat": llm.chat_model()},
    }


def model_options(c) -> dict:
    """The model lists the settings panel offers. Embedder dimensions are only
    known for the active one: probing a server's other models would embed text
    against a model this corpus may not be indexed with, so the UI shows '—' and
    re-indexing settles it."""
    s = effective_settings(c)
    state = embedder_state(c)
    chat_ids = _model_ids(config.CHAT_URL)
    if s["chat_model"] and s["chat_model"] not in chat_ids:
        chat_ids = [s["chat_model"]] + chat_ids
    embed_ids = _model_ids(config.EMBED_URL)
    if s["embed_model"] and s["embed_model"] not in embed_ids:
        embed_ids = [s["embed_model"]] + embed_ids
    serving = s["embed_model"] or state["active_model"]
    return {
        "chat": [{"id": i, "name": i, "active": i == llm.chat_model()} for i in chat_ids],
        "embed": [{"id": i, "name": i,
                   "dim": state["active_dim"] if i == serving else None,
                   "active": i == serving}
                  for i in embed_ids],
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, code=200):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._html((WEB / "index.html").read_text(encoding="utf-8"))
        elif u.path == "/api/searches":
            c = conn()
            rows = c.execute("SELECT slug,title,path,created_at FROM searches ORDER BY created_at DESC").fetchall()
            out = []
            for r in rows:
                n_docs = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (r["slug"],)).fetchone()["n"]
                n_ch = c.execute("SELECT COUNT(*) n FROM chunks WHERE slug=?", (r["slug"],)).fetchone()["n"]
                out.append({**dict(r), "docs": n_docs, "chunks": n_ch})
            self._json(out)
        elif u.path == "/api/stats":
            self._json(corpus_stats(conn()))
        elif u.path == "/api/settings":
            c = conn()
            self._json({"settings": effective_settings(c), "models": model_options(c),
                        "embedder": embedder_state(c),
                        "defaults": DEFAULT_SETTINGS})
        elif u.path == "/api/status":
            slug = parse_qs(u.query).get("slug", [""])[0]
            c = conn()
            if slug:
                rows = c.execute("SELECT * FROM jobs WHERE slug=? ORDER BY id DESC LIMIT 3", (slug,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 5").fetchall()
            self._json([dict(r) for r in rows])
        elif u.path == "/api/history":
            q = parse_qs(u.query)
            sid = q.get("session", [""])[0]
            c = conn()
            rows = c.execute(
                "SELECT role,text,refs_json,created_at FROM turns WHERE session_id=? ORDER BY id LIMIT 200",
                (sid,)).fetchall()
            self._json([{**dict(r)} for r in rows])
        elif u.path == "/api/sessions":
            c = conn()
            rows = c.execute(
                "SELECT s.id, s.slug, s.summary, s.created_at, "
                " (SELECT MIN(t.id) FROM turns t WHERE t.session_id=s.id AND t.role='user') first_turn, "
                " (SELECT COUNT(*) FROM turns t WHERE t.session_id=s.id) turns, "
                " (SELECT COUNT(*) FROM facts f WHERE f.slug=s.slug) facts "
                "FROM sessions s ORDER BY "
                " (SELECT MAX(t.created_at) FROM turns t WHERE t.session_id=s.id) DESC NULLS LAST, "
                " s.created_at DESC LIMIT 50").fetchall()
            out = []
            for r in rows:
                title = ""
                if r["first_turn"]:
                    t = c.execute("SELECT text FROM turns WHERE id=?", (r["first_turn"],)).fetchone()
                    title = (t["text"][:60] + ("…" if len(t["text"]) > 60 else "")) if t else ""
                out.append({"id": r["id"], "slug": r["slug"], "title": title,
                            "turns": r["turns"], "facts": r["facts"],
                            "has_summary": bool(r["summary"]),
                            "created_at": r["created_at"]})
            self._json(out)
        elif u.path == "/api/trace":
            q = parse_qs(u.query)
            rid = q.get("id", [""])[0]
            since = int(q.get("since", ["0"])[0] or 0)
            tr = trace_mod.RUNS.get(rid)
            if not tr:
                self._json({"error": "unknown run"}, 404)
                return
            self._json(tr.wait(since, timeout=3.0))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        try:
            if u.path == "/api/searches":
                slug = _slugify(b.get("slug", ""))
                d = config.SEARCH_DIR / slug
                d.mkdir(parents=True, exist_ok=True)
                (d / "pdf").mkdir(exist_ok=True)
                c = conn()
                db.upsert_search(c, slug, b.get("title", slug), str(d))
                job = db.job_set(c, slug, "index", "queued")

                def run():
                    try:
                        stats = indexer.index_search(c, slug, job_id=job)
                        db.job_update(c, job, "done", json.dumps(stats))
                    except Exception as e:
                        db.job_update(c, job, "error", str(e))
                threading.Thread(target=run, daemon=True).start()
                self._json({"slug": slug, "path": str(d), "job": job})
            elif u.path == "/api/ask":
                from . import agent
                rid = b.get("run_id") or trace_mod.new_run().id
                tr = trace_mod.RUNS[rid]
                # Saved settings are the defaults; a field in this body still wins.
                # slug is taken from the body as-is: null means "search every folder".
                opts = {**effective_settings(conn()),
                        **{k: v for k, v in b.items() if v is not None}}
                opts["slug"] = b.get("slug")
                rank = retriever.rank_opts(opts)

                def work(tr=tr, opts=opts, rank=rank):
                    import time as _t
                    t0 = _t.time()
                    events = []
                    try:
                        res = agent.ask_agentic(conn(), opts["slug"],
                                                opts.get("session_id") or "web",
                                                opts["question"], topk=opts.get("topk"),
                                                log=lambda m: events.append(m),
                                                trace=tr, acquire=bool(opts.get("acquire", True)),
                                                rank=rank,
                                                temperature=opts.get("temperature"))
                        res["agent_events"] = events
                        tr.emit("done", "completed",
                                ms=int((_t.time() - t0) * 1000), count=len(tr.events))
                        tr.finish(res)
                    except Exception as e:
                        tr.emit("error", str(e)[:300])
                        tr.finish({"error": str(e)[:300]})
                threading.Thread(target=work, daemon=True).start()
                self._json({"run_id": rid, "status": "started"})
            elif u.path == "/api/libgen/search":
                self._json(libgen.get_client().search(b["query"], limit=int(b.get("limit", 10))))
            elif u.path == "/api/libgen/get":
                ident = {k: b.get(k) for k in ("md5", "doi", "isbn") if b.get(k)}
                if not ident:
                    self._json({"error": "need md5, doi or isbn"}, 400)
                    return
                slug = b["slug"]
                c = conn()
                job = db.job_set(c, slug, "libgen_get", "queued", str(ident))
                def run_fetch():
                    try:
                        dest = config.SEARCH_DIR / slug / "pdf"
                        dest.mkdir(parents=True, exist_ok=True)
                        dl = libgen.get_client().download(dest_dir=dest, **ident)
                        path = dl.get("path") or dl.get("file") or dl.get("filename") or ""
                        if path and not dl.get("error"):
                            db.job_update(c, job, "indexing", f"saved {Path(path).name}")
                            stats = indexer.index_search(c, slug, job_id=job)
                            db.job_update(c, job, "done", json.dumps(stats))
                        else:
                            db.job_update(c, job, "error", json.dumps(dl)[:300])
                    except Exception as e:
                        db.job_update(c, job, "error", str(e)[:300])
                threading.Thread(target=run_fetch, daemon=True).start()
                self._json({"job": job, "message": "download+index started; poll /api/status"})
            elif u.path == "/api/settings":
                c = conn()
                cur = effective_settings(c)
                num = {"topk": int, "temperature": float, "recency": float}
                for k in DEFAULT_SETTINGS:
                    if k not in b:
                        continue
                    v = b[k]
                    if k in num:
                        try:
                            v = num[k](v)
                        except (TypeError, ValueError):
                            self._json({"error": f"{k} must be a number"}, 400)
                            return
                        if k == "topk" and not 1 <= v <= 50:
                            self._json({"error": "topk must be 1..50"}, 400)
                            return
                        if k == "temperature" and not 0.0 <= v <= 2.0:
                            self._json({"error": "temperature must be 0..2"}, 400)
                            return
                    elif k in ("chat_model", "embed_model"):
                        v = str(v or "")
                    elif k == "acquire":
                        v = bool(v)
                    cur[k] = v
                for k, v in cur.items():
                    db.set_setting(c, k, v)
                apply_settings(c, cur)
                state = embedder_state(c)
                self._json({"settings": cur, "embedder": state,
                            "needs_reindex": state["model_mismatch"] or state["dim_mismatch"]})
            elif u.path == "/api/reindex":
                c = conn()
                slug = b.get("slug")
                if slug in ("", "all", None):
                    slug = None
                elif not c.execute("SELECT 1 FROM searches WHERE slug=?", (slug,)).fetchone():
                    self._json({"error": f"unknown slug {slug}"}, 404)
                    return
                job = db.job_set(c, slug or "all", "reindex", "queued")

                def run_reindex(slug=slug, job=job):
                    try:
                        out = indexer.reembed(conn(), slug, job_id=job)
                        db.job_update(conn(), job, "done", json.dumps(out))
                    except Exception as e:
                        db.job_update(conn(), job, "error", str(e)[:300])
                threading.Thread(target=run_reindex, daemon=True).start()
                self._json({"job": job, "slug": slug or "all",
                            "message": "re-embedding started; poll /api/status"})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_DELETE(self):
        u = urlparse(self.path)
        m = re.match(r"^/api/searches/([A-Za-z0-9_-]{1,64})$", u.path)
        if not m:
            self._json({"error": "not found"}, 404)
            return
        slug = m.group(1)
        c = conn()
        if not c.execute("SELECT 1 FROM searches WHERE slug=?", (slug,)).fetchone():
            self._json({"error": f"unknown slug {slug}"}, 404)
            return
        out = db.drop_corpus(c, slug)
        out["kept_files"] = str(config.SEARCH_DIR / slug)
        out["kept_memory"] = True
        self._json(out)


def main(port=8181):
    print(f"search-bot web on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8181)
