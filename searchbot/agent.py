# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Agentic loop: if the corpus can't answer, ACQUIRE more science and re-ask.

  retrieve -> sufficient?
    yes -> grounded answer
    no  -> libgen_search (LibGen + Anna's Archive + PubMed/arXiv/Crossref)
         -> download best hits into search/{slug}/pdf/ -> index -> retrieve again
         -> max 2 acquire rounds, then answer on whatever exists

Every step is emitted through an Ev (see trace.py) so the web UI can render a
live agent trace (phases: start/retrieve/gate/acquire_*/merge/generate/delta/done).
"""
import re
import time

from . import config, db, libgen, memory, pipeline, retriever

MIN_SCORE = 0.02      # weak RRF floor (2 lanes both top-rank => ~0.032)
MIN_HITS = 3          # fewer than this = corpus too thin
MAX_ROUNDS = 2
MAX_ACQUIRE = 3       # PDFs per round


def _trigrams(s):
    return {s[i:i+3] for i in range(max(len(s) - 2, 0))}


def _fuzzy_covered(term, words):
    """True if term is a corpus word OR a close variant (typo-tolerant):
    shares >=50% of its trigrams with some similarly-sized word."""
    if term in words:
        return True
    t = _trigrams(term)
    if len(t) < 2:
        return True
    for w in words:
        if abs(len(w) - len(term)) > 3:
            continue
        if len(_trigrams(w) & t) >= 0.5 * len(t):
            return True
    return False


def sufficiency(hits, question=None):
    """Good evidence = >=MIN_HITS hits AND EVERY distinctive question term
    appears in at least one hit (titles or chunk text, typo-tolerant).
    Multi-part questions must have every part covered — an uncovered term
    means the answer would be partial, so the agent must acquire it.
    Returns (ok, missing_terms)."""
    if len(hits) < MIN_HITS:
        return False, _distinctive_terms(question) if question else []
    if not question:
        return hits[0]["score"] >= MIN_SCORE, []
    corpus = " ".join((h["title"] or "") + " " + h["text"] for h in hits)
    words = set(re.sub(r"[^a-z0-9 -]", " ", corpus.lower()).split())
    missing = [w for w in _distinctive_terms(question)
               if not _fuzzy_covered(w, words)]
    return (len(missing) == 0), missing


def _merge_per_term(c, hits, question, topk, ev, search_slug=None, rank=None):
    """Ensure every distinctive question term contributes top hits, so a
    multi-part question gets evidence for EACH part (union), not just the
    dominant topic. Re-ranks the union by RRF sum with doc-diversity cap."""
    terms = _distinctive_terms(question)
    companions = " ".join(w for w in _question_keywords(question) if w not in terms)
    for w in terms:
        covered = sum(1 for h in hits if w in (h["title"] or "").lower())
        if covered >= 2:
            continue
        # missing term + the question's own outcome words (a bare one-word
        # query ranks poorly in BM25; 'berberine glucose blood pressure' hits
        # the right papers)
        extra = retriever.retrieve(c, f"{w} {companions}".strip(), slug=search_slug, k=6,
                                 **(rank or {}))
        got = 0
        added = []
        for h in extra:
            if got >= 2:
                break
            if any(h["chunk_id"] == x["chunk_id"] for x in hits):
                continue
            hits.append(h)
            got += 1
            added.append({"slug": h["slug"], "title": (h["title"] or "")[:70],
                          "score": h["score"]})
            ev("merge", f"+'{w}' evidence from {h['slug']}: {(h['title'] or '')[:50]}")
        if added:
            ev("merge_extra", f"term '{w}' covered by {len(added)} extra source(s)",
               term=w, added=added)
    # re-rank union, cap 3 chunks per document so one paper can't flood the block
    ranked = sorted(hits, key=lambda h: -h["score"])
    out, per_doc = [], {}
    for h in ranked:
        dk = h.get("id") or h.get("doc_id") or h["source_file"]
        if per_doc.get(dk, 0) >= 3:
            continue
        per_doc[dk] = per_doc.get(dk, 0) + 1
        out.append(h)
        if len(out) >= (topk or config.FINAL_K):
            break
    # make sure each term still survives the cap
    for h in ranked:
        if len(out) >= (topk or config.FINAL_K) + 4:
            break
        for w in terms:
            if w in (h["title"] or "").lower() and all(h["chunk_id"] != x["chunk_id"] for x in out):
                out.append(h)
                break
    return out


def _distinctive_terms(question):
    """Rarest content words of the question (max 3), for the coverage gate."""
    kws = _question_keywords(question)
    if not kws:
        return []
    # prefer the longest (most scientific) up to 3
    return sorted(set(kws), key=lambda w: (-len(w), w))[:3]


def _question_keywords(question):
    """Content words of the question, rarest-signal first (longer words first)."""
    import re
    from .retriever import _STOP
    words = [w for w in re.sub(r"[^a-zA-Z0-9 -]", "", question.lower()).split()
             if len(w) > 3 and w not in _STOP]
    # drop generic research-speak words that pollute catalogue queries
    generic = {"main", "effects", "effect", "study", "studies", "research", "related",
               "different", "important", "known", "role", "roles", "use", "used",
               "based", "analysis", "data", "results", "mechanisms", "mechanism",
               "compare", "comparison", "versus", "vs", "between", "many", "several"}
    specific = sorted((w for w in dict.fromkeys(words) if w not in generic),
                      key=len, reverse=True)
    # the fallback must exclude generic too: letting them back in is what put
    # "effects study" into catalogue queries in the first place
    fallback = [w for w in dict.fromkeys(words)
                if w not in generic and w not in specific]
    return (specific + fallback)[:4]


def acquire_round(c, slug, question, ev, forced_query=None):
    """Search the catalogue with the question's own keywords, filter by title
    relevance, download+index up to MAX_ACQUIRE papers. Returns count."""
    if forced_query:
        kws = [w for w in re.sub(r"[^a-zA-Z0-9 -]", "", forced_query.lower()).split() if len(w) > 3]
        max_this = 2                      # per-term acquisition: smaller budget
    else:
        kws = _question_keywords(question)
        max_this = MAX_ACQUIRE
    if not kws:
        return 0
    got, tried = 0, set()
    for n in (3, 2, 1):
        used = kws[:n]
        anchor = max(used, key=len)      # distinctive word of THIS query
        query = " ".join(used)
        ev("libgen_search", f"catalogue search: '{query}'", query=query, slug=slug)
        t0 = time.time()
        try:
            res = libgen.get_client().search(query, limit=10, topics=["articles"])
            items = res.get("results", []) if isinstance(res, dict) else []
            lane = "articles"
            if not items:
                # nothing in the articles collection — try the full catalogue
                # (libgen falls over to Anna's Archive/arXiv/Crossref/PubMed itself)
                res = libgen.get_client().search(query, limit=10)
                items = res.get("results", []) if isinstance(res, dict) else []
                lane = "full-catalog (Anna's Archive/arXiv/PubMed…)"
        except Exception as e:
            ev("libgen_search_error", f"search failed: {e}", query=query)
            continue
        ev("libgen_results", f"{len(items)} hit(s) via {lane} in {time.time()-t0:.1f}s",
           query=query, count=len(items),
           hits=[{"title": (it.get('title') or '')[:90], "year": it.get("year"),
                  "md5": (it.get("md5") or "")[:12]} for it in items[:6]])
        for it in items:
            md5 = it.get("md5")
            title = (it.get("title") or "")
            if not md5 or md5 in tried:
                continue
            # relevance gate: title must contain the anchor word
            if anchor not in title.lower():
                continue
            tried.add(md5)
            if c.execute("SELECT 1 FROM acquired WHERE slug=? AND md5=? LIMIT 1",
                         (slug, md5)).fetchone():
                ev("skip_dup", f"already acquired: {title[:70]}")
                continue
            # dedup by normalized title against every doc already indexed for this slug
            norm = re.sub(r"[^a-z0-9]", "", title.lower())
            if norm:
                dup = False
                for row in c.execute("SELECT title FROM docs WHERE slug=?", (slug,)):
                    if re.sub(r"[^a-z0-9]", "", (row["title"] or "").lower())[:60] == norm[:60]:
                        dup = True
                        break
                if dup:
                    ev("skip_dup", f"already indexed: {title[:70]}")
                    continue
            ev("download", f"downloading: {title[:80]}", title=title[:200], slug=slug)
            t1 = time.time()
            try:
                dl = libgen.get_client().download(
                    md5=md5, dest_dir=config.SEARCH_DIR / slug / "pdf")
                c.execute("INSERT OR IGNORE INTO acquired(slug,md5,title,path) VALUES(?,?,?,?)",
                          (slug, md5, title[:300], dl.get("path", "") if isinstance(dl, dict) else ""))
                c.commit()
                path = dl.get("path") or dl.get("file") or ""
                if path and not dl.get("error"):
                    ev("download_done", f"saved {path.split('/')[-1] if path else ''} "
                                        f"({time.time()-t1:.1f}s)", path=path)
                    ev("index", f"indexing new PDF into '{slug}'…", slug=slug)
                    stats = _index(c, slug)
                    ev("index_done", f"+{stats.get('new_docs', 0)} doc(s), "
                                     f"+{stats.get('chunks', 0)} chunk(s)", **stats)
                    got += 1
                    if got >= max_this:
                        return got
                else:
                    ev("download_error", f"failed: {str(dl)[:120]}", title=title[:120])
            except Exception as e:
                ev("download_error", f"failed: {e}", title=title[:120])
        if tried:
            break                        # anchor matched: don't broaden further
    return got


def _index(c, slug):
    from . import indexer
    return indexer.index_search(c, slug, log=lambda m: None)


def pipeline_normalize_slug(slug):
    """'all'/'' -> None (search every folder); memory still gets a stable key."""
    return None if (not slug or slug == "all") else slug


def _derive_folder(question):
    kws = _question_keywords(question)
    return re.sub(r"[^a-z0-9-]", "", (kws[0] if kws else "general").lower())[:24] or "general"


def ask_agentic(c, slug, session_id, question, topk=None, log=None, trace=None,
                acquire=True, rank=None, temperature=None):
    from .trace import Ev
    rank = rank or {}
    ev = Ev(log=log, trace=trace)
    search_slug = pipeline_normalize_slug(slug)
    mem_slug = slug or "all"
    memory.ensure_session(c, session_id, mem_slug)
    question = question.strip()[:pipeline.ASK_LIMIT]
    scope = "ALL folders" if search_slug is None else slug
    ev("start", f"question received — scope: {scope}"
       + ("" if acquire else " (acquire disabled)"), question=question, slug=mem_slug)
    rounds = []

    t0 = time.time()
    hits = retriever.retrieve(c, question, slug=search_slug, k=topk, **rank)
    _emit_hits(ev, "retrieve", f"hybrid retrieval (vec+BM25+RRF): {len(hits)} hit(s)",
               hits, ms=int((time.time()-t0)*1000))
    ok, missing = sufficiency(hits, question)
    ev("gate", ("evidence covers every key term" if ok
                else f"missing coverage: {', '.join(missing)}"),
       ok=ok, missing=missing)

    while acquire and not ok and len(rounds) < MAX_ROUNDS:
        acquired_here = 0
        ev("acquire_plan",
           f"round {len(rounds)+1}: gather sources for "
           f"{', '.join(missing) if missing else 'the question'}",
           missing=missing)
        # one target folder per uncovered distinctive term (ephedra-only
        # corpus asking about berberine -> berberine folder gets the papers)
        for term in (missing or [None]):
            target = search_slug or (term and _folder_for(term, question)) or _derive_folder(question)
            if not search_slug and term:
                ev("acquire_target",
                   f"term '{term}' missing — gathering into NEW folder '{target}'",
                   term=term, folder=target)
            acquired_here += acquire_round(c, target, term or question, ev,
                                           forced_query=term)
        rounds.append({"acquired": acquired_here,
                       "folders": [search_slug or (_folder_for(t, question) if t else None)
                                   for t in (missing or [None])]})
        t0 = time.time()
        hits = retriever.retrieve(c, question, slug=search_slug, k=topk, **rank)
        if search_slug is None:
            hits = _merge_per_term(c, hits, question, topk, ev, None, rank=rank)
        _emit_hits(ev, "retrieve", f"re-retrieval after acquire: {len(hits)} hit(s)",
                   hits, ms=int((time.time()-t0)*1000), round=len(rounds))
        ok, missing = sufficiency(hits, question)
        ev("gate", ("evidence now covers every key term" if ok
                    else f"still missing: {', '.join(missing)}"),
           ok=ok, missing=missing, round=len(rounds))
        if acquired_here == 0:
            break

    if not hits:
        msg = ("No indexed evidence for this question and catalogue "
               "acquisition found nothing usable. Try a different "
               "wording or drop PDFs into the search folder.")
        memory.add_turn(c, session_id, mem_slug, "user", question)
        memory.add_turn(c, session_id, mem_slug, "assistant", msg)
        return {"answer": msg, "evidence": [], "rounds": rounds}
    # generate from current top hits without re-retrieving
    return _answer_with(c, mem_slug, session_id, question, hits, topk, rounds, ev,
                        temperature=temperature)


def _emit_hits(ev, phase, msg, hits, **extra):
    ev(phase, msg, hits=[{"slug": h["slug"], "title": (h["title"] or h["source_file"] or "")[:80],
                          "year": h["year"], "score": h["score"],
                          "lanes": ("vec+fts" if h["in_vec"] and h["in_fts"]
                                    else ("vec" if h["in_vec"] else "fts"))}
                         for h in hits[:8]], **extra)


def _folder_for(term, question):
    return re.sub(r"[^a-z0-9-]", "", term.lower())[:24] or _derive_folder(question)


def _answer_with(c, slug, session_id, question, hits, topk, rounds, ev, temperature=None):
    mem_block = memory.build_memory_block(c, session_id)
    turns = memory.recent_turns(c, session_id)
    messages = [{"role": "system", "content": pipeline.SYSTEM}]
    if mem_block:
        messages.append({"role": "system", "content": "MEMORY (context from earlier)\n" + mem_block})
    for t in turns:
        messages.append({"role": t["role"], "content": t["text"][:1200]})
    ev("prompt",
       f"building prompt: {len(hits)} evidence block(s), "
       f"{len(turns)} memory turn(s), memory_block={'yes' if mem_block else 'no'}",
       evidence_n=len(hits), memory_turns=len(turns),
       has_summary=bool(mem_block))
    messages.append({"role": "user", "content":
                     f"EVIDENCE:\n{pipeline.format_evidence(hits)}\n\nQUESTION: {question}\n\n"
                     "Answer in English citing [E#] tags."})
    from . import llm
    ev("generate", f"asking {llm.chat_model()} to compose the grounded answer…",
       model=llm.chat_model())
    t0 = time.time()
    raw = llm.chat_llm(messages, temperature=(0.2 if temperature is None else temperature),
                       max_tokens=800,
                      on_delta=lambda txt: ev.delta(txt),
                      on_retry=lambda budget: ev("reset",
                          f"first pass had no usable text (thinking ate the budget) — retrying at {budget} tokens…"))
    raw = llm.sanitize(raw)
    ev("generate_done", f"answer composed in {time.time()-t0:.1f}s ({len(raw)} chars)",
       ms=int((time.time()-t0)*1000), chars=len(raw))
    facts = memory.extract_facts(c, session_id, slug, raw)
    if facts:
        ev("memory", f"stored {facts} durable fact(s) for topic '{slug}'", facts=facts)
    shown = re.sub(r"^\s*FACT:.*$", "", raw, flags=re.M | re.I).strip()
    if len(shown) < 40:
        shown = raw.strip()[:1500]
    if not shown:
        shown = ("The model returned no text for this query. Try again or "
                 "rephrase the question.")
    refs = [{"tag": f"E{i}", "title": h["title"], "year": h["year"], "journal": h["journal"],
             "pmcid": h["pmcid"], "doi": h["doi"], "file": h["source_file"], "slug": h["slug"],
             "excerpt": (h["text"] or "")[:280]}
            for i, h in enumerate(hits, 1)]
    memory.add_turn(c, session_id, slug, "user", question)
    memory.add_turn(c, session_id, slug, "assistant", shown, refs)
    ev("memory", "turn history updated (visible in chat history panel)")
    return {"answer": shown, "evidence": refs, "facts_stored": facts, "rounds": rounds}
