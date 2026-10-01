#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Score the whole pipeline, not just the retriever: small model + evidence vs
big model alone, on SciFact claims with human verdicts.

Retrieval metrics can only say how well documents were ranked. This asks the
question the engine exists for: does a small model holding your retrieved
evidence answer better than a bigger model working from memory? Four arms, same
claims, same scorer:

    closed-book        retrieved (this engine's top-k chunks as [E#] evidence)

per model. Answers are scored against the human label — SUPPORTS / REFUTES /
NOT_ENOUGH_EVIDENCE — and the third class is the interesting one: it is the only
label a system can earn by *abstaining*, which is what grounding is supposed to
buy. Every call is appended to --raw as it happens and replayed on a re-run, so
an interrupted sweep resumes instead of restarting.

Usage:
  .venv/bin/python scripts/bench_endtoend.py --limit 50 \
      --models SC117/LFM2.5-2.6B-Uncensored-GGUF DavidAU/Qwen3.5-9B-...
  .venv/bin/python scripts/bench_endtoend.py --split train --limit 200
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchbot import config, db, llm, pipeline, retriever  # noqa: E402
from bench_index import index_beir_corpus, verify_embedder  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=str(config.DATA_DIR / "bench" / "scifact"))
ap.add_argument("--split", default="validation", choices=["train", "validation"])
ap.add_argument("--tag", default="nomic", help="index name (per embedder)")
ap.add_argument("--models", nargs="+", required=True, help="chat model ids to compare")
ap.add_argument("--arms", nargs="+", default=["closed", "retrieved"],
                choices=["closed", "retrieved"])
ap.add_argument("--lanes", nargs="+", default=["hybrid"], choices=["hybrid", "bm25", "vec"])
ap.add_argument("--topk", type=int, default=8, help="evidence chunks given to the model")
ap.add_argument("--max-tokens", type=int, default=800,
                help="answer budget; reasoning models spend it on thinking first "
                     "and will otherwise return an empty answer")
ap.add_argument("--reindex", action="store_true", help="rebuild the index even if it exists")
ap.add_argument("--force", action="store_true",
                help="score an index the current embedder did not build")
ap.add_argument("--limit", type=int, help="only this many claims (pilot)")
ap.add_argument("--workers", type=int, default=4,
                help="chat calls in flight; the server serializes GPU work, so "
                     "raise it until the per-call time stops falling")
ap.add_argument("--raw", default=str(config.DATA_DIR / "bench" / "scifact" / "runs.jsonl"))
ap.add_argument("--json", help="write the summary here")
a = ap.parse_args()

SLUG = "scifact"
config.DB_PATH = config.DATA_DIR / "bench" / f"scifact-{a.tag}.db"
config.RECALL_K = max(config.RECALL_K, a.topk)
rows = lambda name: [json.loads(l) for l in open(Path(a.data) / f"{name}.jsonl")]

corpus = rows("corpus")
LANES = {"hybrid": ("vec", "fts"), "bm25": ("fts",), "vec": ("vec",)}[a.lanes[0]]

# ---------------------------------------------------------------- gold labels
# bigbio ships (claim, abstract, label) triples; a claim's verdict is whatever
# non-NOINFO label its cited abstracts carry. A claim with only NOINFO pairs is
# the abstention class, and one carrying both verdicts is not a question with one
# right answer, so it is dropped rather than guessed. This config calls the
# negative class CONTRADICT; other SciFact exports call it REFUTE, so both are
# accepted and anything else raises rather than silently becoming abstention.
verdicts, evidence_docs = defaultdict(set), defaultdict(set)
for split in ("train", "validation"):
    for p in rows(f"pairs_{split}"):
        if p["label"] != "NOINFO":
            verdicts[p["text_1"]].add(p["label"])
            evidence_docs[p["text_1"]].add(str(p["document_id"]))

LABELS = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES", "REFUTE": "REFUTES"}
claims, dropped = [], 0
for c_ in rows(f"claims_{a.split}"):
    v = verdicts.get(c_["claim"], set())
    if len(v) > 1:
        dropped += 1
    elif v:
        claims.append({"id": c_["id"], "claim": c_["claim"],
                       "gold": LABELS[next(iter(v))],
                       "gold_docs": sorted(evidence_docs[c_["claim"]])})
    else:
        claims.append({"id": c_["id"], "claim": c_["claim"],
                       "gold": "NOT_ENOUGH_EVIDENCE", "gold_docs": []})
if a.limit:
    claims = claims[:a.limit]
gold_dist = Counter(c["gold"] for c in claims)
floor = max(gold_dist.values()) / len(claims)          # always-guess-the-majority

# ------------------------------------------------------------------- the index
c = db.connect()
db.init(c, llm.embed_dim())
if a.reindex:
    for t in ("docs", "chunks", "chunks_fts", "chunks_vec"):
        c.execute(f"DELETE FROM {t} WHERE slug=?", (SLUG,))
    c.commit()
# Called unconditionally: index_beir_corpus skips what is already stored and
# commits as it goes, so an interrupted or partial build finishes on the next run
# instead of being silently reused as half an index.
index_beir_corpus(c, corpus, SLUG)

# Same reason as in bench_nfcorpus: a mismatched embedder does not error, it
# returns confident noise, and here that noise is what the model reads.
if a.lanes[0] != "bm25":
    cos = verify_embedder(c, SLUG)
    if cos < 0.99 and not a.force:
        sys.exit(f"STOP: {config.DB_PATH.name} was built by a different embedder "
                 f"(re-embedded at cosine {cos:.4f}). Point SEARCHBOT_EMBED_* at it, "
                 f"use another --tag, or pass --force.")
    print(f"index embedder verified: cosine {cos:.4f}")
chunk2doc = {r["id"]: r["source_file"] for r in c.execute(
    "SELECT ch.id AS id, d.source_file AS source_file FROM chunks ch "
    "JOIN docs d ON d.id=ch.doc_id WHERE ch.slug=?", (SLUG,))}

SYSTEM = ("You check whether a biomedical claim is supported by the evidence. "
          "Reply with exactly one word: SUPPORTS, REFUTES or NOT_ENOUGH_EVIDENCE. "
          "No explanation.")
PARSE = re.compile(r"NOT_ENOUGH_EVIDENCE|SUPPORTS?|REFUTES?", re.I)


def evidence(claim_text):
    """This engine's top-k chunks, rendered by the same formatter the app prompts with."""
    hits = [h for h in retriever.retrieve(c, claim_text, slug=SLUG, k=a.topk,
                                          lanes=LANES)
            if chunk2doc.get(h["id"])][:a.topk]
    docs = [chunk2doc[h["id"]] for h in hits]
    return pipeline.format_evidence(hits), docs


def ask(claim_text, ev):                 # the model is whichever config.CHAT_MODEL the loop set
    msgs = [{"role": "system", "content": SYSTEM}]
    # The closed-book arm has to be told to answer anyway, or a model trained to
    # call a search tool just emits one: asked to judge a claim with no evidence
    # block, LFM2.5-2.6B returned <|tool_call_start|>[google(query='...')], which
    # sanitize() correctly strips to nothing. Both arms must be under equal
    # pressure to produce a verdict or the comparison measures the prompt.
    msgs.append({"role": "user", "content":
                 f"EVIDENCE:\n{ev}\n\nCLAIM: {claim_text}" if ev else
                 f"CLAIM: {claim_text}\n\nNo evidence is provided for this one. "
                 f"Judge it from what you already know."})
    raw = llm.chat_llm(msgs, temperature=0.0, max_tokens=a.max_tokens)
    m = PARSE.search(raw or "")
    if not m:
        return "UNPARSED", raw
    got = m.group(0).upper()
    return ("SUPPORTS" if got.startswith("SUPPORT") else
            "REFUTES" if got.startswith("REFUTE") else "NOT_ENOUGH_EVIDENCE"), raw


# ------------------------------------------------------------------- the sweep
done = {}
raw_path = Path(a.raw)
raw_path.parent.mkdir(parents=True, exist_ok=True)
if raw_path.exists():
    for line in open(raw_path):
        r = json.loads(line)
        # An ERROR row is a server refusal, not an answer: leaving it out of the
        # resume set means the next pass retries it instead of scoring a refusal
        # as a wrong verdict forever.
        if r["pred"] != "ERROR":
            done[(r["id"], r["model"], r["arm"])] = r["pred"]
fh = open(raw_path, "a")
lock = threading.Lock()

# Retrieval first, serially: the sqlite connection is not shareable across
# threads, it is the cheap half of the work, and having every evidence block in
# hand is what lets the hundreds of chat calls run concurrently.
gold_in_evidence = 0
gold_claim_total = sum(1 for cl in claims if cl["gold_docs"])      # abstention-class claims have none
ev_by_id = {}
t0 = time.time()
if "retrieved" in a.arms:
    for cl in claims:
        ev_text, ev_docs = evidence(cl["claim"])
        ev_by_id[cl["id"]] = ev_text
        if set(ev_docs) & set(cl["gold_docs"]):
            gold_in_evidence += 1

tasks = [(model, arm, cl) for model in a.models for arm in a.arms for cl in claims
         if (cl["id"], model, arm) not in done]
print(f"{len(tasks)} calls to run ({len(done)} already in {raw_path.name}), "
      f"{a.workers} in flight", flush=True)
counted = [0]


def run(task):
    model, arm, cl = task
    try:
        pred, text = ask(cl["claim"], ev_by_id.get(cl["id"]) if arm == "retrieved" else None)
    except Exception as e:                       # a server hiccup must not cost the sweep
        pred, text = "ERROR", str(e)[:200]
    with lock:
        done[(cl["id"], model, arm)] = pred
        fh.write(json.dumps({"id": cl["id"], "model": model, "arm": arm,
                             "pred": pred, "gold": cl["gold"],
                             "raw": (text or "")[:200]}) + "\n")
        fh.flush()
        counted[0] += 1
        if counted[0] % 25 == 0 or counted[0] == len(tasks):
            el = time.time() - t0
            print(f"  {counted[0]}/{len(tasks)} calls, {el:.0f}s "
                  f"({el / max(counted[0], 1):.1f}s/call effective)", flush=True)


# One model at a time, its claims in parallel: llm.chat_llm picks the model from
# config.CHAT_MODEL at call time, so interleaving two models across threads would
# send some requests to whichever thread set the global last.
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    for model in a.models:
        config.CHAT_MODEL = model
        list(pool.map(run, [t for t in tasks if t[0] == model]))
fh.close()

# -------------------------------------------------------------------- scoring
def score(model, arm):
    """accuracy, macro-F1 and per-class recall over the gold-labeled claims."""
    tp = Counter(); fp = Counter(); tot = Counter(); unparsed = 0; answered = 0
    for cl in claims:
        p = done.get((cl["id"], model, arm), "MISSING")
        g = cl["gold"]
        tot[g] += 1
        if p == "MISSING":
            continue
        # A refusal to parse and a server error are both failures to answer, and
        # both count against accuracy — but they are reported separately, because
        # a row whose real story is "the model would not commit to a verdict"
        # must not read as a row whose story is "the model was wrong".
        unparsed += p in ("UNPARSED", "ERROR")
        answered += p not in ("UNPARSED", "ERROR")
        tp[g] += p == g
        fp[p] += p != g
    f1, rec = [], {}
    present = [lab for lab in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_EVIDENCE") if tot[lab]]
    for lab in present:
        prec = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        r = tp[lab] / tot[lab]
        rec[lab] = r
        f1.append(2 * prec * r / (prec + r) if prec + r else 0.0)
    return {"n": len(claims), "answered": answered,
            "acc": round(sum(tp.values()) / len(claims), 4),
            "macroF1": round(sum(f1) / len(present), 4),
            "R_sup": round(rec.get("SUPPORTS", 0.0), 4), "R_ref": round(rec.get("REFUTES", 0.0), 4),
            "R_abstain": round(rec.get("NOT_ENOUGH_EVIDENCE", 0.0), 4),
            "unparsed": unparsed}


results = {f"{m.split('/')[-1][:22]}|{arm}": score(m, arm)
           for m in a.models for arm in a.arms}

print(f"\nSciFact {a.split} | {len(claims)} claims | gold {dict(gold_dist)} | "
      f"majority-class floor {floor:.3f} | dropped {dropped} ambiguous")
print(f"evidence = {a.lanes[0]} lane top-{a.topk} from {config.DB_PATH.name}"
      f" ({c.execute('SELECT COUNT(*) FROM chunks WHERE slug=?', (SLUG,)).fetchone()[0]} chunks)")
qi = config.QUERY_INSTRUCT.strip().replace("\n", " ") or "(none)"
print(f"embed={config.EMBED_URL} model={config.EMBED_MODEL or 'default'} "
      f"query_instruct={qi[:70]}")
if "retrieved" in a.arms:
    print(f"gold abstract reached the evidence block: "
          f"{gold_in_evidence}/{gold_claim_total} claims that have a gold abstract "
          f"({len(claims) - gold_claim_total} are the abstention class and have none by definition)")
print()
print(f"{'system|arm':<28}{'n':>5}{'answered':>9}{'acc':>7}{'macroF1':>9}{'R_sup':>7}"
      f"{'R_ref':>7}{'R_abstain':>10}{'unparsed':>9}")
for name, r in results.items():
    print(f"{name:<28}{r['n']:>5}{r['answered']:>9}{r['acc']:>7}{r['macroF1']:>9}"
          f"{r['R_sup']:>7}{r['R_ref']:>7}{r['R_abstain']:>10}{r['unparsed']:>9}")
print(f"\nR_abstain is the NOT_ENOUGH_EVIDENCE class — the only score a system can")
print(f"earn by declining to guess. Guessing the majority class scores {floor:.3f}.")

if a.json:
    Path(a.json).write_text(json.dumps({"dataset": "scifact", "split": a.split,
                                        "claims": len(claims), "gold": dict(gold_dist),
                                        "majority_floor": round(floor, 4),
                                        "topk": a.topk, "lane": a.lanes[0],
                                        "gold_in_evidence": gold_in_evidence,
                                        "results": results}, indent=1))
    print(f"\nwrote {a.json}")
