#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Fetch the nfcorpus retrieval split (BEIR / MTEB) for benchmarking.

corpus  : 3,633 PubMed documents (title + abstract/MeSH text)
queries : 3,237 consumer health questions
qrels   : graded human relevance judgments (0..3) per split

Lands in data/bench/nfcorpus/ (gitignored, like the rest of data/). The rows
endpoint serves at most 100 per request, so this pages through the whole table.

Usage: .venv/bin/python scripts/fetch_nfcorpus.py
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import config  # noqa: E402

BASE = "https://datasets-server.huggingface.co/rows"
OUT = config.DATA_DIR / "bench" / "nfcorpus"
PAGE = 100


def page(url):
    """One rows-endpoint call, retried through rate limits and CDN hiccups.

    The endpoint answers 429 well before 100 requests go by, so the wait is
    exponential and honours Retry-After when the server sends one.
    """
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            wait = e.headers.get("Retry-After")
            wait = float(wait) if wait and wait.isdigit() else min(90, 3 * 2 ** attempt)
        except Exception:                           # transient CDN failures
            if attempt == 7:
                raise
            wait = min(90, 3 * 2 ** attempt)
        print(f"   retry in {wait:.0f}s", end="\r", flush=True)
        time.sleep(wait)


def fetch(name, dataset, config_name, split):
    """Page a split into data/bench/<dataset>/<name>.jsonl, resuming if told to.

    Rows are appended and flushed per page: a 429 on request 90 of 115 should
    cost a re-run 25 requests, not all 115.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = sum(1 for _ in open(path)) if path.exists() else 0
    fh = open(path, "a")
    offset, total = done, None
    while total is None or offset < total:
        url = (BASE + "?dataset=" + urllib.parse.quote(dataset, safe="")
               + f"&config={config_name}&split={split}&offset={offset}&length={PAGE}")
        blob = page(url)
        batch = blob["rows"]
        total = blob["num_rows_total"]
        if not batch:
            break
        for r in batch:
            fh.write(json.dumps(r["row"], ensure_ascii=False) + "\n")
        fh.flush()
        offset += len(batch)
        print(f"   {name}: {offset}/{total}", end="\r", flush=True)
        time.sleep(0.25)            # the endpoint rate-limits a tight loop
    fh.close()
    print(f"\n{name}: {offset} rows -> {path}")
    return [json.loads(l) for l in open(path)]


if __name__ == "__main__":
    fetch("corpus", "BeIR/nfcorpus", "corpus", "corpus")
    fetch("queries", "BeIR/nfcorpus", "queries", "queries")
    fetch("qrels_test", "mteb/nfcorpus", "default", "test")
    fetch("qrels_dev", "mteb/nfcorpus", "default", "dev")
