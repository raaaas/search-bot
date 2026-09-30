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
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import config  # noqa: E402

BASE = "https://datasets-server.huggingface.co/rows"
OUT = config.DATA_DIR / "bench" / "nfcorpus"
PAGE = 100


def page(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:                      # transient CDN failures
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch(name, dataset, config_name, split):
    rows, offset = [], 0
    while True:
        url = (BASE + "?dataset=" + urllib.parse.quote(dataset, safe="")
               + f"&config={config_name}&split={split}&offset={offset}&length={PAGE}")
        blob = page(url)
        batch = blob["rows"]
        rows.extend(r["row"] for r in batch)
        got = len(batch)
        offset += got
        total = blob["num_rows_total"]
        print(f"   {name}: {offset}/{total}", end="\r", flush=True)
        if got == 0 or offset >= total:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{name}: wrote {len(rows)} rows -> {OUT / (name + '.jsonl')}")
    return rows


if __name__ == "__main__":
    fetch("corpus", "BeIR/nfcorpus", "corpus", "corpus")
    fetch("queries", "BeIR/nfcorpus", "queries", "queries")
    fetch("qrels_test", "mteb/nfcorpus", "default", "test")
    fetch("qrels_dev", "mteb/nfcorpus", "default", "dev")
