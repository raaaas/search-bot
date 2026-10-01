#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Fetch SciFact: claims, the abstract corpus, and per-claim human labels.

SciFact is the benchmark where an end-to-end RAG pipeline can be scored on
something other than relevance: each claim carries a human verdict of SUPPORT,
REFUTE or NOINFO (not enough evidence), so a system's final answer has a gold
label to be wrong against.

The canonical *test* split cannot be used for this. Its 300 claims ship with
evidence and labels withheld (they were the shared task's blind set), so the
only claims with gold verdicts are the ones here. The rows endpoint serves 100
per request, so this pages through all of them.

  corpus.jsonl           5,183 PubMed abstracts, BEIR's field layout
  claims_{train,validation}.jsonl    1,109 claims with their cited doc ids
  pairs_{train,validation}.jsonl     11,436 (claim, abstract, label) judgments

Lands in data/bench/scifact/, gitignored like the rest of data/.

Usage: .venv/bin/python scripts/fetch_scifact.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchbot import config  # noqa: E402

import fetch_nfcorpus  # noqa: E402  reuse its rows-endpoint paging loop

fetch_nfcorpus.OUT = config.DATA_DIR / "bench" / "scifact"

if __name__ == "__main__":
    fetch_nfcorpus.fetch("corpus", "BeIR/scifact", "corpus", "corpus")
    for split in ("train", "validation"):
        fetch_nfcorpus.fetch(f"claims_{split}", "bigbio/scifact",
                             "scifact_claims_source", split)
        fetch_nfcorpus.fetch(f"pairs_{split}", "bigbio/scifact",
                             "scifact_labelprediction_bigbio_pairs", split)
