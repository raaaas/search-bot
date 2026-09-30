# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Central config for the search-bot RAG engine."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root
SEARCH_DIR = ROOT / "search"                           # /{search} folders live here
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "searchbot.db"
LIBGEN_BIN = os.environ.get("SEARCHBOT_LIBGEN_BIN", str(ROOT / "bin" / "libgen-mcp"))
LIBGEN_DOWNLOADS = DATA_DIR / "libgen_downloads"

CHAT_URL = os.environ.get("SEARCHBOT_CHAT_URL", "http://127.0.0.1:8080/v1")
# When SEARCHBOT_CHAT_MODEL is unset, llm.py auto-detects the loaded model
# from /v1/models, so swapping models doesn't need config changes.
CHAT_MODEL = os.environ.get("SEARCHBOT_CHAT_MODEL", "")
EMBED_URL = os.environ.get("SEARCHBOT_EMBED_URL", "http://127.0.0.1:8082/v1")
# Model name sent to the embeddings endpoint. llama.cpp serves one model and
# ignores it; OpenAI-compatible multi-model servers (Ollama, vLLM, hosted) require it.
EMBED_MODEL = os.environ.get("SEARCHBOT_EMBED_MODEL", "")

# Retrieval / prompt budget
CHUNK_CHARS = 1100          # ~300 tokens per chunk
CHUNK_OVERLAP = 140
MAX_CHUNKS_PER_DOC = 40     # cap pathological giant docs
RECALL_K = 30               # per lane (vec / fts) before fusion
FINAL_K = 8                 # chunks in evidence block
RECENT_TURNS = 8            # raw turns re-injected each call
SUMMARY_TRIGGER = 12        # unsummarized turns before compaction
QUERY_TIMEOUT_S = 180

# Ranking signals, applied on top of RRF. Both default to 0.0 so retrieval
# behaves exactly as before until you ask for them; weights are per-call
# overridable (see retriever.retrieve / /api/ask / the `ask` MCP tool).
#   final = rrf * (1 + recency_weight * 0.5**(age/HALF_LIFE)
#                    + citation_weight * log1p(n)/log1p(candidate_max))
# Multiplicative because adjacent RRF ranks sit ~1e-4 apart; an additive bonus
# of any usable size rewrites the ranking outright. Measured optimum is small:
# 0.05 on either signal, more than ~0.2 starts trading relevance for metadata.
RECENCY_WEIGHT = float(os.environ.get("SEARCHBOT_RECENCY_WEIGHT", "0.0"))
RECENCY_HALF_LIFE_YEARS = float(os.environ.get("SEARCHBOT_RECENCY_HALF_LIFE", "10"))
CITATION_WEIGHT = float(os.environ.get("SEARCHBOT_CITATION_WEIGHT", "0.0"))

# OpenAlex is the citation-count source: it batches 40 ids per request and
# matches this corpus on both DOI and PMID. mailto only joins the polite pool.
OPENALEX_URL = os.environ.get("SEARCHBOT_OPENALEX_URL", "https://api.openalex.org/works")
OPENALEX_BATCH = int(os.environ.get("SEARCHBOT_OPENALEX_BATCH", "40"))
OPENALEX_MAILTO = os.environ.get("SEARCHBOT_MAILTO", "")

# Retrieval instruction prefixed to QUERIES only. The "Instruct: ...\nQuery: "
# shape is what embeddinggemma is trained on; most other embedders want either
# their own prefix or none — set SEARCHBOT_QUERY_INSTRUCT="" for those.
QUERY_INSTRUCT = os.environ.get(
    "SEARCHBOT_QUERY_INSTRUCT",
    "Instruct: Given a scientific question, retrieve relevant "
    "passages from research papers\nQuery: ")
