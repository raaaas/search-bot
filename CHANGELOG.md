# Update log

Change history for search-bot, newest first. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is not
version-tagged, so entries are dated.

## 2026-09-30 — retrieval benchmark, lane ablations, signals rescaled

### Added
- **Retrieval benchmark** (`scripts/build_eval_set.py`, `scripts/bench_retrieval.py`)
  — doc-level recall@1/5/8/20, MRR and nDCG@8 over a labeled set drawn from your
  own corpus in three query families (title, verbatim sentence, keyword string),
  reported overall and per family. The generated set lands in `tests/eval/`,
  which is gitignored: it is corpus content (titles and PMIDs), not code.
- **Lane ablation** (`searchbot/retriever.py`) — `retrieve(..., lanes=("vec",))`
  measures one lane at a time, and is the fallback for a corpus whose vector
  index does not match the live embedder.

### Changed
- **`recency` and `citations` are now multiplicative** (`rrf * (1 + w*signal)`)
  instead of added to the RRF score. Adjacent RRF ranks sit ~1e-4 apart, so the
  additive form had no usable range: a `0.02` weight cost 14–26 points of recall
  at rank 1 and `0.2` collapsed it to 0.18. The measured numbers behind both
  forms are in the README's Benchmarks section.
- Test suite is 115 tests: the four signal tests now assert the multiplier
  (`1 + w*signal`) rather than the additive bonus, and lane-restricted retrieval,
  the graceful-scaling property and the per-lane subset relation are covered.

### Measured
- Reference corpus, 354 queries: hybrid R@8 0.932 vs BM25 0.686 (fusion's gain is
  candidate coverage — 17 unanswered queries against 107), and hybrid + citations
  at 0.05 lifts R@1 from 0.644 to 0.732.
- Querying an index built by a different embedder does not error — hybrid R@1
  went to **0.000** with the vector lane fed by a mismatched model, worse than
  disabling the lane (BM25 alone: 0.647).

## 2026-09-30 — license, test suite, ranking signals

### Added
- **Ranking signals** (`searchbot/retriever.py`) — `year_after` / `year_before`
  publication-year bounds, a half-life `recency` boost and a log-scaled
  `citations` boost, all additive on top of RRF. Set per request via the CLI
  (`--after/--before/--recency/--citations`), the `POST /api/ask` body or the MCP
  `ask` tool; defaults live in `SEARCHBOT_RECENCY_WEIGHT`,
  `SEARCHBOT_RECENCY_HALF_LIFE` and `SEARCHBOT_CITATION_WEIGHT`, all `0.0`, so an
  untouched query stays pure RRF and ordering is unchanged.
- **Citation backfill** (`searchbot/citations.py`, `scripts/citations.py`) —
  OpenAlex `cited_by_count` into `docs.citations`, plus `docs.year` when the
  metadata sidecar left it blank. 40 identifiers per request, DOI lane first and
  PMID for whatever DOI missed; unknown works are stamped `openalex:not-found` so
  re-runs skip them, while a failed request leaves rows unstamped and retries.
- `db.migrate()` adds the new `docs` columns to an existing corpus in place — no
  re-indexing required.
- **Test suite** (`tests/`) — 111 `unittest` tests with a throwaway database, a
  deterministic token-hash embedder and stubbed `requests`: no model server, no
  corpus, no network. Covers the control-token scrubber and chat retry ladder,
  RRF fusion and every ranking option, the acquire gate, the OpenAlex lanes and
  batching, schema migration, HTTP routing and trace, MCP schemas and the
  JSON-RPC loop.
- **CI** (`.github/workflows/ci.yml`) — the suite on Python 3.11, 3.12 and 3.13.
- `LICENSE` (GNU GPL-3.0-only) and a short SPDX notice in every source file.
- `requirements.txt`.
- README: ranking signals, citation backfill, configuration table for the new
  variables, development and license sections.

### Fixed
- `POST /api/ask` with `acquire:false` raised `NameError` on an unbound `events`
  list before the answer thread could start.
- The citation backfill query let `OR` precedence swallow the `slug` and
  `only_missing` filters, so a per-folder run considered the whole corpus.
- The not-found stamping `UPDATE` passed its third parameter outside the params
  tuple, so every unmatched doc crashed the run.
- Generic research-speak ("effects", "study", "main") leaked back into catalogue
  queries through the keyword fallback, which triggered downloads on noise.

## 2026-09-30 — first public release

### Added
- Grounded RAG engine over a local PDF/XML corpus: hybrid retrieval (sqlite-vec
  KNN + FTS5 BM25 fused with RRF), citation-forced `[E#]` answers, `FACT:`
  extraction into a per-topic ledger, session summary compaction.
- Agentic acquire loop: when evidence is thin, search the catalogue through
  libgen-mcp, gate by title relevance, dedup against the `acquired` ledger,
  download and index up to 3 papers, re-retrieve (max 2 rounds).
- Live agent trace (`searchbot/trace.py`) rendered by the web UI, and the same
  event list returned by the MCP `ask` tool.
- Web UI + JSON API on `127.0.0.1:8181`, CLI chat, and a stdio MCP server.
- `.gitignore` keeping the corpus, model weights, the libgen-mcp binary and all
  runtime state out of the repository.

### Changed
- Both model endpoints are swappable and vendor-neutral: the chat model name is
  auto-detected from `/v1/models` (or pinned with `SEARCHBOT_CHAT_MODEL`), the
  embeddings request is spec-minimal (`{"input": [...]}`, `model` only when
  configured), vectors are L2-normalized client-side, and the sqlite-vec column
  dimension is read from the first embedding response — so any
  OpenAI-compatible server, local or hosted, works.
