# Update log

Change history for search-bot, newest first. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is not
version-tagged, so entries are dated.

## 2026-10-01 — public benchmark (nfcorpus), lexical lane made disjunctive

### Added
- **nfcorpus benchmark** (`scripts/fetch_nfcorpus.py`, `scripts/bench_nfcorpus.py`) — the
  engine measured against *public* questions and human judgments instead of anything
  derived from this corpus: 323 TREC 2017/2018 biomedical questions, 3,633 PubMed
  documents, 12,334 graded judgments. nDCG@10, MRR@10, Recall@100 and HitRate@100 are
  implemented here in stdlib, at document level. nfcorpus is indexed into a throwaway
  database per embedder under gitignored `data/bench/`, so the real corpus is never
  opened and no corpus content is involved in the numbers.
- **Six baselines, all computed on this machine** (`--systems`): Okapi BM25 at the
  textbook `k1=1.2, b=0.75` and at BEIR's own nfcorpus setting `k1=0.9, b=0.4`;
  BM25 on titles only; TF-IDF cosine (Salton & Buckley); SQLite FTS5's built-in
  `bm25()` with porter stemming; and seeded random order as the floor. Each is
  document-level over the *whole* corpus text (title + body, no chunking), which is
  the unit BEIR scores nfcorpus with, so the engine's chunked lanes meet every
  method at its strongest. Nothing is transcribed from a leaderboard — published
  nfcorpus numbers differ in chunking, title handling and stemming enough to make
  the comparison hollow.
- **`scripts/probe_embedder.py`** — re-embeds text that is already stored and reports the
  cosine against the vector in the table. Two 768-dimension models are indistinguishable
  to SQLite, so this is the only way to know which embedder built an index. Exits 0 on a
  match, 1 on a same-dimension mismatch, 2 on a dimension change.

### Fixed
- **The lexical lane was conjunctive.** Every keyword was emitted as a mandatory quoted
  phrase and FTS5 separates terms by AND, so a document missing one term of the query was
  gone from the candidate set. The engine's BM25 lane scored **0.2109** nDCG@10 on
  nfcorpus while textbook BM25 over the whole corpus scored **0.3069** — the engine's
  lexical lane was 31% *below* the simplest possible baseline, and dragged hybrid down
  with it. Terms are joined with `OR` now: BM25 lane 0.2109 → 0.3154, hybrid 0.2867 →
  0.3190 on the same index, and on the known-item set BM25 R@20 went 0.698 → 0.975 with
  misses 107 → 9.

### Measured (nfcorpus test split, 323 questions, doc-level)

| # | system | nDCG@10 | MRR@10 | Recall@100 | HitRate@100 |
|---|---|---|---|---|---|
| 1 | hybrid — this engine, nomic-embed-text | **0.3516** | **0.5698** | **0.2907** | **0.8421** |
| 2 | vector lane alone, nomic-embed-text | 0.3465 | 0.5468 | 0.2806 | 0.8297 |
| 3 | BM25, SQLite FTS5 + porter, whole docs | 0.3189 | 0.5233 | 0.2475 | 0.7926 |
| 4 | this engine's BM25 lane (chunked) | 0.3154 | 0.5183 | 0.2288 | 0.7802 |
| 5 | Okapi BM25 `k1=1.2 b=0.75`, no stemming | 0.3069 | 0.5151 | 0.2369 | 0.7709 |
| 6 | Okapi BM25 `k1=0.9 b=0.4` (BEIR's config) | 0.3051 | 0.5080 | 0.2376 | 0.7678 |
| 7 | TF-IDF cosine, whole docs | 0.2983 | 0.4935 | 0.2349 | 0.7709 |
| 8 | Okapi BM25 on titles only | 0.2162 | 0.4085 | 0.1726 | 0.6935 |
| 9 | random order (floor) | 0.0098 | 0.0246 | 0.0286 | 0.4334 |

Same engine, other public embedders: hybrid 0.3414 with bge-m3 (dense lane 0.3185),
0.3190 with embeddinggemma-300M (dense lane 0.1957).

- **The engine ranks first of the nine**, +10.3% nDCG@10 over the strongest baseline and
  +17.9% over TF-IDF. `random` at 0.0098 is the check that the metric is not inflating it.
- **Most of that margin belongs to the embedder, not to the engine.** nomic's dense lane
  alone is already +8.7% over the best baseline; fusion contributes the remaining +1.5%.
  With embeddinggemma the hybrid reaches 0.3190 — level with the 0.3189 lexical baseline,
  because its dense lane had nothing to add. Hybrid beats BM25 by about as much as your
  embedder does.
- **Fusion's value is inverse to embedder quality:** hybrid over its own dense lane is
  +1.5% (nomic), +7.2% (bge-m3), +63% (embeddinggemma). Both lanes stay because that is
  cheap insurance, not because fusion is the win.
- **Stemming beat the published tuning.** FTS5 porter (0.3189) outscored both hand-written
  Okapi runs, and BEIR's own `k1=0.9, b=0.4` came in *below* the defaults (0.3051 vs 0.3069).
- **Chunking costs the lexical lane ~1%** (0.3154 chunked vs 0.3189 whole-doc for the same
  method) in exchange for the dense lane's finer granularity.
- Model-name resolution on a multi-model server is not trustworthy: a request for
  `bge-m3` returned nomic-embed-text's 768-d vectors (cosine 1.000000 between the two
  names' outputs, and identical scores to the nomic row, which is how it was caught), while
  a request for a model that does not exist returned HTTP 200 with a 1024-d fallback. The
  server books tokens under the requested name, so its own usage log cannot show the
  substitution. `probe_embedder.py` confirms an index's embedder before trusting a number.

### Superseded
- The `Measured` numbers in the 2026-09-30 entry were taken with the conjunctive lexical
  lane and understate every configuration; the README's known-item table now carries the
  corrected ones (hybrid R@1 0.644 → 0.732, BM25 only 0.647 → 0.879).

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
- **Bearer auth** (`searchbot/llm.py`, `SEARCHBOT_API_KEY`) — the client sent no
  `Authorization` header anywhere, so a keyed provider was simply unreachable and
  the failure looked like a model problem: a 401 on `/v1/models` was swallowed and
  the model id became `"default"`. The token now goes to chat, streaming chat,
  embeddings and model autodetect; a local server ignores the header, so one knob
  covers a mixed local + hosted setup.
- Test suite is 119 tests: the four signal tests now assert the multiplier
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
