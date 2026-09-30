# search-bot

A grounded scientific RAG bot: it answers **only from indexed research PDFs**, and
every claim is cited to an `[E#]` evidence entry.

The bot talks to **any OpenAI-compatible chat endpoint** — a small local model, a
large one, llama.cpp, vLLM, Ollama, or a hosted API. Nothing in the code assumes a
particular model or vendor. All conversational state lives **outside the model** in
SQLite and is re-injected on every call, so quality depends on the corpus and the
retrieval, not on how much the model can hold in context.
Retrieval is hybrid (vector + BM25 with RRF fusion), and when the local corpus is
too thin the agent can fetch and index more papers on its own.

By default everything runs locally: no cloud API is required, and no data leaves
the machine except outbound catalogue/PDF lookups.

---

## Architecture

```
┌─────────────┐   stdio MCP    ┌──────────────────────┐
│ MCP client  │◄──────────────►│ searchbot/mcp_server │
├─────────────┤   HTTP :8181   └──────────┬───────────┘
│ Web UI      │◄──────────────►            │
│ web/*.html  │                  ┌────────▼───────────┐
└─────────────┘                  │  RAG engine        │
                                 │  searchbot/*.py    │
                                 └──┬──────────┬──────┘
              chat endpoint ◄───────┘          └──► embeddings :8082
    any OpenAI-compatible server       (embeddinggemma-300M Q8, mean pooling)
     (you choose, local or hosted)              │
                       SQLite data/searchbot.db│ chunks + FTS5(BM25) + vec0(embedder dim)
                                              │
                                 libgen-mcp ◄─┘  (LibGen + Anna's Archive +
                                               arXiv / Crossref / PubMed / EuropePMC)
```

Two OpenAI-compatible endpoints back the engine:

| role | default | what it can be |
|---|---|---|
| chat / generation | `:8080` | **any** model behind any OpenAI-compatible `/v1/chat/completions` server |
| embeddings | `:8082` | any `/v1/embeddings` server (dev default: `embeddinggemma-300M-Q8_0` + `--embeddings --pooling mean`) |

The chat model name is **auto-detected** from `/v1/models`, so swapping models
needs no code change — or no config change, just point `SEARCHBOT_CHAT_URL`
somewhere else. Embedding dimension is read back from the first response, so a
different embedder size works too (re-index after switching).

## Pipeline per question

1. **Agentic gate** (`searchbot/agent.py`) — retrieve first. If evidence is thin
   (<3 hits, or top RRF score < 0.02), automatically: search the catalogue via
   libgen-mcp, apply a title-anchor relevance gate, dedup against the `acquired`
   ledger and already-indexed titles, download up to 3 PDFs into `search/{slug}/pdf/`,
   index them, and re-retrieve. Max 2 rounds, then answer from whatever exists.
   Disable per-call with `acquire:false`.
2. **Hybrid retrieval** (`searchbot/retriever.py`) — vec0 KNN (query wrapped in the
   embeddinggemma instruction prefix) + FTS5 BM25 → **RRF fusion** → top 8 chunks.
   Year bounds, recency and citation weighting are optional on top of that
   (see [Ranking signals](#ranking-signals)).
3. **Memory assembly** (`searchbot/memory.py`) — session summary + facts ledger +
   the last 8 raw turns injected as system blocks, so the model never has to
   "remember" anything.
4. **Grounded generation** — the model answers in English citing `[E#]`; `FACT:`
   lines are extracted into a per-topic facts table, then stripped from the display.
5. **Compaction** — roughly every 12 turns, older turns are summarized into the
   session summary.

## Ranking signals

RRF alone is relevance-only. Retrieval accepts four optional signals on top of it
— per request in the web API and the MCP `ask` tool, or as flags in the CLI:

| signal | what it does |
|---|---|
| `year_after` / `year_before` | drop candidates outside a publication-year range (a doc with no recoverable year is excluded when a bound is set, rather than passing as "recent enough") |
| `recency` | multiplier `1 + w * 0.5 ** (age / SEARCHBOT_RECENCY_HALF_LIFE)` |
| `citations` | multiplier `1 + w * log1p(n)/log1p(strongest candidate)`, OpenAlex `cited_by_count` |

Both weights default to `0.0`, so an untouched query stays pure RRF and the
ordering is exactly what it was. They are **multiplicative** on purpose: adjacent
RRF ranks sit about `1e-4` apart, so an additive bonus big enough to matter is
also big enough to replace the ranking. Measured useful range is small — `0.05`
on either signal helps, `0.2` starts trading relevance for metadata
(see [Benchmarks](#benchmarks)).

```bash
.venv/bin/python scripts/ask.py ephedra --after 2015 --recency 0.05
.venv/bin/python scripts/ask.py ephedra --citations 0.05

curl -s 127.0.0.1:8181/api/ask -H 'Content-Type: application/json' \
     -d '{"slug":"ephedra","question":"…","year_after":2015,"recency":0.05}'
```

`citations` needs counts in the database, so backfill once per corpus (and
refresh occasionally — counts drift):

```bash
.venv/bin/python scripts/citations.py --dry-run     # coverage report, no requests
.venv/bin/python scripts/citations.py ephedra       # one folder
.venv/bin/python scripts/citations.py --all-again   # re-fetch everything
```

That writes `docs.citations` (and `docs.year`, when the metadata sidecar left it
blank) from OpenAlex: 40 identifiers per request, DOI first and PMID for whatever
DOI missed. Works OpenAlex does not know are stamped `openalex:not-found` so a
re-run does not re-query the same dead ends, while a failed request leaves rows
untouched and retries next time. Set `SEARCHBOT_MAILTO` to use OpenAlex's polite
pool; without it the anonymous rate limit applies.

## Benchmarks

Two sets, and only the first one is comparable to anyone else's numbers.

### nfcorpus — public questions, human judgments

BEIR's `nfcorpus`: 323 real biomedical questions from TREC 2017/2018 over 3,633
PubMed documents, with 12,334 graded relevance judgments (this split uses
grades 1 and 2) made by people who had never seen this codebase. Nothing about
your corpus is involved, so the result is a claim an outsider can re-run.

```bash
.venv/bin/python scripts/fetch_nfcorpus.py              # -> data/bench/nfcorpus/*.jsonl (gitignored)
.venv/bin/python scripts/bench_nfcorpus.py --tag gemma  # one index per embedder, reuse it by tag
```

Documents are chunked, embedded and stored through the same `indexer` functions
the PDF path uses, so this measures the engine, not a hand-tuned variant of it.
The judgments key onto nfcorpus's own documents, not onto your papers, so the
benchmark builds a throwaway database per embedder under `data/bench/` and the
real corpus is never opened.

Every baseline below is **computed on this machine over the same text and the
same qrels**, because leaderboard numbers for nfcorpus are not transferable —
they differ in chunking, title handling and stemming, and transcribing them
compares someone's setup to yours rather than two methods to each other. The
lexical baselines are document-level over the *whole* corpus text (title + body,
no chunking), which is the unit BEIR scores nfcorpus with, so the engine's
chunked lanes face each method at its strongest.

### The field, ranked

Engine rows use nomic-embed-text; the baselines need no embedder at all.

| # | system | nDCG@10 | MRR@10 | Recall@100 | HitRate@100 |
|---|---|---|---|---|---|
| 1 | **hybrid — this engine** (vec + BM25, RRF) | **0.3516** | **0.5698** | **0.2907** | **0.8421** |
| 2 | vector lane alone (nomic-embed-text) | 0.3465 | 0.5468 | 0.2806 | 0.8297 |
| 3 | BM25 via SQLite FTS5, porter stemming, whole docs | 0.3189 | 0.5233 | 0.2475 | 0.7926 |
| 4 | this engine's BM25 lane (chunked) | 0.3154 | 0.5183 | 0.2288 | 0.7802 |
| 5 | Okapi BM25, `k1=1.2 b=0.75`, no stemming | 0.3069 | 0.5151 | 0.2369 | 0.7709 |
| 6 | Okapi BM25, `k1=0.9 b=0.4` (BEIR's nfcorpus config) | 0.3051 | 0.5080 | 0.2376 | 0.7678 |
| 7 | TF-IDF cosine (Salton & Buckley), whole docs | 0.2983 | 0.4935 | 0.2349 | 0.7709 |
| 8 | Okapi BM25 on titles only | 0.2162 | 0.4085 | 0.1726 | 0.6935 |
| 9 | random document order (the floor) | 0.0098 | 0.0246 | 0.0286 | 0.4334 |

### The same engine, three public embedders

| embedder | hybrid | vector lane alone | gain over best baseline (row 3) |
|---|---|---|---|
| nomic-embed-text (768-d) | **0.3516** | 0.3465 | +10.3% |
| bge-m3 (1024-d) | 0.3414 | 0.3185 | +7.1% |
| embeddinggemma-300M (768-d) | 0.3190 | 0.1957 | +0.03% |

What the two tables say, including the part that is not flattering:

- **The engine is the best system measured here**, by 10.3% nDCG@10 over the
  strongest lexical baseline and 17.9% over TF-IDF. That is a real margin, and it
  is not a tie: `random` at 0.0098 shows the metric is not inflating anyone.
- **Almost all of that margin is the embedder, not the engine.** nomic's vector
  lane alone (0.3465) is already 8.7% over the best baseline; RRF fusion adds
  1.5% on top of it. Say this plainly to anyone selling a "state of the art
  hybrid ranker": hybrid beats BM25 by roughly however much *your embedder*
  beats BM25, and with a weak embedder it does not — embeddinggemma's hybrid
  lands at 0.3190, indistinguishable from the lexical baseline at 0.3189, because
  its dense lane (0.1957) had nothing to contribute.
- **Fusion's value is inverse to embedder quality.** Hybrid over its own dense
  lane: +1.5% with nomic, +7.2% with bge-m3, **+63%** with embeddinggemma. A weak
  embedder is carried by the lexical lane; a strong one barely uses it. Keeping
  both lanes is cheap insurance, not the win.
- **Stemming is worth more than the published BM25 tuning.** SQLite's porter
  FTS5 (0.3189) beats both hand-written Okapi variants, and BEIR's own nfcorpus
  `k1=0.9, b=0.4` is *below* the defaults at `k1=1.2, b=0.75` (0.3051 vs 0.3069).
- **Chunking costs the lexical lane ~1%** — 0.3154 chunked against 0.3189 whole-doc
  for the same method — and buys the dense lane its granularity. That trade is
  the honest cost of indexing passages instead of papers.
- **Recall@100 near 0.29 is a property of the dataset, not a defect.** nfcorpus
  averages 38 judged-relevant documents per question (median 16, max 475), so a
  100-document cutoff cannot possibly cover them. HitRate@100 — did *any*
  relevant document reach the candidate pool — is 0.84, and that pool is what the
  prompt's top 8 get selected from.

This benchmark is also what found the lexical-lane bug described in
`CHANGELOG.md`: every keyword was a mandatory quoted phrase (FTS5 space
separators mean AND), so one absent term dropped the document from the
candidate set entirely. The engine's BM25 lane sat **31% below** textbook BM25
(0.2109 vs 0.3069) and dragged hybrid down with it. Joining the terms with `OR`
lifted the lane to 0.3154 and hybrid from 0.2867 to 0.3190 on the same index.

```bash
.venv/bin/python scripts/bench_nfcorpus.py --tag nomic            # everything above
.venv/bin/python scripts/bench_nfcorpus.py --tag nomic \
    --systems hybrid bm25 vec bm25okapi tfidf random              # a subset
```

### Known-item set — retrieval over your own corpus

The second set is generated locally from **your** corpus and is gitignored like
the PDFs it comes from: it is a list of paper titles and PMIDs, which is corpus
content, not code.

```bash
.venv/bin/python scripts/build_eval_set.py --docs 120 --seed 7   # -> tests/eval/known_item.json
.venv/bin/python scripts/bench_retrieval.py --lanes hybrid bm25 vec \
       --recency 0.05 --citations 0.05 --json /tmp/bench.json
```

`build_eval_set.py` derives gold **by construction** in three query families: a
paper's own title (known-item search), a verbatim mid-document sentence, and a
short typed keyword string. It answers a different question than nfcorpus —
*can the engine find a document it is holding, when the query shares words with
it* — and says nothing about paraphrase robustness, which is the axis real
questions stress.

Whole corpus (607 papers / 20,700 chunks), 354 queries, doc-level metric — a hit
is *any* chunk of the right paper reaching the cutoff:

| config | R@1 | R@5 | R@8 | R@20 | MRR | miss |
|---|---|---|---|---|---|---|
| hybrid (vec+BM25, RRF) | 0.732 | 0.912 | 0.949 | 0.975 | 0.811 | 9 |
| BM25 only | 0.879 | 0.944 | 0.952 | 0.975 | 0.908 | 9 |
| vector only | 0.545 | 0.675 | 0.729 | 0.805 | 0.607 | 69 |
| hybrid + recency 0.05 | 0.743 | 0.895 | 0.944 | 0.975 | 0.813 | 9 |
| hybrid + citations 0.05 | **0.757** | 0.907 | **0.949** | 0.975 | **0.825** | 9 |

What the numbers say:

- **On this set BM25 alone now leads.** After the `OR` fix described above the
  lexical lane's cliff disappears — R@20 0.698 → 0.975, misses 107 → 9 — and it
  matches or beats hybrid at every cutoff except the title family, where hybrid
  edges ahead (0.899 vs 0.891). That is expected rather than damning: gold here
  is by construction, so every query shares words with its document by design and
  the set cannot show what a dense lane is for. nfcorpus can, because humans
  typed those questions without the document in front of them: there the vector
  lane scores 0.3465 nDCG@10 against 0.3154 for the lexical lane, and fusion
  lands on top of both at 0.3516.
- **BM25 still wins verbatim prose outright** — 0.871 R@1 on sentence queries
  against hybrid's 0.586, because an exact sentence is a lexical lock and the
  vector lane spends votes on near-neighbour chunks. Per-family tables print
  alongside the overall one for exactly this reason.
- **Both metadata signals help at 0.05.** Under the multiplicative form the cost
  of over-weighting grows smoothly — at `0.2` recency is still worth a point and
  citations costs four, at `1.0` they cost 7 and 19 points of R@1. The additive
  form this replaced had **no usable range**: `0.02` already cost 14 points
  (recency) or 26 (citations), and `0.2` collapsed hybrid R@1 to 0.18.

The corpus index and the query embedder must be the **same model**. Indexing
with embedder A and querying with B does not error, it just returns noise: with
the vector lane fed by a mismatched model, hybrid R@1 was **0.000** across all
354 queries — worse than disabling it, because the garbage votes occupy the top
RRF slots and push BM25's correct chunk down. Two 768-dimension models are
indistinguishable to SQLite, and a server that resolves model names loosely will
answer a request for a model it is not running, cheerfully and with HTTP 200.
`probe_embedder.py` turns that into one number: it re-embeds stored chunk text
and reports the cosine against what is in the table — 0.9999 for the model that
built the index, -0.003 for a same-dimension impostor.

```bash
.venv/bin/python scripts/probe_embedder.py                    # the live corpus
.venv/bin/python scripts/probe_embedder.py --db data/bench/nfcorpus-nomic.db
```

## Agentic trace UI

Every question runs as a background **run** (`searchbot/trace.py`) emitting typed,
timestamped events that the UI renders live:

```
start → retrieve (hits + RRF score + lanes: vec/fts/both)
      → gate (which key terms are covered / missing)
      → acquire_plan → libgen_search → download → index_done (+docs/+chunks)
      → re-retrieve → prompt (evidence N, memory turns)
      → generate → answer streams token-by-token into the bubble
      → memory → done
```

Control tokens are scrubbed from the stream; a `reset` event clears the bubble if
the model needed a retry.

- `POST /api/ask` returns a `run_id` instantly.
- `GET /api/trace?id=&since=n` long-polls for new events (3s) until the run finishes.
- The MCP `ask` tool returns the final result plus the full `agent_events` list.
- `acquire:false` runs the same trace without any catalogue fetching.

## Layout

```
searchbot/            the engine (importable package)
  config.py           ports, budgets, paths
  db.py               schema: docs / chunks / fts5 / vec0 / sessions / turns / facts / jobs
  llm.py              chat client + embedder client (batched, retrying)
  indexer.py          PDF & XML extraction, chunking, embedding
  retriever.py        hybrid vec + BM25, RRF fusion, year/recency/citation ranking
  memory.py           summary compaction, facts ledger, memory block
  pipeline.py         the answer loop (evidence → grounded answer)
  agent.py            the gate/acquire loop
  citations.py        OpenAlex citation-count backfill
  libgen.py           stdio client for libgen-mcp + table parser + term widener
  trace.py            background runs and the event stream
  mcp_server.py       stdio MCP server (JSON-RPC 2.0)
  webserver.py        127.0.0.1:8181 JSON API + static UI
web/index.html        chat UI with evidence panel
scripts/              index.py, ask.py, citations.py,
                      build_eval_set.py + bench_retrieval.py (known-item set),
                      fetch_nfcorpus.py + bench_nfcorpus.py (public benchmark),
                      probe_embedder.py, start_servers.sh
tests/                stdlib unittest suite — no model server, no corpus, no network
tests/eval/           generated locally from your corpus (gitignored)
.github/workflows/    CI: the suite on 3.11 / 3.12 / 3.13
CHANGELOG.md          dated update log
searchbot_mcp.py      standalone MCP entry point for clients that scrub cwd/PYTHONPATH
web_run.py            starts the web UI
```

Not committed (see `.gitignore`): the PDF/XML corpus under `search/`, model weights
in `models/`, the `bin/libgen-mcp` binary, and all runtime state under `data/`.

## Getting started

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. point the bot at a chat model — any OpenAI-compatible server works.
#    e.g. llama.cpp:  llama server -m <your-model> --port 8080 --host 127.0.0.1
#    e.g. Ollama:    export SEARCHBOT_CHAT_URL=http://127.0.0.1:11434/v1
export SEARCHBOT_CHAT_URL=http://127.0.0.1:8080/v1     # optional; this is the default

# 2. bring up an embeddings server on :8082 (embeddinggemma is the reference config;
#    scripts/start_servers.sh is an example, not a requirement)
scripts/start_servers.sh

# 3. drop PDFs into a topic folder and index it
.venv/bin/python scripts/index.py ephedra

# 4. use it
.venv/bin/python web_run.py                   # UI at http://127.0.0.1:8181
.venv/bin/python scripts/ask.py ephedra       # CLI chat
.venv/bin/python searchbot_mcp.py             # MCP server
```

### Adding material

Each topic is `search/{slug}/`. Drop PDFs anywhere inside (`pdf/`, `fulltext_xml/`,
…) — the indexer walks `*.pdf` and `*.xml` recursively, joins metadata from any
`*.csv` sidecar (title / DOI / PMCID / year), chunks at ~1100 chars with overlap,
hard-splits oversized chunks, embeds, and stores. Re-running `index` on a folder
skips already-stored files, so it is safe to run repeatedly.

### Configuration

All via environment variables, no code edits needed:

| variable | default | meaning |
|---|---|---|
| `SEARCHBOT_CHAT_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible chat endpoint |
| `SEARCHBOT_CHAT_MODEL` | *(auto-detect)* | pin a model name instead of detecting from `/v1/models` |
| `SEARCHBOT_EMBED_URL` | `http://127.0.0.1:8082/v1` | OpenAI-compatible embeddings endpoint |
| `SEARCHBOT_EMBED_MODEL` | *(omitted from request)* | model name for multi-model embed servers |
| `SEARCHBOT_QUERY_INSTRUCT` | embeddinggemma `Instruct: …\nQuery: ` | retrieval prefix added to queries only |
| `SEARCHBOT_API_KEY` | *(none)* | Bearer token for a keyed provider, sent to both endpoints; a local server ignores it |
| `SEARCHBOT_RECENCY_WEIGHT` | `0.0` | default recency multiplier for every query |
| `SEARCHBOT_RECENCY_HALF_LIFE` | `10` | years for a paper's recency score to halve |
| `SEARCHBOT_CITATION_WEIGHT` | `0.0` | default citation multiplier for every query |
| `SEARCHBOT_OPENALEX_URL` | `https://api.openalex.org/works` | citation-count source |
| `SEARCHBOT_OPENALEX_BATCH` | `40` | identifiers per request |
| `SEARCHBOT_MAILTO` | *(none)* | contact for OpenAlex's polite pool (higher rate limit) |
| `SEARCHBOT_LIBGEN_BIN` | `bin/libgen-mcp` | path to the libgen-mcp binary |
| `SEARCHBOT_SOCKS` | `socks5h://127.0.0.1:1090` | proxy for catalogue traffic only; `export SEARCHBOT_SOCKS=` to go direct |
| `SEARCHBOT_CHAT_PORT` | `8080` | health-check port used by `start_servers.sh` |
| `SEARCHBOT_LLAMA_BIN` | `~/.local/bin/llama` | llama.cpp binary used by `start_servers.sh` |
| `SEARCHBOT_EMBED_GGUF` | `models/embeddinggemma-300M-Q8_0.gguf` | weights used by `start_servers.sh` |

### Bring your own model

The chat and embedding endpoints are independent and both are plain
OpenAI-compatible HTTP, so nothing has to be local:

```bash
# one server serving both chat and embeddings (Ollama)
export SEARCHBOT_CHAT_URL=http://127.0.0.1:11434/v1
export SEARCHBOT_EMBED_URL=http://127.0.0.1:11434/v1
export SEARCHBOT_EMBED_MODEL=qwen3-embedding:0.6b
export SEARCHBOT_QUERY_INSTRUCT=          # no instruction prefix

# a hosted API
export SEARCHBOT_CHAT_URL=https://api.example.com/v1
export SEARCHBOT_CHAT_MODEL=their-model
export SEARCHBOT_EMBED_URL=https://api.example.com/v1
export SEARCHBOT_EMBED_MODEL=their-embedding-model
export SEARCHBOT_API_KEY=sk-…             # Bearer, both endpoints
export SEARCHBOT_QUERY_INSTRUCT=
```

A server that hosts many models behind one URL (Ollama, Unsloth Studio, vLLM with
`--model-impl` swarms) is the same case: one `CHAT_URL`/`EMBED_URL`, and pick per
request with `SEARCHBOT_CHAT_MODEL` / `SEARCHBOT_EMBED_MODEL`.

Watch the dimension when switching embedders. Deleting and re-indexing is forced
only when the *size* changes — `sqlite-vec` rejects the query — but two 768-d
models are not the same space. Indexing with embeddinggemma and querying with
nomic-embed returns confident nonsense with no error anywhere: measured, that
mistake costs every bit of rank-1 precision (see [Benchmarks](#benchmarks)). If
you swap GGUFs of the same size, re-index anyway.

What the engine sends is the bare minimum of the spec: `{"input": [...]}`
(plus `model` only when you set it) to `/embeddings`, and a standard
`/chat/completions` payload. Vectors are L2-normalized client-side, and the
sqlite-vec column dimension is taken from the first response — so any embedder
size works, but the dimension is fixed per table, so **delete `data/searchbot.db`
and re-index after switching embedders**.

Switching embedders has one quiet failure mode. A different *dimension* errors
out; a different model of the *same* dimension does not — it just retrieves
noise. Multi-model servers make this easy to hit: measured here, a request for
`bge-m3` returned nomic-embed-text's 768-d vectors, and a request for a model
that is not on the server at all returned HTTP 200 with someone else's vectors.
`scripts/probe_embedder.py` settles it in one number by re-embedding text that is
already indexed and reporting the cosine against the stored vector: 0.9999 is the
model that built the index, 0.00 is not.

If your embedder wants its own query prefix (bge, e5, gte), put it in
`SEARCHBOT_QUERY_INSTRUCT`; it is applied to queries only, never to indexed passages.

MCP clients register the server by launching `python -m searchbot.mcp_server` from
the project root with a virtualenv that has the deps installed; it exposes
`list_searches`, `add_search`, `ask`, `index_status`, `libgen_search`,
`libgen_download`.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -t tests
```

The suite builds its own throwaway database in a temp directory, swaps the
embedder for a deterministic token-hash vectoriser and stubs `requests` wherever
an HTTP client is involved — so it needs no model server, no corpus and no
network. Coverage: the control-token scrubber and chat retry ladder, the
vec+BM25+RRF fusion and every ranking option, the acquire gate's term coverage,
the OpenAlex backfill (lanes, batching, dead-end stamping), schema migration, the
HTTP API's routing and trace, and the MCP tool schemas plus JSON-RPC loop.

What changed and when is in [CHANGELOG.md](CHANGELOG.md).

## Notes from building this

Measured with the reference setup (llama.cpp chat + embeddinggemma-300M-Q8_0):

- **Pooling matters.** With `embeddinggemma-300M-Q8_0`: `--pooling mean` gives
  0.73 similarity for related text vs 0.42 for unrelated. `rank`/`last` produce
  degenerate zero vectors; `cls` gives no separation (0.81/0.81). Use `mean`.
- **Embeddings need their own server.** A chat llama.cpp server returns 501 for
  `/v1/embeddings` unless launched with `--embeddings`, which is why `:8080` and
  `:8082` are separate. Not a problem if you use Ollama, vLLM, or a hosted API,
  which serve both from one endpoint.
- **sqlite-vec dimension is fixed per table.** It is read from the first embedding
  response, so switching embedders means deleting and re-indexing the corpus.
  Switching to a different model of the *same* dimension needs no re-index and so
  produces no error — retrieval just degrades to noise. `probe_embedder.py` checks.
- **libgen-mcp returns markdown.** `search` output is a markdown table, parsed to
  JSON by `libgen.py`; `results_per_page` accepts only 25/50/100; `download` needs
  `md5`/`doi`/`isbn` plus a `path` confined by `LIBGEN_MCP_ALLOWED_DOWNLOAD_DIRS`.
  Anna's Archive is already federated into the results.
- Catalogue traffic is routed through a SOCKS proxy only in the libgen-mcp
  subprocess — never in the Python process.
- sqlite-vec KNN and FTS5 behaviour verified on Python 3.12.

## Caveats

- PDFs come from third-party catalogues. **You are responsible for the rights to
  whatever you download**, which is why the corpus and the binary are gitignored.
- The web UI binds to `127.0.0.1` only — it has no authentication. Don't expose it.
- Grounding is enforced by the pipeline (citation validation, fact extraction,
  retry/reset) rather than by the model's own judgment, which keeps weak models
  honest and strong models equally cited. `chat_llm()` retries on empty or
  control-token-garbage output; that path exists for tool-trained local GGUFs and
  is a no-op for well-behaved APIs.
- Answer quality tracks the corpus: expect a weak answer when nothing indexed has
  anything to say.

## License

GNU GPL-3.0-only — see [LICENSE](LICENSE). Every source file carries the matching
SPDX header. Downloaded papers, model weights and any other material you put under
`search/` or `models/` stay outside the repository and outside this license; you
are responsible for their terms.
