# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fully local RAG system over Philippine medical law (statutes, PRC/DOH issuances, Supreme Court
decisions). Three interfaces share one engine: `app.py` (Streamlit web UI, "Haven"), `cli.py`
(`ingest` / `ask` / `chat` / `status`), and `dashboard.py` (a separate Streamlit ops dashboard, own
port, for whoever operates the box rather than the end user). Full narrative walkthrough: `GUIDE.md`.
Migration/ops details for the LLM backend: `HAVEN_VLLM_MIGRATION.md`.

## Commands

```bash
# Setup
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt   # .venv/Scripts/python.exe on Windows

# The LLM server is a separate process (see "LLM backend" below) — start it before
# anything that generates text (ask/chat/app.py). Retrieval-only work doesn't need it.
docker run -d --name vllm --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 \
  vllm/vllm-openai:latest --model QuantTrio/Qwen3.6-27B-AWQ \
  --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 2 \
  --enforce-eager --enable-prefix-caching \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 --reasoning-parser qwen3
# Every flag here is load-bearing for latency — read "Generation speed" below
# before changing any of them, especially --speculative-config.

# Index the corpus (required before ask/chat/app.py return anything)
python cli.py ingest            # add & update
python cli.py ingest --reset    # wipe & rebuild — required after changing EMBED_MODEL/CHUNK_SIZE/CHUNK_OVERLAP

# Run
streamlit run app.py
python cli.py ask "question"
python cli.py chat
python cli.py status            # health check: index size, LLM server, OCR, reranker
streamlit run dashboard.py --server.port 8502   # ops dashboard: health + query metrics

# Tests — no corpus, no LLM, no network required
python tests/test_retrieval.py            # run a single file
for t in tests/*.py; do python "$t"; done  # run all (168 checks across 12 files)
```

There is no lint/format/type-check tooling configured in this repo (no ruff/black/mypy config).

**Test convention:** each `tests/test_*.py` is a standalone script, not a pytest suite (pytest
isn't a dependency and none of these files define `test_*` functions — `pytest tests/` collects
nothing). Each file exposes a `run()` that prints `PASS`/`FAIL` per case and returns a process exit
code via `if __name__ == "__main__": raise SystemExit(run())`. Run files individually with `python`,
not `pytest`. Every test encodes a bug that shipped once; add a case when you fix something.

## Architecture

### The two-journey split

Indexing and answering are fully separate pipelines that only share the on-disk store. Confusing
them is the most common mistake when changing this code:

- **Indexing** (`ragmed/ingest.py`, run via `cli.py ingest`): `loaders.py` reads files (OCR via
  `ocr.py` when a PDF has no text layer) → `chunking.py` splits on legal structure
  (Section/Article/Rule) and tags each chunk with its law/section → `embeddings.py` vectorizes →
  `vectorstore.py` writes to Chroma (`data/chroma/`) and a BM25 index (`data/bm25.pkl`). Rerunning
  `ingest` after any corpus change is required — nothing re-indexes automatically.
- **Answering** (`ragmed/rag.py`, run via `cli.py ask`/`chat`/`app.py`): reads only the store built
  above; never touches `corpus/` files directly.

### The answering pipeline, in call order

`rag.answer()` → `conversation.py` rewrites a dependent follow-up into a standalone query (skipped
for a self-contained question) → `retriever.retrieve()` runs hybrid search (dense via Chroma +
lexical via BM25, fused by `DENSE_WEIGHT`/`LEXICAL_WEIGHT`) → **in this exact order**:
`_dedupe()` → `_rerank()` (cross-encoder) → `_cap_per_source()` (no single document fills the
context) → `_ensure_clause_coverage()` (each *ask* of a compound question gets guaranteed slots) →
`rag.py` builds the grounded prompt (system rules + labeled context chunks + question) →
`llm.generate()` streams the answer.

HyDE (`retriever._hypothetical`) is a conditional side-path before retrieval: when the raw
similarity of the question (or, for compound questions, its worst-scoring individual clause) falls
below a threshold, the LLM drafts a short hypothetical answer and the system searches with that
too, to bridge lay wording to statutory vocabulary. Controlled by `HYDE_MODE` (`off`/`auto`/`always`).

Every pipeline stage above exists because of a specific measured failure (duplicate crowding,
one landmark case eating 8/10 context slots, a compound question's second half never retrieving,
etc.) — the reasoning and measurements live as comments next to each threshold in
`ragmed/config.py`, and the narrative version is in `GUIDE.md`. Don't tune a threshold without
reading why it's set where it is; several were arrived at by writing ~5 answerable and ~5
unanswerable questions and measuring the gap, not by guessing.

### LLM backend: vLLM, not Ollama — this is easy to get backwards from old context

The LLM (`Qwen3.6-27B-AWQ`) is served by **vLLM** in Docker, over an OpenAI-compatible API
(`ragmed/llm.py` uses the `openai` client, not `ollama`). Everything else in the retrieval stack —
embeddings (`bge-base-en-v1.5`) and the cross-encoder reranker (`bge-reranker-base`) — runs on
**CPU**, always, regardless of GPU availability. This is enforced with an explicit `device="cpu"`
in both `ragmed/embeddings.py` and `ragmed/rerank.py`, on every load path including fallbacks —
**if you add or touch a model-loading path in either file, keep that pin.** vLLM is configured to
claim ~93% of GPU VRAM by design (`--gpu-memory-utilization 0.93`), so anything that omits the pin
and auto-detects CUDA will hit a `torch.OutOfMemoryError` there, not gracefully fall back.

Three non-obvious sampling requirements in `ragmed/config.py` / `ragmed/llm.py`, all load-bearing:
- `LLM_TEMPERATURE` must not be `0`. Greedy decoding makes Qwen3-family models loop on repeated
  tokens — the opposite of what "temperature 0 for factual RAG output" usually buys you elsewhere.
- Thinking mode must stay off (`LLM_ENABLE_THINKING=false`, sent via `extra_body` since it isn't a
  standard OpenAI param). Left on, this reasoning model spends the token budget on chain-of-thought
  and can return empty `content`.
- The two calls that FEED RETRIEVAL — the HyDE draft and the follow-up rewrite — pass `seed=True`
  (`LLM_SEED`). They decide what gets *searched for*, so leaving them sampled made the retrieved
  passages themselves random: measured, one unchanged question asked three times returned only 2-3
  of the same 10 chunks. The answer itself is deliberately left unseeded.

### Generation speed — why these vLLM flags, and what was measured

Answering is dominated by decode, so the launch flags above matter more than anything in Python.
Measured on this box (RTX 3090, i9-14900KF), real prompts, token counts taken from the server's
`usage` block rather than by counting SSE chunks:

| Config | Decode | TTFT (warm prefix) |
|---|---|---|
| Original (`--enforce-eager`, no prefix caching, no spec decode) | 19.0 tok/s | 1941 ms |
| Current (adds `--enable-prefix-caching` + ngram `--speculative-config`) | **32.4 tok/s** | 623 ms |

- **N-gram speculative decoding is the big one (1.7x)** and it is *specific to RAG*: the answer
  quotes statutory text that is already in the prompt, so prompt-lookup drafts get accepted at a
  high rate (~2.9 tokens per streamed chunk). Do not remove it to "simplify" the command.
- **Beware benchmarking this by counting streamed chunks.** With spec decode several accepted
  tokens arrive in one chunk, so chunk-counting reports ~16 tok/s and makes the 1.7x win look like
  a regression. Use `stream_options={"include_usage": True}`.
- **`--enforce-eager` stays**, despite the ~15% that CUDA graphs would add. Graphs and the ngram
  drafter compete for the same VRAM: with 19.05 GiB of weights resident, enabling graphs left
  0.94 GiB for KV cache against the 1.33 GiB an 8192-token context needs, and the engine refused to
  start (tried at `--gpu-memory-utilization` 0.93, 0.95 and 0.96). It only fits if `--max-model-len`
  drops to ~4096, which is too close to the real prompt size (~1.9k tokens + 1024 output + history)
  to be safe. Speculative decoding is worth more than graphs, so it wins the memory.
- `--max-model-len 8192` (was 32768) — nothing here comes near 32k, and the smaller window is what
  leaves room for the drafter.
- `--cuda-graph-sizes` does **not** exist in vLLM 0.26; passing it makes the container exit on an
  argparse error that looks nothing like one.

### Process-level caches — what is warm, and what invalidates it

Retrieval's per-query cost used to be dominated by work that had nothing to do with the query.
Four things are now cached per process; when changing any of them, mind the invalidation:

- **HF offline mode** (`config.py`, set at import). sentence-transformers revalidates every model
  file against huggingface.co even when fully cached — ~10s per process, on a system whose whole
  premise is running locally. `HF_OFFLINE=true` sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` before
  huggingface_hub is imported; **it must be set before that import or it is silently ignored**,
  which is why it lives in `config.py` rather than next to the model loads. Set `HF_OFFLINE=false`
  to download a model you have not cached yet.
- **The embedder and reranker** (`embeddings.warmup()` / `rerank.warmup()`). Both are `lru_cache`d,
  so whoever triggers them pays ~6s — by default the first user to ask a question. `app.py` and
  `cli.py chat` now warm them at startup. This was the actual cause of the ~19s "retrieval" times
  in the first `metrics.jsonl` entries: model loading, billed to retrieval.
- **The Chroma client and collection** (`vectorstore._client` / `get_collection`, `lru_cache`d).
  `reset_collection()` clears the collection cache — it must, or every later call in that process
  gets a handle to a deleted collection.
- **The BM25 index** (`retriever._BM25_CACHE`). Keyed by collection size, like the on-disk pickle.
  Size does not change when a file is re-ingested with the same chunk count, so `ingest` calls
  `retriever.invalidate_bm25()` explicitly as well as deleting the pickle.

### Query metrics and the ops dashboard

`ragmed/metrics.py` appends one JSON line per answered question to `data/metrics.jsonl`
(retrieval/generation timings, chunk count, HyDE fired, error) — called from inside
`rag.answer()`, for both the streaming and non-streaming paths. Logging is best-effort and never
raises. `dashboard.py` (a separate Streamlit app, own port) reads this file plus live health probes
(vLLM reachability, GPU VRAM via `nvidia-smi`, the CPU-pinning check above, index size). It has no
auto-refresh — two automatic approaches (`st.fragment(run_every=...)`, an HTML meta-refresh) were
tried and dropped; see its module docstring for why. Refresh is a manual button.

### Restart long-running processes after editing `ragmed/` — this bit the project twice

Python does not hot-reload modules a process has already imported. `streamlit run app.py`,
`streamlit run dashboard.py`, and `cli.py chat` all import from `ragmed/` once at startup; any edit
to a file under `ragmed/` is invisible to an already-running instance of any of them until it's
restarted. This has two distinct failure signatures, both observed in this repo:

- **A crash whose traceback looks like it contradicts the fix.** Python reads traceback source
  lines fresh off disk at *print* time, not from what was actually executed — so a stale process
  can raise an exception through code that, read on screen, already looks fixed. (Happened with the
  `device="cpu"` pinning fix below.)
- **Silent nothing — the more dangerous one.** A new code path (e.g. the metrics logging above)
  simply never runs, with no error at all, because the running process's copy of the module
  predates it. (Happened with `metrics.log_query()` — a Haven session left open since before that
  call was added kept answering questions with the old `rag.py`, correctly and silently.)

Before debugging a change to `ragmed/` that "does nothing" or throws something implausible, check
`ps -o pid,lstart,cmd -p <pid>` against `stat -c '%y' ragmed/whatever.py` for every process that
might hold a stale import, before assuming the code is wrong.

### Configuration

Everything tunable lives in `ragmed/config.py` with env-var overrides (copy `.env.example` to
`.env`). Settings that change the *index itself* (`EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`)
require `cli.py ingest --reset`; retrieval-only settings (`TOP_K`, weights, thresholds) take effect
immediately with no re-index.

### Domain-specific parts, if retargeting to a different corpus

Only `fetch/seeds*.txt` and a few identifier regexes in `ragmed/chunking.py` (`Republic Act No. …`,
`G.R. No. …`, `Rule NN of the Rules of Court`) are Philippine-law-specific. The retrieval pipeline,
prompting, and safeguards are domain-neutral.
