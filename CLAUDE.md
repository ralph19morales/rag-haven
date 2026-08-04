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
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 --reasoning-parser qwen3
# Every flag here is load-bearing — read "Generation speed" below before
# changing any of them. In particular, do NOT add --speculative-config: it was
# tried, it was fast, and it corrupted answers. See below.

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
`rag.py` builds the grounded prompt (system rules + labeled context chunks + question; the
labels are `PASSAGE n` / `Cite as:` and deliberately **not** bracketed — see below) →
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
| Original (`--enforce-eager`, no prefix caching) | 19.0 tok/s | 1941 ms |
| With ngram `--speculative-config` — **REVERTED, see below** | 32.4 tok/s | 623 ms |
| Current (`--enable-prefix-caching`, no spec decode) | ~19 tok/s | ~620 ms |

- **N-gram speculative decoding was removed after it corrupted answers.** It was genuinely 1.7x,
  and it broke the output: fragments already present in the prompt got emitted twice. Measured
  over 15 answers per config on the same questions — **7 repeated fragments in 4/15 answers with
  it on, 0 in 0/15 with it off.** Real examples: `"…course of treatment" the treatment`,
  `**Whatever grave risks of injury** of injury`, and a mangled case number `G.R. No. 210445,0445`.
  A corrupted citation in a legal answer costs more than the speed is worth. **Do not re-enable it
  without re-running that comparison** (`scratchpad/frag_check.py` in the session notes, or
  re-derive: count 1-4 word fragments repeated back-to-back).
- **`--enable-prefix-caching` stays** — it was silently `False` before, is a pure win, and is
  responsible for the TTFT drop (the ~480-token system prompt is identical on every request).
- **Beware benchmarking decode by counting streamed chunks.** With spec decode several accepted
  tokens arrive in one chunk, so chunk-counting reported ~16 tok/s and made the (real, but
  unusable) 1.7x look like a regression. Use `stream_options={"include_usage": True}`.
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

### Don't give the model a citation-shaped label

The retrieved passages were once labelled `[Context 1]`, `[Context 2]`… and the system prompt
told the model its citations had to match an identifier "in a `[Context N]` header line". Both
halves of that were a trap: the label *looks* like a citation in legal writing, and the prompt
spelled the token out for the model to copy. It did — measured, **75 stray markers across 12
answers**, one answer opening every paragraph with `[Context 3] [Context 5]` instead of naming a
law. Those resolve to nothing for a reader who never sees the prompt, so they are worse than an
uncited sentence: they look like a reference and lead nowhere.

The fix is in three parts, and all three matter:
- `_format_context` labels blocks `PASSAGE n` with a separate `Cite as:` line. Unbracketed, so it
  does not read as a citation.
- The prompt points at the `Cite as:` line and forbids passage numbers by name (rule 3). Its own
  wording no longer contains a bracketed label to copy — `tests/test_prompt.py` asserts this.
- `rag.strip_source_labels()` removes any that still get through, on both the streaming and
  non-streaming paths. A prompt rule is a request; this is the guarantee.

Two things that function will not do, both learned by breaking them: it does not strip whole-text
whitespace (it runs per streamed chunk, and a chunk's trailing blank line is the paragraph break —
stripping it glued paragraphs together), and it does not collapse runs of spaces globally (that
un-nested markdown list items anywhere a label had been removed in the same chunk).

### Query metrics and the ops dashboard

`ragmed/metrics.py` appends one JSON line per answered question to `data/metrics.jsonl`
(retrieval/generation timings, chunk count, HyDE fired, error) — called from inside
`rag.answer()`, for both the streaming and non-streaming paths. Logging is best-effort and never
raises. `dashboard.py` (a separate Streamlit app, own port) reads this file plus live health probes
(vLLM reachability, GPU VRAM via `nvidia-smi`, the CPU-pinning check above, index size). It is
presented as a heads-up display (`ui/hud.py`) and deliberately does **not** use the shared
law-library theme in `.streamlit/config.toml` — Haven is read by a worried patient and stays quiet,
this is read at a glance by whoever runs the box. The override is safe only because this is its own
app on its own port. **A state must never be carried by colour, glow, or motion alone**: every
reading is legible with animation disabled, and the two failures this dashboard exists to catch
(LLM down, CPU-pinning regression) also print a full plain-text explanation. 

What it checks, and why each category exists:
- **Liveness** — vLLM reachable, index populated, GPU (`nvidia-smi`), host RAM/load (`/proc`, no
  new dependency), and engine internals from vLLM's own `/metrics` (KV usage, queue depth, prefix
  cache hit rate, decode rate derived from inter-token latency).
- **Regression** — the CPU-pinning check, plus **configuration invariants**: the dashboard reads the
  live container's flags (`docker inspect vllm`) and asserts what this deployment requires —
  speculative decoding OFF, prefix caching ON, thinking OFF, temperature > 0, `LLM_SEED >= 0`. A
  violation turns the overall verdict red and prints why. This exists because the worst regression
  this project has had passed every liveness probe: latency, GPU and index were all green while
  speculative decoding duplicated prompt fragments into answers.
- **Drift** — corpus files newer than the index. Nothing re-indexes automatically, so an
  un-ingested document is invisible: retrieval never returns it and the answer reads as a coverage
  gap rather than an operational mistake.
- **Output** — an on-demand **answer-quality canary**. Asks one known question and inspects the
  reply for the two defects that have actually shipped (leaked `[Context N]`/`PASSAGE n` labels,
  back-to-back repeated fragments) plus that it cited something. On demand only: it costs a full
  generation on the GPU users share.

**When adding a probe, put it in one of those four categories** — and prefer the last two. Liveness
is the easy kind to add and the least likely to catch the next real failure. Refresh is manual by default with an **opt-in Live toggle** (`st.fragment(run_every=…)`, 5/10/30s). This is the third attempt at auto-refresh; the two earlier ones and why they failed are in the module docstring — do not re-try a meta-refresh, it reloads the visible page and now would also wipe the canary output. Live mode defaults to OFF because this page shares a GPU with real users, and the canary is deliberately outside the fragment so it can never auto-fire.

### Never pass a bare `<svg>` to `st.html` — it is silently deleted

`st.html` sanitises with DOMPurify configured `USE_PROFILES: {html: true}` (verifiable in the
shipped bundle: `streamlit/static/static/js/Html.*.js`). The HTML profile does **not** include the
SVG namespace, and `ADD_TAGS` restores only `script` and `style`. So the surrounding `<div>`s and
`<style>` render, and the drawing is thrown away — leaving a caption floating over empty space.

This is the nastiest failure mode in this repo, because **it is invisible from the server**: no
exception, correct element tree, correct markup in `AppTest`. Every server-side check passes. It
shipped three times before anyone noticed — the hero mark, the waiting animation, and the entire
ops HUD, all rendering as blank space.

Route every graphic through `ui/svg.py::svg_img()`, which inlines it as a `data:` URI `<img>`
(`img` is in the html profile, `src` is in Streamlit's `ADD_ATTR`, and DOMPurify permits `data:`
on image tags). Consequence for callers: **the SVG becomes its own document** — page CSS cannot
reach inside it, so all styles and `@keyframes` must live in a `<style>` element within the
`<svg>`. Media queries still work there, so `prefers-reduced-motion` is still honoured.

The general lesson, which cost real time here: *"the server emitted it"* is not *"the browser
rendered it"*. For anything visual, the only real verification is a human looking at the page.

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
