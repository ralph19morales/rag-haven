# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fully local RAG system over Philippine medical law (statutes, PRC/DOH issuances, Supreme Court
decisions). Two interfaces share one engine: `app.py` (Streamlit web UI, "Haven") and `cli.py`
(`ingest` / `ask` / `chat` / `status`). Full narrative walkthrough: `GUIDE.md`. Migration/ops
details for the LLM backend: `HAVEN_VLLM_MIGRATION.md`.

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
  --max-model-len 32768 --gpu-memory-utilization 0.93 --max-num-seqs 2 \
  --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 --reasoning-parser qwen3

# Index the corpus (required before ask/chat/app.py return anything)
python cli.py ingest            # add & update
python cli.py ingest --reset    # wipe & rebuild — required after changing EMBED_MODEL/CHUNK_SIZE/CHUNK_OVERLAP

# Run
streamlit run app.py
python cli.py ask "question"
python cli.py chat
python cli.py status            # health check: index size, LLM server, OCR, reranker

# Tests — no corpus, no LLM, no network required
python tests/test_retrieval.py            # run a single file
for t in tests/*.py; do python "$t"; done  # run all (164 checks across 12 files)
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

Two non-obvious sampling requirements in `ragmed/config.py` / `ragmed/llm.py`, both load-bearing:
- `LLM_TEMPERATURE` must not be `0`. Greedy decoding makes Qwen3-family models loop on repeated
  tokens — the opposite of what "temperature 0 for factual RAG output" usually buys you elsewhere.
- Thinking mode must stay off (`LLM_ENABLE_THINKING=false`, sent via `extra_body` since it isn't a
  standard OpenAI param). Left on, this reasoning model spends the token budget on chain-of-thought
  and can return empty `content`.

If you edit `ragmed/embeddings.py`, `ragmed/rerank.py`, or `ragmed/llm.py` while a long-running
process (`streamlit run app.py`, `cli.py chat`) already has them imported, restart that process —
Python doesn't hot-reload already-imported modules, and a traceback printed after your edit will
still show your fixed source line (tracebacks read source fresh off disk at print time), which can
look like the fix didn't take when it's actually just a stale process.

### Configuration

Everything tunable lives in `ragmed/config.py` with env-var overrides (copy `.env.example` to
`.env`). Settings that change the *index itself* (`EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`)
require `cli.py ingest --reset`; retrieval-only settings (`TOP_K`, weights, thresholds) take effect
immediately with no re-index.

### Domain-specific parts, if retargeting to a different corpus

Only `fetch/seeds*.txt` and a few identifier regexes in `ragmed/chunking.py` (`Republic Act No. …`,
`G.R. No. …`, `Rule NN of the Rules of Court`) are Philippine-law-specific. The retrieval pipeline,
prompting, and safeguards are domain-neutral.
