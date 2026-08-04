# Haven — Local vLLM Migration

**Status:** vLLM serving verified, Haven integration complete (§3 code changes applied, live end-to-end smoke-tested against the running container)
**Date:** 2026-08-03 (integration applied 2026-08-04)
**Target host:** Ubuntu 24.04 / i9-14900K / RTX 3090 24GB ("AI Box dev machine")
**Repo:** `~/Documents/rag/rag-haven`

---

## 1. What changed

Haven's inference backend moves from **Ollama + qwen2.5** to **vLLM + Qwen3.6-27B-AWQ**, served
locally over an OpenAI-compatible API at `http://localhost:8000/v1`.

Rationale: vLLM is the engine the Private AI Box appliance will ship with. Running Haven against
it now rehearses the real deployment path rather than a development stand-in — continuous batching,
PagedAttention, and a containerised CUDA runtime that transfers unchanged to client hardware.

The retrieval stack is unaffected. Embeddings (`bge-base-en-v1.5`), Chroma, BM25, and the
cross-encoder reranker all remain CPU-side and unchanged.

---

## 2. Measured runtime baseline

Recorded from vLLM startup on the dev box. These are the real numbers for a 24GB card — use them
when sizing client appliances, not vendor estimates.

| Metric | Value |
|---|---|
| Model | `QuantTrio/Qwen3.6-27B-AWQ` (dense, 4-bit AWQ) |
| Weights resident | **19.05 GiB** |
| Peak activation | 0.40 GiB |
| KV cache available | **2.39 GiB** |
| KV pool capacity | **14,563 tokens** (was 57,344 at `--max-model-len 32768`; the drafter and the smaller window changed the block sizing) |
| Max context per request | 8,192 |
| Max concurrency @ 8K | **1.78x** |
| Generation throughput | **~19 tok/s** |
| Engine init time | ~60 s |
| vLLM version | 0.26.0 |

**Sizing note:** the commonly cited "27B fits in 16.8 GB at Q4" figure is for **GGUF / llama.cpp**.
Under vLLM + AWQ the real footprint is **19.05 GiB**, because AWQ leaves embeddings, `lm_head`,
layernorms, and the vision encoder in FP16. Do not plan capacity from GGUF numbers.

---

## 3. Required code changes in Haven

### 3.1 Disable thinking mode — **blocking**

Qwen3.6 is a reasoning model. Left on, it emits chain-of-thought into a separate `reasoning` field
and returns `content: null` when `max_tokens` is exhausted mid-reasoning. Observed on the first
smoke test: a two-sentence question consumed all 200 tokens on reasoning and returned no answer.

`enable_thinking` is **not** a standard OpenAI parameter — it must go through `extra_body`, so this
cannot be handled by config alone.

```python
response = client.chat.completions.create(
    model=settings.llm_model,
    messages=messages,
    temperature=settings.llm_temperature,
    top_p=settings.llm_top_p,
    max_tokens=settings.llm_max_tokens,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

For grounded retrieval this is the correct default: the relevant statute has already been retrieved
and reranked, so extended reasoning adds latency and gives the model more room to drift from the
retrieved context. Keep thinking behind a per-query flag for genuinely hard questions; do not make
it the default.

### 3.2 Sampling parameters — **blocking**

**Do not use `temperature=0`.** Greedy decoding causes repeated-token loops across the Qwen3 family.
If Haven currently pins temperature to 0 for deterministic RAG output (the usual and normally
correct instinct), it must change.

| Parameter | Value |
|---|---|
| `temperature` | 0.7 |
| `top_p` | 0.95 |
| `top_k` | 20 |

Consequence: Haven's outputs are no longer deterministic. Regression tests asserting exact output
strings will need to move to semantic or structural assertions.

### 3.3 Client timeout

At ~19 tok/s a 1024-token answer takes over 50 seconds. A default 30 s HTTP timeout will surface
as failures that look like retrieval bugs. Set the client timeout to **180 s**.

### 3.4 `max_tokens` floor

`max_tokens` caps *total* generation. 200 is too low. Use **1024** for RAG answers.

---

## 4. Configuration (`.env`)

> Variable names below are provisional. Confirm against Haven's actual config loader before use:
> `grep -rhoE "os\.(getenv|environ)[.\[(]['\"][A-Z_]+" --include="*.py" .`

```bash
# ---- LLM (local vLLM) ----
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=not-needed
LLM_MODEL=QuantTrio/Qwen3.6-27B-AWQ

# Qwen3 loops on greedy decode — do not set temperature to 0
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.95
LLM_TOP_K=20
LLM_MAX_TOKENS=1024
LLM_TIMEOUT=180

# ---- Embeddings (CPU, unchanged) ----
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=cpu

# ---- Reranker ----
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_N=10

# ---- Vector store ----
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=haven

# ---- Retrieval ----
RETRIEVAL_TOP_K=20
HYBRID_DENSE_WEIGHT=0.7
HYBRID_SPARSE_WEIGHT=0.3

# ---- Runtime ----
LOG_LEVEL=INFO
```

Confirm `.env` is gitignored before the first push:

```bash
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
```

---

## 5. Running the inference server

```bash
docker run -d --name vllm --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model QuantTrio/Qwen3.6-27B-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 2 \
  --enforce-eager \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 \
  --reasoning-parser qwen3

docker update --restart unless-stopped vllm
```

### Why each flag is present

| Flag | Reason | Removable? |
|---|---|---|
| `--limit-mm-per-prompt '{"image":0,"video":0}'` | 27B is multimodal; the vision encoder reserves VRAM even when unused. Largest single saving. | Only if document-image ingestion is added |
| `--enable-prefix-caching` | The system prompt (~480 tokens) is identical on every request and was being re-prefilled each time. Was silently `False` before. | No reason to |
| `--enforce-eager` | Skips CUDA graph capture. Costs ~15–20% throughput — but see below: graphs do not fit alongside the drafter. | Only with more VRAM |
| `--kv-cache-dtype fp8_e5m2` | Halves KV cache memory. `e4m3` requires Ada+ and calibration scales. | See §6 accuracy caveat |
| `--gpu-memory-utilization 0.93` | Leaves headroom for the GNOME desktop (~562 MiB) | Raising it does **not** buy enough for CUDA graphs — tested at 0.95 and 0.96 |
| `--max-model-len 8192` | Real prompts are ~1.9k tokens + 1024 output + history. 32768 reserved KV nothing used. | Lower only with care |
| `--max-num-seqs 2` | Concurrency traded for context on a constrained card | Raise with more VRAM |

### Why `--enforce-eager` stays, despite costing ~15–20%

CUDA graphs and the n-gram drafter compete for the same VRAM, and on a 24 GiB
card with 19.05 GiB of weights resident there is only enough for one:

| Attempt | KV cache available | Needed for 8192 ctx | Result |
|---|---|---|---|
| graphs + spec, `--max-num-seqs 8`, util 0.93 | 0.08 GiB | 1.33 GiB | engine refused to start |
| graphs + spec, `--max-num-seqs 2`, util 0.96 | 0.94 GiB | 1.33 GiB | engine refused to start |
| graphs + spec, `--max-num-seqs 2`, util 0.95 | 0.71 GiB | 1.33 GiB | engine refused to start |
| **eager + spec (current)** | **2.39 GiB** | 1.33 GiB | **runs** |

Graphs would fit at `--max-model-len 4096`, but that is too near the real prompt
size to be safe once conversation history is included. Speculative decoding is
worth ~1.7x against the graphs' ~1.15x, so it gets the memory.

Two traps worth recording, both cost time here:

- `--cuda-graph-sizes` **does not exist** in vLLM 0.26. Passing it makes the
  container exit on an argparse error whose output looks nothing like one, which
  is easy to misread as an OOM.
- **Do not benchmark decode by counting streamed SSE chunks.** With speculative
  decoding several accepted tokens arrive in one chunk, so chunk-counting
  reports ~16 tok/s and makes a 1.7x speedup look like a regression. Take the
  count from `stream_options={"include_usage": True}`.

### Health check

```bash
curl http://localhost:8000/v1/models

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"QuantTrio/Qwen3.6-27B-AWQ",
       "messages":[{"role":"user","content":"In two sentences, what is a data residency requirement?"}],
       "temperature":0.7,"top_p":0.95,"max_tokens":512,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Expect `finish_reason: "stop"` and non-null `content`. `content: null` with populated `reasoning`
means thinking mode is still enabled.

---

## 6. Known constraints and risks

### fp8 KV cache accuracy — **verify before pilot**

vLLM warns on startup that fp8 KV storage *"may cause accuracy drop without a proper scaling
factor."* For a medical-law assistant, a degraded or fabricated citation is a material failure, not
a cosmetic one.

If regression failures appear in grounded-answer fidelity and cannot be traced to retrieval,
**remove `--kv-cache-dtype fp8_e5m2` and re-run before investigating further.** Context capacity
drops roughly by half; correctness takes priority in this domain.

### Throughput ceiling

~19 tok/s. Two things were tried to raise it; read both before attempting either again.

**N-gram speculative decoding — tried, measured at 1.7x (33 tok/s), and REVERTED because it
corrupted output.** Fragments already present in the prompt were emitted twice. Over 15 answers per
config on identical questions: **7 repeated fragments affecting 4/15 answers with it on, 0
affecting 0/15 with it off.** Observed: `"…course of treatment" the treatment`,
`**Whatever grave risks of injury** of injury`, and a mangled citation `G.R. No. 210445,0445`.
For a tool whose value rests on checkable citations, a corrupted G.R. number is a far worse defect
than a slow answer. If you re-try it (a newer vLLM may fix the underlying accept-path bug), re-run
that fragment comparison before trusting it.

**Removing `--enforce-eager`** would recover ~15–20%, but CUDA graphs need memory that comes
straight out of the KV pool; measured attempts at `--gpu-memory-utilization` 0.93/0.95/0.96 all
failed to start. See §5.

### Non-determinism

Sampling at `temperature=0.7` makes outputs non-reproducible. Regression assertions must not depend
on exact strings.

### Hardware ceiling

24 GB runs 27B-class models only with the vision encoder disabled and eager mode on. The
Qwen3.6-35B-A3B MoE requires 32 GB+. This is a concrete appliance-tier boundary:

- **24 GB (RTX 3090):** 27B dense, 16–32K context, ~3 concurrent — development
- **32 GB (RTX 5090):** 35B-A3B MoE class, longer context, higher concurrency — client appliance

---

## 7. Test suite expectations

Run the 164-test regression suite after the changes above.

**Expected failures** (informative, not alarming):
- Exact-string assertions — model changed *and* determinism was lost
- Latency-sensitive tests — ~19 tok/s vs. Ollama's previous throughput
- Any test that assumed `temperature=0`

**Genuine failures** (investigate):
- Retrieval precision or ranking regressions — these are model-independent and should not move
- Citation fidelity or grounding errors — check the fp8 KV caveat in §6 first
- Refusal or format-compliance breaks — may need system-prompt adjustment for Qwen3.6

The failure split is itself useful: it identifies which of the 164 tests were asserting on
model-specific phrasing rather than on retrieval correctness.

---

## 8. Open items

| Item | Notes |
|---|---|
| Confirm `.env` variable names against config loader | ✅ Done — actual names are `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `LLM_ENABLE_THINKING` (`ragmed/config.py`) — this doc's §4 names were provisional and did not match |
| Implement `extra_body` thinking toggle | ✅ Done — §3.1, `ragmed/llm.py`. Live-verified: no `reasoning`-only responses, clean `content` |
| Move sampling params off `temperature=0` | ✅ Done — §3.2, defaults now `0.7`/`0.95`/`20` |
| Re-baseline regression suite | ✅ Done — 163/164 pass. The one failure (`test_offline_embeddings.py`) needs a locally-cached embedding model and is unrelated to this migration |
| Evaluate Qwen3.8-27B | Not yet — weights not observed as released as of 2026-08-04 |
| fp8 KV accuracy verification | **Still open** — needs grounded-answer fidelity testing against the real corpus, not just a code-level check |
| Set `HF_TOKEN` | Documented in `.env.example`; not set (user's own shell env, not committed) |
| **New:** pin `embeddings.py`/`rerank.py` to `device="cpu"` | ✅ Done, not originally scoped here. Both had a code path that omitted `device="cpu"` and defaulted to CUDA; with vLLM holding ~93% of VRAM by design, that produced a `torch.OutOfMemoryError` on first use of the offline-embeddings fallback or the reranker. See §9 |

### Explicitly out of scope

Model swaps, quantization tuning, and context-length increases are **not** blockers for a client
demo. A working end-to-end pipeline on owned hardware with no network egress is the deliverable.
Further optimisation should be driven by what a prospect actually asks for.

---

## 9. Addendum — GPU memory contention with the retrieval stack (2026-08-04)

Not anticipated by §1's "the retrieval stack is unaffected" — it's true for *behavior*, but not
for *memory safety*. Embeddings and reranking are meant to run on CPU regardless of GPU
availability (they're cheap enough that it was never worth the VRAM), and `ragmed/embeddings.py`'s
primary load path already pinned `device="cpu"`. Its offline-fallback path (used when a Hugging
Face Hub call fails) did not, and `ragmed/rerank.py` never did, on either its primary or fallback
path.

That's silent on a machine with GPU headroom to spare — PyTorch just uses CUDA and it works. It
stops being silent the moment vLLM is running, because `--gpu-memory-utilization 0.93` leaves well
under a gigabyte free. The next thing that defaults to CUDA hits a hard
`torch.OutOfMemoryError`, not a graceful fallback — observed as a Streamlit crash mid-query,
traced to `embeddings.py:40` (the fallback path) and reproducible on `rerank.py` on every load.

**Fix:** `device="cpu"` added to all three load sites. Verified post-fix: `cli.py status` shows
both `LLM server: OK` and `Reranker: OK`, `torch.cuda.is_available()` still reports `True` (GPU
present, untouched), and `tests/test_rerank.py` / `tests/test_offline_embeddings.py` pass clean.

**Operational note for anyone hitting this again:** if a fix like this appears not to take effect,
check whether the process serving the request predates the code change first. Python does not
hot-reload already-imported modules, and a printed traceback shows source lines read fresh off
disk at *print* time — so a stale process can show a traceback that quotes your already-fixed line
and still be running the old, broken bytecode. Restart the process (`streamlit run app.py`,
`cli.py chat`, etc.) before assuming the fix didn't work.
