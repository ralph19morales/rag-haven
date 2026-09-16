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
docker run -d --name vllm-qwen14b --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 \
  vllm/vllm-openai:latest --model Qwen/Qwen3-14B-AWQ \
  --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 2 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 --reasoning-parser qwen3
# Every flag here is load-bearing — read "Generation speed" below before
# changing any of them. In particular, do NOT add --speculative-config: it was
# tried on the previous (27B) model, it was fast, and it corrupted answers. See
# below. No --enforce-eager: at this model's size CUDA graphs capture cleanly
# alongside a full KV cache (there's no speculative-decode drafter competing
# for VRAM), so eager mode was dropped once the model changed — see below.

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
for t in tests/*.py; do python "$t"; done  # run all (308 checks across 15 files)
```

```bash
# Evals — measure the RUNNING SYSTEM (needs the index, the models, and usually
# vLLM). Not part of the test suite: slow, non-deterministic, uses the GPU.
python evals/retrieval_authority.py before   # controlling authority reaches top_k
python evals/answer_quality.py before        # leaked labels / repeated fragments
python evals/bench_llm.py before             # TTFT + decode tok/s
```

There is no lint/format/type-check tooling configured in this repo (no ruff/black/mypy config).

**Session handoff lives in `WORKLOG.md`** — current state, open items and where to pick up. Durable
design rules stay here; `WORKLOG.md` is what happened and what is unfinished.

**Before tuning any retrieval constant, sweep it with `evals/retrieval_authority.py` rather than
guessing.** The defaults for `RERANK_PROTECT_TOP` and `MAX_CHUNKS_PER_FAMILY` are the knees of
measured curves, and the reference numbers sit beside each constant in `ragmed/config.py`.

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
context) → `_cap_per_family()` (no single *kind* of authority fills it either) → `_ensure_clause_coverage()` (each *ask* of a compound question gets guaranteed slots) →
`_ensure_fused_head()` (the fused top-`RERANK_PROTECT_TOP` are guaranteed a slot — the reranker
reorders, it does not overrule) →
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

The LLM (`Qwen3-14B-AWQ`) is served by **vLLM** in Docker, over an OpenAI-compatible API
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
The numbers below are current, measured against the running default (`Qwen/Qwen3-14B-AWQ`, no
`--enforce-eager`) on this box (RTX 3090, i9-14900KF) via `evals/bench_llm.py`, real prompts built
from live retrieval, token counts from the server's `usage` block rather than by counting SSE
chunks:

| Metric | Value |
|---|---|
| Weights resident | **9.44 GiB** |
| KV cache available | **12.13 GiB** (158,992 tokens) |
| Max concurrency @ 8K context | **19.41x** |
| Decode throughput | **~79 tok/s** |
| TTFT, warm prefix (cache hit) | **~30 ms** |
| TTFT, cold prefix | **~600-690 ms** |

This project previously ran a `QuantTrio/Qwen3.6-27B-AWQ` model (see `HAVEN_VLLM_MIGRATION.md`),
where weights alone took 19.05 GiB and decode ran ~19 tok/s. Switching to the smaller 14B-class
model (commit "updated model...") didn't just shrink the model — it removed the VRAM pressure that
drove several of the flags below, so re-read this before assuming an old constraint still holds:

- **N-gram speculative decoding is not configured, and stays that way.** On the previous 27B model
  it was tried, measured at a genuine 1.7x (19 → 32 tok/s), and reverted: fragments already present
  in the prompt got emitted twice. Measured over 15 answers per config on the same questions —
  **7 repeated fragments in 4/15 answers with it on, 0 in 0/15 with it off.** Real examples:
  `"…course of treatment" the treatment`, `**Whatever grave risks of injury** of injury`, and a
  mangled case number `G.R. No. 210445,0445`. A corrupted citation in a legal answer costs more than
  the speed is worth. That finding was never re-tested against the 14B model — don't re-enable
  `--speculative-config` without re-running that comparison (re-derive: count 1-4 word fragments
  repeated back-to-back over ~15 answers).
- **`--enable-prefix-caching` stays** — it was silently `False` before it was first added, is a pure
  win, and is why a warm-prefix request (the ~480-token system prompt is identical on every request)
  answers in ~30 ms instead of ~600 ms.
- **Beware benchmarking decode by counting streamed chunks.** With speculative decoding, several
  accepted tokens can arrive in one chunk, so chunk-counting under-reports throughput — this is why
  `evals/bench_llm.py` reads `stream_options={"include_usage": True}` instead. Not currently in play
  (no spec decode configured) but the trap returns the moment it is re-tried.
- **`--enforce-eager` was dropped, and CUDA graphs now capture cleanly.** On the 27B model, graphs
  and the ngram drafter competed for the same VRAM: with 19.05 GiB of weights resident, enabling
  graphs left only 0.94 GiB for KV cache against the 1.33 GiB an 8192-token context needed, and the
  engine refused to start. At 9.44 GiB of weights and no drafter to compete with, both PIECEWISE and
  FULL CUDA graphs capture in well under a second and still leave 12.13 GiB for KV cache — the
  contention that justified eager mode doesn't exist at this model size. If speculative decoding is
  ever revisited, re-run this VRAM math before assuming graphs still fit.
- `--max-model-len 8192` (was 32768 before the vLLM migration) — nothing in this corpus's prompts
  comes near 32k, and the smaller window leaves more of the KV budget usable per concurrent request.
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

### The reranker reorders; it does not overrule

A cross-encoder is strong at "does this passage answer this question" and weak on terse statutory
text matched against a lay narrative — four lines of legislative prohibition read as less
responsive than pages of judicial discussion of the same facts. Measured: asked *"the hospital
won't release my relative's body until we pay"*, hybrid fusion ranked **RA 9439, the Anti-Hospital
Detention Law, third** and the cross-encoder demoted it to **seventeenth**. Five of six context
slots went to case law, three of them about a different statute. The only RA 9439 material left was
the IRR's list of the offence's *elements*, which the model read as a checklist of when detention is
**permitted** — and Haven answered that a hospital may lawfully withhold a body. The exact inverse
of the law, to the person least able to check it.

`_ensure_fused_head()` guarantees the fused top-`RERANK_PROTECT_TOP` (default 4) a place in the
final top_k. Same mechanism and same reasoning as `MIN_CHUNKS_PER_CLAUSE`: when two signals
disagree and each is right in a different regime, tuning their scores against each other fits the
last example you looked at — reserve capacity instead.

**The default is the knee of a measured curve, not a number fitted to the failing query.** Over 7
questions whose controlling authority is known independently (5 statute-governed, 2 deliberately
case-law-governed): `0 → 4/7`, `2 → 6/7`, `3 → 7/7`, `4 → 7/7`, `5 → 7/7`. AUTHORITY saturates at 3,
and both case-law questions hold their exact prior ranks at every setting — that second check is the
one that matters in the other direction, since a fix that merely dragged statutes upward would have
broken them.

Re-measured on the boundary-aware index with a second metric (off-topic chunks on the pure-
jurisprudence informed-consent case), **the default moved 3 → 4**: authority is 7/7 at both, but 4
is the smallest setting that also clears that case's off-topic chunk (`1/6 → 0/6`). The same sweep
corrects something the original was too kind about — going higher does not merely "buy nothing", it
**breaks**: `6 → 6/7` and `7 → 5/7` with statute+IRR lost, because a 6-slot context handed back to
the bi-encoder is the very failure the reranker exists to fix. 5 is the last safe value.

**Instruction sentences are query noise.** People append things like *"Check what philippine law is
saying about this"*. That carries no retrievable content but still shifts the query embedding —
measured, exactly that sentence pulled the Revised Penal Code to fused rank 2 (its Art. 85 concerns
the corpse of an *executed* person) and pushed RA 9439 out of the top_k, so Haven answered "not
covered" with the controlling statute sitting in the index. `_strip_meta_sentences()` removes them
from the SEARCH text only; the user's original wording still reaches the prompt, the same
separation HyDE and the follow-up rewrite already use. It is deliberately narrow — the sentence must
both lead with an assistant-directed imperative AND end on a deictic, so "Explain informed consent"
survives — and it never strips every sentence.

Note the interaction, because it is the kind that hides: once the wrong document was in the fused
head, `_ensure_fused_head` faithfully **protected** it. A guarantee applied to a polluted ranking
propagates the pollution.

**A per-file cap is not a per-authority cap.** `MAX_CHUNKS_PER_SOURCE=3` was satisfied while
jurisprudence still held four of six slots on the deceased-body question — three different case
files, one per file. `_cap_per_family()` limits a corpus family (lawphil / jurisprudence / doh /
prc / billing, via the shared `source_family()`), which is what let the statute and its IRR occupy
the context together: the prohibition lives in one and the interment and document rules in the
other. Measured, it also pulled in the licensure rules naming refusal to release cadavers as a
violation outright (RA 4226, Sec. 17) — a second operative prohibition the answer could not
previously cite.

**Do not raise it on the strength of "more families".** That number is a proxy and it can be gamed
by importing junk: a question one family genuinely answers gets *worse* when breadth is forced.
Informed consent is the case to check — it is pure jurisprudence, and it carries exactly one
off-topic chunk at cap 0, 2 and 3 alike, so the cap swaps which unrelated chunk appears rather than
adding one. Check that question, not the mean.

**A guarantee that runs after a filter must be given the filter's leavings.** `_ensure_fused_head`
and `_ensure_clause_coverage` can only promote a chunk that is still in the list handed to them.
Both caps used to truncate their overflow at `top_k`, which silently disarmed the fused-head
guarantee — measured on the deceased-body question, it protected three chunks and delivered
**one**. The caps and the coverage stage now return their remainder as a TAIL; `top_k` is the point
the backfill must reach, not the length of the result. Second, related break in the same function:
promoting a protected chunk evicted a *different* protected chunk, because it displaced the
lowest-ranked occupant without asking whether that occupant was itself protected. Whatever a
guarantee evicts must be something it was not asked to keep.

**The lesson worth keeping:** a corpus can contain the right law, dense retrieval can rank it
correctly, and the system can still answer the opposite — because a later stage silently overruled
an earlier one. When an answer is wrong, trace the ranking stage by stage before assuming the
corpus is missing something.

**And one stage further back than that: "the right document is in the context" is not "the operative
sentence is in the context".** The deceased-body answer was diagnosed as prompt-level — the model
"treating a condition as a precondition" — on the assumption that the text stating the entitlement
had reached the prompt. It had not. What reached the prompt was the offence-ELEMENTS list from the
same file, which reads exactly like a checklist of conditions the family must satisfy; the
entitlement sentence was capped out three stages earlier. AUTHORITY was green throughout, because
it only asks whether the right *file* appeared. Where a question turns on one sentence, name that
sentence in the eval — `KEY_TEXT` in `evals/retrieval_authority.py` does this.

### Chunking: a document whose headings don't match `SECTION_RE` gets no structure at all

`_split_on_sections` returns the WHOLE document as a single unit when it finds no
SECTION/ARTICLE/RULE header, and `_pack` then falls to its oversized branch. That branch used to
cut at raw character offsets. Most of this corpus lands there: DOH and PRC issuances head their
parts "I. Rationale", "B. Specific Guidelines", "1.", "2.", and Supreme Court decisions have no
section headers — **632 chunks across 19 files**, including all 175 chunks of the landmark
informed-consent case.

The cuts landed mid-word (`who refuse to execute a promisso`, `Detention occ|urs`), and the damage
is not mainly cosmetic: **the chunk's embedding is diluted by whatever the window happened to scoop
up on either side.** The operative sentence of DOH AO 2008-0001 shared a chunk with SSS/GSIS
insurance boilerplate and ranked 9th in fusion as a result. `_split_point` now breaks at the last
paragraph break under the limit, then line break, then sentence end, then space — and `_snap_start`
moves the overlap rewind forward to a word boundary, because that rewind is plain arithmetic and
otherwise reintroduces a severed word at the *head* of the next chunk. Both directions matter; the
first draft of the fix only handled the tail and the tests caught it.

Changing any of this requires `cli.py ingest --reset`. Prefer building into a shadow collection
(`COLLECTION_NAME=... cli.py ingest --reset`) and measuring before swapping — the live index keeps
serving, and the rollback is a rename. Note `data/bm25.pkl` is shared across collections and keyed
by size, so it is rebuilt on first query after a swap.

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

The same three layers were then needed a second time, for the same reason. Answers began explaining
*where* a provision was found instead of citing it: `(as cited in Republic Act No. 9439 in)`,
`(as detailed in ... source file)`, and one that reproduced a `Cite as:` header verbatim, pipe
separator included. **The prompt rule was tried first and did not work** — the leaks persisted and
merely changed shape, which is the clearest evidence in this repo that a prompt rule is a request.
`rag.strip_provenance()` is the guarantee. It is deliberately narrow, because `(as cited in <case>)`
is ordinary legal writing: a parenthetical is removed only when it also names prompt scaffolding
(source file, passage, context, `Cite as`, or the `|` that appears only in a header) or trails off
on a dangling preposition. A milder variant is still unfixed — answers opening "Based on the
provided context, …" — which is a leading hedge and matches neither this nor `trim_trailing_caveat`.

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
presented as a heads-up display (`ui/hud.py`) and deliberately does **not** use the theme in
`.streamlit/config.toml` — Haven is read slowly by a worried patient, this is read at a glance by
whoever runs the box. The override is safe only because this is its own app on its own port. **A state must never be carried by colour, glow, or motion alone**: every
reading is legible with animation disabled, and the two failures this dashboard exists to catch
(LLM down, CPU-pinning regression) also print a full plain-text explanation. 

What it checks, and why each category exists:
- **Liveness** — vLLM reachable, index populated, GPU (`nvidia-smi`), host RAM/load (`/proc`, no
  new dependency), and engine internals from vLLM's own `/metrics` (KV usage, queue depth, prefix
  cache hit rate, decode rate derived from inter-token latency).
- **Regression** — the CPU-pinning check, plus **configuration invariants**: the dashboard reads the
  live container's flags (`docker inspect vllm-qwen14b`, name configurable via
  `VLLM_CONTAINER_NAME`) and asserts what this deployment requires —
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

### Both apps are the same console now — and what that moved onto other shoulders

Haven (`app.py` + `ui/chrome.py`) and the ops dashboard (`dashboard.py` + `ui/hud.py`) are one
register: cyan on near-black, scanlines, corner brackets, glow, monospace, an animated armoured mark
(`ui/robot.py`) that idles on the hero and works during a wait. **`chrome.py` is built on
`hud.hud_css()` and imports the palette from `hud.py` rather than restating it** — keep it that way.
Two copies of a palette that has to agree is precisely how the two pages would drift apart by
accident, and `tests/test_public_surface.py` pins both the shared accent and the single-sourced
stylesheet.

This reverses an earlier rule, and the reason the earlier rule existed did not go away with it. It
read: *Haven carries a disclaimer that has to be believed, and a legal answer rendered in sci-fi
chrome reads as a toy, at which point "verify this before you rely on it" stops landing.* The change
was made deliberately; the risk is now carried **structurally instead of chromatically**, in three
places that are not decoration and must not be traded away for atmosphere:

- **The not-legal-advice line is a panel in the ALERT hue, never the accent** (`chrome.advisory()`).
  On a console, cyan is the colour of everything that is merely working, and the eye stops reading it
  within seconds. The caveat is the one thing on the page that has to survive that.
- **The caveat sentence is never set in tracked-out capitals.** Capitals are for micro-labels and
  headings. A legal caveat styled as a HUD label reads as decoration and gets skipped — which is the
  original failure this whole area is downstream of. The label above the panel may shout; the
  sentence may not.
- **Every state the mark carries in colour is also stated in words, before it.** `robot_hero(alert=…)`
  turns red when the model is unreachable, and `app.py` emits the sentence saying so *above* the
  drawing. The test asserts that ordering, not merely that both exist.

If those three go, nothing is holding the disclaimer up any more. The test file says the same thing
at the top of `run()`; read it before relaxing any of the four checks in that block.

**Never set a font family broadly without excluding the icon spans.** Streamlit draws every
`:material/…:` icon as a **ligature**: the element's text content is the literal string `smart_toy`,
and the font `"Material Symbols Rounded"` is what turns it into a glyph. `hud_css()` sets monospace
on `[class*="st-"]`, which matches those spans — so the names printed as words across *both* apps
(`smart_toy` on every assistant avatar, `keyboard_double_arrow_right` on the sidebar toggle,
`menu_book` on the sources expander), and because Streamlit gives avatars a filled background they
read as coloured blocks of text. `hud_css()` restores the family with `!important` (the emotion classes
outrank a bare attribute selector).

**Match those spans by shape, not by enumeration** — this was fixed twice. The icon component
defaults to `data-testid="stIconMaterial"` but accepts an override, and the overrides are scattered
across the bundle (`stExpanderIconError`, `stFileChipIconSpinner`, `stAlertDynamicIcon`,
`stToastDynamicIcon`, `stElementToolbarButtonIcon`…). A hand-listed set of three selectors shipped,
looked fixed, and left the expander chevron printing `keyboard_arrow_right` in the sidebar. The rule
now uses `$="Icon"` plus two prefixes, and `tests/test_hud_graphics.py` checks **every** icon testid
in the bundle against the selectors it parses out of the stylesheet — re-derive that list with
`grep -rhao 'st[A-Za-z]*Icon[A-Za-z]*' streamlit/static | sort -u` after a Streamlit upgrade. The
symbol font is followed by a real family so an icon slot carrying an emoji rather than a ligature
still resolves; font fallback is per-glyph. A related one in the same family: Streamlit fills the assistant avatar with the
theme's **orangeColor**, which on this palette is the amber warning hue — every answer opened with a
caution block beside it until `chrome.py` overrode it.

**The first page load is held by the nucleus, not by a blank page.** The embedder and reranker load
on first use — ~6s on CPU, paid by whoever opens the page first — and a blank page reads as broken
rather than as busy. `chrome.booting()` fills it with `hud.nucleus()`, the same drawing the console
uses for its verdict; `neutron()` is now a thin wrapper over it, so there is one copy of that
animation rather than two. `app.py` gates the panel on `session_state` rather than on the cache:
`warm_up` is cached per *server*, so a later visitor's placeholder is written and cleared inside one
script run and never paints.

Haven still commits to **one** ground rather than following the reader's light/dark preference, which
is why `.streamlit/config.toml` repeats the same palette into `[theme.light]` and `[theme.dark]`.
That table exists so Streamlit's own widget chrome lands on the console ground instead of fighting
the CSS layered over it. Nothing in `app.py` or `ui/chrome.py` branches on theme.

### Haven shows no machinery — that is a rule, not an accident

`app.py` is the only surface a member of the public sees. It deliberately does not state the model
names, the indexed-passage count, how many passages an answer used, relevance scores, the corpus
filename behind a citation, or the raw transport error when the LLM is down. Every one of those was
on the page at some point and each was removed for the same reason: none of it changes what a reader
should do next, and precision about the apparatus invites someone to defer to the answer instead of
checking it. An operator gets all of it from `cli.py status` and the dashboard.

What must never be stripped in the name of decluttering: the not-legal-advice line (hero *and* under
every answer), the sources section, and the instruction to open the law itself.
`tests/test_public_surface.py` pins both halves — what may not appear, and what must.

The sidebar's "It runs entirely on this computer. Nothing you type is sent anywhere." was removed,
and the test now asserts its **absence**. It is the one claim on the page that a change of
deployment silently falsifies — true of the local setup, false the moment Haven is hosted, and
unlike a wrong citation no reader can check it. The hero's "PRIVATE BY DESIGN" chip was removed for
the same reason. **No claim about the deployment now appears anywhere on the page**, and the test
asserts that in both places. If you ever want one back, derive it from where the app actually runs
rather than hard-coding it into chrome.

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
`<svg>`. Media queries still work there, so `prefers-reduced-motion` is still honoured. It also
takes no hover, so an `<img>` graphic can never carry a tooltip — anything a reader needs must be
direct-labelled into the drawing.

**And `height:auto` is a trap on any graphic wider than it is tall.** An `<img>` derives its
intrinsic ratio from the `viewBox`, so a sparkline drawn into `0 0 100 28` at `width:100%` renders
at 28% of the panel width — roughly 140px tall in a 500px column, about five times the intended
height. That is what made the ops traces read as lumpy area charts. Pass an explicit `height`, set
`preserveAspectRatio="none"` so the drawing fills the box, and give every stroke
`vector-effect="non-scaling-stroke"` or the non-uniform scale renders horizontal and vertical
strokes at different weights. `tests/test_hud_graphics.py` pins all three.

**Charts are the exception to the SVG rule — use Altair, not hand-rolled SVG.** `st.altair_chart`
renders a real Vega-Lite component rather than sanitised HTML, so it is not subject to the trap
above *and* it can carry the hover layer a chart is supposed to have. `hud.chart_theme()` puts one
on the HUD's ground; `hud.SERIES` holds the two categorical slots, which are deliberately NOT the
status hues (an amber line reads as a warning) and were picked by running the computable
colour checks rather than by eye — the measured numbers and the one recorded deviation are beside
the constants in `ui/hud.py`.

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
