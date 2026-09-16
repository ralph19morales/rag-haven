# Philippine Medical Law RAG

A fully local retrieval-augmented generation system over Philippine medical
law — statutes, PRC and DOH issuances, and Supreme Court decisions. Ask a
question in plain language, get an answer built **only** from indexed legal
texts, with the law and section cited for every assertion.

Nothing leaves your machine at query time: local embeddings, a local vector
store, and a local LLM. No API keys, no per-question cost.

> ### ⚖️ Not legal advice
> This is a **search and retrieval** tool. It quotes the law back to you; it
> does not interpret it, does not account for the facts of any case, and does
> not replace a licensed Philippine attorney. It can be **confidently wrong** —
> see [Known limitations](#known-limitations). Every answer ships with its
> sources precisely so a human can check them. Verify against the cited text
> before relying on anything.

---

## Interfaces

**`Haven`** — a web app for non-technical users. Opens with a greeting, offers
common situations to start from ("The hospital won't release my relative's body
until we pay"), and presents sources in plain language.

```bash
streamlit run app.py
```

**CLI** — for development, evaluation, and scripting.

```bash
python cli.py ask "What are the grounds for revoking a physician's certificate of registration?"
python cli.py chat        # interactive
python cli.py status      # health check: index size, LLM server, OCR, reranker
```

**Ops dashboard** — for whoever operates the box, not the end user. LLM
server / GPU / index health at a glance, a dedicated check that the
CPU-pinned embedding and reranker models haven't regressed onto the GPU (see
[Known limitations](#known-limitations) and `HAVEN_VLLM_MIGRATION.md` §9), and
query-latency metrics once questions have been asked. Refresh is manual (the
button) — two automatic approaches were tried and dropped, see the module
docstring in `dashboard.py`. Every query asked via the CLI or Haven logs one
line to `data/metrics.jsonl` (local only, git-ignored —
`METRICS_LOG_QUESTIONS=false` to stop logging question text) — a query is
logged only once its answer has *finished* generating, which can take up to
a minute (see [Performance expectations](#performance-expectations)).

```bash
streamlit run dashboard.py --server.port 8502
```

---

## How it works

```
INDEXING (once, and whenever documents change)
  documents → extract text (OCR for scans) → legal-aware chunking
  → embed → Chroma + BM25 index

ANSWERING (every question)
  question → [rewrite if it's a follow-up] → hybrid search (vector ⊕ BM25)
  → dedupe → cross-encoder rerank → per-source cap → per-clause coverage
  → grounded prompt → local LLM → answer + citations
```

Each retrieval stage exists because of a measured failure, not because it
sounded good. [`GUIDE.md`](GUIDE.md) documents all of them, including the
approaches that were tried and abandoned — that's the part hardest to find
elsewhere. There's also a [published version of the guide][guide-web].

[guide-web]: https://claude.ai/code/artifact/7f184e72-2c36-4811-a437-50087ffa1c10

---

## Requirements

| | |
|---|---|
| Python | 3.11+ |
| GPU / VRAM | **Required.** 24GB (RTX 3090-class) minimum for the 27B model as configured — see [Performance](#performance-expectations) |
| RAM | 16 GB is comfortable — the LLM lives in VRAM now, not system RAM; this covers the OS, Python, and the CPU-side embedding/reranker models |
| Disk | ~21 GB (LLM weights + embedding model + index) |
| [vLLM](https://docs.vllm.ai/) (Docker) | serves the LLM locally over an OpenAI-compatible API |
| Tesseract *(optional)* | OCR for scanned PDFs — `winget install UB-Mannheim.TesseractOCR` |

---

## Quick start

```bash
git clone <your-fork-url> && cd philippine-medical-law
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# ./.venv/bin/python -m pip install -r requirements.txt       # macOS/Linux

docker run -d --name vllm-qwen14b --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-14B-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 2 \
  --enforce-eager \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 \
  --reasoning-parser qwen3               # see HAVEN_VLLM_MIGRATION.md for flag rationale

cp .env.example .env                   # then edit if you want to change models
```

**Build the corpus.** The repository ships *seed lists*, not documents — the
`corpus/` contents are git-ignored, so you fetch them yourself from the
official archives:

```bash
P=.venv/Scripts/python.exe        # or ./.venv/bin/python

$P fetch/fetch_seeds.py --file fetch/seeds.txt                       --subdir lawphil
$P fetch/fetch_seeds.py --file fetch/seeds_prc.txt                   --subdir prc
$P fetch/fetch_seeds.py --file fetch/seeds_doh.txt                   --subdir doh
$P fetch/fetch_seeds.py --file fetch/seeds_jurisprudence.txt          --subdir jurisprudence
$P fetch/fetch_seeds.py --file fetch/seeds_negligence.txt             --subdir jurisprudence
$P fetch/fetch_seeds.py --file fetch/seeds_surgery.txt                --subdir jurisprudence
$P fetch/fetch_seeds.py --file fetch/seeds_billing.txt                --subdir lawphil
$P fetch/fetch_seeds.py --file fetch/seeds_billing_jurisprudence.txt  --subdir jurisprudence
$P fetch/fetch_seeds.py --file fetch/seeds_billing_issuances.txt      --subdir billing

$P fetch/split_rules_of_court.py   # REQUIRED — see below
$P cli.py ingest
```

> **Don't skip `split_rules_of_court.py`.** LawPhil serves Rules of Court 72–109
> as a single 148,000-character page. Indexed whole it produces 199 chunks
> sharing 25 section labels ("Section 2." meaning 27 different things) with no
> citable identity, and the passage answering a question loses to its own
> siblings. The splitter writes one file per rule. Re-fetching restores the
> combined page, so re-run the splitter after any re-fetch.

The fetcher is built for flaky government sites: retries with backoff, a
browser-UA fallback on 403, and tolerance for the broken TLS chains common on
`*.gov.ph`. Re-run it to retry transient failures — already-fetched files are
skipped.

---

## What's in the corpus

Roughly 154 documents / 7,934 passages across five families:

| Family | Contents |
|---|---|
| **Statutes** (LawPhil) | Medical Act, UHC Act, anti-hospital-deposit and anti-detention laws, Civil Code, Revised Penal Code, Data Privacy Act |
| **PRC issuances** | Board of Medicine law, Code of Ethics, disciplinary and licensing rules (several via OCR) |
| **DOH issuances** | Hospital licensing, patients'-rights IRRs, Magna Carta of Public Health Workers |
| **Jurisprudence** (SC E-Library) | Malpractice, informed consent, hospital liability, res ipsa, criminal negligence |
| **Bills & payment** | PhilHealth charter and No-Balance-Billing circulars, HMO cases, senior-citizen/PWD discounts, government medical assistance, small claims, and the deceased-patient chain (estate claims, release of remains) |

Seed lists live in `fetch/seeds*.txt` (`name | url`, one per line) and carry
comments explaining *why* each source is there and what was rejected. Add your
own and re-run the fetcher.

### Curate before you index

A document can download cleanly and still be the wrong *shape*. Two patterns
recur and both wreck retrieval in ways no tuning recovers:

- **Composite pages** — one URL holding many independently-numbered
  sub-documents. Split them per sub-document (see `split_rules_of_court.py`).
- **Bundled duplicate text** — official PDFs often append comparative tables
  restating the same provisions two or three times. One court-rules PDF was 52%
  duplicate. Trim before indexing; the dedupe filter can't catch these because
  multi-column layout makes the copies textually distinct.

Check character counts and outliers right after fetching. Caught later, both
cost a full `ingest --reset`.

---

## Performance expectations

**Be realistic about this before you clone it.** Measured on an RTX 3090 24GB
running `QuantTrio/Qwen3.6-27B-AWQ` under vLLM (see
[`HAVEN_VLLM_MIGRATION.md`](HAVEN_VLLM_MIGRATION.md) for the full baseline):

| Metric | Value |
|---|---|
| Weights resident | 19.05 GiB |
| KV cache available | 2.39 GiB (14,563 tokens) |
| Max context per request | 8,192 |
| Generation throughput | ~19 tok/s |

End-to-end, measured on five representative questions through the full
pipeline (`data/metrics.jsonl` holds both sides of this):

| | Before | After |
|---|---|---|
| Mean time to a complete answer | ~50 s | **~28 s** |
| Decode throughput | 19.0 tok/s | 19.0 tok/s (see note) |
| Time to first token | 1941 ms | **~620 ms** |
| Model load, per process | 15.4 s | 5.6 s (and now off the first question) |
| Retrieval, HyDE question | 14.9 s | 7.4 s |

Where that came from, largest first:

- **Prefix caching** (`--enable-prefix-caching`, silently off before) — the
  ~480-token system prompt is identical on every request and was being
  re-prefilled each time. Time-to-first-token 1941 ms → ~620 ms.
- **N-gram speculative decoding was tried and reverted.** It gave a genuine
  1.7x (19 → 33 tok/s) and corrupted the answers: fragments already in the
  prompt were emitted twice — `"…course of treatment" the treatment`, and once
  a mangled citation, `G.R. No. 210445,0445`. Measured over 15 answers per
  config: 7 repeated fragments in 4/15 answers with it on, **0 in 0/15 with it
  off.** In a tool whose entire value is citations you can check, that is not a
  trade worth making.
- **HyDE draft length** — the draft was writing to its 200-token cap and being
  truncated mid-sentence. Asking for 2-3 sentences cut it to ~95 tokens and
  halved the cost of every question that triggers it (10.5 s → ~4.8 s).
- **`HF_OFFLINE`** — sentence-transformers revalidated both local models
  against huggingface.co on every process start. ~10 s per process, on a system
  that is supposed to run entirely offline.
- **Warming models at startup** and caching the Chroma client and BM25 index
  per process, instead of rebuilding them inside every query.

Remaining levers:

| Change | Effect | Costs you |
|---|---|---|
| More VRAM (32GB+) | CUDA graphs *and* the drafter, longer context, real concurrency | Hardware |
| Shorter answers (prompt or `LLM_MAX_TOKENS`) | Linear — answers run 600-900 tokens | Detail in the answer |
| Lower `TOP_K` | Less to prefill | Less context per answer |
| `RERANK_ENABLED=false` | ~2.5 s | Ranking quality |

24GB runs 27B-class dense models only with the vision encoder disabled
(`--limit-mm-per-prompt`) and eager mode on (`--enforce-eager`). CUDA graphs
would add ~15%, but with 19.05 GiB of weights resident they leave too little KV
cache for an 8k context and the engine refuses to start — measured at
`--gpu-memory-utilization` 0.93, 0.95 and 0.96.

---

## Configuration

Everything is tunable in `ragmed/config.py` or a `.env` file (copy from
`.env.example`). The knobs that matter most:

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen3-14B-AWQ` | Any model your vLLM server is serving |
| `TOP_K` | `6` | Passages given to the LLM |
| `MAX_CHUNKS_PER_SOURCE` | `3` | Stops one landmark document filling the context with itself |
| `MIN_CHUNKS_PER_CLAUSE` | `2` | Slots reserved per *ask* of a compound question |
| `RERANK_ENABLED` | `true` | Cross-encoder relevance rescoring |
| `HYDE_MODE` | `auto` | Bridges lay wording to statutory vocabulary |
| `HISTORY_TURNS` | `3` | Prior turns replayed for follow-ups |

Changing `EMBED_MODEL`, `CHUNK_SIZE` or `CHUNK_OVERLAP` requires
`cli.py ingest --reset`. Retrieval settings take effect immediately.

**If you touch a similarity threshold, measure it.** Write ~5 questions the
corpus answers and ~5 it can't, print their *raw* similarities, and put the
threshold in the gap. If the ranges overlap, the signal is wrong — changing the
number won't save it. `GUIDE.md` §12 has the procedure.

---

## Tests

```bash
for t in tests/*.py; do .venv/Scripts/python.exe "$t"; done
```

164 checks across 12 files. Every one encodes a bug that shipped once —
duplicate crowding, the per-source cap, per-clause slot reservation, the
reranker's fail-open and flat-score guards, the HyDE clause gate, citation
labelling, the fetcher's retry budget, conversation grounding. They need no
corpus, no LLM and no network. Add a case whenever you fix something.

---

## Known limitations

Stated plainly, because they're the difference between a demo and a tool you
can trust:

- **Fluent wrong answers are the real risk**, not refusals. During development
  the system confidently denied a filing deadline that Rule 86 imposes, and
  inverted a statute three runs running — reading perfectly well while being
  wrong. Read the citations, not just the prose.
- **A reranker fixes ordering, not reach.** If the answering passage never
  retrieves, nothing downstream recovers it. Lay phrasing is the usual cause:
  the same cross-encoder separated candidates by 0.1975 on a well-worded query
  and 0.0004 on the lay version of the same question.
- **Coverage is only what you indexed.** The corpus is a curated slice, not all
  of Philippine medical law. Anything not fetched is invisible, and the system
  will say so rather than guess — which is correct behaviour that can still
  look like a wrong answer.
- **English only.** No Filipino or other local-language support.
- **Answer quality tracks the corpus far more than the model.** Adding relevant
  documents beats tuning settings — but don't add documents to fix a *ranking*
  problem; diagnose which you have first (`GUIDE.md` §14).

---

## Project layout

| Path | What it is |
|---|---|
| `ragmed/` | The engine — config, loaders, OCR, chunking, embeddings, vector store, retrieval, reranking, conversation memory, prompting, query-metrics logging |
| `fetch/` | Source fetcher, curated seed lists, the Rules-of-Court splitter |
| `evals/` | Measurements of the running system — retrieval authority, answer quality, LLM throughput. Needs the index and models, so deliberately outside `tests/` (see `evals/README.md`) |
| `WORKLOG.md` | Session handoff: current state, open items, traps found |
| `ui/` | Inline SVG/CSS chrome. `hud.py` is the heads-up display both surfaces are drawn in; `chrome.py` is Haven's layer on top of it; `robot.py` is the animated mark, idling on the hero and working during a wait |
| `tests/` | 308 regression checks |
| `cli.py` / `app.py` / `dashboard.py` | Command line / Haven web app / ops dashboard |
| `GUIDE.md` | Full explainer: how it works and what went wrong |
| `corpus/`, `data/` | Documents, index, and query-metrics log — all git-ignored; `corpus/`/`data/chroma`/`data/bm25.pkl` are regenerable, `data/metrics.jsonl` is operational history and is not |

---

## Using it for another domain

Nothing here is specific to Philippine law except the seed lists and a few
identifier patterns in `ragmed/chunking.py` (`Republic Act No. …`, `G.R. No. …`,
`Rule NN of the Rules of Court`). Swap those and the same engine indexes RFCs,
policy manuals, or internal documentation. The citation-first prompting and the
retrieval safeguards are domain-neutral.

---

## Corpus provenance and licensing

The **code** in this repository is yours to license as you choose — add a
`LICENSE` file before publishing.

The **documents** are not included and are not the project's to license. They
are fetched at build time from [LawPhil](https://lawphil.net) and the
[Supreme Court E-Library](https://elibrary.judiciary.gov.ph), which publish
Philippine legal texts under their own terms. Philippine law itself is public,
but a particular archive's compilation, formatting and headnotes may not be —
review those terms before redistributing fetched documents or hosting this
publicly. The fetcher identifies itself and rate-limits accordingly; please
keep it that way.
