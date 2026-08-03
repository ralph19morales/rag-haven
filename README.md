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

## Two interfaces

**`Haven`** — a web app for non-technical users. Opens with a greeting, offers
common situations to start from ("The hospital won't release my father's body
until we pay"), and presents sources in plain language.

```bash
streamlit run app.py
```

**CLI** — for development, evaluation, and scripting.

```bash
python cli.py ask "What are the grounds for revoking a physician's certificate of registration?"
python cli.py chat        # interactive
python cli.py status      # health check: index size, Ollama, OCR, reranker
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
| RAM | 16 GB minimum for a 7B model; 32 GB comfortable for 14B |
| Disk | ~12 GB (models + index) |
| GPU | **Strongly recommended.** See [Performance](#performance-expectations) |
| [Ollama](https://ollama.com/download) | runs the LLM locally |
| Tesseract *(optional)* | OCR for scanned PDFs — `winget install UB-Mannheim.TesseractOCR` |

---

## Quick start

```bash
git clone <your-fork-url> && cd philippine-medical-law
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# ./.venv/bin/python -m pip install -r requirements.txt       # macOS/Linux

ollama pull qwen2.5:7b-instruct        # or 14b — see Performance
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

Roughly 154 documents / 7,935 passages across five families:

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

**Be realistic about this before you clone it.** Measured on a 24-core CPU with
**no GPU**, `qwen2.5:14b-instruct` (Q4_K_M), models warm:

| Phase | Rate | Per question |
|---|---|---|
| Retrieval + rerank | — | ~4s |
| **Prefill** (reading ~2,200 tokens of context) | ~38 tok/s | **~58s** |
| **Generation** (writing ~400 tokens) | ~7.8 tok/s | **~51s** |
| | | **≈ 110s total** |

On CPU, **prefill is about half the wall clock** and scales with `TOP_K`, not
with model size. Levers, roughly in order of impact:

| Change | Effect | Costs you |
|---|---|---|
| Run on a GPU | ~5–10s per answer | Hardware |
| `qwen2.5:7b-instruct` | ~2× faster | Instruction adherence across the 8 prompt rules |
| Lower `TOP_K` | ~7s per passage dropped | Less context per answer |
| Cap answer length | linear | Shorter answers |
| `RERANK_ENABLED=false` | ~3s | Ranking quality |

A 14B model on CPU cannot be made to feel fast. Pick your quality floor
deliberately.

---

## Configuration

Everything is tunable in `ragmed/config.py` or a `.env` file (copy from
`.env.example`). The knobs that matter most:

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Any model Ollama hosts |
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
| `ragmed/` | The engine — config, loaders, OCR, chunking, embeddings, vector store, retrieval, reranking, conversation memory, prompting |
| `fetch/` | Source fetcher, curated seed lists, the Rules-of-Court splitter |
| `ui/` | The animated hero mark for the web app |
| `tests/` | 164 regression checks |
| `cli.py` / `app.py` | Command line / Haven web app |
| `GUIDE.md` | Full explainer: how it works and what went wrong |
| `corpus/`, `data/` | Documents and index — both git-ignored, both regenerable |

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
