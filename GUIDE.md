# A Field Guide to a Retrieval-Augmented Generation System

> **What this is:** a complete walkthrough of a working RAG system — one that
> answers questions about Philippine medical law from a corpus of statutes,
> regulations and Supreme Court decisions, entirely on a local machine.
>
> **Who it's for:** anyone building, evaluating, or trying to understand a RAG
> system over authoritative documents. It assumes **zero prior knowledge** —
> no machine-learning background needed. By the end you'll understand what RAG
> is, what every component does, how a question becomes a cited answer, and
> which knobs actually matter.
>
> **Why it might be worth your time:** most RAG tutorials stop at "it works."
> This one documents the failures too — the retrieval bugs, the measurements
> that found them, and the approaches that were tried and abandoned. Those
> sections are marked throughout, and they're the parts hardest to find
> elsewhere.
>
> Read it top to bottom the first time. Later, use the Table of Contents to jump
> back to whatever you need.

> ⚖️ **Not legal advice.** This is a research tool for retrieving and citing
> legal texts. It does not interpret the law, does not replace a licensed
> Philippine attorney, and its answers must be verified against the cited
> sources before being relied on for anything.

---

## Table of Contents

1. [The 60-second summary](#1-the-60-second-summary)
2. [What problem are we even solving?](#2-what-problem-are-we-even-solving)
3. [What is RAG? (the core idea)](#3-what-is-rag-the-core-idea)
4. [The mental model: a librarian with a research assistant](#4-the-mental-model-a-librarian-with-a-research-assistant)
5. [The vocabulary you need (plain-English glossary)](#5-the-vocabulary-you-need-plain-english-glossary)
6. [The two journeys: Indexing vs. Answering](#6-the-two-journeys-indexing-vs-answering)
7. [Journey A — Building the index (ingestion), step by step](#7-journey-a--building-the-index-ingestion-step-by-step)
8. [Journey B — Answering a question, step by step](#8-journey-b--answering-a-question-step-by-step)
9. [A tour of every file in this project](#9-a-tour-of-every-file-in-this-project)
10. [Why we chose each technology](#10-why-we-chose-each-technology)
11. [How to actually use it (commands)](#11-how-to-actually-use-it-commands)
12. [Every configuration knob, explained](#12-every-configuration-knob-explained)
12b. [Why it's slow, and what actually helps](#12b-why-its-slow-and-what-actually-helps)
13. [The corpus: what's inside and how to grow it](#13-the-corpus-whats-inside-and-how-to-grow-it)
14. [Troubleshooting & FAQ](#14-troubleshooting--faq)
15. [How to extend it](#15-how-to-extend-it)

---

## 1. The 60-second summary

This is a program that **answers questions about Philippine medical law** by
reading a curated collection of real legal documents (statutes, PRC rules, DOH
orders, Supreme Court decisions) — instead of relying on an AI's memory, which
can be wrong or outdated.

It works in two phases:

- **Indexing (done once, repeated when documents change):** read every
  document, chop it into passages, and store them in a searchable database.
- **Answering (every time someone asks something):** find the handful of
  passages most relevant to the question, hand them to a local AI model, and
  have it write an answer that **cites the specific law and section** it used.

Everything runs **on one machine** — no cloud, no API keys, no per-question
cost, and the documents never leave the computer they're indexed on.

---

## 2. What problem are we even solving?

Imagine you ask a general AI chatbot: *"What are the grounds for revoking a
physician's license in the Philippines?"*

A plain chatbot has three problems here:

1. **It might make things up.** Language models are trained to produce
   *plausible-sounding* text. When they don't know, they often invent a
   confident-sounding answer — a phenomenon called **hallucination**. For legal
   work, a made-up "Section 24" is worse than useless.
2. **Its knowledge is frozen and generic.** It learned from the public internet
   up to some cutoff date. It may not know a 2025 PRC resolution, and it can't
   read *your* specific PDFs.
3. **You can't verify it.** It won't tell you *where* an answer came from, so you
   can't check it against the actual statute.

What you actually want is an assistant that answers **only from a trusted set of
documents you supply**, and **shows its sources** so every claim can be checked.

That is exactly what RAG gives you.

---

## 3. What is RAG? (the core idea)

**RAG** stands for **Retrieval-Augmented Generation**. Break it apart:

- **Generation** = an AI model writing an answer in natural language.
- **Retrieval** = looking up relevant information from a collection of documents.
- **Augmented** = we *feed the retrieved information into* the AI before it
  writes, so its answer is grounded in real source material.

So the one-line definition:

> **RAG = "look up the relevant documents first, then have the AI answer using
> only those documents."**

The AI is no longer answering from memory. It's answering from an **open book**
chosen by whoever curated the corpus. This is why RAG answers can cite sources
and are far less likely to hallucinate — the facts are sitting right there in
front of the model.

**The key insight for a beginner:** the AI model in a RAG system is doing
*reading comprehension*, not *recall*. Its job is "read these passages and
answer the question," not "remember Philippine law." That's a much easier,
safer job — and it's why a modest local model can do legal Q&A well.

---

## 4. The mental model: a librarian with a research assistant

Picture a law library with two staff members:

- **The Librarian (Retrieval).** You ask a question. The librarian doesn't
  answer it — instead they sprint into the stacks and come back with the 6 most
  relevant pages, sticky-noted. They're brilliant at *finding* but don't
  interpret.
- **The Research Assistant (Generation / the LLM).** They take those 6 pages,
  read them, and write you a clear answer — quoting the law and section — and
  they refuse to go beyond what's on the pages.

The whole system is an automated version of this pair, plus a **cataloguing
step** that happens ahead of time so the librarian can find things fast. Keep
this picture in mind; every component below maps onto it.

| Real-world role | In the system |
|---|---|
| Cataloguing the library ahead of time | **Ingestion** (`ragmed/ingest.py`) |
| The card catalog / search index | **Vector store + BM25 index** (`vectorstore.py`, `retriever.py`) |
| The librarian who fetches pages | **Retriever** (`ragmed/retriever.py`) |
| The research assistant who writes the answer | **LLM via vLLM** (`ragmed/llm.py`, `rag.py`) |
| The front desk where questions come in | **CLI / Web UI** (`cli.py`, `app.py`) |

---

## 5. The vocabulary you need (plain-English glossary)

Don't memorize these — just skim now and refer back. Each is expanded later.

- **Corpus** — the whole collection of documents the system knows about. Here it
  lives in the `corpus/` folder.
- **Document** — one source file (a PDF of a statute, a DOH order, etc.).
- **Chunk** — a small slice of a document (a few paragraphs). We search and
  retrieve *chunks*, not whole documents, because whole laws are too big to feed
  to the AI at once and too coarse to pinpoint an answer.
- **Embedding** — a list of numbers (a **vector**) that represents the *meaning*
  of a piece of text. Similar meanings → similar numbers. This is the magic that
  lets a computer find text by *meaning* rather than exact keywords.
- **Vector** — just an array of numbers, e.g. `[0.02, -0.51, 0.13, …]`. An
  embedding is a vector. Yours have 768 numbers each.
- **Vector store / vector database** — a database built to store embeddings and
  instantly find the ones most similar to a query embedding. Yours is **Chroma**.
- **Semantic search (dense retrieval)** — searching by *meaning* using
  embeddings. "Revoke a doctor's license" can match a passage about
  "suspension of a certificate of registration" even with no shared keywords.
- **Lexical search (BM25 / keyword search)** — old-school search by *exact
  words*. Great for things like "RA 2382" or "Section 24" where the exact token
  matters.
- **Hybrid search** — combining semantic + lexical so you get the best of both.
  Yours does this.
- **HyDE (Hypothetical Document Embeddings)** — when your question is worded
  nothing like the law ("they kept changing their story"), the system has the
  LLM draft a short *imaginary* answer in legal vocabulary and searches with
  that too. You never see the draft; it exists only to supply the terminology
  your question lacked.
- **Top-K** — how many chunks are handed to the LLM (yours: 6, tunable). Every
  slot spent on a duplicate or on a fourth passage from the same case is a slot
  the answer doesn't get.
- **Bi-encoder vs cross-encoder** — the two ways to compare a question with a
  chunk. A **bi-encoder** turns each into a vector *separately* and compares the
  vectors; fast enough to search everything, but the chunk's vector never saw
  your question. A **cross-encoder** reads the pair *together* and scores them
  directly; far more accurate, far too slow for a whole corpus. Yours uses a
  bi-encoder to find ~30 candidates and a cross-encoder to order them — the
  standard **retrieve-then-rerank** pattern.
- **LLM (Large Language Model)** — the AI that writes the final answer. Yours is
  **Qwen3-14B-AWQ**, served locally by **vLLM**.
- **vLLM** — a high-throughput local inference server that serves LLMs over an
  OpenAI-compatible API (the same request shape as `api.openai.com`, just
  pointed at `localhost`). It's the engine a production appliance would ship
  with, so running against it locally rehearses the real deployment path
  (continuous batching, PagedAttention, a containerised CUDA runtime) rather
  than standing in with something simpler.
- **Prompt** — the full block of text we send the LLM: your question plus the
  retrieved passages plus instructions ("answer only from this, cite sections").
- **Grounding** — forcing the AI to answer strictly from provided sources.
- **Hallucination** — when an AI confidently invents false information. RAG +
  grounding is our defense against it.
- **OCR (Optical Character Recognition)** — turning a *scanned image* of text
  into actual searchable text. Needed because some PRC PDFs are just photos of
  pages.
- **Ingestion / indexing** — the one-time process of reading documents and
  loading them into the searchable store.

---

## 6. The two journeys: Indexing vs. Answering

This is the single most important thing to internalize. A RAG system has **two
completely separate flows**, and confusing them is the #1 beginner stumbling
block.

### Journey A — Indexing (a.k.a. ingestion). Happens rarely.
You run this when you add or change documents. It reads everything and fills the
database. It does **not** answer questions. Think: *stocking the library.*

```
the documents  →  read text  →  cut into chunks  →  turn each chunk into an
embedding  →  store chunks + embeddings in Chroma
```

Command: `python cli.py ingest`

### Journey B — Answering. Happens every time you ask something.
This uses the database that Journey A already built. It never re-reads your PDFs
from scratch. Think: *using the library.*

```
your question  →  find the most relevant chunks (hybrid search)  →  build a
prompt from them  →  LLM writes a grounded, cited answer  →  show answer + sources
```

Command: `python cli.py ask "..."` (or the web UI)

> **Why separate them?** Reading and embedding documents is slow and only needs
> to happen when documents change. Answering must be fast and happen on demand.
> Splitting the work means every question reuses the expensive indexing you did
> once. The index described here holds **9,066 chunks from 156 documents**.

---

## 7. Journey A — Building the index (ingestion), step by step

When you run `python cli.py ingest`, here's the assembly line. Each stop is a
real file in `ragmed/`.

### Step 1 — Find the files (`loaders.py`)
The system scans the `corpus/` folder (and every subfolder) for supported file
types: **PDF, DOCX, TXT, MD, HTML**. Anything else is ignored.

*Beginner note:* there's a small cleverness here — the web fetchers save both a
clean `.txt` and the raw `.html` of each downloaded page. The loader skips the
raw `.html` when a matching `.txt` exists, so the same document isn't indexed
twice.

### Step 2 — Extract the text (`loaders.py`)
Each file type needs a different reader:
- **PDF** → tried with `pypdf` (fast); if that yields almost nothing, it retries
  with `pdfplumber` (handles trickier layouts).
- **If the PDF is a *scanned image*** (no real text inside — just a picture of a
  page), both readers come back nearly empty. Then **OCR kicks in
  automatically** (`ocr.py`): each page is rendered to an image and Tesseract
  "reads" the text off it. This is how the scanned PRC disciplinary-rules PDFs
  made it into the corpus at all.
- **DOCX** → `python-docx`. **HTML** → `BeautifulSoup` strips tags. **TXT/MD** →
  read directly.

The output is one clean string of text per document.

### Step 3 — Cut into chunks (`chunking.py`)
Now we slice each document into bite-sized passages. But **not** naively every N
characters — that would cut a sentence in half and split "Section 24" from its
own text. Legal documents have structure, so this project uses
**legal-aware chunking**:

- It detects boundaries like `SECTION`, `ARTICLE`, and `RULE` and tries to keep
  each section together in a chunk.
- It **detects the document's identity** — e.g. it recognizes "Republic Act No.
  2382" and tags every chunk from that file with `law = "Republic Act No. 2382"`.
- It records which **section** each chunk came from (e.g. `Section 24. Grounds
  for … revocation`).

This metadata is gold: it's what lets the final answer say *"(Republic Act No.
2382, Sec. 24)"* instead of a vague "the law says…". Chunks target about
**1,100 characters** with **150 characters of overlap** (the overlap prevents an
answer from falling in the crack between two chunks).

Each chunk ends up as: **the text** + **metadata** (`law`, `section`, `title`,
`source` filename, `chunk_index`).

### Step 4 — Turn each chunk into an embedding (`embeddings.py`)
Every chunk of text is fed to a local model called **BAAI/bge-base-en-v1.5**,
which converts it into a **768-number vector** capturing its meaning. This model
downloaded itself the first time you ingested (~440 MB) and now runs entirely
offline on your CPU.

*Why a vector?* Because computers can't compare "meaning" directly, but they
*can* measure whether two vectors point in a similar direction. Two chunks about
license revocation will have vectors close together, even if they use different
words.

### Step 5 — Store everything (`vectorstore.py` → Chroma)
The chunk text, its embedding, and its metadata are saved into **Chroma**, a
local vector database that persists on disk at `data/chroma/`. Chroma is
configured to compare vectors using **cosine similarity** (a standard measure of
"how similar in direction two vectors are").

A separate **BM25 keyword index** is also built from all chunks (cached at
`data/bm25.pkl`) so we can do exact-word search too. It rebuilds automatically
whenever the number of chunks changes.

**That's the whole indexing journey.** After this, the `corpus/` PDFs aren't
needed to answer questions — everything lives in `data/`.

---

## 8. Journey B — Answering a question, step by step

When you run `python cli.py ask "What are the grounds for revoking a physician's
certificate?"`, here's what happens.

### Step 1 — Understand the question as a vector (`embeddings.py`)
Your question is turned into an embedding using the **same** bge model — so it
lives in the same "meaning space" as the chunks. (Small detail: bge models want
a short instruction prefix on *queries only*, which the code adds automatically.)

### Step 2 — Retrieve candidate chunks two ways (`retriever.py`)
This is the librarian sprinting into the stacks. It runs **two searches in
parallel**:

- **Dense (semantic):** ask Chroma for the chunks whose embeddings are closest
  to the question's embedding. Finds meaning-matches.
- **Lexical (BM25):** score chunks by exact word overlap with the question.
  Finds exact-token matches like "RA 2382" or "Section 24".

It pulls ~25 candidates from each.

### Step 3 — Fuse the two result lists (`retriever.py`)
Now it merges them into one ranking. Each candidate gets a combined score:

```
final_score = 0.6 × (semantic score)  +  0.4 × (keyword score)
```

(The 0.6 / 0.4 weights are configurable.) Both score sets are normalized to a
0–1 range first so they're comparable. This **hybrid** approach is why the
system handles both fuzzy conceptual questions *and* precise citation lookups
well.

### Step 3b — Clean up the ranking before cutting it (`retriever.py`)
A raw fused ranking is not yet safe to hand over. Two filters run in order, and
both exist because of failures measured on this corpus.

**Drop exact duplicates.** Legal corpora hold the same passage twice by
design — a resolution on reconsideration reprints the decision it modifies.
Identical text scores identically, so the copies sit adjacent and eat slots in
pairs. One case once took 6 of 10 slots as three duplicate pairs.

**Cap how much any one document may take** (`MAX_CHUNKS_PER_SOURCE`, default 3).
This is the subtler cousin: not duplicate text, but one landmark document that
genuinely discusses its topic across a dozen different passages and out-scores
everything on all of them. Before the cap, *"what is informed consent"* filled
**8 of its 10 slots with Dr. Rubi Li alone** — 3 distinct documents in the whole
context. Across ten test questions the top document's share averaged 47%; with
the cap it's 28%, and that question now returns 5 documents.

The cap **demotes** rather than deletes: once every document has had its three,
the held-back chunks come back to fill any remaining slots. So a question only
one statute answers still gets a full context — the cap changes the *order* of
what's kept, never the amount.

### Step 3c — Ask the *right* question of each chunk (the reranker)
Everything so far compares **two vectors**: one for your question, one computed
for each chunk long before your question existed. That's what makes searching
7,900 chunks fast — and it's also the ceiling. A chunk's vector never saw your
question, so "is about similar things" and "actually answers this" score alike.
Measured here: one question scored **eight different chunks inside a 1.4%
spread**, with the rule that answered it sitting 5th.

A **cross-encoder** reads your question and one chunk *together*, in a single
pass, and scores "does this passage answer this question". It's far too slow to
run over a whole corpus — which is exactly why it runs here, over the ~30
candidates retrieval has already narrowed to. On the same chunks that were tied
within 1.4%, it separated the right one by **0.1975**.

Two safeguards you should know about, because both came from things going wrong:

- **It fails open.** If the model won't load, the previous ordering stands. A
  ranking improvement that can take retrieval down is a bad trade.
- **It ignores itself when it has nothing to say** (`RERANK_MIN_SPREAD`). Asked
  a two-part question, the cross-encoder returned every candidate within
  **0.0004** of 0.5000 — and sorting that noise promoted an unrelated case to
  rank 1. When the spread is that flat, the scores mean nothing and the fused
  order is kept.

Order matters: rerank happens **before** the cap, or the cap would already have
picked which of a document's chunks survive using the very scores the reranker
exists to correct.

### Step 3d — Guarantee every part of a compound question gets heard
A two-part question has a failure mode all the fusion in the world won't fix.
Scores are normalized **against the whole question**, so chunks answering the
neglected half rank below chunks matching the dominant half no matter how many
candidates you fetch. Measured: the top twelve results all scored ≥0.894 and
were *all* about one half.

Tempting fix: fetch more candidates. It doesn't work — raising `CANDIDATE_K`
surfaced the missing half at 60 and **lost it again at 120**, because a bigger
pool moves the normalization window. A number that works at one value and fails
at double is fitted to one question, not a fix.

So each **ask** gets its own search and is *guaranteed* `MIN_CHUNKS_PER_CLAUSE`
slots (default 2). It's a **floor, not a quota** — an ask that legitimately owns
the ranking keeps all its slots. Selection is per-ask; the final order is still
by score, so the strongest evidence still leads.

Only then are the top **6** (configurable, `TOP_K`) chunks kept.

### Step 3e — When the question doesn't speak the corpus's language (HyDE)
A question phrased as a lay narrative — *"they kept changing their story about
my mother"* — embeds nowhere near terse statutory text and offers BM25 no
distinctive words to match. Retrieval returns noise and the system refuses,
which *looks* like honesty and is actually a bug.

**HyDE** (Hypothetical Document Embeddings) fixes it: the LLM drafts a short
imaginary answer in legal vocabulary, and the system retrieves using that too,
unioning the results with the original. The draft is never shown to you — its
only job is to supply the terminology the question lacked. It's deliberately
forbidden from inventing section numbers, because a fabricated "Article 2200" is
a real token BM25 would match hard, aiming retrieval confidently at the wrong
law.

It runs only when needed (`HYDE_MODE=auto`), on two triggers:
1. **The whole question matches poorly** — best raw similarity below
   `HYDE_MIN_SIM`.
2. **Any single *ask* matches poorly** — below `HYDE_CLAUSE_MIN_SIM`. This
   catches the compound question, which trigger 1 structurally cannot see:
   similarity over a whole question is a *maximum*, so one well-matched half
   hides the other. *"Can we refuse to release the body… who can we collect
   from, and is there a deadline?"* scored a healthy 0.736 purely on its first
   half, HyDE stayed off, the estate-claim half was never retrieved — and the
   answer flatly denied a deadline that Rule 86 imposes.

Note which failure that was: not a refusal, a **fluent wrong answer**. Those are
the expensive ones, and they're why the second trigger exists.

> **The one thing to take from 3b–3e:** these four mechanisms fix *different*
> failures, and reaching for the wrong one wastes real time. The cap fixes **one
> document crowding out others**. Clause reservation fixes **one half of a
> question crowding out the other**. The reranker fixes **the right document
> arriving as the wrong chunk**. HyDE fixes **the question not sharing the
> corpus's vocabulary**. None of them fixes a document that isn't there — and
> adding documents fixes none of the four.

### Step 3f — Remembering the conversation (`conversation.py`)
Haven holds the session so follow-ups work. That sounds like one feature. It's
two, and building only the second leaves the first broken.

**Retrieval breaks first.** *"And is there a deadline?"* contains none of the
words that say what it's about — those are in the previous turn. Searched as
written it finds noise, and by the time history reaches the prompt, retrieval
has already failed. So a dependent follow-up is **rewritten into a standalone
question before searching**. The rewrite is forbidden from adding terminology or
facts you didn't say — it resolves references, it doesn't improve your question.
(Same lesson as HyDE: invented vocabulary aims retrieval confidently at the
wrong document.)

**Generation needs the prior turns** to reply coherently instead of restarting
the topic every time.

**And one risk that matters more here than in an ordinary chatbot.** The moment
Haven's earlier answers sit in the prompt, they become a second apparent source
— and it will cite a provision it "remembers stating" a turn ago with no
retrieved passage behind it. To a reader that citation looks exactly like a real
one. Three defences:

1. The block is labelled **"for reference only — not a source, never cite it"**
2. A system-prompt rule forbids citing it, treating an earlier answer as
   established law, or carrying a citation forward between turns
3. The retrieved passages sit **nearer the question** than the history — the
   position the model weighs most, and the one that must win when they disagree

History is bounded (3 turns; assistant replies truncated hardest) because it
shares a fixed context window with the passages, and passages are what the
answer must be grounded in. Every step **fails open**: a failed, empty or
rambling rewrite falls back to your own words.

### Step 4 — Build the prompt (`rag.py`)
The system assembles the text it will send to the AI. It contains three things:

1. **A system prompt** — the rulebook for the AI. Yours instructs it to: answer
   **only** from the provided context, **cite the law and section** for every
   claim, say *"The provided corpus does not cover this"* when the passages don't
   contain the answer (instead of guessing), and add a "this is legal
   information, not legal advice" note when appropriate.
2. **The context** — the 6 retrieved chunks, each labeled with its law/section/
   source so the model can cite precisely.
3. **Your question.**

> **Don't hand the model a citation-shaped label.** The chunks used to be
> labeled `[Context 1]`, `[Context 2]`… and the rulebook told the model its
> citations had to match an identifier "in a `[Context N]` header line". Both
> halves were a trap. A bracketed label *looks* like a citation to anything
> trained on legal or academic writing, and spelling the token out in the rules
> handed the model a template to copy. It copied it: **75 stray markers across
> 12 answers**, with one answer opening every paragraph `[Context 3] [Context 5]`
> instead of naming a law. To the reader — who never sees the prompt — those are
> references that resolve to nothing, which is worse than no citation at all,
> because they *look* checkable.
>
> The fix is three-layered, and the third layer is the point: the labels became
> unbracketed (`PASSAGE 1` with a separate `Cite as:` line), the rules now name
> the offence explicitly, and `strip_source_labels()` removes any that still
> escape. **A prompt rule is a request; only code is a guarantee.** Anything a
> user must never see needs an enforcement layer outside the model, however
> firmly the prompt asks.

### Step 5 — The LLM writes the answer (`llm.py` → vLLM)
The prompt goes to **vLLM**, which serves **Qwen3-14B-AWQ** locally over an
OpenAI-compatible API. The model reads the passages and streams back an answer,
token by token, grounded in and citing the supplied sections.

Two settings here are less obvious than they look. Sampling is **not** greedy
(`temperature=0.7`, `top_p=0.95`, `top_k=20`): Qwen3-family models loop on
repeated tokens under greedy decoding, so "turn the randomness off for a more
factual answer" — the usual instinct for grounded RAG — actually breaks
generation. And **thinking mode is explicitly disabled**
(`enable_thinking: False`, sent via `extra_body` since it isn't a standard
OpenAI parameter): Qwen3.6 is a reasoning model, and left on it burns the
token budget on chain-of-thought before writing an answer — observed directly
during setup, where a two-sentence question spent all 200 tokens reasoning and
returned no content at all. Since the answer is already grounded in retrieved,
reranked passages, extended reasoning adds latency without adding accuracy.

### Step 6 — Show the answer + sources (`cli.py` / `app.py`)
You get the written answer **plus a list of the source chunks** it drew from
(law, section, filename, and relevance score), so you can verify every point
against the real document.

> **The honesty guarantee:** because the model is told to answer only from the
> retrieved text and to admit when the corpus is silent, you get "I don't have
> that" instead of a confident fabrication. That's the whole point of RAG for
> legal work.

---

## 9. A tour of every file in this project

Here's what each piece is and why it exists. Think of `ragmed/` as the *engine*,
and `cli.py`/`app.py` as the *steering wheels*.

### The engine — `ragmed/`
| File | Plain-English job |
|---|---|
| `config.py` | The single control panel. Every setting (which models, chunk size, how many results, OCR options) lives here and can be overridden with a `.env` file. Start here when you want to change behavior. |
| `loaders.py` | Turns any supported file (PDF/DOCX/HTML/TXT) into clean text. Calls OCR when a PDF is a scanned image. |
| `ocr.py` | Reads text off *scanned* PDFs by rendering pages to images and running Tesseract. Only used when normal text extraction fails. |
| `chunking.py` | Cuts documents into passages along legal boundaries (Section/Article/Rule) and tags each with its law + section. |
| `embeddings.py` | Converts text ↔ meaning-vectors using the local bge model. |
| `vectorstore.py` | Talks to the Chroma database: create the collection, add chunks, search, count. |
| `retriever.py` | The hybrid search brain: dense + BM25, then fuse, rerank, cap, reserve per-ask slots, and rank. |
| `rerank.py` | The cross-encoder that re-scores retrieved candidates by reading question and chunk together. Fails open — if the model won't load, retrieval's ordering stands. |
| `conversation.py` | Session memory: rewrites dependent follow-ups into standalone search queries, and fences replayed turns so they can't be cited as a source. |
| `llm.py` | Talks to vLLM over its OpenAI-compatible API; also checks whether the server is up and the configured model is loaded. |
| `rag.py` | The conductor: retrieve → build the grounded prompt → get the answer. Holds the all-important system prompt. Also times retrieval and generation and hands them to `metrics.py` after every answer. |
| `metrics.py` | Appends one JSON line per answered question to `data/metrics.jsonl` (timings, chunk count, HyDE fired, error) — read by `dashboard.py`. Logging is best-effort: a write failure never breaks an answer. |
| `ingest.py` | The indexing pipeline that runs Steps 1–5 of Journey A over the whole corpus. |
| `__init__.py` | Marks `ragmed` as a Python package (plumbing; nothing to configure). |

### The steering wheels
| File | Job |
|---|---|
| `cli.py` | Command-line interface. Subcommands: `ingest`, `ask`, `chat`, `status`. This is your main tool. |
| `app.py` | A **Streamlit** web chat UI — same engine, friendlier face, with expandable source citations. |
| `dashboard.py` | A separate **Streamlit** ops dashboard (`streamlit run dashboard.py --server.port 8502`) — LLM/GPU/index health and query metrics, for whoever operates the box rather than the end user. See §11 and §14. |

### The document-fetching tools — `fetch/`
| File | Job |
|---|---|
| `fetch_seeds.py` | Downloads documents listed in a seed file into `corpus/_fetched/`. Robust against flaky government websites (retries, browser fallback, broken-certificate tolerance for `.gov.ph`). |
| `common.py` | Shared download helpers used by the fetcher (fetch, clean HTML, save PDF/text). |
| `seeds.txt` | Curated list of **statutes** from LawPhil (Medical Act, UHC Act, Civil Code, Penal Code, etc.). |
| `seeds_prc.txt` | Curated list of **PRC** issuances (Board of Medicine ethics, licensing, disciplinary rules). |
| `seeds_doh.txt` | Curated list of **DOH** issuances (hospital licensing, patients'-rights IRRs, health-worker rules), sourced from the Supreme Court E-Library. |
| `seeds_jurisprudence.txt`, `seeds_negligence.txt`, `seeds_surgery.txt` | Supreme Court decisions: malpractice, informed consent, hospital liability, records/certificates, surgical cases. |
| `seeds_billing.txt`, `seeds_billing_jurisprudence.txt`, `seeds_billing_issuances.txt` | **Medical bills, payment and non-payment** — PhilHealth, HMO cases, senior/PWD discounts, government assistance, small claims, and the deceased-patient chain (estate claims, release of remains). |
| `split_rules_of_court.py` | **Required after fetching the Rules of Court.** LawPhil serves Rules 72–109 as one 148k-character page; this splits it into one file per rule. See the warning below. |

### The data folders
| Folder | What's inside |
|---|---|
| `corpus/` | Your source documents. You drop files here; fetchers save into `corpus/_fetched/`. This is the *input*. |
| `data/chroma/` | The Chroma vector database (the searchable index). Auto-generated. |
| `data/bm25.pkl` | The cached keyword index. Auto-generated. |
| `.venv/` | The isolated Python environment with all the installed libraries. |

### The support files
| File | Job |
|---|---|
| `requirements.txt` | The list of Python libraries the project needs. |
| `.env.example` | A template showing every setting you can override. Copy to `.env` to customize. |
| `.gitignore` | Tells git which files *not* to track (the big regenerable stuff: `data/`, downloaded docs, the virtual env). |
| `README.md` | The quick-start / reference (setup + commands). |
| `GUIDE.md` | **This document** — the deep, beginner-friendly explanation. |

> **The regenerable vs. precious distinction:** `corpus/` (the source documents) and
> the code are precious. `data/`, `.venv/`, and `corpus/_fetched/` are all
> *regenerable* — you can delete them and rebuild with `pip install` + `fetch` +
> `ingest`. That's why they're git-ignored.

---

## 10. Why we chose each technology

Understanding the *why* makes the whole thing click, and helps you swap parts
later.

### Why "fully local / offline"?
A deliberate trade. Benefits: **privacy** (documents never leave the machine),
**no API keys**, **no per-question cost**, and it works without internet once set
up. Cost: a GPU with meaningful VRAM (vLLM), a Docker install, and Tesseract for
OCR. On a 24GB consumer card, generation runs at ~79 tok/s with the default
14B model — a complete answer lands in a few seconds of decode time (longer
questions with more retrieval or HyDE add a few seconds more). Embeddings and reranking still run on
CPU regardless of GPU size (see below — they're small enough that it isn't
worth the VRAM). For legal research over sensitive or proprietary documents,
privacy plus zero marginal cost is usually the right call — but if you are
indexing public documents at high volume, a cloud backend is the reasonable
choice, and the engine is written to swap.

### Why vLLM + Qwen3-14B-AWQ for the LLM?
**vLLM** is what a production/appliance deployment of this kind of system
actually ships with, so developing against it — rather than a simpler local
stand-in — rehearses the real serving path: continuous batching,
PagedAttention, a containerised CUDA runtime that transfers unchanged to
client hardware. **Qwen3-14B-AWQ** is a 4-bit-quantized 14B model; AWQ
leaves embeddings, `lm_head`, layernorms and the vision encoder in FP16. Its
real resident footprint, measured the same way, is **9.44 GiB** — well under
the **19.05 GiB** the earlier `Qwen3.6-27B-AWQ` default needed on the same
24GB card (full 27B baseline in `HAVEN_VLLM_MIGRATION.md` §2; the
commonly-quoted "27B fits in ~17GB" figure is a GGUF/llama.cpp number and
doesn't transfer to vLLM's AWQ path). The smaller footprint isn't just less
VRAM used — it removed the contention that used to force `--enforce-eager`:
with 9.44 GiB of weights there's room for CUDA graphs *and* a full KV cache,
so the current launch command no longer passes it (see §5 flags below). Swap
the model with one flag (`--model` in the `docker run` command) and matching
`LLM_MODEL` in `.env`; re-run `evals/bench_llm.py` after any model swap rather
than assuming these numbers scale by parameter count. Sizing math for a
different model class is in `HAVEN_VLLM_MIGRATION.md` §6.

### Why bge-base for embeddings?
`BAAI/bge-base-en-v1.5` is a well-regarded open embedding model with a great
quality-to-speed balance on CPU. "Base" is the middle size; you can drop to
`bge-small` (faster) or `bge-large` (more accurate) by changing `EMBED_MODEL`.

> **Why CPU, specifically, and why it matters now.** Embeddings and the
> cross-encoder reranker (`ragmed/embeddings.py`, `ragmed/rerank.py`) are both
> explicitly pinned to `device="cpu"` in code, not just left to whatever
> PyTorch picks by default. This didn't matter under Ollama, which ran on CPU
> too — but vLLM claims most of the GPU by design
> (`--gpu-memory-utilization 0.93`, leaving under a gigabyte free), so if
> either model auto-detects CUDA and tries to load there, it hits an
> out-of-memory crash mid-query instead of gracefully falling back. This
> actually shipped once: `embeddings.py`'s offline-fallback path and
> `rerank.py`'s loader both omitted `device="cpu"`, so a Hugging Face Hub
> hiccup (or, for the reranker, every load) sent them straight into the wall
> vLLM had already built. See the troubleshooting entry below if you see a
> `torch.OutOfMemoryError` pointing at `embeddings.py` or `rerank.py`.

### Why Chroma for the vector store?
Chroma is a local, persistent, zero-configuration vector database. It just writes
to a folder (`data/chroma/`) — no separate server to run. Perfect for a
single-machine project.

### Why *hybrid* retrieval instead of just semantic?
Legal questions are special: exact tokens matter enormously. "RA 2382",
"Section 24", "PRC" must match precisely — and pure semantic search can miss
exact identifiers. Pure keyword search, meanwhile, misses paraphrases. Combining
both (dense + BM25) covers both needs. This is a genuinely important design
choice for a *legal* RAG specifically.

### Why legal-aware chunking?
Because a citation is only useful if it points to the right Section. By splitting
on `SECTION`/`ARTICLE`/`RULE` and tagging chunks with their law and section, the
system can produce real citations and keep each provision intact rather than
sliced across chunks.

### Why OCR?
Many official PRC/DOH documents are *scanned images* — a photo of a page with no
digital text inside. Without OCR they're invisible to search. OCR (via Tesseract)
reads the text off the image so those documents — including the PRC
disciplinary-procedure rules most relevant to "PRC complaints" — become
searchable.

---

## 11. How to actually use it (commands)

All commands are run from the project folder, using the virtual environment's
Python. Examples below use the Windows path `.\.venv\Scripts\python.exe`; on
macOS or Linux that is `./.venv/bin/python`.

### "Fully local" means at RUN time, not at INSTALL time

This is worth being precise about, because "runs locally" is easy to misread as
"never needs the internet".

**Setup needs the internet once**, to bring the following onto the machine (the
LLM row below dominates the total — see the note after the table):

| What | Size | Where it lands |
|------|------|----------------|
| Python packages (PyPI) | ~2 GB | `.venv\` |
| Embedding model `BAAI/bge-base-en-v1.5` | 439 MB | `%USERPROFILE%\.cache\huggingface\hub` |
| LLM `Qwen/Qwen3-14B-AWQ` (pulled by the vLLM container) | ~9.3 GiB checkpoint (9.44 GiB resident once loaded — smaller than the 19.05 GiB the earlier 27B default needed) | same Hugging Face cache, mounted into the container |
| Tesseract OCR (optional, scanned PDFs only) | ~60 MB | `C:\Program Files\Tesseract-OCR` (or `apt install tesseract-ocr` on Linux) |
| The corpus itself (LawPhil / SC E-Library) | 36 MB | `corpus\` |

**After that, answering is fully offline.** Nothing in the query path leaves the
machine: the LLM is vLLM on `localhost:8000`, the vector store is Chroma on
disk (`data\chroma`, telemetry explicitly disabled), the BM25 index is a local
pickle, and the embedding model is read from the cache above. Verified by
running a full query with all outbound HTTP blackholed — only `localhost`
reachable — and getting a complete, correctly cited answer.

One wrinkle that had to be fixed: `sentence-transformers` contacts
huggingface.co on load *even when the model is already cached*, so a DNS hiccup
used to crash a live query. `ragmed/embeddings.py` now falls back to loading the
cached snapshot directly from disk (see `tests/test_offline_embeddings.py`) —
and that fallback, like the primary load, is pinned to `device="cpu"` so it
can't collide with vLLM for GPU memory (see the callout in §10).

To move this to a machine that will never have internet, copy `.venv\`, the
Hugging Face cache folder (it now holds both the embedding and LLM weights),
and the project directory — plus the `vllm/vllm-openai` Docker image, pulled
separately.

### One-time setup
```powershell
# 1. Python libraries
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# 2. vLLM — needs Docker + an NVIDIA GPU. Pulls the model on first run; the
#    flags matter (see CLAUDE.md's "Generation speed" section for why each one
#    is there — most claw back VRAM vLLM would otherwise waste on unused
#    multimodal support). No --enforce-eager: at the 14B model's size, CUDA
#    graphs capture cleanly alongside a full KV cache (no speculative-decode
#    drafter competing for VRAM the way there was under the earlier 27B
#    default) — see HAVEN_VLLM_MIGRATION.md's superseded-note for that history.
docker run -d --name vllm-qwen14b --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-14B-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kv-cache-dtype fp8_e5m2 \
  --reasoning-parser qwen3

docker update --restart unless-stopped vllm-qwen14b   # survives a reboot
```

```powershell
# 3. Tesseract for OCR — only needed for scanned PDFs
winget install UB-Mannheim.TesseractOCR

# 4. First run downloads the embedding model (439 MB) automatically.
.\.venv\Scripts\python.exe cli.py status
```

### Everyday commands
```powershell
# Check the system's health (index size, is vLLM up, is OCR ready)
.\.venv\Scripts\python.exe cli.py status

# Build / refresh the index after adding documents
.\.venv\Scripts\python.exe cli.py ingest          # add & update
.\.venv\Scripts\python.exe cli.py ingest --reset  # wipe & rebuild from scratch

# Ask a single question
.\.venv\Scripts\python.exe cli.py ask "What are the grounds for revoking a physician's certificate of registration?"

# Have a back-and-forth conversation
.\.venv\Scripts\python.exe cli.py chat

# Launch the web chat interface (opens in your browser)
.\.venv\Scripts\python.exe -m streamlit run app.py

# Launch the ops dashboard — a separate app, separate port; see §14 for what it shows
.\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.port 8502
```

### Fetching more documents
```powershell
.\.venv\Scripts\python.exe fetch\fetch_seeds.py --file fetch\seeds.txt     --subdir lawphil
.\.venv\Scripts\python.exe fetch\fetch_seeds.py --file fetch\seeds_prc.txt --subdir prc
.\.venv\Scripts\python.exe fetch\fetch_seeds.py --file fetch\seeds_doh.txt --subdir doh
# then re-index:
.\.venv\Scripts\python.exe cli.py ingest
```

> **Golden rule:** whenever you add, remove, or change documents in `corpus/`,
> run `ingest` again. The index doesn't update itself.

---

## 12. Every configuration knob, explained

All of these live in `ragmed/config.py` with sensible defaults. To change one
without editing code, copy `.env.example` to `.env` and set it there.

### Models
- `EMBED_MODEL` — which embedding model. Default `BAAI/bge-base-en-v1.5`.
  Alternatives: `bge-small` (faster), `bge-large` (better). *Changing this
  requires a full `ingest --reset`* because old and new embeddings aren't
  comparable.
- `LLM_BASE_URL` — where vLLM's OpenAI-compatible API is listening. Default
  `http://localhost:8000/v1`.
- `LLM_MODEL` — which model vLLM is serving. Default
  `Qwen/Qwen3-14B-AWQ`. Must match the `--model` the container was
  started with.
- `LLM_TEMPERATURE` / `LLM_TOP_P` / `LLM_TOP_K` — sampling. Defaults `0.7` /
  `0.95` / `20`. **Don't set temperature to 0** — despite being the usual
  instinct for factual RAG output, greedy decoding makes Qwen3-family models
  loop on repeated tokens.
- `LLM_MAX_TOKENS` — caps *total* generation per answer. Default `1024`.
- `LLM_TIMEOUT` — client HTTP timeout in seconds. Default `180`. At ~79 tok/s
  with the current default model a full 1024-token answer takes well under
  20s, but the margin was set generously (the default `openai` client timeout
  is far shorter) back when the larger 27B default ran at ~19 tok/s and a long
  answer took ~50s — it's still safe headroom, just no longer a tight one.
- `LLM_SEED` — fixed sampling seed for the two calls that feed *retrieval* (the
  HyDE draft and the follow-up rewrite), not for the answer. Without it those
  calls are sampled, which makes the retrieved passages themselves random:
  measured, one unchanged question asked three times returned only 2-3 of the
  same 10 chunks. Negative disables.
- `HF_OFFLINE` — default `true`. Stops sentence-transformers revalidating the
  cached embedder and reranker against huggingface.co on every process start,
  which cost ~10s per process. Set `false` to download a new model.
- `LLM_ENABLE_THINKING` — whether Qwen3.6 is allowed to reason before
  answering. Default `false`. Left on, it spends `LLM_MAX_TOKENS` on
  chain-of-thought and can return no content at all — see §8, Step 5.

### Chunking (affects the index; re-ingest after changing)
- `CHUNK_SIZE` — target characters per chunk. Default `1100`.
- `CHUNK_OVERLAP` — characters shared between neighbors. Default `150`.

### Retrieval (affects answers; no re-ingest needed)
- `TOP_K` — how many chunks are fed to the LLM. Default `6`. More = more context
  but slower and noisier.
- `CANDIDATE_K` — how many candidates each search fetches before fusion. Default
  `25`.
- `DENSE_WEIGHT` / `LEXICAL_WEIGHT` — the 0.6 / 0.4 balance between semantic and
  keyword search. Raise `LEXICAL_WEIGHT` if you care more about exact citations.
- `HYDE_CLAUSE_MIN_SIM` — the compound-question trigger for HyDE. Default `0.70`.
  Similarity over a whole question is a *maximum*, so one well-matched clause
  hides every other: the deceased-patient question scored 0.736 (above
  `HYDE_MIN_SIM`) purely because its first half matches the anti-detention law,
  and the estate-claim half was never retrieved — the answer then denied a
  deadline that Rule 86 imposes. Each **ask** is now scored separately and HyDE
  fires when the worst falls below this. Only the asks: the weakest clause of an
  answerable question is its narrative setup, not its question, so scoring every
  clause made answerable and unanswerable questions indistinguishable.
- `RERANK_ENABLED` / `RERANK_MODEL` — the cross-encoder that rescores retrieved
  candidates (default `BAAI/bge-reranker-base`, ~1.1 GB, downloaded once). Set
  `RERANK_ENABLED=false` to trade ranking quality for ~4s per query, or use
  `cross-encoder/ms-marco-MiniLM-L-6-v2` for most of the benefit at ~10× smaller.
- `RERANK_CANDIDATES` — how many of the retrieved chunks get rescored. Default
  `30`; the tail keeps its fused order behind them.
- `RERANK_MIN_SPREAD` — ignore the reranking when its scores are too flat to
  mean anything. Default `0.02`. This exists because a compound question once
  produced scores spanning 0.0004, and sorting that noise promoted an unrelated
  case to rank 1.
- `HISTORY_TURNS` / `HISTORY_USER_CHARS` / `HISTORY_ASSISTANT_CHARS` — how much
  of the conversation is replayed. Bounded on purpose: history shares the
  context window with the retrieved passages, and the passages are what the
  answer must be grounded in.
- `HISTORY_REWRITE` — rewrite a dependent follow-up into a standalone search
  query before retrieving. Costs one short extra LLM call on follow-ups only;
  set `false` to trade follow-up accuracy for speed.
- `CLAUSE_CANDIDATE_K` / `MIN_CHUNKS_PER_CLAUSE` — per-ask search size, and the
  slots each ask of a compound question is guaranteed. Defaults `10` / `2`; set
  the latter to `0` to rank purely by score.
- `MAX_CHUNKS_PER_SOURCE` — most chunks one document may take in the top-K.
  Default `3`; `0` turns the cap off. A landmark case discusses its doctrine
  across many passages and out-scores everything on all of them: before the cap,
  "what is informed consent" gave **8 of 10 slots to Dr. Rubi Li**, leaving 3
  distinct documents in the whole context. With it, that question returns 5
  documents and the top document's share across a 10-question sample fell from
  47% to 28%. Overflow is demoted rather than dropped, so a question only one
  statute answers still fills its context.

### OCR
- `OCR_ENABLED` — turn OCR on/off. Default `true`.
- `OCR_LANGUAGE` — Tesseract language(s). Default `eng`.
- `OCR_DPI` — image resolution for OCR; higher = more accurate but slower.
  Default `300`.
- `OCR_MIN_CHARS` — if a PDF yields fewer characters than this, treat it as
  scanned and try OCR. Default `200`.
- `TESSERACT_CMD` — explicit path to `tesseract.exe` if auto-detection fails.

---

## 12b. Why it's slow, and what actually helps

This used to open with "even on a dedicated GPU, this is single-user readable speed, not
interactive-fast" — true of the project's earlier 27B default (measured on an RTX 3090 24GB
running `QuantTrio/Qwen3.6-27B-AWQ`; full baseline in `HAVEN_VLLM_MIGRATION.md` §2), and no longer
true of the current default. Measured with `evals/bench_llm.py` against the running config
(`Qwen/Qwen3-14B-AWQ`, no `--enforce-eager`, same 24GB card):

| Metric | Value |
|---|---|
| Weights resident | 9.44 GiB |
| KV cache available | 12.13 GiB (158,992 tokens) |
| Max context per request | 8,192 |
| Max concurrency @ 8K context | 19.41× |
| Generation throughput | ~79 tok/s |
| Time to first token, warm prefix | ~30 ms |
| Time to first token, cold prefix | ~600-690 ms |
| Engine init (container startup) | ~8-14s, one-time |

A 1024-token answer (`LLM_MAX_TOKENS`, the default cap) now takes well under 15 seconds of decode
— down from the "can take over a minute" of the 27B era, which is why `LLM_TIMEOUT` still defaults
to a generous `180` even though the tight margin that number was originally sized for no longer
applies. Retrieval + reranking is still cheap by comparison, running in low single-digit seconds on
CPU — with generation this much faster, it's now a larger share of total wall-clock than before,
not because it got slower but because generation got faster around it.

**Prefill rate** (how fast the model reads the retrieved-passages prompt before it starts writing)
still hasn't been isolated from total wall-clock on this stack — the table above times full
generations, not prefill alone. Per this guide's own rule in §15 — if you touch a threshold,
measure it — the same applies to performance claims: don't take the table above as the last word on
*your* hardware, and re-run `evals/bench_llm.py` after any model or flag change rather than
assuming a ratio from these numbers.

Levers, roughly in order of value:

| Change | Effect | Costs you |
|---|---|---|
| Raise `--max-num-seqs` | real concurrency — measured headroom is 19× at 8K context, and it's currently capped at 2 | more requests sharing the same decode throughput |
| More VRAM (32GB+) | room for a MoE-class model, longer context, more concurrency | hardware |
| Lower `TOP_K` | less prompt to prefill | less context per answer |
| Lower `LLM_MAX_TOKENS` | linear | shorter answers |
| `RERANK_ENABLED=false` | ~2.5s | ranking quality |

`--enforce-eager` is no longer part of the launch command (see §5 flags) — at the 14B model's
footprint, CUDA graphs capture cleanly alongside a full KV cache, so there's no longer a tradeoff
here to make. That wasn't true of the 27B default: measured, CUDA graphs and the n-gram
speculative-decode drafter competed for the same VRAM and could not both fit, which is why eager
mode used to be required and speculative decoding used to be the one worth keeping. Speculative
decoding isn't configured at all now (see the reverted-and-not-retested note in CLAUDE.md's
"Generation speed" section), so that specific tradeoff no longer applies either.

### Don't benchmark against yourself
The failure mode that actually bit this project, not a hypothetical one:
`ragmed/embeddings.py` and `ragmed/rerank.py` both load small models that are
meant to run on CPU regardless of what GPU is available — but one code path
in each omitted the explicit `device="cpu"` pin, so they auto-detected CUDA
instead. That's invisible on a machine with GPU headroom to spare. It is not
invisible once vLLM is running, because `--gpu-memory-utilization 0.93`
leaves well under a gigabyte free — so the very next thing that tries to
touch the GPU (the embedder, on an offline-fallback path; the reranker, on
every load) hits a hard `torch.OutOfMemoryError` mid-query. The fix was two
lines (add `device="cpu"` to both loaders), but the lesson generalises: once
one process on a box is deliberately saturating a resource, *anything else
that silently defaults to the same resource* will fail in a way that looks
unrelated. Check what's already holding the GPU (or CPU, or RAM) before
chasing a crash as if it were a logic bug.

---

## 13. The corpus: what's inside and how to grow it

The index described here holds **9,066 chunks from 156 documents**, spanning five
sources:

- **Statutes** (from LawPhil): the Medical Act of 1959, the UHC Act,
  Anti-Hospital Deposit/Detention laws, Data Privacy Act, Civil Code,
  Revised Penal Code, and more.
- **PRC issuances**: Board of Medicine Board Law, Code of Ethics, resolutions,
  and the disciplinary/administrative-investigation rules (these last ones came
  in via OCR).
- **DOH issuances** (from the Supreme Court E-Library): hospital licensing
  orders, the implementing rules for the patients'-rights laws, and the Magna
  Carta of Public Health Workers.
- **Jurisprudence** (Supreme Court E-Library): medical malpractice, informed
  consent, hospital liability, and the negligence/surgery bundles.
- **Billing, payment and non-payment** (added 2026-07-29): PhilHealth's charter
  and No Balance Billing circulars, HMO payment cases, senior-citizen and PWD
  discount laws with their IRRs, government medical assistance, small-claims
  procedure, and the deceased-patient chain — Rule 86 claims against the
  estate, the Family Code's support and community-property rules, and PD 856
  on disposal of the remains.

### Three ways to add more
1. **Drop your own files** into `corpus/` (any subfolder), then `ingest`.
   Text-based PDFs work best; scanned ones are handled by OCR automatically.
2. **Add URLs to a seed file** (`fetch/seeds*.txt`, format `name | url`) and run
   the fetcher. Good sources: **LawPhil** (statutes & jurisprudence), the
   **Supreme Court E-Library** (statutes, IRRs, DOH orders — reliable full text).
3. **Mix both.**

### Check the SHAPE of what you fetched, not just that it downloaded
A file can arrive complete, correct, and on-topic, and still be the wrong
*shape* to index. Two patterns recur, both cheap to fix now and expensive later
— catching either after indexing costs a full `ingest --reset` rebuild.

**One page holding many numbered documents.** LawPhil publishes Rules of Court
72–109 as a single 148k-character page. Indexed whole it produced **199 chunks
sharing just 25 section labels** — every rule restarts its own numbering, so
`Section 2.` meant 27 different things — and no citable identity at all. The
passage answering a real question lost to its own 198 siblings: retrieval found
the right *file* and the wrong *section*, and the answer denied a rule the file
plainly contains. `fetch/split_rules_of_court.py` splits it into one file per
rule; Rule 86 went from competing among 199 chunks to 13, every one citable.

**PDFs padded with restated text.** Official PDFs often append comparative
tables or consolidated redlines. The 2022 small-claims rules PDF was 484k
characters, of which **252k (52%) was the same rule text printed three times**
side by side. Trimmed to the operative text before indexing. Note the duplicate
filter can't help here — three-column layout makes the copies textually
different.

How to spot both, right after fetching:
```powershell
# outlier file sizes
Get-ChildItem corpus\_fetched\*\*.txt | Sort-Object Length | Select-Object -Last 5 Name,Length
# numbering that restarts = many documents in one file
Select-String -Path "<suspect-file>" -Pattern "^Section 1\." | Measure-Object
```

Whenever you trim or split, **say so in the seed file** — what you removed and
why. And remember re-fetching restores the original, so a split has to be a
re-runnable script, not a one-off edit.

### A note on document quality
Garbage in, garbage out. A thin corpus gives thin answers. The single best thing
you can do to improve answer quality is **add more relevant, high-quality
documents** — far more impactful than tweaking model settings.

One caveat worth internalising, because it cuts the other way: **don't add
documents to fix a ranking problem.** If an answer is wrong, first find out
whether the source is missing, present but out-ranked, or present and retrieved
as the wrong chunk. Those need opposite fixes, and adding sources only helps the
first. See the troubleshooting entry below.

---

## 14. Troubleshooting & FAQ

**Q: I ran `ask` and it says the LLM server isn't reachable.**
The vLLM container isn't running, hasn't finished loading (`docker logs vllm-qwen14b`
— engine init takes ~60s), or `LLM_BASE_URL`/`LLM_MODEL` don't match how it
was started. Check `docker ps`, then `curl http://localhost:8000/v1/models`
directly — you should see your model listed. `cli.py status` should then show
`LLM server: OK`.

**Q: `torch.OutOfMemoryError: CUDA out of memory`, pointing at `embeddings.py`
or `rerank.py`.**
vLLM reserves most of the GPU on purpose (`--gpu-memory-utilization 0.93`), so
there's normally under a gigabyte free. Embeddings and the reranker are meant
to stay on CPU regardless — if you see this, something in that path is trying
to auto-detect CUDA instead of using the explicit `device="cpu"` pin (this
happened once already; see the callout in §10). If you just edited
`ragmed/embeddings.py` or `ragmed/rerank.py` to fix exactly this and it's
*still* happening: restart whatever process is running the query
(`streamlit run app.py`, `cli.py chat`, etc.). Python doesn't reload code
you've already imported into a running process — and confusingly, a
traceback printed *after* your edit will still show your fixed source line,
because Python reads tracebacks fresh off disk at print time, not from what
was actually executed. A traceback that looks like it contradicts the code
you're staring at is a sign to check whether the process predates the fix,
not to doubt the fix.

**Q: I added/changed something in `ragmed/` and nothing happened — no error,
just silence (e.g. `dashboard.py`'s Query metrics never gains a new row after
asking a question in Haven).**
Same root cause as the entry above, minus the crash — which makes it easier
to miss. A long-running process (`streamlit run app.py`, `streamlit run
dashboard.py`, `cli.py chat`) only has the version of `ragmed/*.py` that was
on disk when it *started*; Python does not hot-reload modules it has already
imported. If a Haven session has been open since before you added, say, the
query-metrics logging in `rag.py`, every question it answers runs the old
`rag.py` with no logging call in it at all — correctly, silently, and with
nothing to point at the real cause. Check `ps -o pid,lstart,cmd -p <pid>`
against `stat -c '%y' ragmed/whatever.py` for the file(s) you changed; if the
process started first, restart it. This is worth checking *before* debugging
the feature itself — it is the more likely explanation than a logic bug for
anything RAG-adjacent that "does nothing" rather than errors.

**Q: The answer says "The provided corpus does not cover this."**
That's the system being *honest*, not broken. It means the retrieved chunks
didn't contain the answer. Fix by adding the relevant document to `corpus/` and
re-ingesting. This is a feature — it's refusing to make things up.

**Q: The answer is confident but wrong — or it answered only half my question.**
This is the failure mode to take most seriously, because nothing looks broken.
Don't start adding documents; first find out *which* of three things went wrong,
because they need opposite fixes:

1. **The source is missing** → corpus problem. Add it, re-ingest.
2. **The source is there but out-ranked** → retrieval problem. Adding more
   documents makes this *worse*.
3. **The source was retrieved, but the wrong chunk of it** → chunking or
   document-shape problem (see "Check the SHAPE" above).

Two cheap probes tell them apart:

```powershell
# (a) Re-ask just the failing part, in the corpus's own vocabulary
.\.venv\Scripts\python.exe cli.py ask "What is the deadline for filing a money claim against the estate of a deceased person?"

# (b) Look at raw retrieval, bypassing the LLM entirely
.\.venv\Scripts\python.exe -c "from ragmed import retriever; [print(f'{r.score:.3f} {r.metadata.get(\"source\")}') for r in retriever.retrieve('your question', top_k=25)]"
```

A real example from this corpus: a compound question about a deceased patient's
unpaid bill gave a fluent answer that denied any filing deadline existed. Probe
(a) — the same question asked narrowly — ranked the correct document **#2**.
Probe (b) showed the compound question had it **nowhere in the top 25**. That
one comparison proved the corpus was fine and pointed the fix at retrieval,
saving a pointless hunt for documents already sitting in the index.

Also glance at the **source list under every answer**: if 8 of 10 citations name
the same document, the context was crowded even if this particular answer came
out right. That's what `MAX_CHUNKS_PER_SOURCE` is for.

**Q: A document I added isn't showing up in answers.**
Three usual causes: (1) you forgot to run `ingest` after adding it; (2) it's a
scanned PDF and OCR is off or Tesseract isn't found (check `cli.py status`); (3)
it's an unsupported file type. Run `ingest` and read its summary — it lists any
skipped files and why.

**Q: How do I know the answer is correct?**
Every answer lists its **sources** (law, section, file). Open the cited section
in the actual document and verify. Never trust a legal answer you haven't
checked against the source — RAG makes this easy by design, but it's still on you
to look.

**Q: I changed `EMBED_MODEL` / `CHUNK_SIZE` and things got weird.**
Those change the *index itself*. Run `cli.py ingest --reset` to rebuild cleanly.
Retrieval-only settings (`TOP_K`, weights) don't need a re-ingest.

**Q: Is any of my data being sent to the cloud?**
No. Embeddings, the vector store, and the LLM all run locally. The only time the
system touches the internet is (a) the first download of the embedding model,
(b) when *you* run a fetcher to download documents, and (c) when the vLLM
container pulls model weights from Hugging Face on first start. Everyday
questions are 100% offline.

**Q: Is this legal advice?**
No. It is a *legal-information retrieval tool* over a corpus someone curated. It
does not interpret the law, does not account for facts specific to a case, and
does not replace a licensed Philippine attorney. The system prompt requires it
to say so, and every answer ships with the sources it used precisely so a human
can check them. **Verify against the cited text before relying on anything.**

**Q: How accurate is it, honestly?**
Accuracy depends far more on the corpus than on the model. When a question is
squarely covered by an indexed document, answers are generally well-grounded and
correctly cited. When it isn't, the system is prompted to refuse — but the
failure mode to watch for is the one documented in §14 above: a *fluent, wrong*
answer that reads exactly like a correct one. This guide records several real
examples, including one where the system confidently denied a filing deadline
that the Rules of Court impose. Read the citations, not just the prose.

---

## 15. How to extend it

Natural directions once the basics are clear, roughly easiest first:

1. **Point it at a different corpus.** Nothing above is specific to Philippine
   law except the seed lists and a few identifier patterns in `chunking.py`.
   Swap those and the same engine indexes RFCs, policy manuals, or internal
   documentation.
2. **Grow the corpus** — the jurisprudence, negligence, surgery and billing
   bundles ship as seed lists; add whatever else the domain needs. Check the
   *shape* of anything new before ingesting (§13).
3. **Tune retrieval** — if exact-citation questions matter most to you, nudge
   `LEXICAL_WEIGHT` up; if conceptual questions dominate, keep semantic higher.
   Try `TOP_K = 8` for more context.

   **If you touch a similarity threshold, measure it — don't guess.** Write ~5
   questions the corpus answers and ~5 it can't, print their *raw* similarities
   (fused scores are normalised per-query and can't be compared), and put the
   threshold in the gap between the two ranges. If the ranges overlap, your
   *signal* is wrong and no number will save it — that's exactly what happened
   when tuning `HYDE_CLAUSE_MIN_SIM`: scoring every clause of a question gave
   total overlap, scoring only its *asks* separated cleanly and the threshold
   fell out on its own.
4. **Run the tests after changing the engine** — `tests/` holds 308 checks, each
   encoding a bug that shipped once (duplicate crowding, the per-source cap,
   per-ask slot reservation, the reranker's fail-open and flat-score guards, the
   HyDE clause gate, citation labelling, the fetcher's retry budget). Add a case
   whenever you fix something.
5. **The reranker is already in** (§8, step 3c) — if queries feel slow, that's
   where the seconds go: `RERANK_ENABLED=false` or the MiniLM model. Before
   tuning it, work out whether your problem is **ordering** (right document,
   wrong chunk — the reranker's job) or **reach** (the chunk never retrieved at
   all — it can't help). They look identical in a bad answer and need opposite
   fixes.
6. **Improve citations** — parse section numbers even more precisely, or add
   clickable links back to the source documents in the web UI.
7. **Try a different LLM** — swap `--model` in the vLLM `docker run` command
   and match `LLM_MODEL` in `.env`. Sizing is the real constraint: the current
   14B default leaves most of a 24GB card free (9.44 GiB weights resident); a
   24GB card can still go up to a 27B-dense model with the vision encoder
   disabled, though that size needs `--enforce-eager` again once speculative
   decoding or CUDA graphs compete for the same VRAM (see §12b), and a
   35B-class MoE needs 32GB+. See `HAVEN_VLLM_MIGRATION.md` §6 for the
   measured boundary before assuming a bigger model just drops in.

### The one-paragraph recap to lock it in
> Your system **indexes** documents once (read → chunk → embed → store) and then
> **answers** questions on demand (embed the question → hybrid-search for the
> best chunks → feed them to a local LLM with strict "cite your sources, don't
> make things up" instructions → return the answer plus its sources). Everything
> runs on one machine. The librarian finds the pages; the assistant writes the
> answer from those pages only. That's RAG.

That is the whole system. Every component above exists to make one of those
arrows trustworthy.
