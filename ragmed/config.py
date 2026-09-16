"""Central configuration for the Philippine medical-law RAG system.

Everything tunable lives here. Override any value with an environment
variable (or a .env file in the project root) of the same name.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file from the project root if present.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# --- Hugging Face offline mode --------------------------------------------
# sentence-transformers revalidates every model file against huggingface.co on
# load, even when the model is fully cached on disk. Measured here: 9.9s to load
# bge-base and 5.6s for bge-reranker-base, against 4.1s and 1.5s with the
# network calls skipped — roughly 10 SECONDS of pure round-trips per process,
# on a system whose entire premise is that it runs locally. `cli.py ask` starts
# a fresh process per question and paid it every time.
#
# This MUST be set before huggingface_hub is first imported: the library reads
# the variable into a module constant at import time, so setting it later is
# silently ignored (the same trap documented in embeddings._model). config.py is
# imported before any model code, which is why it lives here and not there.
#
# setdefault, so an explicit environment value still wins. Set HF_OFFLINE=false
# when you need to DOWNLOAD a model you have not cached yet — with it on, a
# missing model fails to load rather than being fetched.
if _env("HF_OFFLINE", "true").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# --- Paths -----------------------------------------------------------------
CORPUS_DIR = PROJECT_ROOT / "corpus"          # raw documents you drop in
FETCHED_DIR = CORPUS_DIR / "_fetched"          # scraper output
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"               # persistent vector store
BM25_PATH = DATA_DIR / "bm25.pkl"              # cached lexical index

# --- Embeddings (local, via sentence-transformers) -------------------------
# bge-base is a good quality/speed balance on CPU. Swap to
# "BAAI/bge-small-en-v1.5" (384-dim, faster) or "BAAI/bge-large-en-v1.5"
# (higher quality, slower) via the EMBED_MODEL env var.
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# bge models are trained to prepend this instruction to *queries* only.
EMBED_QUERY_PREFIX = _env(
    "EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

# --- OCR (for scanned / image-only PDFs) -----------------------------------
# When a PDF yields little/no extractable text, fall back to OCR via
# Tesseract (pages are rendered with pypdfium2 — no Poppler needed).
def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


OCR_ENABLED = _env_bool("OCR_ENABLED", True)
OCR_LANGUAGE = _env("OCR_LANGUAGE", "eng")     # Tesseract lang code(s)
OCR_DPI = _env_int("OCR_DPI", 300)             # render resolution for OCR
OCR_MIN_CHARS = _env_int("OCR_MIN_CHARS", 200)  # below this, try OCR
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 0)   # 0 = all pages
# Optional explicit path to tesseract.exe; auto-detected if unset.
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = _env("COLLECTION_NAME", "ph_medical_law")

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1100)      # target chars per chunk
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)  # char overlap between chunks

# --- Retrieval -------------------------------------------------------------
TOP_K = _env_int("TOP_K", 6)                   # chunks passed to the LLM
CANDIDATE_K = _env_int("CANDIDATE_K", 25)      # candidates before fusion
# Diversity cap: at most this many chunks from any ONE source file may occupy
# the top_k. A landmark document chunks into many near-identical passages and
# wins them all — measured on this corpus, the informed-consent question filled
# 8 of 10 slots with Dr. Rubi Li alone, leaving 3 distinct documents in the
# whole context. The cap DEMOTES the overflow rather than dropping it: if
# nothing else scores, the demoted chunks backfill, so a question that only one
# document answers still gets a full context. 0 disables the cap.
MAX_CHUNKS_PER_SOURCE = _env_int("MAX_CHUNKS_PER_SOURCE", 3)
# The same idea one level up: most chunks any one corpus FAMILY (lawphil,
# jurisprudence, doh, prc, billing) may take in the top_k. A per-FILE cap does
# not constrain this — the deceased-body question filled four of six slots with
# jurisprudence drawn from three different case files, satisfying the per-source
# cap while case law still owned two thirds of the context, leaving no room for
# the statute and its IRR to answer together. Demotes rather than drops, so a
# question only one family answers still fills its context. 0 disables.
#
# Measured over the 7-question authority set, sweeping the cap:
#
#   cap  authority  statute+IRR together  families/top_k  off-topic (body)
#     0      7/7            0/1               2.00             5/6
#     2      7/7            1/1               2.86             4/6
#     3      7/7            1/1               2.29             4/6
#
# 2, on the strength of the statute+IRR column and one concrete gain it is easy
# to miss in the averages: at 2 the licensure rules reach the prompt, and they
# name "refusal ... to release cadavers ... for non-payment of hospital bills"
# as a violation outright (RA 4226, Sec. 17). That is a second operative
# prohibition the answer could not previously cite.
#
# What the sweep does NOT show, and was checked separately: "more families" is
# a proxy that can be gamed by importing junk. The informed-consent question —
# genuinely answered by jurisprudence alone — carries exactly one off-topic
# chunk at cap 0, 2 and 3 alike, so the cap swaps WHICH unrelated chunk appears
# rather than adding one. Re-check that case, not just the mean, before raising
# this: forcing breadth on a question one family answers is the failure mode.
MAX_CHUNKS_PER_FAMILY = _env_int("MAX_CHUNKS_PER_FAMILY", 2)
# Per-clause retrieval, for questions that ask more than one thing. Each ask
# gets its own vector query (CLAUSE_CANDIDATE_K hits) and is then GUARANTEED
# MIN_CHUNKS_PER_CLAUSE slots in the top_k.
# Why a guarantee rather than more candidates: min-max normalisation scores
# everything against the same question, so chunks answering a neglected half
# normalise below chunks matching the dominant half no matter how many are
# fetched. On the deceased-patient question the top twelve all scored >=0.894
# and were all one half of it. Raising CANDIDATE_K "fixed" it at 60 and broke
# again at 120 — a number fitted to one query, since a bigger pool moves the
# normalisation window. Reserving capacity is a mechanism instead.
# Set MIN_CHUNKS_PER_CLAUSE=0 to disable and rank purely by fused score.
CLAUSE_CANDIDATE_K = _env_int("CLAUSE_CANDIDATE_K", 10)
MIN_CHUNKS_PER_CLAUSE = _env_int("MIN_CHUNKS_PER_CLAUSE", 2)

# --- Reranking (cross-encoder) ---------------------------------------------
# Retrieval embeds question and chunk separately, so it cannot tell "answers
# this question" from "is about similar things". Measured here: one ask scored
# eight chunks inside a 1.4% spread and put the answering rule 5th. A
# cross-encoder reads the pair TOGETHER and scores relevance directly. Too slow
# for a whole corpus, which is why it runs only over candidates retrieval has
# already narrowed. Adds seconds per query on CPU — set RERANK_ENABLED=false to
# trade that accuracy back for speed.
RERANK_ENABLED = _env_bool("RERANK_ENABLED", True)
# bge-reranker-base pairs with the bge embedder and handles formal register
# well. "cross-encoder/ms-marco-MiniLM-L-6-v2" is ~10x smaller and much faster
# if latency matters more than ranking quality.
RERANK_MODEL = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 30)  # how many to rescore
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 16)
# Minimum score spread for a reranking to be trusted. A cross-encoder that
# cannot tell its candidates apart returns them all at ~0.5, and sorting that is
# sorting noise — measured, it promoted an unrelated case to rank 1 on a
# compound question whose candidates spanned 0.0004. The same model separates by
# 0.1975 when the query names what it asks about. Below this, the fused ordering
# stands.
RERANK_MIN_SPREAD = float(_env("RERANK_MIN_SPREAD", "0.02"))
# How many of the FUSED top results are guaranteed to survive reranking.
#
# The reranker is allowed to reorder, not to overrule. Measured failure: asked
# "the hospital won't release my relative's body until we pay", hybrid fusion
# put RA 9439 — the Anti-Hospital Detention Law, which is the whole answer — at
# rank 3, and the cross-encoder demoted it to rank 17, out of the prompt. Five
# of the six slots went to case law, three of them about a DIFFERENT statute.
# The only RA 9439 material left was the IRR's list of the offence's elements,
# which the model then read as a checklist of when detention is ALLOWED, and it
# answered that a hospital may lawfully withhold a body. The exact inverse of
# the law, to the exact person least able to check it.
#
# Why this shape of fix. A cross-encoder is strong at "does this passage answer
# this question" and weak on terse statutory text against a lay narrative: four
# lines of legislative prohibition look less responsive than pages of judicial
# discussion around the same facts. That is precisely the case where the
# bi-encoder was right and the reranker was wrong, so the reranker must not get
# the last word alone. Rather than tune scores against each other, reserve
# capacity — the same mechanism, and the same reasoning, as
# MIN_CHUNKS_PER_CLAUSE.
#
# The KNEE of a measured curve, not a number fitted to the failing question.
# Measured over 7 questions whose controlling authority is known independently
# (5 statute-governed, 2 deliberately case-law-governed). Originally, on the
# pre-chunk-fix index, by AUTHORITY alone:
#
#   protect=0  4/7      protect=3  7/7
#   protect=2  6/7      protect=4  7/7      protect=5  7/7
#
# which saturates at 3, and 3 was the default for that reason. Re-measured
# 2026-08-05 on the boundary-aware index, adding the off-topic count on the
# informed-consent case (pure jurisprudence — the guard against a change that
# merely drags statutes upward):
#
#   protect=3  auth 7/7, consent off-topic 1/6
#   protect=4  auth 7/7, consent off-topic 0/6   <- default
#   protect=5  auth 7/7, consent off-topic 0/6
#   protect=6  auth 6/7                          <- breaks
#   protect=7  auth 5/7, statute+IRR 0/1         <- reranker bypassed entirely
#
# AUTHORITY still saturates at 3, so the move to 4 is bought by the second
# metric alone: it is the smallest setting that also clears the off-topic chunk
# out of the case-law question. Note what the newer sweep corrects in the older
# one — "going higher buys nothing" was too kind. Past 5 it actively breaks,
# because a 6-slot context handed back to the bi-encoder is the failure the
# reranker was added to fix. 5 is the last safe value; 4 is the chosen one.
# 0 disables the guarantee.
RERANK_PROTECT_TOP = _env_int("RERANK_PROTECT_TOP", 4)

# --- Conversation memory ---------------------------------------------------
# How many prior user+assistant PAIRS are replayed to the model. Kept small on
# purpose: history shares a fixed context window with the retrieved passages,
# and passages are what the answer must be grounded in. Budget at TOP_K=10 and
# an 8192-token window leaves roughly 3,900 tokens for history — 3 pairs at the
# limits below use well under that.
HISTORY_TURNS = _env_int("HISTORY_TURNS", 3)
HISTORY_USER_CHARS = _env_int("HISTORY_USER_CHARS", 400)
# Assistant replies are the long part of a transcript and the least useful to
# replay whole — their substance is in the passages, which are re-retrieved
# every turn regardless.
HISTORY_ASSISTANT_CHARS = _env_int("HISTORY_ASSISTANT_CHARS", 500)
# Rewrite a context-dependent follow-up into a standalone search query before
# retrieving. Costs one extra (short) LLM call on follow-ups only. Without it,
# "and is there a deadline?" is embedded as written and retrieves noise — no
# amount of history in the generation prompt can repair that, because retrieval
# has already happened by then. Set false to trade follow-up accuracy for speed.
HISTORY_REWRITE = _env_bool("HISTORY_REWRITE", True)
HISTORY_REWRITE_MAX_TOKENS = _env_int("HISTORY_REWRITE_MAX_TOKENS", 60)
# Weight of dense (semantic) vs lexical (BM25) scores in hybrid fusion.
DENSE_WEIGHT = float(_env("DENSE_WEIGHT", "0.6"))
LEXICAL_WEIGHT = float(_env("LEXICAL_WEIGHT", "0.4"))

# --- HyDE (hypothetical document embeddings) -------------------------------
# A question asked in lay narrative ("they kept changing their story") does not
# embed near terse statutory text, and carries no distinctive tokens for BM25 —
# so retrieval returns noise and the system falsely refuses. HyDE has the LLM
# draft a short hypothetical ANSWER in domain vocabulary and retrieves with
# that instead, bridging the vocabulary gap.
#   off    — never (fastest; original behaviour)
#   auto   — only when the question retrieves poorly (default)
#   always — every query, costs one extra LLM call each time
HYDE_MODE = _env("HYDE_MODE", "auto").strip().lower()
# Auto trigger: run HyDE when the best RAW cosine similarity is below this.
# Raw similarity is used because the fused scores are min-max normalised and
# therefore not comparable across queries. Measured on this corpus: well-posed
# questions scored 0.696–0.789, lay-narrative ones 0.594–0.653. 0.68 sits in
# that gap — re-measure if you change the embedding model or the corpus.
HYDE_MIN_SIM = float(_env("HYDE_MIN_SIM", "0.68"))
# Compound questions defeat the check above: similarity over the whole text is a
# MAXIMUM, so one well-matched clause hides the rest. "Can we refuse to release
# the body ... who can we collect from, and is there a deadline?" scored 0.736 —
# above the threshold — because its first half matches the anti-detention law
# almost exactly; the estate-claim half was never retrieved and the answer said
# there was no deadline, though Rule 86 sets one. So each clause is also scored
# on its own and HyDE fires when the WORST one falls below this. Kept as a
# separate knob because a lone clause is shorter than a full question and need
# not share its threshold — re-measure before moving either.
# Measured on this corpus over a 9-question sample: asks the corpus answers
# scored 0.721-0.783, asks it does not 0.605-0.678. 0.70 sits in that gap.
# (Scoring EVERY clause instead of just the asks was tried and abandoned — the
# two groups overlapped completely, because the weakest clause of an answerable
# question is its narrative setup, not its question.)
HYDE_CLAUSE_MIN_SIM = float(_env("HYDE_CLAUSE_MIN_SIM", "0.70"))
# Keep the draft short — this is the single most expensive step in retrieval.
# The draft is generated at ~19 tok/s, serially, before retrieval can finish, so
# every token is ~53ms on the critical path: the old 200-token draft cost 10.5s
# of a 50s answer, and the model wrote to the cap every time (i.e. it was being
# truncated mid-sentence, the tell that the CAP and not the prompt was setting
# the length).
#
# What actually fixed that was the prompt, not this number: `_HYDE_SYSTEM` now
# asks for "2-3 sentences, and stop" and leads with the terms of art, which is
# the only part that does any work — HyDE contributes domain VOCABULARY, and
# that is front-loaded. Drafts now come back at ~70 words / ~95 tokens on their
# own, in 4.2-5.3s. So 128 is a safety ceiling, no longer the binding
# constraint; lowering it further would start truncating again without buying
# anything.
#
# Do NOT expect to A/B this cap by comparing retrieved chunks: HyDE is sampled,
# and the sampling noise is far larger than the cap. Measured over 3 runs of one
# unchanged question at a FIXED cap, only 2-3 of 10 retrieved ids were stable
# run to run. See LLM_SEED below, which is what makes that comparison possible
# at all.
HYDE_MAX_TOKENS = _env_int("HYDE_MAX_TOKENS", 128)

# --- LLM (local, via vLLM — OpenAI-compatible API) --------------------------
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = _env("LLM_API_KEY", "not-needed")  # vLLM ignores it; the client just requires a non-empty string
LLM_MODEL = _env("LLM_MODEL", "Qwen/Qwen3-14B-AWQ")
# Qwen3 loops on greedy decode — do not set temperature to 0.
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(_env("LLM_TOP_P", "0.95"))
LLM_TOP_K = _env_int("LLM_TOP_K", 20)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)  # caps TOTAL generation, not just the reply
# Measured ~14 tok/s on a 24GB card — a 1024-token answer can take over a
# minute, so the default client timeout surfaces as a false retrieval failure.
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 180)
# Name of the running vLLM Docker container. Read only by dashboard.py, to
# `docker inspect` the container's actual launch flags for the config-invariant
# checks (speculative decoding off, prefix caching on, etc.) — not used to
# reach the API, which always goes through LLM_BASE_URL.
VLLM_CONTAINER_NAME = _env("VLLM_CONTAINER_NAME", "vllm-qwen14b")
# Qwen3.6 is a reasoning model: left on, it spends max_tokens on chain-of-
# thought (in a separate `reasoning` field) and can return content=None. The
# retrieved/reranked context has already narrowed the answer, so extended
# reasoning adds latency without adding grounding. Not a standard OpenAI
# param — sent via extra_body in ragmed/llm.py.
LLM_ENABLE_THINKING = _env_bool("LLM_ENABLE_THINKING", False)
# Fixed sampling seed for the two auxiliary calls that FEED RETRIEVAL — the
# HyDE draft and the follow-up rewrite. Not applied to the answer itself.
#
# Those two calls decide what gets searched for, so sampling them makes the
# retrieved passages themselves random. Measured: asking one unchanged
# HyDE-triggering question three times, only 2-3 of its 10 retrieved chunk ids
# were the same across runs — so the same person asking the same thing twice was
# shown a substantially different set of sources, with no way to tell why. It
# also makes every retrieval change untestable, because the noise floor is
# bigger than most effects being measured.
#
# A seed fixes this WITHOUT greedy decoding, which matters: temperature 0 is not
# available to us (see LLM_TEMPERATURE — Qwen3 loops on it). Same question and
# same corpus now give the same passages; a different question still samples
# normally. Set LLM_SEED to a negative number to go back to unseeded drafts.
LLM_SEED = _env_int("LLM_SEED", 1729)

# --- Query metrics (for the ops dashboard, dashboard.py) -------------------
METRICS_PATH = DATA_DIR / "metrics.jsonl"
METRICS_ENABLED = _env_bool("METRICS_ENABLED", True)
# The question text is genuinely useful for an ops view (which question was
# slow / errored), but this app exists to handle people's private medical
# situations, so logging it is opt-out rather than an unexamined default.
# Everything still stays on this machine either way — data/ is git-ignored
# and nothing here is transmitted — this only controls what's on disk.
METRICS_LOG_QUESTIONS = _env_bool("METRICS_LOG_QUESTIONS", True)


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to."""
    for d in (CORPUS_DIR, FETCHED_DIR, DATA_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
