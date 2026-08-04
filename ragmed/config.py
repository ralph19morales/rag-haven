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
HYDE_MAX_TOKENS = _env_int("HYDE_MAX_TOKENS", 200)  # keep the draft short

# --- LLM (local, via vLLM — OpenAI-compatible API) --------------------------
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = _env("LLM_API_KEY", "not-needed")  # vLLM ignores it; the client just requires a non-empty string
LLM_MODEL = _env("LLM_MODEL", "QuantTrio/Qwen3.6-27B-AWQ")
# Qwen3 loops on greedy decode — do not set temperature to 0.
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(_env("LLM_TOP_P", "0.95"))
LLM_TOP_K = _env_int("LLM_TOP_K", 20)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)  # caps TOTAL generation, not just the reply
# Measured ~14 tok/s on a 24GB card — a 1024-token answer can take over a
# minute, so the default client timeout surfaces as a false retrieval failure.
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 180)
# Qwen3.6 is a reasoning model: left on, it spends max_tokens on chain-of-
# thought (in a separate `reasoning` field) and can return content=None. The
# retrieved/reranked context has already narrowed the answer, so extended
# reasoning adds latency without adding grounding. Not a standard OpenAI
# param — sent via extra_body in ragmed/llm.py.
LLM_ENABLE_THINKING = _env_bool("LLM_ENABLE_THINKING", False)


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to."""
    for d in (CORPUS_DIR, FETCHED_DIR, DATA_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
