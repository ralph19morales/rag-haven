"""Cross-encoder reranking: score (question, chunk) pairs directly.

Why this exists — the failure it was added to fix:

Bi-encoder retrieval embeds the question and the chunk SEPARATELY and compares
the two vectors. That is what makes it fast enough to search a whole corpus,
and also what limits it: the chunk's vector is computed without ever seeing the
question, so two chunks that use similar language score alike even when only one
of them answers what was asked. Measured on this corpus, the ask "who can we
collect from, and is there a deadline?" scored eight different chunks inside a
1.4% spread (0.588-0.596) — Rule 86's filing deadline sat 5th, below chunks that
merely shared its lay phrasing. Every retrieval-side fix (splitting the source,
clause-aware HyDE gating, the per-source cap, per-clause slot reservation) moved
the right DOCUMENTS into the prompt and still landed on the wrong CHUNKS within
them: Boston Equity arrived as its compulsory-joinder section, the Civil Code as
its article on pledges.

A cross-encoder reads the question and the chunk TOGETHER in one forward pass,
so it can judge "does this passage answer this question" rather than "is this
passage about similar things". It is far too slow to run over a whole corpus —
which is exactly why it runs here, over the handful of candidates retrieval has
already narrowed to.

Degrades gracefully: if the model cannot load, reranking is skipped and the
fused ordering stands. A ranking improvement must never be able to take the
system down.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _model():
    """Load the cross-encoder, falling back to the on-disk cache when the
    Hugging Face hub is unreachable — same pattern as `embeddings._model`,
    and for the same reason: sentence-transformers contacts huggingface.co on
    load even when the model is already cached, so a DNS hiccup would otherwise
    break a system whose whole point is running locally."""
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(config.RERANK_MODEL, max_length=512, device="cpu")
    except Exception:  # noqa: BLE001 - network, DNS, TLS, hub errors
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(config.RERANK_MODEL,
                                      local_files_only=True)
        return CrossEncoder(local_dir, max_length=512, device="cpu")


def is_available() -> bool:
    """True when reranking is enabled AND the model actually loads.

    Used by `cli.py status` so a silently-disabled reranker is visible rather
    than being mistaken for a reranker that is working badly."""
    if not config.RERANK_ENABLED:
        return False
    try:
        _model()
        return True
    except Exception:  # noqa: BLE001
        return False


def _sigmoid(x: float) -> float:
    import math

    # Cross-encoders emit a raw logit. Squashing to 0-1 keeps the score in the
    # same range users already see next to every citation, so a reranked score
    # and a fused score read on the same scale.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)                      # avoids overflow on large negatives
    return e / (1.0 + e)


def score(query: str, texts: list[str]) -> list[float] | None:
    """Score each text against the query. Returns None if reranking is off or
    the model is unavailable, so callers can keep their existing ordering."""
    if not config.RERANK_ENABLED or not texts:
        return None
    try:
        model = _model()
        raw = model.predict([(query, t) for t in texts],
                            batch_size=config.RERANK_BATCH_SIZE,
                            show_progress_bar=False)
    except Exception:  # noqa: BLE001 - model load or inference failure
        return None
    return [_sigmoid(float(s)) for s in raw]
