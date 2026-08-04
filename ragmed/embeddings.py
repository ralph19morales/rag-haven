"""Local embedding model wrapper (sentence-transformers).

Loaded lazily and cached so the model is only read from disk once per
process. Uses the bge query-instruction convention: queries get a prefix,
documents do not.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _model():
    """Load the embedding model, falling back to the on-disk cache when the
    Hugging Face hub is unreachable.

    sentence-transformers contacts huggingface.co on load even when the model
    is already cached, so a DNS hiccup takes down a system whose whole point is
    running locally — observed here as a hard crash mid-query.

    The fallback resolves the cached snapshot to a local DIRECTORY and loads
    that. Setting HF_HUB_OFFLINE at this point does NOT work: huggingface_hub
    reads it into a module constant at import time, so a late env change is
    ignored and the retry goes back to the network. A filesystem path bypasses
    hub lookups entirely.

    The first-ever load still needs the network — nothing is cached yet, so
    `local_files_only` raises and the original error surfaces."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(config.EMBED_MODEL, device="cpu")
    except Exception:  # noqa: BLE001 - network, DNS, TLS, hub errors
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(config.EMBED_MODEL,
                                      local_files_only=True)
        return SentenceTransformer(local_dir, device="cpu")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages/chunks for storage."""
    model = _model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,   # cosine == dot product
        show_progress_bar=len(texts) > 64,
        batch_size=32,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    """Embed a single search query (with the bge query prefix)."""
    model = _model()
    vec = model.encode(
        config.EMBED_QUERY_PREFIX + text,
        normalize_embeddings=True,
    )
    return vec.tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed several search queries at once (with the bge query prefix).

    One batched encode rather than a loop of `embed_query` calls: the clause
    coverage check in the retriever embeds every clause of a compound question,
    and on CPU the per-call overhead dominates at these sizes."""
    if not texts:
        return []
    model = _model()
    vecs = model.encode(
        [config.EMBED_QUERY_PREFIX + t for t in texts],
        normalize_embeddings=True,
        batch_size=16,
    )
    return [v.tolist() for v in vecs]


def embedding_dim() -> int:
    return _model().get_sentence_embedding_dimension()
