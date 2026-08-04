"""Chroma-backed persistent vector store.

We pass our own embeddings in (embedding_function=None) so the same local
bge model is used for both indexing and querying, and so Chroma never tries
to phone home for a default model.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _client():
    """The Chroma client, cached for the life of the process.

    Constructing a PersistentClient re-opens the sqlite file and reloads the
    HNSW index descriptor — measured at ~410ms here. `retriever.retrieve()`
    calls `get_collection()` on every query, so an uncached client put that
    410ms on the critical path of every single question. (app.py wrapped its
    own call in st.cache_resource, but the retriever never saw that cache —
    it calls this module directly.)"""
    import chromadb
    from chromadb.config import Settings

    config.ensure_dirs()
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


@lru_cache(maxsize=1)
def get_collection():
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Delete and recreate the collection (full reindex)."""
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    # The cached handle now points at a deleted collection — every later call
    # in this process would get a dead object back. Drop it before recreating.
    get_collection.cache_clear()
    return get_collection()


def add(collection, ids, embeddings, documents, metadatas):
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(collection, query_embedding, n_results: int):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def count(collection) -> int:
    return collection.count()


def all_documents(collection):
    """Return (ids, documents, metadatas) for the whole collection.

    Used to (re)build the BM25 lexical index.
    """
    got = collection.get(include=["documents", "metadatas"])
    return got["ids"], got["documents"], got["metadatas"]
