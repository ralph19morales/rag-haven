"""Chroma-backed persistent vector store.

We pass our own embeddings in (embedding_function=None) so the same local
bge model is used for both indexing and querying, and so Chroma never tries
to phone home for a default model.
"""
from __future__ import annotations

from . import config


def _client():
    import chromadb
    from chromadb.config import Settings

    config.ensure_dirs()
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection():
    client = _client()
    return client.get_or_create_collection(
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
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


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
