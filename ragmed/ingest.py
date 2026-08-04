"""Ingestion pipeline: files -> text -> chunks -> embeddings -> Chroma.

Idempotent per file: each chunk id is derived from a hash of the file path +
chunk index, so re-ingesting the same file replaces its chunks rather than
duplicating them. Use --reset for a clean full rebuild.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import config, embeddings, vectorstore
from .chunking import chunk_document
from .loaders import iter_corpus_files, load_file


def _chunk_id(rel_path: str, idx: int) -> str:
    h = hashlib.sha1(f"{rel_path}::{idx}".encode()).hexdigest()[:16]
    return f"{h}"


def ingest_corpus(reset: bool = False, verbose: bool = True) -> dict:
    """Ingest every supported file under the corpus directory.

    Returns a summary dict with counts.
    """
    config.ensure_dirs()
    collection = vectorstore.reset_collection() if reset else \
        vectorstore.get_collection()

    files = list(iter_corpus_files(config.CORPUS_DIR))
    total_chunks = 0
    processed = 0
    skipped: list[str] = []

    iterator = files
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Ingesting", unit="file")
        except ImportError:
            pass

    for path in iterator:
        rel = str(path.relative_to(config.CORPUS_DIR))
        try:
            text = load_file(path)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{rel}: {e}")
            continue
        if not text.strip():
            skipped.append(f"{rel}: no extractable text (scanned image?)")
            continue

        chunks = chunk_document(text, rel)
        if not chunks:
            continue

        ids = [_chunk_id(rel, c.metadata["chunk_index"]) for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        vecs = embeddings.embed_documents(docs)

        # Upsert semantics: delete any existing chunks for this file first.
        try:
            collection.delete(where={"source": rel})
        except Exception:
            pass
        vectorstore.add(collection, ids, vecs, docs, metas)

        total_chunks += len(chunks)
        processed += 1

    # Invalidate the BM25 cache so it rebuilds on next query — both copies.
    # The pickle on disk is the one that outlives this run; the in-process one
    # matters when something ingests and then queries without restarting.
    if config.BM25_PATH.exists():
        try:
            config.BM25_PATH.unlink()
        except OSError:
            pass
    from . import retriever  # local import: ingest must stay usable without the
    retriever.invalidate_bm25()   # retrieval stack's heavier dependencies

    return {
        "files_processed": processed,
        "files_found": len(files),
        "chunks_indexed": total_chunks,
        "collection_size": vectorstore.count(collection),
        "skipped": skipped,
    }
