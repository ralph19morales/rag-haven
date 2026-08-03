"""Hybrid retrieval: dense semantic search + BM25 lexical search.

Legal queries mix two needs:
  * semantic ("grounds for revoking a physician's license") -> dense vectors
  * exact tokens ("RA 2382", "Section 24", "PRC")             -> BM25 keywords

We fetch candidates from both, min-max normalise each score set, and fuse
with configurable weights. The BM25 index is built from the whole Chroma
collection and cached to disk; it is rebuilt automatically when the
collection size changes.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, replace

from . import config, embeddings, vectorstore

# --- HyDE: bridge the lay-question / formal-source vocabulary gap ----------

_HYDE_SYSTEM = (
    "You draft a SHORT hypothetical passage that could plausibly appear in a "
    "Philippine legal source answering the user's question. Write it in the "
    "vocabulary and register of statutes and Supreme Court decisions — name "
    "the doctrines, causes of action, and legal terms of art involved.\n\n"
    "Rules:\n"
    "1. Do NOT invent specific article, section, republic act, or G.R. "
    "numbers. Name doctrines and concepts in words instead.\n"
    "2. 3-5 sentences. No preamble, no caveats, no bullet points.\n"
    "3. This is a retrieval aid, not an answer to the user — it is never "
    "shown to anyone. Accuracy of detail does not matter; using the right "
    "TERMINOLOGY does."
)


def _hypothetical(question: str) -> str | None:
    """Draft a hypothetical answer whose wording is closer to the corpus than
    the raw question is. Returns None if drafting fails — HyDE is an
    enhancement, so any error must degrade to plain retrieval, never break it.

    Rule 1 of the prompt matters more than it looks: an invented "Article 2200"
    is a real, distinctive token that BM25 would match hard, pointing retrieval
    confidently at the wrong provision. Terminology helps; fabricated citations
    actively hurt.

    The draft is written for the WHOLE question. Aiming it at one ask of a
    compound question was tried twice and lost both times, which is worth
    recording so it is not retried:
      * the ask ALONE ("who can we collect from, and is there a deadline?") no
        longer says the debtor died, so the draft came back about sureties and
        prescription and pulled general contract law to the top;
      * the ask WITH the question as context did name the estate, but also
        invented a procedural rule and a filing period, and its hits displaced
        the statute the other half needed.
    Both made the measured result worse than no targeting at all. The draft is
    also nondeterministic, so letting it steer slot reservation made the
    reservation nondeterministic too — the opposite of what that mechanism is
    for. Coverage is handled deterministically by `_ensure_clause_coverage`;
    HyDE stays a whole-question vocabulary bridge.
    """
    from . import llm  # local import: keeps the module importable without a
                       # configured LLM (ingest, tests, offline tooling).
    try:
        text = llm.generate(_HYDE_SYSTEM, question, stream=False,
                            max_tokens=config.HYDE_MAX_TOKENS)
    except Exception:  # noqa: BLE001 - any LLM/transport failure
        return None
    text = (text or "").strip()
    return text or None


# Split on sentence terminators, KEEPING the terminator: "?" is what separates
# a thing being asked from the narrative setting it up.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.?!;])\s+")


def _split_clauses(question: str) -> list[str]:
    """Split a question into its sentences, terminators retained.

    Sentence terminators and semicolons only. Splitting on "and" was rejected:
    "release the body and the death certificate" is ONE ask, and breaking it
    produces fragments that match nothing and would fire HyDE on every query.

    Fragments under four words are dropped — "If not" and a stray "P400,000"
    carry no retrievable content, and scoring them would make every compound
    question look uncovered."""
    parts = []
    for raw in _CLAUSE_SPLIT_RE.split(question):
        part = raw.strip()
        if len(part.split()) >= 4 and re.search(r"[a-zA-Z]", part):
            parts.append(part)
    return parts


def _ask_clauses(question: str) -> list[str]:
    """The clauses worth scoring for coverage: the ones actually ASKING.

    Measured on this corpus, scoring every clause does not work. The
    lowest-scoring clause of a fully-answerable question is its narrative
    setup, not its question — "A patient arrived at the emergency room without
    money" scores 0.617 while its ask, "Can the hospital require a deposit
    before treating him", scores 0.729. Across a 9-question sample the worst
    clause of answerable questions (0.617-0.717) overlapped the worst clause of
    unanswerable ones (0.520-0.632) completely: no threshold could separate
    them. Restricting to the asks separates cleanly — answerable asks scored
    0.721-0.783, unanswerable ones 0.605-0.678.

    A question with no "?" at all is lay narrative ("they kept changing their
    story"), which has no ask to isolate. Every clause is returned then, which
    is the case HyDE was built for and which the whole-question check already
    catches."""
    clauses = _split_clauses(question)
    asks = [c for c in clauses if c.rstrip().endswith("?")]
    return asks or clauses


def _should_hyde(dense_sims: list[float],
                 clause_sims: list[float] | None = None) -> bool:
    """Decide whether this question needs the vocabulary bridge.

    Uses the best RAW cosine similarity, which is comparable across queries —
    the fused scores are min-max normalised and are not. (An earlier guess that
    a "flat" fused ranking marks a bad query did not survive measurement: two
    lay-narrative questions produced spreads indistinguishable from well-posed
    ones. Raw similarity separated them cleanly.)

    `clause_sims` covers the COMPOUND question, which the whole-question check
    cannot see. Similarity over the full text is a maximum, so one well-matched
    clause hides every other one: "can we refuse to release the body ... who can
    we collect from, and is there a deadline?" scored 0.736 — comfortably above
    the 0.68 threshold, because its first half matches the anti-detention law
    almost exactly. HyDE stayed off, the estate-claim half was never retrieved,
    and the answer asserted there was no deadline when Rule 86 sets one. Judging
    the WORST-covered clause instead of the best-covered question fixes that:
    a question is only well-posed if every part of it is."""
    if config.HYDE_MODE == "always":
        return True
    if config.HYDE_MODE != "auto":
        return False
    if dense_sims and max(dense_sims) < config.HYDE_MIN_SIM:
        return True
    return bool(clause_sims) and min(clause_sims) < config.HYDE_CLAUSE_MIN_SIM


@dataclass
class Retrieved:
    id: str
    text: str
    metadata: dict
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# --- BM25 lexical index ----------------------------------------------------

def _build_bm25(collection):
    from rank_bm25 import BM25Okapi

    ids, docs, metas = vectorstore.all_documents(collection)
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    index = {"bm25": bm25, "ids": ids, "docs": docs, "metas": metas,
             "n": len(ids)}
    config.ensure_dirs()
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


def _load_bm25(collection):
    """Load cached BM25 index, rebuilding if stale or missing."""
    current_n = vectorstore.count(collection)
    if config.BM25_PATH.exists():
        try:
            with open(config.BM25_PATH, "rb") as f:
                index = pickle.load(f)
            if index.get("n") == current_n:
                return index
        except Exception:
            pass
    return _build_bm25(collection)


def _dedupe_key(text: str) -> str:
    """Normalised chunk text, used to spot the same passage indexed twice."""
    return " ".join(text.lower().split())


def _dedupe(results: list["Retrieved"]) -> list["Retrieved"]:
    """Drop chunks whose text duplicates a higher-scoring chunk already kept.

    A legal corpus holds the same passage more than once by design: a Supreme
    Court resolution on reconsideration reproduces long stretches of the
    decision it modifies, so both documents chunk to byte-identical text that
    scores identically. Left alone, one case can take 6 of 10 slots as three
    duplicate PAIRS and crowd out the case that actually answers the question.

    Keeps the first (highest-scoring) copy, so the surviving chunk still points
    at a real source. Exact match after whitespace/case normalisation only —
    passages that merely overlap in part are left alone rather than risk
    merging two genuinely different chunks."""
    seen: set[str] = set()
    out: list["Retrieved"] = []
    for r in results:
        key = _dedupe_key(r.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _cap_per_source(results: list["Retrieved"], cap: int,
                    top_k: int) -> list["Retrieved"]:
    """Limit how many chunks any single source file may take in the top_k.

    `_dedupe` above only catches passages that are byte-identical, so it cannot
    touch the commoner failure: one document contributing many DIFFERENT chunks.
    A landmark case discusses its doctrine across a dozen passages and out-scores
    everything else on all of them. Measured on this corpus, "what is informed
    consent" filled 8 of 10 slots with Dr. Rubi Li, and the deceased-patient
    question gave 9 of 25 candidates to a single case — so the rule that
    actually answered the second half never reached the prompt.

    Overflow is DEMOTED, not discarded. A question that genuinely has one source
    (a single statute's licensing procedure, say) must still fill its context,
    so once every source has had its `cap`, the held-back chunks backfill in
    score order. The cap therefore changes the ORDER of the top_k, never its
    size.

    Assumes `results` is already sorted best-first; the kept chunk for each
    source is thus always its highest-scoring one."""
    if cap <= 0:
        return results

    kept: list["Retrieved"] = []
    overflow: list["Retrieved"] = []
    seen: dict[str, int] = {}
    for r in results:
        src = r.metadata.get("source", "?")
        if seen.get(src, 0) < cap:
            seen[src] = seen.get(src, 0) + 1
            kept.append(r)
        else:
            overflow.append(r)

    if len(kept) < top_k:
        kept.extend(overflow[: top_k - len(kept)])
    return kept


def _rerank(query: str, results: list["Retrieved"],
            limit: int) -> list["Retrieved"]:
    """Re-score the top `limit` candidates with the cross-encoder and re-sort.

    Only the head of the ranking is rescored: the cross-encoder is orders of
    magnitude slower than a vector comparison, and anything retrieval placed
    far down was not going to be selected anyway. The untouched tail keeps its
    fused scores and stays behind the rescored head, which is where it already
    was.

    Failure is a no-op by design — `rerank.score` returns None when the model is
    disabled or will not load, and the fused ordering simply stands."""
    from . import rerank

    head = results[:limit]
    if len(head) < 2:
        return results
    scores = rerank.score(query, [r.text for r in head])
    if scores is None:
        return results

    # A near-flat distribution means the cross-encoder found nothing it can
    # tell apart — reordering by it would be sorting noise. Measured: on a
    # compound question every candidate came back within 0.0004 of 0.5000, and
    # sorting that spread promoted an unrelated case to rank 1. The same model
    # separates by 0.1975 when the query names what it is asking about. So the
    # spread is the signal about whether the scores mean anything: below the
    # threshold, keep the ordering retrieval already produced.
    if max(scores) - min(scores) < config.RERANK_MIN_SPREAD:
        return results

    # Replace rather than mutate: `Retrieved` objects are shared with the pool
    # and with the caller, and writing a rerank score onto them destroys the
    # fused score in lists this function was only asked to read. (A test caught
    # exactly that — a later call saw scores an earlier one had overwritten.)
    rescored = [replace(r, score=s) for r, s in zip(head, scores)]
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored + results[limit:]


def _rerank_clause_ids(clauses: list[str], clause_ids: list[list[str]],
                       by_id: dict[str, "Retrieved"]) -> list[list[str]]:
    """Reorder each ask's reserved candidates by how well they answer THAT ask.

    This is the half of reranking that fixes the measured "right document,
    wrong chunk" failure. Slot reservation guarantees an ask reaches the prompt,
    but it picked that ask's chunks by the same bi-encoder similarity that could
    not tell them apart — so Boston Equity arrived as its compulsory-joinder
    section and the Civil Code as its article on pledges. Scoring each
    candidate against the ask itself puts the passage that answers it first."""
    from . import rerank

    out: list[list[str]] = []
    for clause, ids in zip(clauses, clause_ids):
        texts, kept = [], []
        for cid in ids:
            r = by_id.get(cid)
            if r is not None:
                kept.append(cid)
                texts.append(r.text)
        if len(kept) < 2:
            out.append(ids)
            continue
        scores = rerank.score(clause, texts)
        if scores is None or (max(scores) - min(scores)
                              < config.RERANK_MIN_SPREAD):
            out.append(ids)      # nothing to tell apart — keep retrieval order
            continue
        order = sorted(range(len(kept)), key=lambda i: scores[i], reverse=True)
        out.append([kept[i] for i in order])
    return out


def _ensure_clause_coverage(ranked: list["Retrieved"],
                            clause_ids: list[list[str]],
                            top_k: int, min_per_clause: int) -> list["Retrieved"]:
    """Reserve slots in the top_k for every ask of a compound question.

    Fusing per-clause candidates into the pool is not enough on its own, and the
    reason is worth stating: min-max normalisation scores every chunk against
    the SAME question, so a chunk that answers the neglected half perfectly
    still normalises below chunks that match the dominant half. Measured on the
    deceased-patient question, the top twelve results all scored >=0.894 and were
    all about one half of it; the other half's documents sat below a cliff at
    0.563 and never appeared. Raising CANDIDATE_K was tried and is not a fix —
    it worked at 60 and reverted at 120, because a larger pool shifts the
    normalisation window. That is a number tuned to one query, not a mechanism.

    So this stops competing on score and reserves capacity instead: each ask
    contributes at least `min_per_clause` of its own best-matching chunks, and
    the rest of the top_k fills from the global ranking as usual. Selection is
    by clause; the final ORDER is still by fused score, so the prompt still
    leads with the strongest evidence.

    `clause_ids` holds each clause's hits best-first for THAT clause. Ids absent
    from `ranked` (dropped as duplicates or by the per-source cap) are skipped
    rather than resurrected — those filters exist for their own good reasons."""
    if not clause_ids or min_per_clause <= 0 or len(clause_ids) < 2:
        return ranked

    by_id = {r.id: r for r in ranked}
    selected: list["Retrieved"] = []
    seen: set[str] = set()

    for ids in clause_ids:
        taken = 0
        for cid in ids:
            if taken >= min_per_clause or len(selected) >= top_k:
                break
            r = by_id.get(cid)
            if r is None or cid in seen:
                continue
            seen.add(cid)
            selected.append(r)
            taken += 1

    for r in ranked:                      # fill the remainder globally
        if len(selected) >= top_k:
            break
        if r.id in seen:
            continue
        seen.add(r.id)
        selected.append(r)

    selected.sort(key=lambda r: r.score, reverse=True)
    return selected


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def retrieve(question: str, top_k: int | None = None,
             debug: dict | None = None) -> list[Retrieved]:
    """Return the top_k most relevant chunks using hybrid fusion.

    Pass a dict as `debug` to receive diagnostics (whether HyDE fired, the
    draft it used, the best raw similarity before and after)."""
    top_k = top_k or config.TOP_K
    collection = vectorstore.get_collection()
    if vectorstore.count(collection) == 0:
        return []

    cand = config.CANDIDATE_K

    # --- Dense candidates ---
    q_vec = embeddings.embed_query(question)
    dense = vectorstore.query(collection, q_vec, cand)
    dense_ids = dense["ids"][0]
    dense_docs = dense["documents"][0]
    dense_metas = dense["metadatas"][0]
    # cosine distance -> similarity
    dense_sims = [1.0 - d for d in dense["distances"][0]]

    pool: dict[str, dict] = {}
    for i, cid in enumerate(dense_ids):
        pool[cid] = {
            "text": dense_docs[i],
            "meta": dense_metas[i],
            "dense": dense_sims[i],
            "lexical": 0.0,
        }

    if debug is not None:
        debug["raw_sim"] = max(dense_sims) if dense_sims else 0.0
        debug["hyde"] = False

    # --- Clause coverage: is any PART of a compound question unretrieved?
    # Only computed when the whole-question check has already passed and the
    # question really has several clauses — so a plain question costs nothing
    # extra, and the price of a compound one is a batched encode plus a vector
    # query per clause (no LLM call unless the check then fires).
    # The trigger is a question of MORE THAN ONE sentence, but the clauses
    # scored are only its asks. Gating on "more than one ask" instead would miss
    # the common shape "here is the situation. What do I do?", whose single ask
    # can still be uncovered; gating on a single-sentence question would apply
    # the clause threshold to the whole question and quietly move the legacy
    # one.
    #
    # The SAME per-clause query serves two purposes, which is why they are
    # computed together: its best hit scores the clause for the gate, and its
    # top ids become that clause's reserved candidates (see coverage below).
    clause_sims: list[float] = []
    clause_ids: list[list[str]] = []
    clauses: list[str] = []
    if len(_split_clauses(question)) > 1:
        clauses = _ask_clauses(question)
    if clauses:
        # One ask needs no per-clause work: querying it would just repeat the
        # whole-question query and double the cost for nothing.
        n_hits = config.CLAUSE_CANDIDATE_K if len(clauses) > 1 else 1
        for vec in embeddings.embed_queries(clauses):
            res = vectorstore.query(collection, vec, n_hits)
            dists = res["distances"][0]
            ids = res["ids"][0]
            clause_sims.append(1.0 - dists[0] if dists else 0.0)
            clause_ids.append(list(ids))
            # Fold this clause's hits into the pool so they are fused and
            # scored like any other candidate. Without this they could be
            # reserved a slot below but carry no score to be ranked by.
            if len(clauses) > 1:
                docs = res["documents"][0]
                metas = res["metadatas"][0]
                for i, cid in enumerate(ids):
                    sim = 1.0 - dists[i]
                    entry = pool.get(cid)
                    if entry is None:
                        pool[cid] = {"text": docs[i], "meta": metas[i],
                                     "dense": sim, "lexical": 0.0}
                    else:
                        entry["dense"] = max(entry["dense"], sim)
        if debug is not None:
            debug["clauses"] = clauses
            debug["clause_sims"] = clause_sims

    # --- HyDE: retrieve again with a hypothetical answer, and UNION the pools.
    # Union rather than replace: the original question stays authoritative, and
    # the draft only adds reach. A chunk found by both keeps its better score.
    lexical_query = question
    if _should_hyde(dense_sims, clause_sims):
        draft = _hypothetical(question)
        if draft:
            lexical_query = f"{question} {draft}"
            h_dense = vectorstore.query(collection, embeddings.embed_query(draft), cand)
            h_sims = [1.0 - d for d in h_dense["distances"][0]]
            for i, cid in enumerate(h_dense["ids"][0]):
                entry = pool.get(cid)
                if entry is None:
                    pool[cid] = {
                        "text": h_dense["documents"][0][i],
                        "meta": h_dense["metadatas"][0][i],
                        "dense": h_sims[i],
                        "lexical": 0.0,
                    }
                else:
                    entry["dense"] = max(entry["dense"], h_sims[i])
            if debug is not None:
                debug.update(hyde=True, draft=draft,
                             hyde_raw_sim=max(h_sims) if h_sims else 0.0)

    # --- Lexical candidates ---
    # Scored over question + draft: supplying the domain terms the lay question
    # lacked is precisely what lets BM25 contribute anything at all here.
    index = _load_bm25(collection)
    bm25 = index["bm25"]
    if bm25 is not None:
        scores = bm25.get_scores(_tokenize(lexical_query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i in ranked[:cand]:
            cid = index["ids"][i]
            entry = pool.get(cid)
            if entry is None:
                pool[cid] = {
                    "text": index["docs"][i],
                    "meta": index["metas"][i],
                    "dense": 0.0,
                    "lexical": float(scores[i]),
                }
            else:
                entry["lexical"] = float(scores[i])

    # --- Fuse ---
    ids = list(pool.keys())
    dense_norm = _minmax([pool[i]["dense"] for i in ids])
    lex_norm = _minmax([pool[i]["lexical"] for i in ids])
    results: list[Retrieved] = []
    for j, cid in enumerate(ids):
        score = (config.DENSE_WEIGHT * dense_norm[j]
                 + config.LEXICAL_WEIGHT * lex_norm[j])
        results.append(Retrieved(
            id=cid,
            text=pool[cid]["text"],
            metadata=pool[cid]["meta"],
            score=score,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    # Order matters: dedupe first so an identical passage cannot burn one of a
    # source's capped slots, then cap, then cut to top_k.
    # Order matters throughout. Dedupe first so an identical passage cannot burn
    # a capped slot; RERANK next, so the cap and the clause reservation both act
    # on true relevance rather than on bi-encoder similarity; then cap; then
    # reserve. Reranking after the cap would be too late — the cap would already
    # have chosen which of a document's chunks survive, using the very scores
    # the reranker exists to correct.
    deduped = _dedupe(results)
    reranked = _rerank(question, deduped, config.RERANK_CANDIDATES)
    if clause_ids:
        by_id = {r.id: r for r in reranked}
        clause_ids = _rerank_clause_ids(clauses, clause_ids, by_id)
    capped = _cap_per_source(reranked, config.MAX_CHUNKS_PER_SOURCE, top_k)
    covered = _ensure_clause_coverage(capped, clause_ids, top_k,
                                      config.MIN_CHUNKS_PER_CLAUSE)
    if debug is not None:
        debug["sources_in_top_k"] = len(
            {r.metadata.get("source", "?") for r in covered[:top_k]})
    return covered[:top_k]
