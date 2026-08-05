"""Tests for cross-encoder reranking in ragmed.retriever / ragmed.rerank.

Why this exists: bi-encoder retrieval embeds question and chunk separately, so
it cannot distinguish "answers this question" from "is about similar things".
Measured on this corpus, one ask scored eight chunks inside a 1.4% spread and
put the answering rule 5th. A cross-encoder reads the pair together and, given a
query that names what it asks about, separated the same chunks by 0.1975.

Two behaviours matter more than the reranking itself and are what these tests
pin down:

  * It must FAIL OPEN. A ranking improvement that can take retrieval down is a
    bad trade, so an unavailable model leaves the fused order untouched.
  * It must ignore its own output when that output is meaningless. Asked a
    compound question, the cross-encoder returned every candidate within 0.0004
    of 0.5000; sorting that spread promoted an unrelated case to rank 1. Below
    RERANK_MIN_SPREAD the fused ordering stands.

The model is stubbed here, so these run offline and in milliseconds.

Run:  .venv/Scripts/python.exe tests/test_rerank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed import config, rerank, retriever  # noqa: E402
from ragmed.retriever import (Retrieved, _ensure_fused_head,  # noqa: E402
                              _rerank, _rerank_clause_ids)


def chunk(cid: str, score: float, text: str | None = None) -> Retrieved:
    return Retrieved(id=cid, text=text or cid,
                     metadata={"source": f"{cid}.txt"}, score=score)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def ids(rs: list[Retrieved]) -> list[str]:
    return [r.id for r in rs]


def run() -> int:
    r = []
    orig_score = rerank.score
    orig_score_pairs = rerank.score_pairs
    orig_spread = config.RERANK_MIN_SPREAD
    orig_enabled = config.RERANK_ENABLED
    config.RERANK_MIN_SPREAD = 0.02
    config.RERANK_ENABLED = True
    try:
        # Retrieval's order is wrong; the cross-encoder disagrees clearly.
        def fused():                     # fresh objects per scenario, so a
            return [chunk("wrong1", 0.99),   # mutation bug cannot hide behind
                    chunk("wrong2", 0.98),   # state left by an earlier call
                    chunk("right", 0.42)]

        rerank.score = lambda q, texts: [0.50 if t != "right" else 0.70
                                         for t in texts]
        out = _rerank("q", fused(), limit=10)
        r.append(check("a clear disagreement reorders the ranking",
                       ids(out)[0], "right"))
        r.append(check("scores are replaced by the reranker's",
                       round(out[0].score, 2), 0.70))
        r.append(check("nothing is dropped", len(out), 3))

        # The measured regression: flat scores must NOT reorder anything.
        # The caller's own objects must come back unchanged.
        original = fused()
        rerank.score = lambda q, texts: [0.50 if t != "right" else 0.70
                                         for t in texts]
        _rerank("q", original, limit=10)
        r.append(check("reranking does not mutate the caller's chunks",
                       [round(x.score, 2) for x in original],
                       [0.99, 0.98, 0.42]))

        rerank.score = lambda q, texts: [0.5000, 0.5004, 0.5001][:len(texts)]
        out = _rerank("q", fused(), limit=10)
        r.append(check("a flat distribution keeps the fused order",
                       ids(out), ids(fused())))
        r.append(check("and leaves the fused scores intact",
                       [round(x.score, 2) for x in out], [0.99, 0.98, 0.42]))

        # Fail open: an unavailable model must not disturb anything.
        rerank.score = lambda q, texts: None
        r.append(check("unavailable model leaves the ranking untouched",
                       ids(_rerank("q", fused(), limit=10)), ids(fused())))

        # Only the head is rescored; the tail keeps its place behind it.
        rerank.score = lambda q, texts: [0.9 - i / 100 for i in range(len(texts))]
        many = [chunk(f"c{i}", 1.0 - i / 100) for i in range(6)]
        out = _rerank("q", list(many), limit=3)
        r.append(check("only `limit` candidates are rescored",
                       ids(out)[3:], ["c3", "c4", "c5"]))

        # Degenerate inputs.
        r.append(check("a single candidate is returned as-is",
                       ids(_rerank("q", [chunk("only", 1.0)], 10)), ["only"]))
        r.append(check("empty input is handled", _rerank("q", [], 10), []))

        # --- per-clause reranking -----------------------------------------
        # These stub `score_pairs`, not `score`: per-clause reranking sends
        # EVERY clause's pairs through the cross-encoder in one batched call
        # (each pair carries its own query), which is what removed two
        # sequential CPU passes per compound question. Stubbing `score` here
        # silently tested nothing once that changed — the real model ran
        # instead of the fake and two cases failed for the wrong reason.
        by_id = {c.id: c for c in [chunk("a", .5, "joinder of parties"),
                                   chunk("b", .5, "time within which claims"),
                                   chunk("c", .5, "pledged thing")]}
        rerank.score_pairs = lambda pairs: [
            0.70 if "claims" in t else 0.50 for _, t in pairs]
        out = _rerank_clause_ids(["is there a deadline?"],
                                 [["a", "b", "c"]], by_id)
        r.append(check("an ask's candidates are reordered to answer IT",
                       out[0][0], "b"))

        # Batching must not cross-contaminate: one flat score list comes back
        # for all clauses and has to be split at the right offsets, so each ask
        # is ranked by ITS OWN scores. Getting the arithmetic wrong here would
        # silently rank clause 2 by clause 1's scores — the exact failure the
        # per-clause rerank exists to prevent.
        calls = []

        def _spy(pairs):
            calls.append(list(pairs))
            # clause 1 favours "b"; clause 2 favours "c".
            return [{"a": .50, "b": .90, "c": .50}[t[-1]] if q == "one"
                    else {"a": .50, "b": .50, "c": .90}[t[-1]]
                    for q, t in pairs]

        rerank.score_pairs = _spy
        by_letter = {c.id: c for c in [chunk("a", .5, "chunk a"),
                                       chunk("b", .5, "chunk b"),
                                       chunk("c", .5, "chunk c")]}
        out = _rerank_clause_ids(["one", "two"],
                                 [["a", "b", "c"], ["a", "b", "c"]], by_letter)
        r.append(check("every clause is scored in ONE batched call",
                       len(calls), 1))
        r.append(check("all clauses' pairs go in that call",
                       len(calls[0]), 6))
        r.append(check("each clause is ranked by its own slice of the scores",
                       [out[0][0], out[1][0]], ["b", "c"]))

        rerank.score_pairs = lambda pairs: [0.5000] * len(pairs)
        r.append(check("flat clause scores keep retrieval order",
                       _rerank_clause_ids(["q"], [["a", "b", "c"]], by_id),
                       [["a", "b", "c"]]))

        rerank.score_pairs = lambda pairs: None
        r.append(check("unavailable model keeps clause order",
                       _rerank_clause_ids(["q"], [["a", "b", "c"]], by_id),
                       [["a", "b", "c"]]))

        rerank.score_pairs = lambda pairs: [0.70 if "claims" in t else 0.50
                                            for _, t in pairs]
        r.append(check("ids missing from the ranking are dropped, not crashed",
                       _rerank_clause_ids(["q"], [["a", "ghost"]], by_id),
                       [["a", "ghost"]]))

        # --- the fused head survives reranking ------------------------------
        # The bug: asked why a hospital was withholding a body, hybrid fusion
        # ranked RA 9439 (the Anti-Hospital Detention Law) THIRD and the
        # cross-encoder demoted it to seventeenth. The prohibition never
        # reached the prompt, the only surviving RA 9439 chunk was the IRR's
        # list of the offence's ELEMENTS, and the model read that as a
        # checklist of when detention is permitted — answering that a hospital
        # may lawfully withhold a body. The exact inverse of the law.
        fused = [chunk(c, 1.0 - i / 10) for i, c in
                 enumerate(["a", "b", "statute", "d", "e", "f", "g"])]
        # Reranking has shoved the statute to the back.
        demoted = [c for c in fused if c.id != "statute"] + \
                  [c for c in fused if c.id == "statute"]
        r.append(check("a protected fused hit is pulled back into the top_k",
                       ids(_ensure_fused_head(demoted, ["a", "b", "statute"],
                                              top_k=4))[:4].count("statute"), 1))
        r.append(check("it leads the context rather than trailing it",
                       ids(_ensure_fused_head(demoted, ["a", "b", "statute"],
                                              top_k=4))[0], "statute"))
        r.append(check("nothing is dropped from the pool",
                       len(_ensure_fused_head(demoted, ["statute"], top_k=4)),
                       len(demoted)))
        # Must be a FLOOR, not a reordering: anything already inside the top_k
        # keeps the position reranking gave it.
        r.append(check("an already-present hit is left where it is",
                       ids(_ensure_fused_head(fused, ["a", "b"], top_k=4)),
                       ids(fused)))
        r.append(check("disabled by 0 protected ids",
                       ids(_ensure_fused_head(demoted, [], top_k=4)),
                       ids(demoted)))

        # --- rerank.score itself -------------------------------------------
        rerank.score = orig_score
        rerank.score_pairs = orig_score_pairs
        config.RERANK_ENABLED = False
        r.append(check("disabled reranking returns None",
                       rerank.score("q", ["some text"]), None))
        r.append(check("is_available is False when disabled",
                       rerank.is_available(), False))
        config.RERANK_ENABLED = True
        r.append(check("empty text list returns None",
                       rerank.score("q", []), None))
        r.append(check("sigmoid squashes a logit into 0-1",
                       0.0 < rerank._sigmoid(-8.0) < rerank._sigmoid(8.0) < 1.0,
                       True))
    finally:
        rerank.score = orig_score
        rerank.score_pairs = orig_score_pairs
        config.RERANK_MIN_SPREAD = orig_spread
        config.RERANK_ENABLED = orig_enabled

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
