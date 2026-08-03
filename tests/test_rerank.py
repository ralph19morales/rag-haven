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
from ragmed.retriever import Retrieved, _rerank, _rerank_clause_ids  # noqa: E402


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
        by_id = {c.id: c for c in [chunk("a", .5, "joinder of parties"),
                                   chunk("b", .5, "time within which claims"),
                                   chunk("c", .5, "pledged thing")]}
        rerank.score = lambda q, texts: [
            0.70 if "claims" in t else 0.50 for t in texts]
        out = _rerank_clause_ids(["is there a deadline?"],
                                 [["a", "b", "c"]], by_id)
        r.append(check("an ask's candidates are reordered to answer IT",
                       out[0][0], "b"))

        rerank.score = lambda q, texts: [0.5000] * len(texts)
        r.append(check("flat clause scores keep retrieval order",
                       _rerank_clause_ids(["q"], [["a", "b", "c"]], by_id),
                       [["a", "b", "c"]]))

        rerank.score = lambda q, texts: None
        r.append(check("unavailable model keeps clause order",
                       _rerank_clause_ids(["q"], [["a", "b", "c"]], by_id),
                       [["a", "b", "c"]]))

        r.append(check("ids missing from the ranking are dropped, not crashed",
                       _rerank_clause_ids(["q"], [["a", "ghost"]], by_id),
                       [["a", "ghost"]]))

        # --- rerank.score itself -------------------------------------------
        rerank.score = orig_score
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
        config.RERANK_MIN_SPREAD = orig_spread
        config.RERANK_ENABLED = orig_enabled

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
