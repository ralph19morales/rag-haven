"""Tests for per-clause slot reservation in ragmed.retriever.

Why this exists: min-max normalisation scores every chunk against the SAME
question, so a chunk answering the neglected half of a compound question
normalises below chunks matching the dominant half — however many candidates
are fetched. Measured on the deceased-patient question, the top twelve results
all scored >=0.894 and were all about one half of it; the other half's documents
sat below a cliff at 0.563 and never appeared, and the answer denied a filing
deadline the Rules of Court impose.

Raising CANDIDATE_K is NOT the fix and this is worth recording: it surfaced the
missing half at 60 and lost it again at 120, because a larger pool moves the
normalisation window. That is a number fitted to one query. Reserving capacity
per ask is a mechanism, and it is what these tests pin down.

Run:  .venv/Scripts/python.exe tests/test_clause_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.retriever import Retrieved, _ensure_clause_coverage  # noqa: E402


def chunk(cid: str, score: float, source: str | None = None) -> Retrieved:
    return Retrieved(id=cid, text=cid,
                     metadata={"source": source or f"{cid}.txt"}, score=score)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def ids(results: list[Retrieved]) -> list[str]:
    return [r.id for r in results]


def run() -> int:
    r = []

    # The measured shape: one half of the question owns every high score, the
    # other half's evidence sits far below and never reaches the prompt.
    dominant = [chunk(f"detain{i}", 0.99 - i / 100) for i in range(10)]
    neglected = [chunk("estate1", 0.46), chunk("estate2", 0.45)]
    ranked = dominant + neglected
    clause_ids = [[c.id for c in dominant], ["estate1", "estate2"]]

    out = _ensure_clause_coverage(ranked, clause_ids, top_k=10,
                                  min_per_clause=2)
    r.append(check("neglected ask reaches the top_k at all",
                   {"estate1", "estate2"} <= set(ids(out)), True))
    r.append(check("top_k size is unchanged", len(out[:10]), 10))
    r.append(check("results stay ordered by fused score",
                   ids(out[:10]) == sorted(ids(out[:10]), key=lambda i:
                                      -next(x.score for x in out if x.id == i)),
                   True))
    r.append(check("the dominant ask keeps the rest of the slots",
                   sum(1 for i in ids(out[:10]) if i.startswith("detain")), 8))

    # Reservation is a FLOOR, not a quota: an ask that legitimately owns the
    # ranking is not cut back to min_per_clause.
    out = _ensure_clause_coverage(ranked, clause_ids, top_k=10,
                                  min_per_clause=1)
    r.append(check("floor of 1 still admits the neglected ask",
                   "estate1" in ids(out), True))
    r.append(check("floor of 1 leaves 9 slots to the dominant ask",
                   sum(1 for i in ids(out[:10]) if i.startswith("detain")), 9))

    # Disabled / degenerate inputs fall through untouched.
    r.append(check("min_per_clause=0 disables reservation",
                   ids(_ensure_clause_coverage(ranked, clause_ids, 10, 0)),
                   ids(ranked)))
    r.append(check("a single ask needs no reservation",
                   ids(_ensure_clause_coverage(ranked, [clause_ids[0]], 10, 2)),
                   ids(ranked)))
    r.append(check("no clause ids at all is a no-op",
                   ids(_ensure_clause_coverage(ranked, [], 10, 2)),
                   ids(ranked)))
    r.append(check("empty ranking is handled",
                   _ensure_clause_coverage([], clause_ids, 10, 2), []))

    # Ids a filter already removed are skipped, not resurrected: dedupe and the
    # per-source cap ran for their own reasons and must not be undone here.
    out = _ensure_clause_coverage(ranked, [["detain0"], ["gone", "estate1"]],
                                  top_k=5, min_per_clause=2)
    r.append(check("ids missing from the ranking are skipped",
                   "gone" in ids(out[:5]), False))
    r.append(check("the ask still gets its available evidence",
                   "estate1" in ids(out[:5]), True))

    # A clause whose evidence is already winning must not be double-counted
    # into extra slots.
    both = [["detain0", "detain1"], ["detain0", "detain2"]]
    out = _ensure_clause_coverage(ranked, both, top_k=5, min_per_clause=2)
    r.append(check("shared evidence is not selected twice",
                   len(ids(out[:5])), len(set(ids(out[:5])))))
    r.append(check("top_k is still filled when asks overlap", len(out[:5]), 5))

    # top_k smaller than the total reservation must not overflow.
    out = _ensure_clause_coverage(ranked, clause_ids, top_k=3,
                                  min_per_clause=2)
    r.append(check("reservation never exceeds top_k", len(out[:3]), 3))

    # Demoted is not deleted. The reservation hands its result to
    # _ensure_fused_head, which can only promote a chunk still in the list —
    # truncating here disarmed that guarantee for every compound question, the
    # same way the two caps did for simple ones.
    out = _ensure_clause_coverage(ranked, clause_ids, top_k=3, min_per_clause=2)
    r.append(check("the remainder is kept behind the selection",
                   sorted(ids(out)), sorted(ids(ranked))))
    # top_k=3 has room for one reserved chunk, not both — so estate1 is inside
    # the selection and estate2 is what the tail exists to preserve.
    r.append(check("the reserved chunk is inside the top_k",
                   "estate1" in ids(out[:3]), True))
    r.append(check("what did not fit survives in the tail",
                   "estate2" in ids(out[3:]), True))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
