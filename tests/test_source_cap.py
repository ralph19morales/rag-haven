"""Tests for the per-source diversity cap in ragmed.retriever.

Why this exists: `_dedupe` only collapses byte-identical passages, so it cannot
touch the commoner crowding failure — one document contributing many DIFFERENT
chunks. A landmark case discusses its doctrine across a dozen passages and
out-scores everything else on all of them. Measured on this corpus before the
cap: "what is informed consent" filled 8 of 10 slots with Dr. Rubi Li (3
distinct documents in the entire context), and the deceased-patient question
gave 9 of 25 candidates to one case, so the rule answering its second half
never reached the prompt.

The cap must DEMOTE, never discard: a question only one statute answers still
has to fill its context. That backfill behaviour is what most of these tests
pin down.

Run:  .venv/Scripts/python.exe tests/test_source_cap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.retriever import Retrieved, _cap_per_source  # noqa: E402


def chunk(source: str, score: float, text: str = "") -> Retrieved:
    return Retrieved(id=f"{source}:{score}", text=text or f"{source} {score}",
                     metadata={"source": source}, score=score)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def sources(results: list[Retrieved]) -> list[str]:
    return [r.metadata["source"] for r in results]


def run() -> int:
    r = []

    # The measured failure: one landmark case takes 8 of 10 slots.
    rubi = [chunk("rubi-li.txt", 1.0 - i / 100) for i in range(8)]
    others = [chunk("rosit.txt", 0.90), chunk("cereno.txt", 0.89)]
    out = _cap_per_source(rubi + others, cap=3, top_k=10)
    r.append(check("capped source keeps only `cap` chunks up front",
                   sources(out[:3]), ["rubi-li.txt"] * 3))
    r.append(check("the crowded-out documents now make the top_k",
                   sources(out[3:5]), ["rosit.txt", "cereno.txt"]))
    r.append(check("top_k is still filled (overflow backfills)",
                   len(out[:10]), 10))

    # Backfill order: demoted chunks return in score order, after the diverse
    # ones — a full context is worth more than a strictly-diverse short one.
    r.append(check("backfill resumes with the best demoted chunk",
                   out[5].metadata["source"], "rubi-li.txt"))
    r.append(check("backfill is in score order",
                   [x.score for x in out[5:8]],
                   [rubi[3].score, rubi[4].score, rubi[5].score]))

    # A question genuinely answered by ONE document must still fill its context.
    only = [chunk("medical-act.txt", 1.0 - i / 100) for i in range(10)]
    out = _cap_per_source(only, cap=3, top_k=10)
    r.append(check("single-source question still fills top_k", len(out), 10))
    r.append(check("single-source order is unchanged",
                   [x.score for x in out], [x.score for x in only]))

    # Enough diversity already: the cap must be a no-op.
    diverse = [chunk(f"doc{i}.txt", 1.0 - i / 100) for i in range(10)]
    r.append(check("already-diverse ranking is untouched",
                   sources(_cap_per_source(diverse, cap=3, top_k=10)),
                   sources(diverse)))

    # The highest-scoring chunk of each source is the one kept, so the citation
    # still points at that document's most relevant passage.
    mixed = [chunk("a.txt", 1.0), chunk("a.txt", 0.99), chunk("a.txt", 0.98),
             chunk("a.txt", 0.97), chunk("b.txt", 0.5)]
    out = _cap_per_source(mixed, cap=3, top_k=5)
    r.append(check("kept chunks are the source's best three",
                   [x.score for x in out[:3]], [1.0, 0.99, 0.98]))
    r.append(check("a different source is promoted above the overflow",
                   out[3].metadata["source"], "b.txt"))

    # cap=0 disables the feature entirely (documented escape hatch).
    r.append(check("cap=0 disables capping",
                   sources(_cap_per_source(rubi + others, cap=0, top_k=10)),
                   sources(rubi + others)))

    # Degenerate inputs must not raise.
    r.append(check("empty input is handled", _cap_per_source([], 3, 10), []))
    no_meta = [Retrieved(id="x", text="t", metadata={}, score=1.0),
               Retrieved(id="y", text="u", metadata={}, score=0.9)]
    r.append(check("chunks without a source key do not crash",
                   len(_cap_per_source(no_meta, cap=1, top_k=2)), 2))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
