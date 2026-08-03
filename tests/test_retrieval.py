"""Tests for retrieval-time deduplication in ragmed.retriever.

Why this exists: a legal corpus holds the same passage more than once by
design. A Supreme Court resolution on reconsideration reproduces long stretches
of the decision it modifies, so both chunk to byte-identical text and score
identically. Measured on this corpus before the fix, one case took 6 of 10
slots as three duplicate PAIRS.

Run:  .venv/Scripts/python.exe tests/test_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.retriever import Retrieved, _dedupe, _dedupe_key  # noqa: E402

DECISION = ("It is good medical practice for the anesthesiologist to see the "
            "patient a day before the surgery.")
OTHER = ("The hospital is liable under the doctrine of corporate negligence "
         "for failing to supervise its consultants.")


def chunk(text: str, score: float, source: str) -> Retrieved:
    return Retrieved(id=f"{source}:{score}", text=text,
                     metadata={"source": source}, score=score)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def run() -> int:
    results = []

    # The real shape: the same passage from a decision and from the resolution
    # that modified it, scoring identically.
    pair = [chunk(DECISION, 1.0, "ramos-1999.txt"),
            chunk(DECISION, 1.0, "ramos-2002-resolution.txt"),
            chunk(OTHER, 0.9, "psi-agana.txt")]
    out = _dedupe(pair)
    results.append(check("duplicate pair collapses to one", len(out), 2))
    results.append(check("highest-scoring copy is the one kept",
                         out[0].metadata["source"], "ramos-1999.txt"))
    results.append(check("distinct chunk survives",
                         out[1].metadata["source"], "psi-agana.txt"))

    # Three duplicate pairs — the measured failure — must free three slots.
    six = []
    for i, score in enumerate((1.0, 0.96, 0.93)):
        for src in ("decision.txt", "resolution.txt"):
            six.append(chunk(f"passage number {i}", score, src))
    results.append(check("three duplicate pairs collapse to three",
                         len(_dedupe(six)), 3))

    # Normalisation: whitespace and case differences are still duplicates.
    results.append(check(
        "whitespace and case variants are treated as duplicates",
        len(_dedupe([chunk(DECISION, 1.0, "a.txt"),
                     chunk(DECISION.upper().replace(" ", "\n  "), 0.9, "b.txt")])),
        1))

    # Must NOT over-merge: different passages are all kept, in order.
    distinct = [chunk(DECISION, 1.0, "a.txt"), chunk(OTHER, 0.9, "b.txt"),
                chunk("A third, unrelated passage entirely.", 0.8, "c.txt")]
    kept = _dedupe(distinct)
    results.append(check("distinct passages are all kept", len(kept), 3))
    results.append(check("score order is preserved",
                         [r.score for r in kept], [1.0, 0.9, 0.8]))

    # A passage that merely OVERLAPS is deliberately not merged — collapsing
    # partial overlaps risks discarding a genuinely different chunk.
    results.append(check(
        "partial overlap is not merged",
        len(_dedupe([chunk(DECISION, 1.0, "a.txt"),
                     chunk(DECISION + " The surgeon must also attend.", 0.9,
                           "b.txt")])),
        2))

    results.append(check("empty input is handled", _dedupe([]), []))
    results.append(check("key normalises whitespace",
                         _dedupe_key("  A  B \n c "), "a b c"))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
