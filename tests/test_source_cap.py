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

from ragmed.retriever import (Retrieved, _cap_per_family,  # noqa: E402
                              _cap_per_source, _ensure_fused_head,
                              source_family)


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
    # --- per-FAMILY cap -----------------------------------------------------
    # The per-source cap limits one FILE. It does not stop one kind of
    # authority owning the context: the deceased-body question filled four of
    # six slots with jurisprudence drawn from three different case files, so
    # the per-source cap was satisfied while the statute and its IRR — which
    # answer that question together — could not both fit.
    fam_pool = [
        Retrieved(id="j1", text="", metadata={"source": "_fetched/jurisprudence/a.txt"}, score=.99),
        Retrieved(id="j2", text="", metadata={"source": "_fetched/jurisprudence/b.txt"}, score=.98),
        Retrieved(id="j3", text="", metadata={"source": "_fetched/jurisprudence/c.txt"}, score=.97),
        Retrieved(id="s1", text="", metadata={"source": "_fetched/lawphil/ra9439.txt"}, score=.60),
        Retrieved(id="d1", text="", metadata={"source": "_fetched/doh/irr.txt"}, score=.50),
    ]
    r.append(check("family is read from the fetch path convention",
                         source_family("_fetched/jurisprudence/a.txt"), "jurisprudence"))
    r.append(check("hand-added files fall back to the top directory",
                         source_family("statutes/RA-2382.txt"), "statutes"))
    capped = _cap_per_family(fam_pool, cap=2, top_k=4)
    r.append(check("one family cannot own the whole top_k",
                         [r.id for r in capped][:4], ["j1", "j2", "s1", "d1"]))
    # "Demoted, not dropped" means the overflow BACKFILLS when the capped
    # selection would otherwise come up short — matching _cap_per_source. With
    # top_k=4 above there was no room, so j3 legitimately falls away; give it
    # one more slot and it must come back, behind the other families.
    r.append(check("overflow backfills when there is room",
                   [x.id for x in _cap_per_family(fam_pool, cap=2, top_k=5)],
                   ["j1", "j2", "s1", "d1", "j3"]))
    # A question only one family answers must still get a full context.
    only_j = fam_pool[:3]
    r.append(check("a single-family pool still fills the top_k",
                         [r.id for r in _cap_per_family(only_j, cap=2, top_k=3)],
                         ["j1", "j2", "j3"]))
    r.append(check("cap of 0 disables the family cap",
                         [r.id for r in _cap_per_family(fam_pool, 0, 4)],
                         [r.id for r in fam_pool]))

    # --- demoted is not deleted --------------------------------------------
    # Both caps hand their result to `_ensure_clause_coverage` and
    # `_ensure_fused_head`, which can only promote a chunk that is still in the
    # list. The caps used to truncate their overflow at top_k, which silently
    # disarmed the fused-head guarantee — on the deceased-body question it
    # protected three chunks and delivered one, because two had already been
    # deleted here. The top_k itself is unchanged by this; what changes is that
    # a later stage still has something to promote.
    r.append(check("family cap keeps the overflow as a tail",
                   [x.id for x in _cap_per_family(fam_pool, cap=2, top_k=4)],
                   ["j1", "j2", "s1", "d1", "j3"]))
    r.append(check("source cap keeps the overflow as a tail",
                   len(_cap_per_source(rubi + others, cap=3, top_k=4)),
                   len(rubi + others)))
    r.append(check("neither cap loses a chunk",
                   sorted(x.id for x in _cap_per_family(fam_pool, 2, 1)),
                   sorted(x.id for x in fam_pool)))

    # The end-to-end invariant, on the real stage order: a chunk in the fused
    # head that the RERANKER pushed down must still reach the top_k after both
    # caps have run. This is the assertion CLAUDE.md states as a guarantee.
    reranked = [
        Retrieved(id="x1", text="", metadata={"source": "_fetched/jurisprudence/x.txt"}, score=.99),
        Retrieved(id="x2", text="", metadata={"source": "_fetched/jurisprudence/x.txt"}, score=.98),
        Retrieved(id="d1", text="", metadata={"source": "_fetched/doh/irr.txt"}, score=.80),
        Retrieved(id="d2", text="", metadata={"source": "_fetched/doh/other.txt"}, score=.79),
        Retrieved(id="s1", text="", metadata={"source": "_fetched/lawphil/ra9439.txt"}, score=.70),
        Retrieved(id="b1", text="", metadata={"source": "_fetched/billing/san.txt"}, score=.60),
        Retrieved(id="j1", text="", metadata={"source": "_fetched/jurisprudence/a.txt"}, score=.50),
        Retrieved(id="j2", text="", metadata={"source": "_fetched/jurisprudence/b.txt"}, score=.49),
    ]
    fused_head = ["j1", "j2", "s1"]     # what fusion ranked first, pre-rerank
    out = _cap_per_source(reranked, 3, 6)
    out = _cap_per_family(out, 2, 6)
    out = _ensure_fused_head(out, fused_head, 6)
    in_top_k = [i for i in fused_head if i in [x.id for x in out[:6]]]
    r.append(check("the whole fused head survives both caps", in_top_k, fused_head))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
