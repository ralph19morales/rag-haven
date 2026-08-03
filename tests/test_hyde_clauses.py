"""Tests for the clause-coverage half of the HyDE gate in ragmed.retriever.

Why this exists: similarity over a whole question is a MAXIMUM, so one
well-matched clause hides every other one. Measured on this corpus, "can we
refuse to release the body ... who can we collect from, and is there a
deadline?" scored 0.736 — above the 0.68 gate — because its first half matches
the anti-detention law almost exactly. HyDE stayed off, the estate-claim half
was never retrieved, and the answer asserted there was no deadline although
Rule 86 sets one.

The fix scores each ASK on its own. Scoring every clause was tried first and
abandoned: the weakest clause of an answerable question is its narrative setup
("A patient arrived at the emergency room without money", 0.617), not its
question (0.729), so the answerable and unanswerable groups overlapped
completely and no threshold could separate them. Restricted to asks they
separate: answerable 0.721-0.783, unanswerable 0.605-0.678.

No LLM and no corpus needed — these exercise the splitting and the gate only.

Run:  .venv/Scripts/python.exe tests/test_hyde_clauses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed import config, retriever  # noqa: E402

DECEASED = ("A patient died in our hospital with an unpaid bill of P400,000. "
            "Can we refuse to release the body and the death certificate until "
            "the family pays? If not, who can we collect from, and is there a "
            "deadline?")
COVERED = ("A patient arrived at the emergency room without money. Can the "
           "hospital require a deposit before treating him?")
NARRATIVE = ("My father passed away last month. The hospital is now calling me "
             "every day demanding that I settle his bill personally.")

WELL_POSED = [0.79, 0.75, 0.71]   # whole-question similarity, comfortably fine


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def run() -> int:
    r = []
    orig = (config.HYDE_MODE, config.HYDE_MIN_SIM, config.HYDE_CLAUSE_MIN_SIM)
    config.HYDE_MODE = "auto"
    config.HYDE_MIN_SIM = 0.68
    config.HYDE_CLAUSE_MIN_SIM = 0.70
    try:
        # --- splitting ------------------------------------------------------
        r.append(check("compound question splits into its sentences",
                       len(retriever._split_clauses(DECEASED)), 3))
        r.append(check("terminators are retained (asks stay identifiable)",
                       retriever._split_clauses(DECEASED)[1].endswith("?"),
                       True))
        r.append(check("only the asks are scored",
                       len(retriever._ask_clauses(DECEASED)), 2))
        r.append(check("narrative setup is excluded from the asks",
                       any(c.startswith("A patient died")
                           for c in retriever._ask_clauses(DECEASED)),
                       False))
        r.append(check("a question with no '?' falls back to every clause",
                       len(retriever._ask_clauses(NARRATIVE)), 2))
        r.append(check("short fragments are dropped ('If not')",
                       any(c.strip() == "If not"
                           for c in retriever._split_clauses(DECEASED)),
                       False))
        r.append(check("a bare amount is not a clause",
                       retriever._split_clauses("P400,000. Can the hospital "
                                                "still collect the balance?"),
                       ["Can the hospital still collect the balance?"]))
        r.append(check("single-sentence question yields one clause",
                       len(retriever._split_clauses(
                           "What are the elements of medical negligence?")), 1))
        r.append(check("empty question is handled",
                       retriever._split_clauses(""), []))

        # --- the gate -------------------------------------------------------
        # The measured failure: whole question fine, one ask uncovered.
        r.append(check("uncovered ask fires despite a strong whole-question "
                       "score", retriever._should_hyde(WELL_POSED,
                                                       [0.678, 0.596]), True))
        r.append(check("all asks covered does not fire",
                       retriever._should_hyde(WELL_POSED, [0.729, 0.768]),
                       False))
        r.append(check("exactly at the clause threshold does not fire",
                       retriever._should_hyde(WELL_POSED, [0.70]), False))
        r.append(check("just under the clause threshold fires",
                       retriever._should_hyde(WELL_POSED, [0.699]), True))
        r.append(check("no clause scores leaves legacy behaviour intact",
                       retriever._should_hyde(WELL_POSED, []), False))
        r.append(check("a poor whole question still fires without clauses",
                       retriever._should_hyde([0.60, 0.58], []), True))

        # Mode gating must still win over the clause check.
        config.HYDE_MODE = "off"
        r.append(check("off: an uncovered ask does not fire",
                       retriever._should_hyde(WELL_POSED, [0.40]), False))
        config.HYDE_MODE = "always"
        r.append(check("always: fires with every ask covered",
                       retriever._should_hyde(WELL_POSED, [0.99]), True))
        config.HYDE_MODE = "auto"

        # The covered two-sentence question is the regression guard: it used to
        # false-fire when every clause was scored.
        asks = retriever._ask_clauses(COVERED)
        r.append(check("covered question exposes only its ask", len(asks), 1))
        r.append(check("and that ask is the question, not the setup",
                       asks[0].startswith("Can the hospital"), True))
    finally:
        (config.HYDE_MODE, config.HYDE_MIN_SIM,
         config.HYDE_CLAUSE_MIN_SIM) = orig

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
