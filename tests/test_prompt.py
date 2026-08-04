"""Policy assertions on the grounded SYSTEM_PROMPT in ragmed.rag.

These are not behaviour tests — they pin constraints that were each added to
fix a real, observed failure, so that a later edit cannot quietly drop one.

Run:  .venv/Scripts/python.exe tests/test_prompt.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.rag import SYSTEM_PROMPT  # noqa: E402

P = SYSTEM_PROMPT.lower()

# Foreign decisions quoted inside Philippine rulings. The model cited these as
# if they were Philippine authority until rule 3 was added. Listing them HERE
# is safe; listing them in the prompt would not be (see the parroting guard).
FOREIGN_CASES = ["canterbury", "cobbs", "spence", "darling", "charleston",
                 "schloendorff", "nathanson", "salgo"]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def run() -> int:
    r = []

    # --- rule numbering is well-formed (a renumbering slip is easy to miss) --
    nums = [int(n) for n in re.findall(r"^(\d+)\.\s", SYSTEM_PROMPT, re.M)]
    r.append(check("rules are numbered consecutively from 1",
                   nums == list(range(1, len(nums) + 1)), f"(found {nums})"))

    # --- citations must come from the context headers -----------------------
    r.append(check(
        'citations are restricted to identifiers on "Cite as:" lines',
        "cite as:" in P and "never cite" in P))

    # The passage label must not be citation-shaped, and must be forbidden by
    # name. "[Context 3]" both LOOKED like a legal citation and was written into
    # the prompt's own wording, so the model copied it: 75 stray markers across
    # 12 answers, one answer opening every paragraph with "[Context 3]
    # [Context 5]". Those resolve to nothing for a reader who cannot see the
    # prompt, which is worse than an uncited sentence.
    r.append(check("prompt does not itself contain a bracketed passage label",
                   not re.search(r"\[\s*(?:context|passage|source)\s*n?\d*\s*\]",
                                 P)))
    r.append(check("prompt forbids citing by passage number",
                   "never write" in P and "context 3" in P))

    # --- foreign material is not authority ----------------------------------
    r.append(check("prompt tells the model to cite Philippine authority only",
                   "philippine authority only" in P))
    r.append(check("prompt explains quoted foreign material is not authority",
                   "persuasive" in P and "foreign" in P))

    # --- the parroting landmine: no concrete examples the model can copy -----
    # A concrete example refusal sentence in this prompt was previously copied
    # verbatim by the 14b model as its generic refusal on unrelated questions.
    # The same risk applies to naming a foreign case: name one and it may start
    # appearing in answers. Rules must stay behavioural.
    named = [c for c in FOREIGN_CASES if c in P]
    r.append(check("no foreign case is named in the prompt", not named,
                   f"(found {named})" if named else ""))

    r.append(check(
        "no canned refusal sentence for the model to parrot",
        "the corpus does not specify a prescriptive period" not in P))

    # --- constraints from earlier fixes must survive ------------------------
    r.append(check("still forbids answering from outside the context",
                   "only using the provided context" in P))
    r.append(check("still requires a SPECIFIC refusal naming what is missing",
                   "name the" in P and "missing" in P))
    r.append(check("still forbids trailing caveats after a real answer",
                   "stop there" in P))
    r.append(check("still keeps the legal-information-not-advice note",
                   "not legal advice" in P))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
