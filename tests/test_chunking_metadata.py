"""Tests for document-identity detection in ragmed.chunking.

Why this exists: `law` metadata is what the grounded prompt tells the model to
cite. A document the detector does not recognise is indexed with law='?' and
becomes UNCITABLE — it still retrieves, so the answer looks fine while the
citation is missing or vague. An index audit found 433 of 5024 chunks (8.6%)
in that state, including the entire Revised Penal Code.

Run:  .venv/Scripts/python.exe tests/test_chunking_metadata.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.chunking import detect_doc_metadata  # noqa: E402

# Heads as they actually appear after fetching (E-Library page chrome included,
# because that chrome is what the detector really has to see past).
CHROME = (" - Supreme Court E-Library Supreme Court E-Library Information At "
          "Your Fingertips HOME PHILIPPINE REPORTS E-BOOKS REPUBLIC ACTS "
          "CHIEF JUSTICES NEWS & ADVISORIES SITE MAP ABOUT US")


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def law_of(head: str) -> str:
    return detect_doc_metadata(head, "f.txt").get("law", "?")


def run() -> int:
    r = []

    # --- the regressions this file was written for ---------------------------
    r.append(check(
        "Act No. 3815 (Revised Penal Code) is recognised",
        law_of("Act No. 3815 - AN ACT REVISING THE PENAL CODE AND OTHER PENAL "
               "LAWS." + CHROME),
        "Act No. 3815"))

    r.append(check(
        "abbreviated 'R.A. 7305' is recognised",
        law_of("DOH - REVISED IMPLEMENTING RULES AND REGULATIONS ON THE MAGNA "
               "CARTA OF PUBLIC HEALTH WORKERS OR R.A. 7305*" + CHROME),
        "Republic Act No. 7305"))

    r.append(check(
        "a Code of Ethics is labelled by name",
        law_of("Board of Medicine Code of Ethics Article I GENERAL PRINCIPLES "
               "Section 1. The primary objectives of the practice of medicine"),
        "Board of Medicine Code of Ethics"))

    # --- ordering guard: 'Republic Act No. 386' contains 'Act No. 386' -------
    r.append(check(
        "Republic Act wins over the bare Act pattern",
        law_of("Republic Act No. 386 - AN ACT TO ORDAIN AND INSTITUTE THE "
               "CIVIL CODE OF THE PHILIPPINES" + CHROME),
        "Republic Act No. 386"))

    r.append(check(
        "R.A. short form also wins over the bare Act pattern",
        law_of("IRR of R.A. 9439, an Act prohibiting detention" + CHROME),
        "Republic Act No. 9439"))

    # --- a court decision must never be hijacked by a statute it cites ------
    r.append(check(
        "G.R. number wins over any statute quoted inside the ruling",
        law_of("G.R. No. 172406 - CONCEPCION ILAO-ORETA, PETITIONER, VS. "
               "SPOUSES RONQUILLO. Citing Republic Act No. 386 and Act No. "
               "3815 throughout." + CHROME),
        "G.R. No. 172406"))

    r.append(check(
        "court decisions are typed as jurisprudence",
        detect_doc_metadata("G.R. No. 126297 - PROFESSIONAL SERVICES, INC.",
                            "f.txt").get("law_type"),
        "jurisprudence"))

    # --- Rules of Court -----------------------------------------------------
    # LawPhil serves Rules 72-109 as one page. Indexed whole it produced 199
    # chunks sharing 25 section labels ("Section 2." meant 27 different things)
    # and law=None, so the six-to-twelve month deadline in Rule 86 Sec. 2 was
    # retrievable but uncitable. fetch/split_rules_of_court.py splits the page
    # per rule and writes this header; the detector has to read it.
    RULE_HEAD = ("Rules of Court of the Philippines\n"
                 "PART II - SPECIAL PROCEEDINGS\n"
                 "RULE 86 - Claims Against Estate\n\n"
                 "RULE 86\nClaims Against Estate\n"
                 "Section 1. Notice to creditors to be issued by court.")
    r.append(check("a split Rule of Court is citable",
                   law_of(RULE_HEAD), "Rule 86 of the Rules of Court"))
    r.append(check("Rules of Court are typed as RULE",
                   detect_doc_metadata(RULE_HEAD, "f.txt").get("law_type"),
                   "RULE"))

    # Narrowness guards. A decision citing Rule 65, or a statute mentioning a
    # rule, must keep its own identity — relabelling those would be worse than
    # the law=None it fixes.
    r.append(check(
        "a decision citing a Rule keeps its G.R. number",
        law_of("G.R. No. 173946 - BOSTON EQUITY RESOURCES, INC. A petition "
               "under Rule 65 of the Rules of Court was filed." + CHROME),
        "G.R. No. 173946"))
    r.append(check(
        "a statute mentioning the Rules of Court keeps its RA number",
        law_of("Republic Act No. 9439 - AN ACT PROHIBITING THE DETENTION OF "
               "PATIENTS. Proceedings shall follow the Rules of Court, and "
               "RULE 86 shall apply to claims." + CHROME),
        "Republic Act No. 9439"))
    r.append(check(
        "a bare 'RULE 86' with no Rules-of-Court context is not claimed",
        law_of("RULE 86\nSome unrelated document that happens to start this "
               "way and never names the source of the rule."),
        "?"))

    # --- other identities still work ----------------------------------------
    for head, want in [
        ("Presidential Decree No. 223 - CREATING THE PROFESSIONAL REGULATION "
         "COMMISSION", "Presidential Decree No. 223"),
        ("Administrative Order No. 70-A s. 2002 - RULES GOVERNING LICENSURE",
         "Administrative Order No. 70-A"),
    ]:
        r.append(check(f"still detects {want}", law_of(head + CHROME), want))

    # --- must not invent an identity ----------------------------------------
    r.append(check("plain prose stays unlabelled",
                   law_of("A note about how to organise this folder."), "?"))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
