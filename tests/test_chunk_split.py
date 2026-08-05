"""Tests for how ragmed.chunking splits an OVERSIZED unit.

Why this exists: `_split_on_sections` returns the whole document as a single
unit whenever its headings do not match SECTION_RE — and most of this corpus
does not match. DOH and PRC issuances head their parts "I. Rationale",
"B. Specific Guidelines", "1.", "2."; Supreme Court decisions have no section
headers at all. Those documents fell to `_pack`'s oversized branch, which used
to cut at raw character offsets: 632 chunks across 19 files on this corpus,
including all 175 chunks of the landmark informed-consent case.

The cuts landed mid-word — "…surviving relatives who refuse to execute a
promisso", "Detention occ|urs when all of the following are present". That costs
twice: the fragment is unreadable if it reaches the prompt, and the chunk's
embedding is diluted by whatever unrelated text the window happened to scoop up.
Measured consequence: the operative sentence of DOH AO 2008-0001 (a relative who
REFUSES to sign a promissory note may still claim the body) sat in a chunk whose
first half was SSS/GSIS insurance boilerplate, ranked 9th in fusion, and never
reached the prompt — while the offence-elements list from the same file did, and
the model read it as a checklist of conditions the family must satisfy.

Run:  python tests/test_chunk_split.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.chunking import _pack, _split_point  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def eq(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def pieces(text: str, size: int, overlap: int) -> list[str]:
    return [t for t, _ in _pack([text], size, overlap)]


def run() -> int:
    r = []

    # --- the mid-word cut, which is the whole point ------------------------
    # Real shape from DOH AO 2008-0001: sentences of very uneven length with no
    # blank lines, which is what defeated the old arithmetic split.
    ao = ("3. In the case of a deceased patient, any of his/her surviving "
          "relatives who refuse to execute a promissory note shall be allowed "
          "to claim the cadaver and can demand the issuance of death "
          "certificate and other pertinent documents for interment purposes. "
          "Documents for other purposes shall be released only after execution "
          "of a promissory note. ") * 4
    out = pieces(ao, 400, 40)
    # Stated as a property of the BOUNDARY, not of the first character. An
    # earlier version of this test asserted each piece starts with a capital,
    # which is not the invariant: a piece may legitimately open mid-sentence
    # ("to claim the cadaver and can demand…") having broken at a clean space.
    # Every boundary must fall on whitespace in the source — that, and only
    # that, is what stops a word being severed.
    severed = []
    for p in out:
        i = ao.find(p)
        if i > 0 and not ao[i - 1].isspace():
            severed.append(p[:30])
        j = i + len(p)
        if j < len(ao) and not ao[j].isspace():
            severed.append(p[-30:])
    r.append(check("no boundary severs a word", not severed,
                   f"(severed: {severed[:2]})" if severed else ""))

    # --- paragraph breaks are preferred over line and sentence breaks ------
    # The break itself is consumed by the piece that precedes it (and stripped),
    # so the returned index sits AFTER it — 402 is the "\n\n", 404 is past it.
    para = "A" * 200 + "\n\n" + "B" * 200 + "\n\n" + "C" * 200
    r.append(eq("breaks at the last paragraph break under the limit",
                _split_point(para, 0, 500), 404))

    line = "A" * 200 + "\n" + "B" * 200 + "\n" + "C" * 200
    r.append(eq("falls back to a line break", _split_point(line, 0, 500), 402))

    sent = "Aa " * 100 + "End of it. " + "Bb " * 100
    r.append(check("falls back to a sentence end",
                   sent[:_split_point(sent, 0, 400)].endswith("End of it. ")))

    nospace = "x" * 1000
    r.append(eq("a unit with no break at all still cuts at the limit",
                _split_point(nospace, 0, 400), 400))

    # A legal abbreviation's period must not be mistaken for a sentence end.
    abbrev = "word " * 60 + "under Sec. 24 of R.A. No. 9439 the rule applies. tail"
    pt = _split_point(abbrev, 0, 340)
    r.append(check("an abbreviation's period is not a sentence end",
                   not abbrev[:pt].rstrip().endswith(("Sec.", "No.", "R.A.")),
                   f"(ends {abbrev[:pt][-14:]!r})"))

    # --- invariants that must hold for any input ---------------------------
    # Every character of the source has to survive into some chunk. A splitter
    # that silently drops text is the worst possible failure here: retrieval
    # simply never sees the provision and the answer reads as a corpus gap.
    body = ("Section text here. " * 30 + "\n\n"
            + "Another paragraph that runs on. " * 30 + "\n"
            + "Final part without a trailing break " * 20)
    for size, overlap in ((300, 50), (500, 100), (250, 0), (1100, 150)):
        out = pieces(body, size, overlap)
        joined = "".join(out)
        lost = [w for w in set(re.findall(r"\w+", body))
                if w not in joined]
        r.append(check(f"no text is lost (size={size}, overlap={overlap})",
                       not lost, f"(missing {lost[:3]})" if lost else ""))
        r.append(check(f"every piece is within size (size={size})",
                       all(len(p) <= size for p in out),
                       f"(max {max(len(p) for p in out)})"))
        r.append(check(f"no empty piece (size={size})", all(p.strip() for p in out)))

    # Termination: a break found at or before start+overlap would rewind the
    # cursor and loop forever. Pathological input that invites exactly that.
    evil = ("a. " * 400)          # a break candidate every three characters
    out = pieces(evil, 200, 190)  # overlap almost as large as size
    r.append(check("a large overlap still terminates", len(out) > 0,
                   f"({len(out)} pieces)"))
    r.append(check("large overlap does not lose text",
                   "".join(out).count("a.") >= 400))

    # Overlap is honoured: consecutive pieces share text.
    out = pieces("Sentence number one is here. " * 40, 400, 100)
    r.append(check("consecutive pieces overlap", len(out) > 2 and any(
        out[i][-40:] in out[i + 1] or out[i + 1][:40] in out[i]
        for i in range(len(out) - 1))))

    # A unit that already fits is passed through untouched.
    r.append(eq("a unit under size is not split",
                pieces("short enough", 400, 50), ["short enough"]))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
