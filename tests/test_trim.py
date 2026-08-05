"""Tests for the trailing-caveat post-processor in ragmed.rag.

Run:  .venv/Scripts/python.exe -m pytest tests -q
      (or plain: .venv/Scripts/python.exe tests/test_trim.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed.rag import (_stream_trim, strip_provenance,  # noqa: E402
                        strip_source_labels, trim_trailing_caveat)

# A real refusal the system produced: it opens by naming the gap, cites what it
# DID find, then closes by restating the gap. Nothing here may be trimmed.
REFUSAL = (
    "The provided contexts do not directly address whether it is ethical for "
    "doctors and hospitals to recommend the transfer of critically ill "
    "patients solely based on financial considerations.\n\n"
    "Republic Act No. 10932 provides guidelines for transferring patients "
    "when there is an immediate danger to their health (Sec. 2). However, it "
    "does not address ethical considerations related to financial status.\n\n"
    "Therefore, the corpus does not cover the ethics of financially motivated "
    "transfers."
)

ANSWER = (
    "A physician may be administratively liable for immorality, dishonourable "
    "conduct, or malpractice (Republic Act No. 2382, Sec. 24). The Board of "
    "Medicine may reprimand, suspend, or revoke the certificate of "
    "registration after due notice and hearing (Sec. 24)."
)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


def run() -> int:
    results = []

    # --- regression: never cut into the middle of a sentence ---------------
    frag = (ANSWER + " Therefore, it is concluded that the provided context "
            "does not address the ethics of such transfers.")
    results.append(check(
        "mid-sentence caveat is dropped whole, not decapitated",
        trim_trailing_caveat(frag), ANSWER))

    for tail in (
        "In summary, the corpus does not specify an answer to this question.",
        "Therefore, the excerpts do not discuss this topic.",
        "Accordingly, its scope means the documents do not cover that point.",
    ):
        results.append(check(
            f"whole-sentence trim: {tail[:34]!r}",
            trim_trailing_caveat(f"{ANSWER} {tail}"), ANSWER))

    # An abbreviation's period must not pose as a sentence boundary — vital in
    # a legal corpus full of "Sec. 24" / "Republic Act No. 2382". With no real
    # boundary before it, the hedge-like clause stays put rather than being
    # severed from the sentence it belongs to.
    abbrev = (ANSWER + " The penalty is set out in Sec. 24 and the documents "
              "do not cover appeals to the Commission.")
    results.append(check(
        "abbreviation period is not treated as a sentence end",
        trim_trailing_caveat(abbrev), abbrev))

    # --- regression: a genuine refusal survives intact ---------------------
    results.append(check(
        "multi-paragraph refusal is left alone (non-streaming)",
        trim_trailing_caveat(REFUSAL), REFUSAL))

    streamed = "".join(_stream_trim(iter([REFUSAL])))
    results.append(check(
        "multi-paragraph refusal is left alone (streaming)",
        streamed, REFUSAL))

    # token-by-token streaming must behave identically
    toks = [REFUSAL[i:i + 7] for i in range(0, len(REFUSAL), 7)]
    results.append(check(
        "multi-paragraph refusal is left alone (streaming, tokenised)",
        "".join(_stream_trim(iter(toks))), REFUSAL))

    results.append(check(
        "short standalone refusal is left alone",
        trim_trailing_caveat("The corpus does not specify a prescriptive "
                             "period for a PRC administrative complaint."),
        "The corpus does not specify a prescriptive period for a PRC "
        "administrative complaint."))

    # --- still trims what it was built to trim -----------------------------
    results.append(check(
        "trailing hedge after a real answer is trimmed",
        trim_trailing_caveat(ANSWER + " However, the context does not provide "
                             "further detail on the appeal period."), ANSWER))

    results.append(check(
        "trailing sourcing meta-comment is trimmed",
        trim_trailing_caveat(ANSWER + " This is based on the provided "
                             "context."), ANSWER))

    results.append(check(
        "caveat as its own final paragraph is trimmed while streaming",
        "".join(_stream_trim(iter([
            ANSWER, "\n\n", "The provided context does not elaborate further."
        ]))).rstrip(), ANSWER))

    # --- must NOT trim ------------------------------------------------------
    advice = ANSWER + " This is legal information, not legal advice."
    results.append(check(
        "legal-advice disclaimer is preserved", trim_trailing_caveat(advice),
        advice))

    cited = (ANSWER + " The hospital does not incur liability where it did "
             "not cover the procedure under Sec. 4.")
    results.append(check(
        "a substantive final sentence containing 'not cover' is preserved",
        trim_trailing_caveat(cited), cited))

    # --- passage labels emitted AS citations --------------------------------
    # Observed in a real answer: every paragraph opened "[Context 3]
    # [Context 5]" instead of naming a law. Those labels name blocks of the
    # PROMPT, which the reader never sees, so they read as references that
    # resolve to nothing. The prompt now forbids them and the context blocks are
    # no longer bracketed; this is the guarantee behind that request.
    results.append(check(
        "leading passage labels are stripped",
        strip_source_labels(
            "[Context 3] [Context 5] The physician must disclose risks."),
        "The physician must disclose risks."))
    results.append(check(
        "a label mid-sentence leaves no double space",
        strip_source_labels("The duty [Context 2] is to disclose."),
        "The duty is to disclose."))
    results.append(check(
        "a parenthesised label takes its space before the full stop",
        strip_source_labels("This is required (Context 2)."),
        "This is required."))
    results.append(check(
        "the disclaimer survives having a label peeled off it",
        strip_source_labels(
            "[Context 6] This is legal information, not legal advice."),
        "This is legal information, not legal advice."))

    # Must NOT touch real citations, ordinary prose, or markdown structure.
    keep = "The ground is in (Republic Act No. 2382, Sec. 24) and G.R. No. 165279."
    results.append(check("real citations are untouched",
                         strip_source_labels(keep), keep))
    prose = "In this context, 5 years is the prescriptive period."
    results.append(check("'context' in ordinary prose is untouched",
                         strip_source_labels(prose), prose))
    # Regression: an earlier version collapsed every run of spaces in the text,
    # which un-nested list items anywhere a label was removed in the same chunk.
    nested = "[Context 1] The duties are:\n\n- disclose\n    - material risks"
    results.append(check(
        "nested list indentation survives a label removal",
        strip_source_labels(nested),
        "The duties are:\n\n- disclose\n    - material risks"))

    results.append(check("streaming strips labels too",
                         "".join(_stream_trim(iter([
                             "[Context 2] A duty exists.\n\n",
                             "[Context 4] And a second one."]))),
                         "A duty exists.\n\nAnd a second one."))

    # --- provenance parentheticals ------------------------------------------
    # The answer explaining WHERE it found a provision instead of citing it.
    # Rule 3 forbids this and does not succeed, so this is the guarantee — the
    # same three-layer treatment the passage labels get. Both examples below are
    # verbatim from measured answers.
    results.append(check(
        "a parenthetical naming the source file is removed",
        strip_provenance(
            "Furthermore, **Republic Act No. 9439** (as detailed in the "
            "implementing rules in **Republic Act No. 9439** source file) "
            "explicitly states that detention is unlawful."),
        "Furthermore, **Republic Act No. 9439** explicitly states that "
        "detention is unlawful."))
    results.append(check(
        "a parenthetical reproducing a Cite as: header is removed",
        strip_provenance(
            "**Republic Act No. 4226, Sec. 17** (as referenced in **Republic "
            "Act No. 4226 | SECTION 17. Violations.**) lists the refusal."),
        "**Republic Act No. 4226, Sec. 17** lists the refusal."))
    results.append(check(
        "a parenthetical trailing off mid-phrase is removed",
        strip_provenance("The rule (as cited in **Republic Act No. 9439** in) "
                         "applies."),
        "The rule applies."))
    results.append(check(
        "the space before punctuation is repaired after removal",
        strip_provenance("It is unlawful (as stated in the context) ."),
        "It is unlawful."))

    # --- must NOT touch -----------------------------------------------------
    # "(as cited in X)" is ordinary legal writing. Only a parenthetical that
    # names prompt scaffolding, or collapses mid-phrase, is a leak.
    keep = ("The doctrine was adopted (as cited in Ramos v. Court of Appeals) "
            "by this Court.")
    results.append(check("a genuine case parenthetical is preserved",
                         strip_provenance(keep), keep))
    keep2 = "Detention is unlawful in the case of a deceased patient."
    results.append(check("ordinary prose is untouched",
                         strip_provenance(keep2), keep2))
    results.append(check("empty input is handled", strip_provenance(""), ""))
    results.append(check("text with no parenthesis is returned as-is",
                         strip_provenance("No parens here."), "No parens here."))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
