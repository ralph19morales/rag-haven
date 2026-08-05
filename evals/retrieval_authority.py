"""Does the CONTROLLING authority reach the prompt, and is the context varied?

UNLIKE tests/, these need the real index and the real models — they measure the
system, not the code. Run them before and after any change to retrieval
ordering; the numbers in ragmed/config.py's comments came from here.

    python evals/retrieval_authority.py before
    # ...change something...
    python evals/retrieval_authority.py after

Each case names the instrument that actually governs the question, established
independently of what the system happens to return. Two cases are deliberately
case-law-governed: the doctrine lives in jurisprudence there, so a "fix" that
merely drags statutes upward must be seen to break them.

Metrics, because a ranking change trades them against each other:
  AUTHORITY  the controlling instrument is in the top_k        (must not fall)
  FAMILIES   distinct corpus families in the top_k             (a PROXY — see below)
  SUPPORT    for a question a statute and its IRR answer TOGETHER, both present
  KEY        the OPERATIVE SENTENCE reached the prompt         (see below)

On KEY: "the right file is in the top_k" is a weaker claim than it sounds, and
believing it cost this project a session. The deceased-body answer was diagnosed
as a prompt-level fault — the model "treating a condition as a precondition" —
on the assumption that the text stating the entitlement had reached the prompt.
It had not. The IRR chunk that arrived was the offence-ELEMENTS list, from the
same file; the chunk carrying the entitlement was capped out three stages
earlier. AUTHORITY was green throughout. Where a question turns on one specific
sentence, name that sentence here, not just its file.

On FAMILIES: it is gameable. Breadth can be manufactured by importing junk, and
a question one family genuinely answers gets worse when breadth is forced. Never
raise MAX_CHUNKS_PER_FAMILY on the strength of this number alone — check the
informed-consent case (pure jurisprudence) has not gained off-topic chunks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed import config, embeddings, rerank, retriever  # noqa: E402

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"

# (question, controlling authority, [instruments that should appear TOGETHER])
CASES = [
    ("The hospital won't release my relative's body until we pay the bill. Can they do that?",
     # The STATUTE specifically: it carries the operative prohibition. The IRR
     # chunk that used to reach the prompt was the offence-ELEMENTS list, which
     # the model read as a permission checklist — so "some RA 9439 file
     # appeared" is not good enough.
     ["anti-hospital-detention-of-patients-ra-9439"],
     ["anti-hospital-detention-of-patients-ra-9439", "doh-ao-2008-0001"]),
    ("Can a hospital refuse to treat me in an emergency because I can't pay a deposit?",
     ["anti-hospital-deposit-law-ra-8344", "ra-10932"], []),
    ("Am I entitled to a senior citizen discount on my hospital bill?",
     ["expanded-senior-citizens-act-of-2010-ra-9994"], []),
    ("What are the grounds for revoking a physician's certificate of registration?",
     ["RA-2382-Medical-Act-1959-excerpt", "medical-act-of-1959-ra-2382"], []),
    ("Do I have a right to see my own medical records?",
     ["data-privacy-act-of-2012-ra-10173"], []),
    # Case-law-governed ON PURPOSE. Jurisprudence in the top_k is CORRECT here.
    ("My doctor never explained the risks before my operation. What was I owed?",
     ["dr-rubi-li-v-spouses-soliman"], []),
    ("Is a hospital liable for the negligence of its doctors?",
     ["ramos-v-court-of-appeals", "professional-services", "nogales"], []),
]

# The sentence a question actually turns on, where there is one. Keyed by case
# index. Matched against the retrieved TEXT (whitespace-normalised), not the
# filename — that is the whole point of the metric.
KEY_TEXT = {
    # DOH AO 2008-0001, B.2.3. The unconditional half of the rule: a relative
    # who REFUSES to sign a promissory note may still claim the cadaver and the
    # interment documents. Without this sentence the only IRR text available is
    # the elements-of-the-offence list, which reads as a checklist of conditions
    # the family must satisfy — and the answer hedges accordingly.
    0: "who refuse to execute a promissory note shall be allowed to claim the cadaver",
}

# Chunks that are plainly unrelated to the question, for the guard described
# above. Keyed by case index.
RELEVANT = {
    0: ["9439", "4226", "2008-0001"],
    5: ["rubi-li", "casumpang", "bontilao", "informed", "nogales", "ramos",
        "professional-services"],
}


def run() -> int:
    embeddings.warmup()
    rerank.warmup()
    auth = fams = support_hit = support_total = 0
    key_hit = key_total = 0

    print(f"protect_top={config.RERANK_PROTECT_TOP} "
          f"family_cap={config.MAX_CHUNKS_PER_FAMILY} "
          f"source_cap={config.MAX_CHUNKS_PER_SOURCE} top_k={config.TOP_K}")
    print(f"{'hit':<5}{'rank':<6}{'fams':<6}{'off':<5}{'key':<5}question")
    print("-" * 84)

    for i, (q, expected, both) in enumerate(CASES):
        chunks = retriever.retrieve(q)
        srcs = [c.metadata.get("source", "") for c in chunks]
        rank = next((n for n, s in enumerate(srcs, 1)
                     if any(e in s for e in expected)), None)
        auth += rank is not None
        n_fams = len({retriever.source_family(s) for s in srcs})
        fams += n_fams
        off = ""
        if i in RELEVANT:
            n_off = sum(1 for s in srcs
                        if not any(k in s for k in RELEVANT[i]))
            off = f"{n_off}/{len(srcs)}"
        key = ""
        if i in KEY_TEXT:
            key_total += 1
            body = " ".join(" ".join(c.text.split()) for c in chunks)
            found = KEY_TEXT[i] in body
            key_hit += found
            key = "YES" if found else "NO"
        if both:
            support_total += 1
            support_hit += all(any(b in s for s in srcs) for b in both)
        print(f"{'YES' if rank else 'no ':<5}{str(rank or '-'):<6}"
              f"{n_fams:<6}{off:<5}{key:<5}{q[:52]}")

    n = len(CASES)
    print("-" * 84)
    print(f"[{LABEL}] authority {auth}/{n} | mean families {fams / n:.2f} | "
          f"statute+IRR together {support_hit}/{support_total} | "
          f"key sentence {key_hit}/{key_total}")
    # Reference numbers on this corpus, 2026-08-05:
    #   protect=0 family=0 -> authority 4/7, families 2.00, support 0/1
    #   protect=3 family=2 -> authority 7/7, families 2.86, support 1/1
    # KEY was added 2026-08-05 (later): it was 0/1 at every setting above, which
    # is what the "authority is green, the answer is still wrong" state was.
    return 0 if auth == n and key_hit == key_total else 1


if __name__ == "__main__":
    raise SystemExit(run())
