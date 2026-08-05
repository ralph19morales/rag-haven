"""Does the system still produce a correct-looking ANSWER?

The gap this closes: every other check in this project measures whether the
machine is working. Both of the worst defects shipped past all of them.

  * n-gram speculative decoding duplicated fragments already present in the
    prompt into the answer — including a mangled case number, `G.R. No.
    210445,0445`. Latency was healthy the entire time.
  * The prompt's own passage labels were emitted AS citations
    ("[Context 3] [Context 5] The physician must…"), which resolve to nothing
    for a reader who never sees the prompt.

Both were found by a human reading a reply. This makes that a command.

Needs the index and a running LLM, and costs one generation per question — so
it is not part of tests/. The ops dashboard runs a single-question version of
the same checks behind its "Run canary" button.

    python evals/answer_quality.py            # 3 questions x 2 samples
    python evals/answer_quality.py label 4    # 4 samples each
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed import embeddings, rag, rerank  # noqa: E402

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 2

QUESTIONS = [
    "The hospital won't release my relative's body until we pay the bill. Can they do that?",
    "My doctor never explained the risks before my operation. What was I owed?",
    "What are the grounds for revoking a physician's certificate of registration?",
]

# Prompt-internal labels. Rule 3 of SYSTEM_PROMPT forbids these and
# rag.strip_source_labels removes them; anything found here means both failed.
LABELS = re.compile(r"\[Context\s*\d+\]|\bPASSAGE\s+\d+\b", re.I)
# A 1-4 word fragment repeated back-to-back — the signature of a mis-accepted
# speculative draft.
FRAGMENT = re.compile(
    r"""\b([A-Za-z][\w']*(?:\s+[\w']+){0,3})\b[\s"'’,.;:]{1,6}\1\b""", re.I)
# Something that looks like a real Philippine citation.
CITATION = re.compile(
    r"(?:Republic Act No\.|G\.R\. No\.|Sec\.|Section|Article|Rule)\s*\S+")
# The answer describing the PROMPT's own scaffolding instead of just citing.
# Same defect family as the leaked passage labels: machinery the reader cannot
# see, surfacing in the reply. Observed on the deceased-body question as two
# dangling parentheticals in a single answer — "(as cited in **Republic Act No.
# 9439** in)" and "**Republic Act No. 4226, Section 17** (as cited in)" — which
# are worse than a bare citation because they promise a provenance and then
# stop mid-phrase. `Cite as:` is a prompt-internal label, so any mention of it
# is by definition a leak.
PROMPT_META = re.compile(
    r"\bas cited in\b|\bcite as\b|\bthe (?:provided|above|given) (?:context|passages?)\b"
    r"|\bsource file\b",
    re.I)
# Benign: a bolded list heading restating its own word, "**Insanity**: Insanity".
def _is_heading_echo(text: str, frag: str) -> bool:
    return bool(re.search(r"\*\*\s*" + re.escape(frag) + r"\s*\*\*\s*:", text, re.I))


def run() -> int:
    embeddings.warmup()
    rerank.warmup()
    labels = frags = uncited = meta = 0
    n = 0
    for q in QUESTIONS:
        for _ in range(SAMPLES):
            text = rag.answer(q, stream=False).text
            n += 1
            found_labels = LABELS.findall(text)
            found_frags = [m.group(1) for m in FRAGMENT.finditer(text)
                           if not _is_heading_echo(text, m.group(1))]
            found_meta = PROMPT_META.findall(text)
            cited = CITATION.findall(text)
            labels += len(found_labels)
            frags += len(found_frags)
            meta += len(found_meta)
            uncited += not cited
            if found_labels or found_frags or found_meta or not cited:
                print(f"  [{q[:34]}…] labels={len(found_labels)} "
                      f"fragments={found_frags[:3]} meta={found_meta[:3]} "
                      f"citations={len(cited)}")

    print(f"\n[{LABEL}] over {n} answers: leaked labels={labels}  "
          f"repeated fragments={frags}  prompt-meta={meta}  "
          f"uncited answers={uncited}")
    # Reference on this corpus, 2026-08-05: 0 / 0 / 0.
    # With speculative decoding enabled it was 7 fragments across 15 answers.
    # prompt-meta was added later the same day, after an answer carried two
    # dangling "(as cited in …)" parentheticals; it was NOT measured at 0.
    return 0 if not (labels or frags or meta or uncited) else 1


if __name__ == "__main__":
    raise SystemExit(run())
