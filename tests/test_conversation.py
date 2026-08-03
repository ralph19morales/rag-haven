"""Tests for conversation memory in ragmed.conversation and ragmed.rag.

Why this exists: a chat assistant over a legal corpus has two failure modes
that a naive "just append the history" implementation walks straight into.

  * A follow-up is searched as written. "And is there a deadline?" carries none
    of the words that say what it is about, so retrieval returns noise and the
    answer is built on the wrong passages. The rewrite step exists to stop
    that, and it must fire on dependent questions and stay out of the way on
    self-contained ones.

  * History becomes a laundered source. Once prior answers are in the prompt,
    the model can cite a provision it "remembers saying" with no retrieved
    passage behind it — a citation a reader cannot distinguish from a real one.
    The prompt must fence history explicitly, and the retrieved context must
    sit nearer the question than the history does.

The LLM is stubbed, so these run offline and in milliseconds.

Run:  .venv/Scripts/python.exe tests/test_conversation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed import config, conversation, llm, rag  # noqa: E402
from ragmed.retriever import Retrieved  # noqa: E402


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


HIST = [
    {"role": "user", "content": "A patient died with an unpaid hospital bill."},
    {"role": "assistant", "content": "The hospital may not detain the body."},
]


def run() -> int:
    r = []
    orig_generate = llm.generate
    orig = (config.HISTORY_REWRITE, config.HISTORY_TURNS,
            config.HISTORY_ASSISTANT_CHARS, config.HISTORY_USER_CHARS)
    try:
        # --- which questions need resolving against earlier turns ----------
        for q in ["And is there a deadline?", "What about the spouse?",
                  "so who pays it?", "Is that the same for a private room?",
                  "Can they do that?"]:
            r.append(check(f"dependent: {q!r}", conversation.needs_rewrite(q), True))
        for q in ["What are the grounds for revoking a physician's licence?",
                  "Can a hospital require a deposit before emergency treatment?"]:
            r.append(check(f"self-contained: {q[:38]!r}…",
                           conversation.needs_rewrite(q), False))
        r.append(check("empty question needs nothing",
                       conversation.needs_rewrite("   "), False))

        # --- history formatting is bounded ---------------------------------
        config.HISTORY_ASSISTANT_CHARS = 20
        long_hist = HIST + [{"role": "assistant", "content": "x" * 500}]
        out = conversation.format_history(long_hist)
        r.append(check("assistant replies are truncated",
                       any(line.endswith("…") for line in out.splitlines()), True))
        r.append(check("every line is attributed",
                       all(l.startswith(("User:", "Assistant:"))
                           for l in out.splitlines()), True))
        config.HISTORY_ASSISTANT_CHARS = orig[2]

        config.HISTORY_TURNS = 1
        many = [{"role": "user", "content": f"q{i}"} for i in range(10)]
        r.append(check("only the last N turns are kept",
                       len(conversation.recent_turns(many)), 2))
        config.HISTORY_TURNS = orig[1]
        r.append(check("no history formats to empty",
                       conversation.format_history([]), ""))

        # --- the rewrite, and its fail-open guarantees ----------------------
        config.HISTORY_REWRITE = True
        llm.generate = lambda *a, **k: "Is there a deadline for filing a claim?"
        r.append(check("dependent question is rewritten",
                       conversation.contextualize("And is there a deadline?", HIST),
                       "Is there a deadline for filing a claim?"))
        r.append(check("self-contained question is not sent for rewrite",
                       conversation.contextualize(
                           "Can a hospital require a deposit before emergency "
                           "treatment?", HIST),
                       "Can a hospital require a deposit before emergency "
                       "treatment?"))

        def boom(*a, **k):
            raise RuntimeError("model down")

        llm.generate = boom
        r.append(check("a failed rewrite falls back to the user's words",
                       conversation.contextualize("And the deadline?", HIST),
                       "And the deadline?"))
        llm.generate = lambda *a, **k: "   "
        r.append(check("an empty rewrite falls back",
                       conversation.contextualize("And the deadline?", HIST),
                       "And the deadline?"))
        llm.generate = lambda *a, **k: "Here is what I think you meant: " + "y" * 400
        r.append(check("a rambling rewrite is rejected",
                       conversation.contextualize("And the deadline?", HIST),
                       "And the deadline?"))
        llm.generate = lambda *a, **k: '"Quoted rewrite"'
        r.append(check("surrounding quotes are stripped",
                       conversation.contextualize("And it?", HIST),
                       "Quoted rewrite"))

        config.HISTORY_REWRITE = False
        llm.generate = lambda *a, **k: "SHOULD NOT BE USED"
        r.append(check("rewrite disabled -> question passes through",
                       conversation.contextualize("And the deadline?", HIST),
                       "And the deadline?"))
        config.HISTORY_REWRITE = True
        r.append(check("no history -> question passes through",
                       conversation.contextualize("And the deadline?", []),
                       "And the deadline?"))

        # --- the grounding fence -------------------------------------------
        chunk = Retrieved(id="1", text="Section 2. Text of the provision.",
                          metadata={"law": "RA 9439", "section": "SEC. 2.",
                                    "source": "ra9439.txt"}, score=1.0)
        p = rag.build_prompt("Q?", [chunk], history="User: earlier\nAssistant: reply")
        r.append(check("history is labelled as not a source",
                       "not a source" in p, True))
        r.append(check("retrieved context sits nearer the question than history",
                       p.index("CONVERSATION SO FAR") < p.index("CONTEXT:")
                       < p.index("QUESTION:"), True))
        r.append(check("no history -> no conversation block at all",
                       "CONVERSATION SO FAR" in rag.build_prompt("Q?", [chunk]),
                       False))
        r.append(check("system prompt forbids citing earlier turns",
                       "Never cite it" in rag.SYSTEM_PROMPT, True))
    finally:
        llm.generate = orig_generate
        (config.HISTORY_REWRITE, config.HISTORY_TURNS,
         config.HISTORY_ASSISTANT_CHARS, config.HISTORY_USER_CHARS) = orig

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
