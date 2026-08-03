"""Tests for HyDE query expansion in ragmed.retriever.

Why this exists: a question asked in lay narrative ("they kept changing their
story") does not embed near terse statutory text and carries no distinctive
tokens for BM25, so retrieval returns noise and the system falsely refuses.
HyDE drafts a short hypothetical answer in domain vocabulary and retrieves with
that too. Measured on this corpus, the failing question went from best raw
similarity 0.653 (flat ranking, refusal) to 0.776 with the relevant cases
surfacing.

The LLM is mocked here, so these run with Ollama down.

Run:  .venv/Scripts/python.exe tests/test_hyde.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragmed import config, llm, retriever  # noqa: E402


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got:  {got!r}\n      want: {want!r}")
    return ok


HIGH = [0.79, 0.75, 0.71]   # a well-posed question
LOW = [0.65, 0.62, 0.60]    # a lay-narrative question


def run() -> int:
    r = []
    orig_mode, orig_min = config.HYDE_MODE, config.HYDE_MIN_SIM
    orig_generate = llm.generate
    config.HYDE_MIN_SIM = 0.68
    try:
        # --- mode gating ----------------------------------------------------
        config.HYDE_MODE = "off"
        r.append(check("off: never fires, even on a poor match",
                       retriever._should_hyde(LOW), False))

        config.HYDE_MODE = "always"
        r.append(check("always: fires even on a strong match",
                       retriever._should_hyde(HIGH), True))

        config.HYDE_MODE = "auto"
        r.append(check("auto: skips a well-posed question",
                       retriever._should_hyde(HIGH), False))
        r.append(check("auto: fires on a lay-narrative question",
                       retriever._should_hyde(LOW), True))
        r.append(check("auto: no candidates at all does not fire",
                       retriever._should_hyde([]), False))
        r.append(check("auto: exactly at the threshold does not fire",
                       retriever._should_hyde([0.68]), False))
        r.append(check("auto: just under the threshold fires",
                       retriever._should_hyde([0.679]), True))

        # An unrecognised mode must behave like 'off', not crash or default on.
        config.HYDE_MODE = "typo"
        r.append(check("an unknown mode is treated as off",
                       retriever._should_hyde(LOW), False))
        config.HYDE_MODE = "auto"

        # --- drafting, and failing open -------------------------------------
        llm.generate = lambda *a, **k: "  The doctrine of quasi-delict applies. "
        r.append(check("draft is returned, stripped",
                       retriever._hypothetical("q"),
                       "The doctrine of quasi-delict applies."))

        def boom(*a, **k):
            raise RuntimeError("ollama is down")

        llm.generate = boom
        r.append(check("LLM failure degrades to plain retrieval (returns None)",
                       retriever._hypothetical("q"), None))

        llm.generate = lambda *a, **k: "   \n  "
        r.append(check("an empty draft is treated as no draft",
                       retriever._hypothetical("q"), None))

        llm.generate = lambda *a, **k: None
        r.append(check("a None draft is handled",
                       retriever._hypothetical("q"), None))

        # The token cap must actually be passed through — an uncapped draft is
        # slow and dilutes the embedding it exists to produce.
        seen = {}

        def capture(system, prompt, stream=False, max_tokens=None):
            seen["max_tokens"] = max_tokens
            seen["system"] = system
            return "drafted"

        llm.generate = capture
        retriever._hypothetical("q")
        r.append(check("the draft is length-capped",
                       seen["max_tokens"], config.HYDE_MAX_TOKENS))

        # Policy guard: a fabricated "Article 2200" is a distinctive token that
        # BM25 matches hard, aiming retrieval confidently at the wrong law.
        # The prompt must keep forbidding invented identifiers.
        prompt = retriever._HYDE_SYSTEM.lower()
        r.append(check("the draft prompt forbids inventing citation numbers",
                       ("do not invent" in prompt and "number" in prompt), True))
    finally:
        config.HYDE_MODE, config.HYDE_MIN_SIM = orig_mode, orig_min
        llm.generate = orig_generate

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
