"""Conversation memory: make follow-up questions work without breaking grounding.

A chat assistant over a legal corpus has TWO problems that look like one, and
solving only the second is the usual mistake.

  1. RETRIEVAL. "And is there a deadline?" is meaningless on its own. Embedded
     as written it retrieves noise, because the words that say what it is about
     ("claim against the estate of a deceased patient") are in the PREVIOUS
     turn. No amount of conversation history in the generation prompt fixes
     this — by then retrieval has already failed. So a context-dependent
     question is rewritten into a standalone one BEFORE it is searched.

  2. GENERATION. The model needs the prior turns to write a coherent reply
     rather than restarting the topic each time.

And one risk that matters more here than in a general chatbot:

  3. GROUNDING. The moment prior assistant answers go into the prompt, they
     become a second apparent source — and the model will happily cite a
     provision it "remembers stating" a turn ago, with no retrieved passage
     behind it. That laundered citation is indistinguishable from a real one to
     a reader, which is exactly the failure this whole system exists to
     prevent. History is therefore fenced and labelled as NOT a source, and the
     system prompt is given a matching rule.

Both LLM-facing steps fail open: if the rewrite call errors, the original
question is searched; if history cannot be formatted, the turn proceeds without
it. Conversation memory is an enhancement, never a dependency.
"""
from __future__ import annotations

import re

from . import config

# Markers that a question leans on the previous turn to be understood. Kept
# deliberately cheap — this runs before every turn, and the cost of a false
# positive is one short LLM call, while the cost of a false negative is a
# failed search. Bias generous.
_CONTINUATION = re.compile(
    r"^\s*(?:and|but|so|then|also|what about|how about|ok(?:ay)?[, ]|"
    r"that|those|it|they|he|she|this)\b",
    re.IGNORECASE,
)
_ANAPHORA = re.compile(
    r"\b(?:it|its|that|this|those|these|they|them|their|he|she|his|her|"
    r"the same|instead|as well|too|there)\b",
    re.IGNORECASE,
)

_REWRITE_SYSTEM = """You rewrite a follow-up question so it stands on its own.

Rules:
1. Use the conversation ONLY to resolve what the question refers to — who, \
what, and which situation.
2. Keep the user's own words wherever you can. Do NOT add legal terminology, \
statute names, section numbers, or any fact the user did not say. You are \
resolving references, not improving the question.
3. If the question already stands on its own, return it EXACTLY as written.
4. Output only the rewritten question. No preamble, no explanation, no quotes."""


def needs_rewrite(question: str) -> bool:
    """Whether this question probably depends on earlier turns to make sense."""
    q = question.strip()
    if not q:
        return False
    words = q.split()
    if _CONTINUATION.match(q):
        return True
    # Short questions carrying a pronoun are the classic dependent follow-up
    # ("is there a deadline for that?"). Long ones usually restate their own
    # subject even when they contain a pronoun.
    return len(words) <= 12 and bool(_ANAPHORA.search(q))


def recent_turns(messages: list[dict], max_turns: int | None = None) -> list[dict]:
    """The last few turns, oldest first. `messages` is the app's display history
    ({"role", "content"} dicts); anything else on those dicts is ignored."""
    max_turns = max_turns or config.HISTORY_TURNS
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    return turns[-(max_turns * 2):]


def format_history(messages: list[dict]) -> str:
    """Render prior turns compactly, with assistant replies truncated.

    Assistant answers are the long part of a transcript and the least useful to
    replay in full — their substance is in the retrieved passages, which are
    re-retrieved each turn anyway. Truncating them is what keeps history from
    crowding out the passages inside a fixed context window.
    """
    turns = recent_turns(messages)
    if not turns:
        return ""
    lines = []
    for m in turns:
        who = "User" if m["role"] == "user" else "Assistant"
        text = " ".join((m.get("content") or "").split())
        limit = (config.HISTORY_USER_CHARS if m["role"] == "user"
                 else config.HISTORY_ASSISTANT_CHARS)
        if len(text) > limit:
            text = text[:limit].rstrip() + " …"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def contextualize(question: str, messages: list[dict]) -> str:
    """Rewrite a follow-up into a standalone search query.

    Returns the original question unchanged when there is nothing to resolve,
    when the rewrite is disabled, or when the model call fails — retrieval must
    never be blocked by this step.
    """
    if not config.HISTORY_REWRITE or not messages:
        return question
    if not needs_rewrite(question):
        return question
    hist = format_history(messages)
    if not hist:
        return question

    from . import llm  # local import: keeps this module importable without an
                       # LLM configured (tests, ingest, offline tooling).
    user = f"Conversation so far:\n{hist}\n\nFollow-up question: {question}"
    try:
        out = llm.generate(_REWRITE_SYSTEM, user, stream=False,
                           max_tokens=config.HISTORY_REWRITE_MAX_TOKENS)
    except Exception:  # noqa: BLE001 - any LLM/transport failure
        return question
    out = (out or "").strip().strip('"').strip()
    # A rewrite that came back empty, or that ballooned into an explanation,
    # is not trustworthy — searching the user's own words beats searching the
    # model's commentary about them.
    if not out or len(out) > len(question) * 6 + 120:
        return question
    return out
