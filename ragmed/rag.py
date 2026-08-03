"""Orchestration: retrieve -> build grounded prompt -> generate answer.

The prompt is engineered for legal QA: the model must answer ONLY from the
retrieved context, cite the law and section it relied on, and explicitly say
when the corpus does not cover the question (rather than inventing law).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import conversation, llm, retriever
from .retriever import Retrieved

# A trailing "the context/corpus does not provide/mention/... more" hedge that
# the model sometimes appends AFTER a perfectly good, cited answer (against the
# prompt's rule 5). This matches only that trailing sentence — NOT a genuine
# full refusal (guarded by _opens_with_refusal + a minimum-answer check) and NOT
# the "legal information, not legal advice" note (different wording).
#
# Every pattern must start at a SENTENCE boundary: without this the match can
# begin mid-sentence ("Therefore, it is concluded that the context does not
# cover X.") and trimming decapitates the sentence, leaving a dangling fragment
# ("Therefore, it is concluded that"). The negative lookbehinds stop a legal
# abbreviation's period ("Sec. 2", "No. 10932") from posing as a boundary.
_BOUNDARY = (
    r"(?:^|(?<=[.!?])|(?<=\n))"
    r"(?<!\bNo\.)(?<!\bNos\.)(?<!\bSec\.)(?<!\bSecs\.)(?<!\bArt\.)(?<!\bArts\.)"
    r"(?<!\bpar\.)(?<!\bInc\.)(?<!\bv\.)"
    r"\s*"
)
# A short lead-in the hedge may hide behind INSIDE the same sentence
# ("Therefore, it is concluded that …", "In summary, …"). It cannot cross a
# sentence terminator, so a match still starts at the boundary and the whole
# sentence — lead-in included — is what gets removed.
_LEAD_IN = r"[^.!?\n]{0,90}?"
# Core of the negative hedge, unanchored — reused to detect a refusal OPENING.
_REFUSAL_CORE = (
    r"(?:the\s+)?(?:provided\s+|given\s+|above\s+)?"
    r"(?:contexts?|corpus|excerpts?|passages?|sources?|documents?|"
    r"information\s+(?:provided|given|available)|text\s+provided)"
    r"\b[^.]*?\b(?:do(?:es)?|did|is|are|was|were)\b[^.]*?\bn[o']?t\b[^.]*?"
    r"(?:provide|contain|specif|mention|include|cover|detail|address|elaborat|"
    r"state|indicat|give|discuss|describe|offer|go into)"
)
# (a) negative hedge: "the context does not provide/mention/... more".
_CAVEAT_TAIL = re.compile(
    _BOUNDARY + _LEAD_IN + _REFUSAL_CORE
    + r"[^.]*\.\s*$",
    re.IGNORECASE,
)
# (b) positive sourcing meta-comment: "This information is derived from the
#     provided context." / "The above is based on the passages."
_META_TAIL = re.compile(
    _BOUNDARY + _LEAD_IN
    + r"(?:this|the above|the foregoing|all of (?:this|the above))\b[^.]{0,70}?"
      r"\b(?:based|derived|drawn|taken|sourced)\b[^.]*?"
      r"\b(?:contexts?|corpus|passages?|excerpts?|sources?|documents?|"
      r"provided\s+\w+|information provided)"
      r"[^.]*\.\s*$",
    re.IGNORECASE,
)
_TAIL_PATTERNS = (_CAVEAT_TAIL, _META_TAIL)
_MIN_KEEP = 60   # don't trim if it would leave less than a real answer behind
_HEAD_WINDOW = 250  # chars of the answer's opening inspected for a refusal

_OPENING_REFUSAL = re.compile(_REFUSAL_CORE, re.IGNORECASE)


def _opens_with_refusal(head: str) -> bool:
    """True when the answer LEADS with "the context does not cover X".

    Such an answer is a genuine refusal from end to end — its closing sentence
    restates the gap and is part of the answer, not a stray caveat appended to
    a substantive one. Trimming there is what produced the dangling-fragment
    bug, because the earlier paragraphs of a long refusal satisfied the
    prior_len guard."""
    return bool(_OPENING_REFUSAL.search(head[:_HEAD_WINDOW]))


def trim_trailing_caveat(text: str, prior_len: int = 0,
                         head: str | None = None) -> str:
    """Remove up to a few trailing meta sentences the model appends after a
    substantive answer — negative hedges ("the context does not provide…") and
    positive sourcing notes ("this is based on the provided context"). Leaves
    genuine refusals and deliberate disclaimers (e.g. the legal-advice note)
    intact, and never cuts into the middle of a sentence.

    prior_len = chars of real answer already emitted before `text` (used by the
    streaming path, where the answer may live in earlier, already-flushed
    paragraphs). The guard keeps a caveat only when little TOTAL answer remains.
    head = the answer's opening (defaults to `text`); the streaming path passes
    the already-flushed start so a refusal opening is still visible here."""
    out = text.rstrip()
    if _opens_with_refusal(out if head is None else head):
        return out
    for _ in range(3):
        m = None
        for pat in _TAIL_PATTERNS:
            m = pat.search(out)
            if m:
                break
        if not m:
            break
        candidate = out[:m.start()].rstrip()
        if len(candidate) + prior_len < _MIN_KEEP:  # answer too small → keep it
            break
        out = candidate
    return out


def _stream_trim(token_gen):
    """Stream tokens live but hold back the final paragraph so a trailing
    caveat in it can be trimmed before it is ever shown. Tracks how much real
    answer was already emitted so a caveat that is its OWN final paragraph is
    still trimmable, and keeps the answer's opening so a whole-answer refusal
    is recognised."""
    buf = ""
    head = ""
    emitted = 0
    for tok in token_gen:
        buf += tok
        idx = buf.rfind("\n\n")           # emit everything before the last blank line
        if idx != -1:
            chunk = buf[:idx + 2]
            yield chunk
            if len(head) < _HEAD_WINDOW:
                head += chunk
            emitted += len(chunk.strip())
            buf = buf[idx + 2:]
    tail = trim_trailing_caveat(buf, prior_len=emitted, head=head + buf)
    if tail:
        yield tail

SYSTEM_PROMPT = """You are a careful legal research assistant specialising in \
Philippine medical law, health policy, and the regulation of medical \
professionals (including PRC licensing and disciplinary matters, relevant \
civil-law and criminal-law provisions, and Department of Health / PRC \
issuances).

Rules you must follow:
1. Answer ONLY using the provided CONTEXT. Do not use outside knowledge or \
guess at provisions that are not shown.
2. Cite the specific law and section for every legal assertion, e.g. \
"(Republic Act No. 2382, Sec. 24)". Every citation must use an identifier that \
appears in a [Context N] header line above. Never cite a case, law, or section \
whose identifier is not in one of those headers — including ones named inside \
the body of a passage.
3. Cite Philippine authority only. Philippine decisions in the context often \
quote or discuss foreign rulings and foreign doctrine; that material is \
persuasive reasoning inside a Philippine ruling, not authority of its own. \
Attribute the point to the Philippine case or statute in the header, never to \
the foreign source it quotes. If the corpus shows a doctrine only as quoted \
foreign material, say that plainly instead of citing it as binding.
4. If the context does not answer the question, say so plainly and name the \
specific thing THIS question asked about that is missing (the provision, \
topic, figure, or item). Describe the gap in your own words for the actual \
question asked — do NOT reuse any fixed or example wording, and never carry \
over a sentence about a topic the user did not ask about. Never invent a rule, \
number, deadline, or provision. Use a "not covered" statement ONLY when you \
genuinely cannot answer; never attach it as a caveat to an answer you were \
able to give.
5. When you HAVE answered from the context, stop there. Do not append trailing \
disclaimers, notes about what the context "does not provide", or suggestions \
to "consult the full text".
6. Do not give personalised legal advice or predict case outcomes. You may \
explain what the law says. Add a one-line note that this is legal information, \
not legal advice, when the user seems to be asking about their own situation.
7. Be precise and concise. Quote key statutory language when it matters.
8. A CONVERSATION SO FAR block may appear. It is there so you can tell what a \
follow-up refers to and avoid repeating yourself. It is NOT a source. Never \
cite it, never treat anything you said earlier as established law, and never \
carry a citation forward from an earlier turn — if a provision matters to this \
answer it must appear in the CONTEXT below, or you cannot rely on it. When the \
context no longer supports something you said earlier, say so plainly."""


@dataclass
class Answer:
    text: str
    sources: list[Retrieved]


def _format_context(chunks: list[Retrieved]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        law = c.metadata.get("law", "")
        section = c.metadata.get("section", "")
        title = c.metadata.get("title", "")
        source = c.metadata.get("source", "")
        header_bits = [b for b in (law, section) if b]
        header = " | ".join(header_bits) if header_bits else (title or source)
        blocks.append(f"[Context {i}] {header}\nSource file: {source}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, chunks: list[Retrieved],
                 history: str = "") -> str:
    context = _format_context(chunks)
    # History goes FIRST and context LAST so the passages sit nearest the
    # question — the position the model weighs most heavily, and the one that
    # must win when the two disagree.
    head = (f"CONVERSATION SO FAR (for reference only — not a source, never "
            f"cite it):\n{history}\n\n" if history else "")
    return (
        f"{head}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above. Cite the law and section for "
        "each point. If the context is insufficient, say so."
    )


def answer(question: str, top_k: int | None = None, stream: bool = False,
           history: list[dict] | None = None):
    """Run the full RAG pipeline.

    Returns an Answer when stream=False, or a (generator, sources) tuple when
    stream=True so callers can render tokens live and show sources after.

    `history` is the caller's prior turns ({"role", "content"} dicts). When
    given, a context-dependent follow-up is rewritten into a standalone query
    BEFORE retrieval — otherwise "and is there a deadline?" is searched as
    written and finds nothing — and the recent turns are replayed to the model,
    fenced as reference rather than as a source. See ragmed/conversation.py.
    """
    search_query = question
    history_block = ""
    if history:
        search_query = conversation.contextualize(question, history)
        history_block = conversation.format_history(history)

    chunks = retriever.retrieve(search_query, top_k=top_k)

    if not chunks:
        msg = ("The corpus is empty or nothing matched. Ingest documents "
               "first with:  python cli.py ingest")
        if stream:
            return (iter([msg]), [])
        return Answer(text=msg, sources=[])

    # The ORIGINAL question is what the model answers — the rewrite exists to
    # aim retrieval, and showing the user's own words keeps the reply in their
    # framing rather than the rewriter's.
    prompt = build_prompt(question, chunks, history_block)

    if stream:
        return (_stream_trim(llm.generate(SYSTEM_PROMPT, prompt, stream=True)),
                chunks)

    text = llm.generate(SYSTEM_PROMPT, prompt, stream=False)
    return Answer(text=trim_trailing_caveat(text), sources=chunks)
