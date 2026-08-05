"""Orchestration: retrieve -> build grounded prompt -> generate answer.

The prompt is engineered for legal QA: the model must answer ONLY from the
retrieved context, cite the law and section it relied on, and explicitly say
when the corpus does not cover the question (rather than inventing law).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from . import config, conversation, llm, metrics, retriever
from .retriever import Retrieved

# A trailing "the context/corpus does not provide/mention/... more" hedge that
# the model sometimes appends AFTER a perfectly good, cited answer (against the
# prompt's rule 6). This matches only that trailing sentence — NOT a genuine
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


# Passage labels the model sometimes emits AS a citation — "[Context 3]",
# "(Passage 5)", "PASSAGE 2". They name a block of the prompt, which the reader
# has never seen, so they are worse than an uncited sentence: they look like a
# reference and resolve to nothing. Rule 3 forbids them; this removes the ones
# that get through, because a prompt rule is a request and this is a guarantee.
#
# Deliberately narrow. Only the three label words, only when followed by a
# number, and only bracketed/parenthesised or in the SHOUTING form the prompt
# itself uses — so ordinary prose ("in this context, 5 years") is untouched.
_SOURCE_LABEL = re.compile(
    r"""(?:
          [\[(]\s*(?:context|passage|source)\s*\#?\s*\d+\s*[\])]   # [Context 3]
        | \bPASSAGE\s+\d+\b                                        # PASSAGE 3
      )""",
    re.IGNORECASE | re.VERBOSE,
)
# The commonest shape by far is a run of labels opening a paragraph
# ("[Context 3] [Context 5] The physician must…"), so that case is handled
# first and as a unit — including the indentation it would otherwise leave
# behind. Markdown reads four leading spaces as a code block, so a stray label
# at a line start could silently turn a paragraph into a grey monospace box.
_LEADING_LABELS = re.compile(
    r"^[ \t]*(?:" + _SOURCE_LABEL.pattern + r"[ \t]*)+",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)
# A label in the MIDDLE of a sentence takes its surrounding spaces with it and
# leaves exactly one, so the words either side do not run together. Repair has
# to be scoped to the removal site like this: an earlier version collapsed every
# run of spaces in the answer, which silently un-nested markdown list items
# anywhere a label had been removed elsewhere in the same chunk.
_INLINE_LABEL = re.compile(
    r"[ \t]*(?:" + _SOURCE_LABEL.pattern + r")[ \t]*",
    re.IGNORECASE | re.VERBOSE,
)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?)])")


# A parenthetical that explains WHERE a provision was found, rather than citing
# it. Rule 3 forbids these and does not succeed: measured after the rule was
# added, one answer still carried "(as detailed in the implementing rules in
# Republic Act No. 9439 source file)" and "(as referenced in Republic Act No.
# 4226 | SECTION 17. Violations.)" — the second reproducing a "Cite as:" header
# verbatim, pipe separator included. Same three-layer treatment as the passage
# labels, and for the same reason: the reader never sees the prompt, so a
# reference into it resolves to nothing.
#
# Deliberately narrow, because "(as cited in X)" is legitimate legal writing.
# The parenthetical is removed only when it ALSO shows one of the two marks of
# a leak: it names prompt scaffolding (a source file, a passage, the context,
# a "Cite as" line, or the "|" that only ever appears in a header), or it
# trails off on a dangling preposition, which is what a citation collapsing
# mid-phrase looks like.
_PROVENANCE = re.compile(
    r"""\s*\(\s*(?:as\s+)?
        (?:cited|detailed|referenced|quoted|stated|found|shown|listed|
           described|provided|set\s+out|mentioned)
        \s+(?:in|on|at|from)\b
        (?P<body>[^)]*)
        \)""",
    re.IGNORECASE | re.VERBOSE,
)
_SCAFFOLD = re.compile(
    r"\bsource\s+file\b|\bpassages?\b|\bcontexts?\b|\bcite\s+as\b|\|",
    re.IGNORECASE,
)
_DANGLING = re.compile(r"\b(?:in|on|at|from|of|the)\s*$", re.IGNORECASE)


def _is_leak(m: re.Match) -> bool:
    body = m.group("body")
    return bool(_SCAFFOLD.search(body) or _DANGLING.search(body.rstrip(" *")))


def strip_provenance(text: str) -> str:
    """Remove parentheticals that point INTO the prompt instead of citing.

    Leaves a genuine "(as cited in <case>)" alone — only a parenthetical naming
    prompt scaffolding, or one that trails off mid-phrase, is removed. The
    citation itself is untouched, because it sits outside the parentheses."""
    if not text or "(" not in text:
        return text
    out = _PROVENANCE.sub(lambda m: "" if _is_leak(m) else m.group(0), text)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", out) if out != text else text


def strip_source_labels(text: str) -> str:
    """Remove prompt-internal passage labels the model used as citations.

    Only whitespace ADJACENT TO A REMOVED LABEL is touched. Blanket-stripping
    every line's indentation would have been simpler and wrong: markdown nests
    list items by leading spaces, and the answers here are frequently lists of
    obligations, so flattening them would silently restructure the reply."""
    if not text:
        return text
    out = _LEADING_LABELS.sub("", text)
    out = _INLINE_LABEL.sub(" ", out)
    if out == text:
        return text                      # nothing removed, nothing to repair
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    # Per-line rstrip, NOT a strip() of the whole string: this runs on each
    # streamed chunk, and a chunk's trailing blank line is the paragraph break
    # that separates it from the next one. Stripping it glued paragraphs
    # together ("A duty exists.And a second one.").
    return "\n".join(ln.rstrip() for ln in out.split("\n"))


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
            # Scrubbing per emitted chunk is safe because chunks are only ever
            # cut at a blank line and a passage label contains no newline, so a
            # label can never straddle the boundary.
            chunk = strip_provenance(strip_source_labels(buf[:idx + 2]))
            yield chunk
            if len(head) < _HEAD_WINDOW:
                head += chunk
            emitted += len(chunk.strip())
            buf = buf[idx + 2:]
    tail = strip_provenance(strip_source_labels(
        trim_trailing_caveat(buf, prior_len=emitted, head=head + buf)))
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
appears on a "Cite as:" line above. Never cite a case, law, or section whose \
identifier is not on one of those lines — including ones named inside the body \
of a passage.
3. The passages are numbered only to keep them apart. That numbering is \
invisible to the reader, so it can never serve as a citation: never write \
"PASSAGE 3", "Context 3", or any bare bracketed number in your answer. Cite by \
law and section, using the "Cite as:" wording, and nothing else. Give the \
citation and nothing about where you found it — the reader cannot see this \
prompt, so any phrase describing how it is laid out, or trailing off into where \
a provision was quoted from, names something that is not there.
4. Cite Philippine authority only. Philippine decisions in the context often \
quote or discuss foreign rulings and foreign doctrine; that material is \
persuasive reasoning inside a Philippine ruling, not authority of its own. \
Attribute the point to the Philippine case or statute in the header, never to \
the foreign source it quotes. If the corpus shows a doctrine only as quoted \
foreign material, say that plainly instead of citing it as binding.
5. If the context does not answer the question, say so plainly and name the \
specific thing THIS question asked about that is missing (the provision, \
topic, figure, or item). Describe the gap in your own words for the actual \
question asked — do NOT reuse any fixed or example wording, and never carry \
over a sentence about a topic the user did not ask about. Never invent a rule, \
number, deadline, or provision. Use a "not covered" statement ONLY when you \
genuinely cannot answer; never attach it as a caveat to an answer you were \
able to give.
6. When you HAVE answered from the context, stop there. Do not append trailing \
disclaimers, notes about what the context "does not provide", or suggestions \
to "consult the full text".
7. Do not give personalised legal advice or predict case outcomes. You may \
explain what the law says. Add a one-line note that this is legal information, \
not legal advice, when the user seems to be asking about their own situation.
8. Be precise and concise. Quote key statutory language when it matters.
9. Do not turn a description into a requirement. A provision listing the \
elements of an offence, or the circumstances in which conduct becomes \
unlawful, states when a duty has been BREACHED — it is not a checklist the \
person asking must satisfy before the duty exists at all. In the same way, \
where the context grants something in unconditional terms, do not attach to it \
a condition taken from a different provision or a different sentence. If a \
condition genuinely governs, quote the words that impose it and say exactly \
what it applies to; where the context sets out both an unconditional \
entitlement and a narrower conditional one, give the unconditional one first \
and keep the two apart.
10. A CONVERSATION SO FAR block may appear. It is there so you can tell what a \
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
    """Lay out the retrieved passages for the prompt.

    The passage label is deliberately NOT bracketed. It used to be
    "[Context 3]", which looks exactly like a citation in legal writing — and
    the model treated it as one, opening paragraphs with "[Context 3]
    [Context 5]" instead of naming the law. Measured on this corpus: 75 stray
    markers across 12 answers, one answer carrying 65. The reader cannot see
    these labels at all, so a citation made of them is worse than no citation.
    A bare "PASSAGE n" separates the blocks without offering the model a
    citation-shaped token to copy. See rule 2 and the `_SOURCE_LABEL` scrub."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        law = c.metadata.get("law", "")
        section = c.metadata.get("section", "")
        title = c.metadata.get("title", "")
        source = c.metadata.get("source", "")
        header_bits = [b for b in (law, section) if b]
        header = " | ".join(header_bits) if header_bits else (title or source)
        blocks.append(f"PASSAGE {i}\nCite as: {header}\n"
                      f"Source file: {source}\n{c.text}")
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
    t0 = time.perf_counter()
    search_query = question
    history_block = ""
    if history:
        search_query = conversation.contextualize(question, history)
        history_block = conversation.format_history(history)

    debug: dict = {}
    chunks = retriever.retrieve(search_query, top_k=top_k, debug=debug)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    def _log(generation_ms: float, error: str | None) -> None:
        metrics.log_query(
            question=question if config.METRICS_LOG_QUESTIONS else None,
            question_len=len(question),
            num_chunks=len(chunks),
            hyde_fired=debug.get("hyde", False),
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
            total_ms=round(retrieval_ms + generation_ms, 1),
            error=error,
        )

    if not chunks:
        msg = ("The corpus is empty or nothing matched. Ingest documents "
               "first with:  python cli.py ingest")
        _log(generation_ms=0.0, error="empty_index")
        if stream:
            return (iter([msg]), [])
        return Answer(text=msg, sources=[])

    # The ORIGINAL question is what the model answers — the rewrite exists to
    # aim retrieval, and showing the user's own words keeps the reply in their
    # framing rather than the rewriter's.
    prompt = build_prompt(question, chunks, history_block)

    if stream:
        gen_t0 = time.perf_counter()

        def _timed_stream():
            try:
                yield from _stream_trim(llm.generate(SYSTEM_PROMPT, prompt, stream=True))
            except Exception as e:
                _log(generation_ms=(time.perf_counter() - gen_t0) * 1000,
                     error=str(e)[:200])
                raise
            else:
                _log(generation_ms=(time.perf_counter() - gen_t0) * 1000, error=None)

        return (_timed_stream(), chunks)

    gen_t0 = time.perf_counter()
    try:
        text = llm.generate(SYSTEM_PROMPT, prompt, stream=False)
    except Exception as e:
        _log(generation_ms=(time.perf_counter() - gen_t0) * 1000, error=str(e)[:200])
        raise
    _log(generation_ms=(time.perf_counter() - gen_t0) * 1000, error=None)
    return Answer(text=strip_provenance(strip_source_labels(
        trim_trailing_caveat(text))), sources=chunks)
