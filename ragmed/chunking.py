"""Legal-aware chunking.

Philippine statutes and rules are structured as SECTIONs (and articles,
rules, etc.). Naive fixed-size chunking splits a Section mid-sentence and
loses the "Section N" anchor that a legal answer must cite. This module:

  1. Detects document-level metadata (e.g. "Republic Act No. 2382", title).
  2. Splits text on legal-structure boundaries (SECTION / ARTICLE / RULE).
  3. Packs those units into chunks near CHUNK_SIZE, keeping each chunk's
     originating section label so it can be carried into citations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

# Matches headers like:  "SECTION 12.", "Sec. 3.", "ARTICLE IV", "RULE III"
SECTION_RE = re.compile(
    r"^\s*(?:SEC(?:TION|\.)?|ARTICLE|ART\.|RULE)\s+"
    r"(?:[0-9]+|[IVXLCDM]+)\.?",
    re.IGNORECASE | re.MULTILINE,
)

# Document identifiers commonly found near the top of a legal text.
# Accept the abbreviated "R.A. 7305" as well as "Republic Act No. 7305" — DOH
# and PRC issuances title themselves with the short form, and requiring the
# spelled-out words left them untagged (uncitable) in the index.
RA_RE = re.compile(
    r"(?:Republic\s+Act|\bR\.\s?A\.)\s*(?:No\.?\s*)?(\d+)", re.IGNORECASE)
# "Act No. 3815" (the Revised Penal Code). MUST be tested after RA_RE, since
# "Republic Act No. 386" also contains the substring "Act No. 386".
ACT_RE = re.compile(r"\bAct\s+No\.?\s*(\d+)", re.IGNORECASE)
# Codes of ethics carry no number; label them by name so they stay citable.
CODE_RE = re.compile(
    r"((?:[A-Z][\w'&-]*\s+){0,4}Code\s+of\s+Ethics)", re.IGNORECASE)
PD_RE = re.compile(r"Presidential\s+Decree\s+(?:No\.?\s*)?(\d+)", re.IGNORECASE)
BP_RE = re.compile(r"Batas\s+Pambansa\s+(?:Blg\.?\s*)?(\d+)", re.IGNORECASE)
# A Rule of Court carries no RA/PD number, so without this it indexes with
# law=None — retrievable but UNCITABLE, the failure this detector exists to
# prevent. Deliberately narrow: it requires the words "Rules of Court" nearby,
# so a decision or statute that merely mentions "Rule 65" is not relabelled.
# Tested after the statute patterns for the same reason.
RULE_RE = re.compile(
    r"Rules\s+of\s+Court.{0,400}?^\s*RULE\s+(\d+)\b",
    re.IGNORECASE | re.DOTALL | re.MULTILINE)
AO_RE = re.compile(r"Administrative\s+Order\s+(?:No\.?\s*)?([\w\-]+)", re.IGNORECASE)
RESO_RE = re.compile(r"Resolution\s+(?:No\.?\s*)?([\w\-]+)", re.IGNORECASE)
# Supreme Court decisions are identified by a G.R. (or "L-") docket number.
# A case that *quotes* a statute must NOT be mislabelled as that statute, so
# this is detected first and, when present, wins.
GR_RE = re.compile(r"G\.?\s*R\.?\s+Nos?\.?\s+(L?-?\s?\d[\d\-]*)", re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def detect_doc_metadata(text: str, filename: str) -> dict:
    """Best-effort extraction of the document's legal identity."""
    head = text[:3000]
    meta: dict = {"source": filename}

    # Court decision? Identify by docket number and stop — do not let a statute
    # cited inside the ruling hijack the document's identity.
    if m := GR_RE.search(head):
        num = re.sub(r"\s+", "", m.group(1))
        meta["law"] = f"G.R. No. {num}"
        meta["law_type"] = "jurisprudence"
    elif m := RA_RE.search(head):
        meta["law"] = f"Republic Act No. {m.group(1)}"
        meta["law_type"] = "RA"
    elif m := PD_RE.search(head):
        meta["law"] = f"Presidential Decree No. {m.group(1)}"
        meta["law_type"] = "PD"
    elif m := BP_RE.search(head):
        meta["law"] = f"Batas Pambansa Blg. {m.group(1)}"
        meta["law_type"] = "BP"
    elif m := ACT_RE.search(head):
        meta["law"] = f"Act No. {m.group(1)}"
        meta["law_type"] = "ACT"
    elif m := RULE_RE.search(head):
        meta["law"] = f"Rule {m.group(1)} of the Rules of Court"
        meta["law_type"] = "RULE"
    elif m := AO_RE.search(head):
        meta["law"] = f"Administrative Order No. {m.group(1)}"
        meta["law_type"] = "AO"
    elif m := RESO_RE.search(head):
        meta["law"] = f"Resolution No. {m.group(1)}"
        meta["law_type"] = "RESO"
    elif m := CODE_RE.search(head):
        meta["law"] = " ".join(m.group(1).split())
        meta["law_type"] = "CODE"

    # First non-empty, reasonably long line is often the title.
    for line in text.splitlines():
        line = line.strip()
        if 15 <= len(line) <= 200:
            meta.setdefault("title", line)
            break

    return meta


def _current_section_label(unit: str) -> str | None:
    """Return the section header at the start of a unit, if any."""
    m = SECTION_RE.match(unit)
    if not m:
        return None
    # Grab the header line, trimmed to something citation-sized.
    line = unit.splitlines()[0].strip()
    return line[:80]


def _split_on_sections(text: str) -> list[str]:
    """Split text so each piece begins at a SECTION/ARTICLE/RULE boundary."""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [text]
    units: list[str] = []
    # Preamble before the first section (title, whereas-clauses, etc.).
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            units.append(pre)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit = text[start:end].strip()
        if unit:
            units.append(unit)
    return units


# A sentence end it is safe to break a chunk after. The negative lookbehinds
# stop a legal abbreviation's period ("Sec. 2", "No. 9439", "Art. 5") from
# posing as one — the same guard, for the same reason, as rag._BOUNDARY.
#
# `(?<![A-Z])` covers the initial-shaped abbreviations in one stroke — "R.A.",
# "G.R.", "P.D.", "B.P." — which the word-shaped lookbehinds cannot reach, since
# the character before that period is a lone capital. It also declines to break
# after a trailing acronym ("...issued by the DOH."), which is a real but cheap
# loss: the splitter simply takes the previous sentence end instead.
_SENTENCE_END = re.compile(
    r"(?<!\bNo)(?<!\bNos)(?<!\bSec)(?<!\bSecs)(?<!\bArt)(?<!\bArts)"
    r"(?<!\bpar)(?<!\bpars)(?<!\bInc)(?<!\bv)(?<![A-Z])"
    r"\.\s")

# Where to prefer breaking an oversized unit, best first. A paragraph break is
# worth more than a line break, and a line break more than a sentence end.
_PARA_BREAK = re.compile(r"\n\s*\n")
_LINE_BREAK = re.compile(r"\n")


def _split_point(unit: str, start: int, size: int) -> int:
    """Absolute, exclusive index at which to end the piece beginning at `start`.

    Prefers the last paragraph break before the size limit, then the last line
    break, then the last sentence end, then the last space — and only cuts at
    the raw offset when the text offers no break at all.

    Why this exists: `_split_on_sections` returns the WHOLE document as one unit
    whenever a file's headings do not match SECTION_RE, and DOH/PRC issuances
    ("I. Rationale", "B. Specific Guidelines", "1.", "2.") and Supreme Court
    decisions mostly do not. Those documents were therefore sliced at blind
    character offsets — 632 chunks across 19 files on this corpus, including all
    175 chunks of the landmark informed-consent case. The cuts landed mid-word
    ("…who refuse to execute a promisso", "Detention occ|urs when…"), which
    costs twice over: the fragment is unreadable if it reaches the prompt, and
    the chunk's embedding is diluted by whatever unrelated material the window
    happened to scoop up on either side. That dilution is what kept the
    operative sentence of DOH AO 2008-0001 out of the top_k while the elements
    list from the same file got in.

    The floor at half `size` keeps a break from producing a runt chunk; without
    it an early paragraph break would be preferred over a good later one."""
    hard = start + size
    if hard >= len(unit):
        return len(unit)
    floor = start + max(1, size // 2)      # never produce a piece under half size
    for pat in (_PARA_BREAK, _LINE_BREAK, _SENTENCE_END):
        last = None
        for m in pat.finditer(unit, floor, hard):
            last = m
        if last is not None:
            return last.end()
    cut = unit.rfind(" ", floor, hard)
    return cut + 1 if cut != -1 else hard


def _snap_start(unit: str, pos: int) -> int:
    """Move a piece's start forward to the next word boundary.

    The overlap rewind (`end - overlap`) is plain arithmetic and lands wherever
    it lands — which is mid-word about as often as not, reintroducing at the
    START of a chunk exactly the severed word `_split_point` removes from its
    end. Snapping FORWARD rather than back is deliberate: it can only increase
    `pos`, so the packing loop is still guaranteed to make progress."""
    if pos <= 0 or pos >= len(unit):
        return pos
    if unit[pos - 1].isspace() or unit[pos].isspace():
        return pos
    nxt = pos
    while nxt < len(unit) and not unit[nxt].isspace():
        nxt += 1
    while nxt < len(unit) and unit[nxt].isspace():
        nxt += 1
    return nxt


def _pack(units: list[str], size: int, overlap: int) -> list[tuple[str, str | None]]:
    """Pack section-units into ~size chunks, remembering each unit's section.

    Returns list of (chunk_text, section_label). Oversized units are split at
    the nearest real text boundary — see `_split_point`.
    """
    out: list[tuple[str, str | None]] = []
    buf = ""
    buf_label: str | None = None

    def flush():
        nonlocal buf, buf_label
        if buf.strip():
            out.append((buf.strip(), buf_label))
        buf = ""

    for unit in units:
        label = _current_section_label(unit)

        # A single unit larger than `size`: split it at text boundaries, with
        # overlap. `end` is chosen by _split_point rather than by arithmetic, so
        # the advance has to be derived from it — and forced to make progress,
        # since a break found at or before `start + overlap` would otherwise
        # rewind and loop forever.
        if len(unit) > size:
            flush()
            start = 0
            while start < len(unit):
                end = _split_point(unit, start, size)
                piece = unit[start:end].strip()
                if piece:
                    out.append((piece, label))
                if end >= len(unit):
                    break
                start = _snap_start(unit, max(end - overlap, start + 1))
            continue

        if len(buf) + len(unit) + 1 > size:
            flush()
        if not buf:
            buf_label = label
        buf += ("\n\n" if buf else "") + unit

    flush()
    return out


def chunk_document(text: str, filename: str) -> list[Chunk]:
    """Full pipeline: detect metadata, split on sections, pack into chunks."""
    doc_meta = detect_doc_metadata(text, filename)
    units = _split_on_sections(text)
    packed = _pack(units, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    chunks: list[Chunk] = []
    for idx, (chunk_text, section_label) in enumerate(packed):
        meta = dict(doc_meta)
        meta["chunk_index"] = idx
        if section_label:
            meta["section"] = section_label
        chunks.append(Chunk(text=chunk_text, metadata=meta))
    return chunks
