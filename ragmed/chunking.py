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


def _pack(units: list[str], size: int, overlap: int) -> list[tuple[str, str | None]]:
    """Pack section-units into ~size chunks, remembering each unit's section.

    Returns list of (chunk_text, section_label). Oversized units are hard-split.
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

        # A single unit larger than `size`: split it with char overlap.
        if len(unit) > size:
            flush()
            start = 0
            while start < len(unit):
                piece = unit[start : start + size]
                out.append((piece.strip(), label))
                start += size - overlap
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
