"""Turn source files into plain text.

Supports PDF, DOCX, HTML and TXT. Each loader returns a single cleaned
string; page/section structure is preserved as best-effort by inserting
form-feed (\\f) markers between pages so the chunker can reason about them.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def _clean(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def load_pdf(path: Path) -> str:
    """Extract text from a PDF. Tries pypdf first, falls back to pdfplumber."""
    pages: list[str] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception:
        pages = []

    # If pypdf produced almost nothing (scanned/complex layout), try pdfplumber.
    if sum(len(p.strip()) for p in pages) < 40:
        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
        except Exception:
            pass

    text = _clean("\f".join(pages))

    # Still (almost) empty -> likely a scanned image PDF. Fall back to OCR.
    if len(text) < config.OCR_MIN_CHARS and config.OCR_ENABLED:
        from . import ocr

        if ocr.is_available():
            ocr_text = _clean(ocr.ocr_pdf(path))
            if len(ocr_text) > len(text):
                return ocr_text

    return text


def load_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    return _clean("\n".join(parts))


def load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return _clean(soup.get_text("\n"))


def load_text(path: Path) -> str:
    return _clean(path.read_text(encoding="utf-8", errors="ignore"))


def load_file(path: Path) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    if ext in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def iter_corpus_files(root: Path):
    """Yield every supported file under `root`, recursively.

    Skips a raw .html/.htm file when a sibling .txt with the same stem exists
    — the fetchers save both a cleaned .txt and the raw .html archive, and we
    only want to index the clean text once.

    Also skips README/NOTES files: those describe how to USE the corpus
    directory and are not source documents. Indexed, they surface as
    uncitable chunks that can only mislead an answer.
    """
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        if path.stem.lower() in {"readme", "notes", "index"}:
            continue
        if path.suffix.lower() in {".html", ".htm"}:
            if path.with_suffix(".txt").exists():
                continue
        yield path
