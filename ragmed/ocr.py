"""OCR fallback for scanned / image-only PDFs.

Many older PRC and DOH issuances are scanned documents with no text layer,
so pypdf/pdfplumber extract nothing. Here we render each page to a bitmap
with pypdfium2 (pure-Python, no Poppler/Ghostscript needed) and run Tesseract
OCR on it via pytesseract.

Only Tesseract itself is an external dependency — install the binary from
https://github.com/UB-Mannheim/tesseract/wiki (or `winget install
UB-Mannheim.TesseractOCR`). Everything else is pip-installed.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from . import config

# Common Windows install locations to auto-probe if tesseract isn't on PATH.
_WINDOWS_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


@lru_cache(maxsize=1)
def _configure_tesseract() -> bool:
    """Point pytesseract at the tesseract binary. Returns True if found."""
    try:
        import pytesseract
    except ImportError:
        return False

    import shutil

    # 1) Explicit override from config/env.
    if config.TESSERACT_CMD and Path(config.TESSERACT_CMD).exists():
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        return True
    # 2) On PATH.
    on_path = shutil.which("tesseract")
    if on_path:
        pytesseract.pytesseract.tesseract_cmd = on_path
        return True
    # 3) Known Windows locations.
    for cand in _WINDOWS_CANDIDATES:
        if cand and Path(cand).exists():
            pytesseract.pytesseract.tesseract_cmd = cand
            return True
    return False


def is_available() -> bool:
    """True if OCR can actually run (pytesseract + pypdfium2 + binary)."""
    if not _configure_tesseract():
        return False
    try:
        import pypdfium2  # noqa: F401
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def unavailable_reason() -> str:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return "pytesseract not installed (pip install pytesseract)"
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return "pypdfium2 not installed (pip install pypdfium2)"
    if not _configure_tesseract():
        return ("Tesseract binary not found. Install it (winget install "
                "UB-Mannheim.TesseractOCR) or set TESSERACT_CMD.")
    return "unknown OCR error"


def ocr_pdf(path: Path) -> str:
    """Render each page and OCR it. Returns text with \\f page separators."""
    import pypdfium2 as pdfium
    import pytesseract

    if not _configure_tesseract():
        return ""

    scale = config.OCR_DPI / 72.0  # pdfium scale is relative to 72 dpi
    pages_text: list[str] = []

    pdf = pdfium.PdfDocument(str(path))
    try:
        n = len(pdf)
        if config.OCR_MAX_PAGES > 0:
            n = min(n, config.OCR_MAX_PAGES)
        for i in range(n):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            text = pytesseract.image_to_string(
                pil_image, lang=config.OCR_LANGUAGE)
            pages_text.append(text)
    finally:
        pdf.close()

    return "\f".join(pages_text)
