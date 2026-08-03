"""Shared helpers for the source fetchers.

Design goals: be polite (identifiable User-Agent, rate-limited), be
resumable (skip files already downloaded), and save both the raw HTML and a
cleaned .txt so the ingester can pick it up from corpus/_fetched.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

# corpus/_fetched
FETCHED_DIR = Path(__file__).resolve().parent.parent / "corpus" / "_fetched"

HEADERS = {
    "User-Agent": (
        "ph-medlaw-rag/0.1 (personal legal-research indexing; "
        "contact: local user)"
    )
}

# Some government hosts (WordPress + security plugins) reject non-browser
# User-Agents with a 403/406. For public documents we fall back to a
# browser-style header set on those responses.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": ("text/html,application/xhtml+xml,application/pdf,"
               "application/xml;q=0.9,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Be a good citizen — official sites are public-service infrastructure.
REQUEST_DELAY_SECONDS = 2.0


def slugify(text: str, maxlen: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:maxlen].strip("-") or "document"


def _is_gov_ph(url: str) -> bool:
    """True for *.gov.ph hosts — where we tolerate a broken TLS chain."""
    host = (urlparse(url).hostname or "").lower()
    return host.endswith(".gov.ph") or host == "gov.ph"


def _get_with_retry(url: str, timeout: int, retries: int = 3):
    """GET with retry + backoff. Government sites drop connections, have
    flaky DNS, filter User-Agents, and often ship a broken TLS certificate
    chain — none of which should abort a run for a public document.

    The UA and TLS fallbacks do NOT consume the retry budget. They used to:
    a host needing both left only one real attempt, so a single flaky
    connection after the switches failed the URL outright. That silently drops
    a source from the corpus — which is far worse than a slow fetch, because
    nothing downstream reports the missing document."""
    last_exc = None
    headers = HEADERS
    verify = True
    attempt = 0          # genuine retries (transient failures) only
    ua_switched = False  # each fallback may fire once, so this cannot loop
    tls_relaxed = False
    while attempt < retries:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout,
                                verify=verify)
            # UA-based blocking: switch to a browser header set and retry.
            if resp.status_code in (403, 406) and not ua_switched:
                headers = BROWSER_HEADERS
                ua_switched = True
                continue  # a fallback switch, NOT a retry — see below
            resp.raise_for_status()
            return resp
        except requests.exceptions.SSLError as e:
            last_exc = e
            # Broken cert chain on a government host: retry without
            # verification (public legal text, low tampering risk).
            if not tls_relaxed and _is_gov_ph(url):
                urllib3.disable_warnings(
                    urllib3.exceptions.InsecureRequestWarning)
                verify = False
                tls_relaxed = True
                continue  # a fallback switch, NOT a retry
            attempt += 1
            if attempt < retries:
                time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            last_exc = e
            attempt += 1
            if attempt < retries:
                time.sleep(2 * attempt)  # 2s, 4s backoff
    raise last_exc


def fetch_html(url: str, timeout: int = 30) -> str:
    resp = _get_with_retry(url, timeout)
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def fetch_bytes(url: str, timeout: int = 60) -> tuple[bytes, str]:
    """GET a URL, returning (content_bytes, content_type)."""
    resp = _get_with_retry(url, timeout)
    ctype = resp.headers.get("Content-Type", "").lower()
    return resp.content, ctype


def _looks_like_pdf(url: str, ctype: str, content: bytes) -> bool:
    return (
        url.lower().split("?")[0].endswith(".pdf")
        or "application/pdf" in ctype
        or content[:5] == b"%PDF-"
    )


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def save_document(name: str, html: str, subdir: str = "") -> Path:
    """Save cleaned text (and raw HTML) under corpus/_fetched/<subdir>/."""
    out_dir = FETCHED_DIR / subdir if subdir else FETCHED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base = slugify(name)
    txt_path = out_dir / f"{base}.txt"
    html_path = out_dir / f"{base}.html"

    txt_path.write_text(html_to_text(html), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    return txt_path


def save_pdf(name: str, content: bytes, subdir: str = "") -> Path:
    """Save a raw PDF under corpus/_fetched/<subdir>/ for the ingester."""
    out_dir = FETCHED_DIR / subdir if subdir else FETCHED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{slugify(name)}.pdf"
    pdf_path.write_bytes(content)
    return pdf_path


def download(name: str, url: str, subdir: str = "") -> Path:
    """Fetch a URL and save it in the right format (PDF stays PDF, HTML -> txt).

    Returns the path the ingester will pick up.
    """
    content, ctype = fetch_bytes(url)
    if _looks_like_pdf(url, ctype, content):
        return save_pdf(name, content, subdir)
    # Treat as HTML/text.
    html = content.decode("utf-8", errors="ignore")
    return save_document(name, html, subdir)


def already_fetched(name: str, subdir: str = "") -> bool:
    out_dir = FETCHED_DIR / subdir if subdir else FETCHED_DIR
    base = slugify(name)
    return (out_dir / f"{base}.txt").exists() or (out_dir / f"{base}.pdf").exists()


def polite_sleep():
    time.sleep(REQUEST_DELAY_SECONDS)
