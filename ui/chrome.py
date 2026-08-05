"""Haven's chrome — a heads-up display, the same register as the ops console.

WHAT CHANGED, AND WHAT DID NOT. This module used to be the deliberate opposite
of `ui/hud.py`: neutral near-black, a single brass accent, soft surfaces, no
console cues, on the reasoning that a legal answer rendered in sci-fi chrome
reads as a toy and stops its disclaimer landing. That split was removed by
request — Haven is now the console: cyan on near-black, scanlines, corner
brackets, glow, monospace, and an armoured mark that idles and works
(`ui/robot.py`).

The reasoning behind the old split has not been thrown away, because the risk it
named is real and is now unmanaged by the palette. It is handled here instead,
in three places, and these are the parts of this file NOT to trade away for
atmosphere:

  * `advisory()` renders the not-legal-advice line as an alert-weight panel —
    the heaviest thing on the page, above the ask box, in the alert hue rather
    than the ambient cyan. On a console, ambient chrome is what the eye learns
    to skip; a caveat has to sit outside it.
  * the caveat SENTENCE is never set in tracked-out capitals. Uppercase
    letterspacing is used for micro-labels and headings only. A legal caveat set
    as a HUD label reads as decoration and gets skipped, which is the failure
    this whole module exists downstream of.
  * `source_card()` still names the law the way the law names itself, and still
    refuses a filename. Chrome changed; what a reader can carry to a lawyer
    did not.

The palette is IMPORTED from `ui/hud.py`, not restated. The two surfaces are one
register now, and a second copy of a palette that has to agree is exactly how
they would drift back apart by accident.

Same offline rule as the rest of `ui/`: inline CSS and SVG only, no webfont, no
remote asset.
"""
from __future__ import annotations

import html

from .hud import (ALERT, CYAN, CYAN_DIM, CYAN_GHOST, INK_2, LINE, MONO, TEXT,
                  TEXT_DIM, hud_css, nucleus)

# --- palette ---------------------------------------------------------------
# Aliases, so callers on this page read in Haven's terms rather than the
# dashboard's. They are the SAME values by design — see the module docstring.
ACCENT = CYAN
GROUND_2 = INK_2
LABEL = TEXT_DIM


def chrome_css() -> str:
    """Page chrome: the shared console base, then Haven's own components.

    Built on `hud_css()` rather than beside it. Everything structural — the
    ground, the scanline overlay, monospace throughout, the uppercase headings,
    the alert and button treatments — is the console's, stated once. What
    follows is only what Haven has and the dashboard does not: a conversation,
    example situations, source cards and an advisory panel.
    """
    return hud_css() + f"""
<style>
  /* --- chat: each turn is a HUD panel ---------------------------------- */
  [data-testid="stChatMessage"] {{
    background: linear-gradient(180deg, rgba(20,52,68,.30), rgba(10,17,25,.18));
    border: 1px solid {LINE}; border-radius: 0;
    padding: 14px 16px; margin-bottom: 12px; position: relative;
  }}
  [data-testid="stChatMessage"]::before,
  [data-testid="stChatMessage"]::after {{
    content: ""; position: absolute; width: 12px; height: 12px;
    border-color: {CYAN}; border-style: solid; opacity: .8;
  }}
  [data-testid="stChatMessage"]::before {{
    top: -1px; left: -1px; border-width: 2px 0 0 2px;
  }}
  [data-testid="stChatMessage"]::after {{
    bottom: -1px; right: -1px; border-width: 0 2px 2px 0;
  }}
  /* Body copy inside a turn is the one place letterspacing is NOT applied:
     these are paragraphs of law, read line by line, not glanced at. */
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li {{
    letter-spacing: 0; line-height: 1.62;
  }}

  /* Avatars. Streamlit fills the assistant's with the theme's ORANGE, which on
     this page is the amber warning hue — so every answer opened with a caution
     block beside it. Square, ink-filled, cyan glyph instead. Only the
     background and colour are touched: the font-family belongs to the icon
     ligature and is restored in hud_css(), and setting one here would print
     `smart_toy` as text again. */
  [data-testid^="stChatMessageAvatar"] {{
    background: {INK_2} !important; border: 1px solid {LINE};
    border-radius: 0; color: {CYAN} !important;
    box-shadow: 0 0 14px rgba(79,216,255,.16);
  }}
  [data-testid="stChatMessageAvatarUser"] {{
    color: {TEXT_DIM} !important; box-shadow: none;
  }}

  /* The answer carries a live rail down its left edge — the console's
     equivalent of the hairline the old chrome used. */
  [class*="st-key-hv-answer"] {{
    border-left: 2px solid {CYAN}; padding-left: 15px;
    box-shadow: -7px 0 18px -12px {CYAN};
  }}

  /* --- micro-label. METADATA AND HEADINGS ONLY — never the caveat. ----- */
  .hv-label {{
    font-family: {MONO}; font-size: .64rem; letter-spacing: .24em;
    text-transform: uppercase; color: {CYAN}; font-weight: 600;
    display: flex; align-items: center; gap: 9px; margin: 4px 0 11px;
  }}
  .hv-label::after {{
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, {CYAN_GHOST}, transparent);
  }}

  /* --- source card: a readout line with the law as its value ----------- */
  .hv-card {{
    position: relative; background: {INK_2};
    border: 1px solid {LINE}; border-left: 2px solid {CYAN};
    padding: 11px 14px 12px; margin: 0 0 8px;
  }}
  .hv-card::after {{
    content: ""; position: absolute; bottom: -1px; right: -1px;
    width: 10px; height: 10px; border: solid {CYAN_DIM};
    border-width: 0 2px 2px 0; opacity: .8;
  }}
  .hv-n {{
    font-family: {MONO}; font-size: .62rem; letter-spacing: .18em;
    color: {CYAN}; font-variant-numeric: tabular-nums; font-weight: 600;
    text-shadow: 0 0 12px rgba(79,216,255,.45);
  }}
  .hv-law {{
    font-family: {MONO}; font-weight: 600; font-size: .92rem; line-height: 1.4;
    margin: 5px 0 0; color: {TEXT}; letter-spacing: .02em;
  }}
  .hv-sec {{
    font-family: {MONO}; font-size: .8rem; color: {TEXT_DIM};
    margin: 4px 0 0; line-height: 1.45;
  }}

  /* --- advisory: the not-legal-advice panel.
     Deliberately NOT the ambient cyan. On a console the accent is the colour of
     everything that is merely working, and the eye stops reading it within
     seconds; this is the one thing on the page that must survive that. Alert
     hue, heavier rule, its own ground. The SENTENCE stays sentence case — see
     the module docstring for why the label above it may be capitals and the
     caveat itself may not. --- */
  .hv-advisory {{
    position: relative; border: 1px solid {ALERT}; border-left-width: 3px;
    background: linear-gradient(90deg, rgba(255,95,109,.10), transparent 78%),
                {INK_2};
    padding: 13px 16px 14px; margin: 4px 0 16px;
  }}
  .hv-advisory .hv-adv-label {{
    font-family: {MONO}; font-size: .64rem; letter-spacing: .26em;
    text-transform: uppercase; color: {ALERT}; font-weight: 600;
    margin: 0 0 7px;
  }}
  .hv-advisory p {{
    margin: 0; color: {TEXT}; font-size: .92rem; line-height: 1.6;
    letter-spacing: 0;
  }}
  .hv-advisory b {{ color: {ALERT}; }}

  /* --- hero meta strip -------------------------------------------------- */
  .hv-meta {{
    display: flex; justify-content: center; flex-wrap: wrap;
    gap: 0; margin: 14px 0 18px;
  }}
  .hv-meta span {{
    font-family: {MONO}; font-size: .62rem; letter-spacing: .2em;
    text-transform: uppercase; color: {TEXT_DIM}; padding: 0 15px;
    border-right: 1px solid {LINE};
  }}
  .hv-meta span:last-child {{ border-right: none; }}
  .hv-meta b {{ color: {CYAN}; font-weight: 600; }}

  /* --- the opening situations.
     Numbered with a CSS counter rather than by baking "01" into each label:
     the number is presentation, and putting it in the button text would send it
     to screen readers as part of the question. --- */
  .st-key-hv-situations {{ counter-reset: hv-sit; }}
  .st-key-hv-situations .stButton {{
    counter-increment: hv-sit; position: relative;
  }}
  .st-key-hv-situations .stButton::before {{
    content: counter(hv-sit, decimal-leading-zero);
    position: absolute; left: 15px; top: 50%; transform: translateY(-50%);
    font-family: {MONO}; font-size: .62rem; letter-spacing: .14em;
    color: {CYAN_DIM}; font-variant-numeric: tabular-nums;
    pointer-events: none; z-index: 1; transition: color .18s ease;
  }}
  /* These override the console's button rule on one point only: a full
     question is not a control label, so it keeps sentence case and normal
     tracking. Everything else — square corners, cyan frame, glow on hover —
     is inherited. */
  .st-key-hv-situations .stButton > button {{
    position: relative; text-align: left; justify-content: flex-start;
    padding: 14px 16px 14px 48px; min-height: 0; height: auto;
    white-space: normal; line-height: 1.55; text-transform: none;
    letter-spacing: .01em; font-size: .86rem; color: {TEXT};
    border: 1px solid {LINE}; background: {INK_2};
  }}
  .st-key-hv-situations .stButton > button:hover {{
    border-color: {CYAN}; background: rgba(79,216,255,.08);
    box-shadow: 0 0 18px rgba(79,216,255,.22); color: {TEXT};
  }}
  .st-key-hv-situations .stButton:hover::before {{ color: {CYAN}; }}

  /* --- ask box ---------------------------------------------------------- */
  [data-testid="stChatInput"] {{
    border: 1px solid {CYAN_DIM}; border-radius: 0; background: {INK_2};
  }}
  [data-testid="stChatInput"] textarea {{
    font-family: {MONO}; letter-spacing: .02em;
  }}

  /* --- expander (the sources) ------------------------------------------- */
  [data-testid="stExpander"] details {{
    border: 1px solid {LINE}; border-radius: 0; background: transparent;
  }}
  [data-testid="stExpander"] summary {{
    font-family: {MONO}; text-transform: uppercase; letter-spacing: .16em;
    font-size: .7rem; color: {CYAN};
  }}
</style>
"""


def hero_meta() -> str:
    """What is behind the answers, in words rather than numbers.

    Takes no arguments. It used to lead with the indexed passage count, which is
    a number the reader cannot act on and cannot check: "8,600 passages" tells a
    worried person nothing about whether their situation is covered, and invites
    them to read size as authority. On a console that pull is stronger, not
    weaker — a readout implies the number was worth reading out.

    It also carried a "PRIVATE BY DESIGN" chip, removed for a sharper reason: it
    is a claim about the DEPLOYMENT, not about the law, and it is only true while
    Haven runs on the reader's own machine. Host this for real users and the chip
    keeps asserting it, silently. A claim a change of hosting can falsify does
    not belong in permanent chrome.
    """
    return ('<div class="hv-meta">'
            '<span>Philippine <b>medical law</b></span>'
            '<span>Statutes · regulations · court rulings</span>'
            '</div>')


def label(text: str) -> str:
    """A metadata micro-label with a rule trailing off to the right."""
    return f'<div class="hv-label">{html.escape(text)}</div>'


def booting() -> str:
    """What fills the page while the search models load.

    The first request into a fresh process pays about six seconds of embedder
    and reranker on CPU (see `app.py::warm_up`), and until now that was six
    seconds of blank page — which reads as broken rather than as busy, and is
    the worst possible first impression for a page someone arrived at worried.

    The nucleus is `hud.nucleus()`, the same drawing the ops console uses for
    its verdict, borrowed here purely as a mark for "working". It is imported
    rather than reimplemented; see that function.

    Deliberately says nothing about WHAT is loading. "Starting up" is the whole
    of what a reader can act on, and naming the machinery here would put back
    exactly what the rest of this page strips out.
    """
    return ('<div style="display:flex;flex-direction:column;align-items:center;'
            'padding:34px 0 10px" role="status" aria-live="polite">'
            + nucleus(CYAN, "Starting up")
            + f'<p style="margin:10px 0 0;font-family:{MONO};font-size:.78rem;'
              f'color:{TEXT_DIM};text-align:center;max-width:34ch;'
              f'line-height:1.6">This takes a few seconds the first time '
              f'someone opens the page.</p></div>')


def advisory(body_html: str, label_text: str = "Advisory") -> str:
    """The not-legal-advice panel.

    `body_html` is trusted markup written by `app.py` — a fixed sentence with a
    `<b>` in it, not user or corpus content. `label_text` is escaped anyway; it
    costs nothing and this function should stay safe if it is ever called with
    something dynamic.

    The label is capitals; the body is not. That asymmetry is the point of the
    function: the console voice announces the panel, and then gets out of the
    way of the sentence a reader actually has to believe.
    """
    return (f'<div class="hv-advisory">'
            f'<div class="hv-adv-label">{html.escape(label_text)}</div>'
            f'<p>{body_html}</p></div>')


def source_card(n: int, law: str, section: str) -> str:
    """One passage the answer was drawn from, named the way the law names itself.

    The originating FILENAME used to be printed under each card in monospace
    (`_fetched/doh/doh-ao-2008-0001-irr-of-ra-9439-…txt`). It is meaningful when
    tuning the corpus and noise to everyone else — a reader cannot open it and
    cannot verify anything with it. The whole page is monospace now, which makes
    a path look even more like it belongs; it still does not.

    Everything is escaped: `law` and `section` come from document metadata built
    during ingestion, derived from fetched files — not content this app authored,
    and therefore not content it should trust into raw HTML.
    """
    sec = (f'<div class="hv-sec">{html.escape(section)}</div>'
           if section else "")
    return (
        f'<div class="hv-card">'
        f'<div class="hv-n">{n:02d}</div>'
        f'<div class="hv-law">{html.escape(law or "Untitled document")}</div>'
        f'{sec}'
        f'</div>'
    )
