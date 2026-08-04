"""Haven's chrome — the ops dashboard's visual language, taken down several stops.

`ui/hud.py` gives the ops console a heads-up display: dark ground, cyan accent,
scanlines, rotating rings. That is right for a screen read at a glance, from
across a room, to answer "is anything broken". It would be wrong here. Haven is
read slowly, up close, by someone who is worried — and the page carries a
disclaimer that has to be believed. A legal answer rendered in sci-fi chrome
reads as a toy, and the moment it does, the "verify this before relying on it"
line stops landing.

So this borrows the HUD's STRUCTURE and none of its voltage:

  kept       corner brackets, hairline rules that fade out, uppercase
             letterspaced micro-labels for metadata, tabular figures, a
             precise instrument-like layout for the source cards
  dropped    the dark ground, the high-chroma accent, scanlines, glow on text,
             anything that rotates, monospace body copy

Colours stay the law-library palette from `.streamlit/config.toml` — slate
indigo, brass, blue-black on cool pearl — and are passed in per theme rather
than hard-coded, because unlike the dashboard (its own app, forced dark) Haven
honours whichever mode the reader is in.

Same offline rule as the rest of `ui/`: inline CSS and SVG only, no webfont, no
remote asset.
"""
from __future__ import annotations

import html


def chrome_css(dark: bool) -> str:
    """Page-level chrome. `dark` selects the palette; both are first-class."""
    if dark:
        line = "#28313d"
        rule = "#323d4c"
        accent = "#8fa4d4"
        label = "#8b95a3"
        card = "rgba(26,32,41,.55)"
        card_line = "#2b3542"
    else:
        line = "#d5dae1"
        rule = "#c3cbd6"
        accent = "#38476b"
        label = "#616b7a"
        card = "rgba(255,255,255,.62)"
        card_line = "#d9dee5"

    return f"""
<style>
  /* --- micro-label: the one typographic tic carried over from the HUD.
     Uppercase, letterspaced, small. Used only for METADATA — never for the
     answer, never for the disclaimer. Those stay in plain sentence case,
     because a legal caveat set in tracked-out capitals reads as decoration
     and gets skipped. --- */
  .hv-label {{
    font-size: .625rem; letter-spacing: .19em; text-transform: uppercase;
    color: {label}; font-weight: 600; display: flex; align-items: center;
    gap: 9px; margin: 0 0 9px;
  }}
  .hv-label::after {{
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, {rule}, transparent);
  }}

  /* --- source card: the direct descendant of the dashboard's hud-panel.
     Same two-corner bracket construction, at a fraction of the contrast. --- */
  .hv-card {{
    position: relative; border: 1px solid {card_line}; background: {card};
    padding: 12px 14px 11px; margin: 0 0 9px;
  }}
  .hv-card::before, .hv-card::after {{
    content: ""; position: absolute; width: 9px; height: 9px;
    border-color: {accent}; border-style: solid; opacity: .5;
  }}
  .hv-card::before {{ top: -1px; left: -1px; border-width: 1.5px 0 0 1.5px; }}
  .hv-card::after  {{ bottom: -1px; right: -1px; border-width: 0 1.5px 1.5px 0; }}

  .hv-n {{
    font-size: .625rem; letter-spacing: .14em; color: {label};
    font-variant-numeric: tabular-nums;
  }}

  /* --- hero meta strip: the one place the machinery is stated on the front
     page. Facts only — how much law is indexed, and that it never leaves the
     machine. Both are things a worried reader has an actual reason to want,
     which is what keeps this from being chrome for its own sake. --- */
  .hv-meta {{
    display: flex; justify-content: center; flex-wrap: wrap;
    gap: 0; margin: 14px 0 4px;
  }}
  .hv-meta span {{
    font-size: .62rem; letter-spacing: .17em; text-transform: uppercase;
    color: {label}; padding: 0 14px; border-right: 1px solid {line};
    font-variant-numeric: tabular-nums;
  }}
  .hv-meta span:last-child {{ border-right: none; }}
  .hv-meta b {{ color: {accent}; font-weight: 600; }}

  /* --- the four opening situations, as bracketed cards.
     Numbered with a CSS counter rather than by baking "01" into each label:
     the number is presentation, and putting it in the button text would send
     it to screen readers as part of the question. --- */
  .st-key-hv-situations {{ counter-reset: hv-sit; }}
  .st-key-hv-situations .stButton {{
    counter-increment: hv-sit; position: relative;
  }}
  .st-key-hv-situations .stButton::before {{
    content: counter(hv-sit, decimal-leading-zero);
    position: absolute; left: 15px; top: 50%; transform: translateY(-50%);
    font-size: .62rem; letter-spacing: .12em; color: {label};
    font-variant-numeric: tabular-nums; pointer-events: none; z-index: 1;
  }}
  .st-key-hv-situations .stButton > button {{
    position: relative; text-align: left; justify-content: flex-start;
    padding: 13px 15px 13px 46px; min-height: 0; height: auto;
    white-space: normal; line-height: 1.45;
    border: 1px solid {card_line}; background: {card}; border-radius: 0;
    transition: border-color .16s, background .16s;
  }}
  .st-key-hv-situations .stButton > button:hover {{
    border-color: {accent}; background: {card};
  }}
  .st-key-hv-situations .stButton > button::before,
  .st-key-hv-situations .stButton > button::after {{
    content: ""; position: absolute; width: 8px; height: 8px;
    border-color: {accent}; border-style: solid; opacity: .45;
  }}
  .st-key-hv-situations .stButton > button::before {{
    top: -1px; left: -1px; border-width: 1.5px 0 0 1.5px;
  }}
  .st-key-hv-situations .stButton > button::after {{
    bottom: -1px; right: -1px; border-width: 0 1.5px 1.5px 0;
  }}

  /* --- answer frame: a single hairline down the left of every reply.
     Deliberately one rule and not a full bracketed box — the answer is the
     thing being read, and boxing it competes with the chat bubble already
     around it. --- */
  [class*="st-key-hv-answer"] {{
    border-left: 2px solid {accent}; padding-left: 15px;
  }}
  .hv-law {{
    font-weight: 600; font-size: .95rem; line-height: 1.35; margin: 2px 0 0;
  }}
  .hv-sec {{ font-size: .86rem; opacity: .82; margin: 4px 0 0; }}
  .hv-file {{
    font-size: .68rem; opacity: .58; margin: 7px 0 0;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    word-break: break-all;
  }}
</style>
"""


def hero_meta(passages: int, local: bool = True) -> str:
    """The one-line statement of what is behind the answers.

    Kept to facts a reader has a reason to want — how much law is indexed, and
    that it stays on this machine. Model names and chunk settings belong in
    "How this works" in the sidebar, not on the front page (see this app's
    module docstring)."""
    bits = [f"<span><b>{passages:,}</b> passages indexed</span>",
            "<span>Philippine medical law</span>"]
    if local:
        bits.append("<span>Runs <b>on this computer</b></span>")
    return f'<div class="hv-meta">{"".join(bits)}</div>'


def label(text: str) -> str:
    """A metadata micro-label with a rule trailing off to the right."""
    return f'<div class="hv-label">{html.escape(text)}</div>'


def source_card(n: int, law: str, section: str, source: str) -> str:
    """One retrieved passage, as an instrument readout rather than a list item.

    Everything is escaped: `law`, `section` and `source` come from document
    metadata built during ingestion, which is derived from fetched files —
    not content this app authored, and therefore not content it should trust
    into raw HTML.
    """
    sec = (f'<div class="hv-sec">{html.escape(section)}</div>'
           if section else "")
    return (
        f'<div class="hv-card">'
        f'<div class="hv-n">SOURCE {n:02d}</div>'
        f'<div class="hv-law">{html.escape(law or "Untitled document")}</div>'
        f'{sec}'
        f'<div class="hv-file">{html.escape(source)}</div>'
        f'</div>'
    )
