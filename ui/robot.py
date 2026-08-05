"""Haven's mark: an armoured assistant, idling or working.

Replaces `ui/hero.py` (the scales settling into balance) and `ui/thinking.py`
(the same scales, still weighing) — one figure now covers both states, which is
the point: the hero and the waiting mark are the SAME robot in two modes, so the
transition from "sitting here" to "reading the law" reads as the thing you are
looking at starting work rather than as one graphic being swapped for another.

Two modes, one drawing:

  idle     head tilts slowly, eyes hold a soft pulse, the chest reactor turns,
           the antenna tips blink out of step. Nothing is urgent.
  working  reactor spins up, eyes pulse hard and fast, and a scan line sweeps
           down the visor, clipped to the helmet so it reads as the faceplate's
           own display rather than a bar crossing the page.

CONSTRUCTION — read `ui/svg.py` before editing. This leaves as a data-URI
`<img>`, never a bare `<svg>`: Streamlit's sanitiser deletes the SVG namespace
outright, and this project shipped that bug three times. The consequence for
this file is that the document is SELF-CONTAINED — page CSS cannot reach inside
it, so every colour, every rule and every `@keyframes` is written into the
`<style>` element below. Media queries still work in there, which is why
`prefers-reduced-motion` is still honoured against the reader's real setting.

The palette is imported from `ui/hud.py` rather than restated. Haven and the ops
dashboard are now the same console register on purpose (see `ui/chrome.py`), and
two copies of a palette that must agree is how they would silently drift apart.

Colour never carries meaning here. The robot is decoration and a waiting signal;
the phase caption under it is real page text, and every state this mark is in is
also stated in words nearby.
"""
from __future__ import annotations

import html

from .hud import ALERT, CYAN, CYAN_DIM, CYAN_GHOST, LINE, MONO, TEXT_DIM
from .svg import svg_img

# The helmet outline, reused three times: as the drawn faceplate, as the clip
# region for the visor scan, and as the faint aura behind the head. Stated once
# because a clip path that has drifted from the shape it clips produces a sweep
# that leaks past the jaw, which looks like a rendering fault.
_HELMET = ("M120 24 L152 38 L157 68 L150 95 L133 110 L107 110 "
           "L90 95 L83 68 L88 38 Z")


def _robot_svg(working: bool, glow: str = CYAN) -> str:
    """The figure. `working` picks the mode; `glow` recolours the live parts.

    `glow` exists for the one case where the mark has to say something is wrong
    (the model unreachable), and it is never the ONLY carrier of that — the page
    states it in a sentence directly above.
    """
    # Two timings, one drawing. The idle figure moves slowly enough to be
    # ignorable while someone reads; the working figure is faster because it is
    # the only honest signal available during a wait.
    tilt = "9s" if not working else "3.4s"
    spin = "26s" if not working else "6s"
    spin_rev = "17s" if not working else "4.2s"
    eye = "4.2s" if not working else "1.15s"
    pulse = "3.6s" if not working else "1.4s"

    scan = ""
    if working:
        # Clipped to the faceplate so it reads as the helmet's own display.
        scan = ('<g clip-path="url(#visor)">'
                '<rect class="scan" x="78" y="-34" width="84" height="34" '
                'fill="url(#sweep)"/></g>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 192"
     role="img" aria-label="Haven">
  <style>
    .bob {{ animation: bob 7s ease-in-out infinite; }}
    .head {{ transform-origin: 120px 72px; animation: tilt {tilt} ease-in-out infinite; }}
    .eye {{ animation: eye {eye} ease-in-out infinite; }}
    .tip {{ animation: blink 2.8s steps(1, end) infinite; }}
    .tip2 {{ animation-delay: 1.4s; }}
    .ring-o {{ transform-origin: 120px 151px; animation: spin {spin} linear infinite; }}
    .ring-m {{ transform-origin: 120px 151px; animation: spinr {spin_rev} linear infinite; }}
    .core {{ animation: core {pulse} ease-in-out infinite; }}
    .scan {{ animation: sweep 1.9s linear infinite; }}
    @keyframes bob {{ 0%,100% {{ transform: translateY(0); }}
                      50% {{ transform: translateY(-3px); }} }}
    @keyframes tilt {{ 0%,100% {{ transform: rotate(-2.2deg); }}
                       50% {{ transform: rotate(2.2deg); }} }}
    @keyframes eye {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .42; }} }}
    @keyframes blink {{ 0%,55% {{ opacity: 1; }} 56%,100% {{ opacity: .12; }} }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes spinr {{ to {{ transform: rotate(-360deg); }} }}
    @keyframes core {{ 0%,100% {{ opacity: .30; }} 50% {{ opacity: .10; }} }}
    @keyframes sweep {{ 0% {{ transform: translateY(0); }}
                        100% {{ transform: translateY(148px); }} }}
    /* Motion is a nicety; legibility is not. Everything below resolves to the
       drawing's resting state, and the scan — which has no resting state
       because it is a moving highlight — is removed rather than frozen
       mid-visor. */
    @media (prefers-reduced-motion: reduce) {{
      .bob, .head, .eye, .tip, .ring-o, .ring-m, .core {{ animation: none; }}
      .scan {{ display: none; }}
    }}
  </style>
  <defs>
    <filter id="gl" x="-70%" y="-70%" width="240%" height="240%">
      <feGaussianBlur stdDeviation="3.1" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <clipPath id="visor"><path d="{_HELMET}"/></clipPath>
    <linearGradient id="sweep" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{glow}" stop-opacity="0"/>
      <stop offset="0.5" stop-color="{glow}" stop-opacity="0.42"/>
      <stop offset="1" stop-color="{glow}" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <g class="bob" fill="none" stroke-linecap="round" stroke-linejoin="round">

    <!-- torso and pauldrons: drawn under the head so the neck tucks behind -->
    <g stroke="{CYAN_DIM}" stroke-width="2.2">
      <path d="M108 108 v14 M132 108 v14"/>
      <path d="M78 176 L84 143 Q92 129 108 122 L132 122 Q148 129 156 143 L162 176"/>
      <path d="M84 143 Q60 147 53 176 L78 176"/>
      <path d="M156 143 Q180 147 187 176 L162 176"/>
      <path d="M53 176 L187 176" stroke="{LINE}"/>
    </g>
    <!-- Chest seams sit OUTSIDE the reactor's outer ring on purpose. Drawn at
         the plate centres they framed it in a box, and the gauge stopped
         reading as the thing set into the chest. -->
    <g stroke="{CYAN_GHOST}" stroke-width="1.5">
      <path d="M90 156 L90 172 M150 156 L150 172"/>
      <path d="M64 160 h8 M168 160 h8"/>
    </g>

    <!-- chest reactor -->
    <g class="ring-o">
      <circle cx="120" cy="151" r="23" fill="none" stroke="{CYAN_DIM}"
              stroke-width="1.3" stroke-dasharray="3 7" opacity="0.85"/>
    </g>
    <g class="ring-m">
      <circle cx="120" cy="151" r="17.5" fill="none" stroke="{glow}"
              stroke-width="1.6" stroke-dasharray="13 9" opacity="0.8"/>
    </g>
    <circle class="core" cx="120" cy="151" r="12" fill="{glow}" opacity="0.22"/>
    <circle cx="120" cy="151" r="7" fill="{glow}" filter="url(#gl)"/>

    <!-- head. The faceplate detail is CYAN_DIM, not CYAN_GHOST: the ghost tone
         is a hairline colour (1.5:1) and at this size the brow, ridge and
         grille rendered as smudges rather than as plating. -->
    <g class="head">
      <path d="{_HELMET}" fill="none" stroke="{CYAN}" stroke-width="2.4"/>
      <path d="M88 51 L106 56 L134 56 L152 51" fill="none"
            stroke="{CYAN_DIM}" stroke-width="1.8"/>
      <path d="M120 61 V79" fill="none" stroke="{CYAN_DIM}" stroke-width="1.5"/>
      <g stroke="{CYAN_DIM}" stroke-width="1.5" fill="none" opacity="0.85">
        <path d="M105 91 h30"/><path d="M107 98 h26"/><path d="M111 104 h18"/>
      </g>
      <g class="eye" fill="{glow}" filter="url(#gl)">
        <path d="M97 66 L114 61 L114 73 L97 74 Z"/>
        <path d="M143 66 L126 61 L126 73 L143 74 Z"/>
      </g>
      <g stroke="{CYAN_DIM}" stroke-width="1.8" fill="none">
        <path d="M94 33 L88 16"/><path d="M146 33 L152 16"/>
      </g>
      <circle class="tip" cx="88" cy="14" r="3" fill="{glow}" filter="url(#gl)"/>
      <circle class="tip tip2" cx="152" cy="14" r="3" fill="{glow}"
              filter="url(#gl)"/>
      {scan}
    </g>
  </g>
</svg>"""


def robot_hero(alert: bool = False) -> str:
    """The figure at rest — the largest thing on an empty page.

    `alert` recolours it for the model-unreachable state. That is deliberately
    the only thing that changes its colour, and the page says so in words
    immediately above it: nothing here may be the sole carrier of a state.
    """
    glow = ALERT if alert else CYAN
    return ('<div style="display:flex;justify-content:center;padding:6px 0 2px">'
            + svg_img(_robot_svg(False, glow), "212px", "Haven")
            + '</div>')


def robot_thinking(caption: str) -> str:
    """The figure working, over the phase the pipeline is actually in.

    `caption` is escaped: a phase line can carry a retrieved document's title,
    which is corpus text and not something this app authored.

    The caption stays OUTSIDE the image, as real page text — an `<img>` carries
    no selectable text and no hover, so anything a reader needs has to be direct
    page content or drawn into the graphic (see `ui/svg.py`).
    """
    label = html.escape(caption)
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;
            gap:2px;padding:8px 0 14px" role="status" aria-live="polite">
  {svg_img(_robot_svg(True), "132px", "")}
  <p style="margin:8px 0 0;font-family:{MONO};font-size:.72rem;
            letter-spacing:.16em;text-transform:uppercase;color:{TEXT_DIM};
            text-align:center">{label}</p>
</div>
"""
