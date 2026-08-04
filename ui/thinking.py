"""The waiting mark: the scales, still weighing.

Why this exists at all. Sources used to be rendered ABOVE the answer, on the
reasoning that the model spent about a minute reading before it wrote a word,
so the passages were the only real thing to put on screen during the wait. That
was true when it was written. It is not any more — retrieval finishes in 2-8s
and the first token follows about 2s later, so the wait is now short enough to
be *held* rather than filled, and the answer can take the position a reader
expects it in, with the sources under it.

Something still has to occupy those few seconds, and a bare spinner says only
"busy". This says what the system is actually doing, in the same figure the
page already opens with: the scales weighing. The caption changes with the
phase, so the motion is not decorative — it is the one honest signal available
while a local model works.

Shares the hero's construction (inline SVG, theme-coloured strokes, no remote
asset — nothing leaves the machine) but keeps its own `ph-think-` class prefix:
the hero mark can be on screen at the same time on a first turn, and two
animations fighting over `.ph-beam` would drive both.
"""
from __future__ import annotations

import html

from .svg import svg_img


def thinking_mark(stroke: str, accent: str, faint: str, caption: str) -> str:
    """A small pair of scales rocking gently, over a caption.

    `caption` is the phase the pipeline is in — user-facing text, escaped
    because a phase line can carry a retrieved document's title.

    Delivered as a data-URI <img> (see ui/svg.py): a bare <svg> is stripped by
    Streamlit's sanitiser, which is why this mark rendered as blank space until
    that was diagnosed. The caption stays OUTSIDE the image so it remains real,
    selectable page text for a screen reader.
    """
    label = html.escape(caption)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 130">
  <style>
    .beam {{ transform-origin:120px 52px; animation:weigh 2.4s ease-in-out infinite; }}
    .panl {{ animation:dipl 2.4s ease-in-out infinite; }}
    .panr {{ animation:dipr 2.4s ease-in-out infinite; }}
    .ring {{ transform-origin:120px 62px; animation:turn 18s linear infinite; }}
    @keyframes weigh {{ 0%,100%{{transform:rotate(-6deg);}} 50%{{transform:rotate(6deg);}} }}
    @keyframes dipl  {{ 0%,100%{{transform:translateY(8px);}} 50%{{transform:translateY(-8px);}} }}
    @keyframes dipr  {{ 0%,100%{{transform:translateY(-8px);}} 50%{{transform:translateY(8px);}} }}
    @keyframes turn  {{ to {{ transform:rotate(360deg); }} }}
    @media (prefers-reduced-motion: reduce) {{
      .beam,.panl,.panr,.ring {{ animation:none; }}
    }}
  </style>
  <g class="ring" fill="none" stroke="{faint}" stroke-width="1.1" opacity=".55">
    <circle cx="120" cy="62" r="54" stroke-dasharray="9 13"/>
  </g>
  <g fill="none" stroke="{stroke}" stroke-width="2.4"
     stroke-linecap="round" stroke-linejoin="round">
    <path d="M96 116 h48" stroke="{accent}" stroke-width="3"/>
    <path d="M108 116 c2-14 4-22 12-26 8 4 10 12 12 26"/>
    <path d="M120 90 V44"/>
    <circle cx="120" cy="38" r="5" stroke="{accent}"/>
    <g class="beam">
      <path d="M50 52 H190" stroke="{accent}" stroke-width="3"/>
      <path d="M50 52 v8"/><path d="M190 52 v8"/>
    </g>
    <g class="panl">
      <path d="M50 60 L34 78 M50 60 L66 78" stroke="{faint}"/>
      <path d="M30 78 h40 a20 20 0 0 1 -40 0 z" stroke="{accent}"/>
    </g>
    <g class="panr">
      <path d="M190 60 L174 78 M190 60 L206 78" stroke="{faint}"/>
      <path d="M170 78 h40 a20 20 0 0 1 -40 0 z" stroke="{accent}"/>
    </g>
  </g>
</svg>"""
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;
            gap:2px;padding:10px 0 14px" role="status" aria-live="polite">
  {svg_img(svg, "116px", "")}
  <p style="margin:6px 0 0;font-size:.84rem;opacity:.72;text-align:center">{label}</p>
</div>
"""
