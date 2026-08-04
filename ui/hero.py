"""The hero mark: a hand-drawn scales of justice that settles into balance.

It is delivered as a data-URI <img> (see ui/svg.py), NOT as a bare
<svg> element — Streamlit's sanitiser strips the SVG namespace outright, so
this mark silently failed to render at all until that was found.

Why an inline SVG and not an image file: this app's premise is that nothing
leaves the machine, so a remote illustration is out, and a bundled raster would
need two versions for the light and dark grounds. Vector strokes take their
colour from the active theme and stay crisp at any size.

Why scales rather than a figure of Justice: a full figure rendered in flat
strokes reads as clip-art unless it is drawn very well, and a crude one would
undercut everything else on the page. The scales carry the same meaning and can
be executed precisely.

The motion is the idea of the page, not decoration: the beam swings and settles
level — weighing, then balance. It runs once, takes under three seconds, and is
skipped entirely for anyone who has asked for reduced motion.
"""
from __future__ import annotations

from .svg import svg_img


def scales_svg(stroke: str, accent: str, faint: str) -> str:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 168"
     role="img" aria-label="A pair of scales settling into balance">
  <style>
    .beam {{ transform-origin:120px 52px;
             animation:weigh 2.6s cubic-bezier(.34,.62,.32,1) both; }}
    .panl {{ animation:dipl 2.6s cubic-bezier(.34,.62,.32,1) both; }}
    .panr {{ animation:dipr 2.6s cubic-bezier(.34,.62,.32,1) both; }}
    @keyframes weigh {{
      0%{{transform:rotate(-11deg);}} 38%{{transform:rotate(7deg);}}
      66%{{transform:rotate(-3deg);}} 86%{{transform:rotate(1deg);}}
      100%{{transform:rotate(0deg);}}
    }}
    @keyframes dipl {{
      0%{{transform:translateY(15px);}} 38%{{transform:translateY(-9px);}}
      66%{{transform:translateY(4px);}} 86%{{transform:translateY(-1px);}}
      100%{{transform:translateY(0);}}
    }}
    @keyframes dipr {{
      0%{{transform:translateY(-15px);}} 38%{{transform:translateY(9px);}}
      66%{{transform:translateY(-4px);}} 86%{{transform:translateY(1px);}}
      100%{{transform:translateY(0);}}
    }}
    /* Motion is a nicety; legibility is not. Anyone who has asked their
       system to stop animating gets the balanced end state immediately. */
    @media (prefers-reduced-motion: reduce) {{
      .beam,.panl,.panr {{ animation:none; }}
    }}
  </style>
  <g fill="none" stroke="{stroke}" stroke-width="2.4"
     stroke-linecap="round" stroke-linejoin="round">
    <path d="M96 150 h48" stroke="{accent}" stroke-width="3"/>
    <path d="M108 150 c2-14 4-22 12-26 8 4 10 12 12 26"/>
    <path d="M120 124 V44"/>
    <circle cx="120" cy="38" r="5" stroke="{accent}"/>
    <g class="beam">
      <path d="M50 52 H190" stroke="{accent}" stroke-width="3"/>
      <path d="M50 52 v8"/><path d="M190 52 v8"/>
    </g>
    <g class="panl">
      <path d="M50 60 L30 84 M50 60 L70 84" stroke="{faint}"/>
      <path d="M26 84 h48 a24 24 0 0 1 -48 0 z" stroke="{accent}"/>
    </g>
    <g class="panr">
      <path d="M190 60 L170 84 M190 60 L210 84" stroke="{faint}"/>
      <path d="M166 84 h48 a24 24 0 0 1 -48 0 z" stroke="{accent}"/>
    </g>
  </g>
</svg>"""
    return ('<div style="display:flex;justify-content:center;padding:4px 0 2px">'
            + svg_img(svg, "178px", "A pair of scales settling into balance")
            + '</div>')
