"""HUD chrome for the ops dashboard — a heads-up display, not a report.

WHY THIS EXISTS, AND WHY ONLY HERE. Haven (`app.py`) is deliberately quiet: a
law-library palette, serif headings, nothing that competes with the text of the
law. This module is the opposite on purpose, and it is scoped to `dashboard.py`
alone, whose audience is the person operating the box. A status console is the
one surface where a glowing readout is the *right* register — it is read at a
glance, from a distance, to answer "is anything wrong", and that is a different
job from "what does the law say about my relative's body".

Same hard constraints as the rest of the UI:
  * nothing leaves the machine — no webfonts, no CDN, no remote assets. Every
    glyph is an inline SVG and every face is a built-in monospace keyword.
  * motion is a nicety. Everything animated is disabled under
    prefers-reduced-motion, and every reading is legible without it: the
    animation never *carries* a value, it only decorates one.

The palette is intentionally NOT the shared `.streamlit/config.toml` theme.
That theme is the law-library one both apps otherwise share; a HUD needs a dark
ground and a single high-chroma accent to work at all. Because dashboard.py
runs as its own app on its own port, overriding it here cannot leak into Haven.
"""
from __future__ import annotations

import html

from .svg import svg_img


# --- palette ---------------------------------------------------------------
# One accent (cyan) carries "normal". Deviations get their own hue so a problem
# is a COLOUR CHANGE, not a smaller number — readable at a glance across a room,
# which is the entire point of a status console.
INK = "#0a1119"
INK_2 = "#0e1822"
LINE = "#16394a"
CYAN = "#4fd8ff"
CYAN_DIM = "#2b7f9e"
CYAN_GHOST = "#12303d"
TEXT = "#bfe4f2"
TEXT_DIM = "#6f97a8"
AMBER = "#ffb648"
ALERT = "#ff5f6d"
GOOD = "#35e0a1"

MONO = ('"Cascadia Mono",ui-monospace,SFMono-Regular,Consolas,'
        '"Liberation Mono",monospace')

# --- categorical series colours, for the plotted charts only ----------------
# The palette above is CHROME: one accent plus three status hues. A chart that
# draws two measures at once needs two colours that mean "this series" rather
# than "this state", and the status hues are reserved — an amber line would read
# as a warning. So these two slots exist, assigned in fixed order and never
# cycled; a third measure gets its own chart, not a third hue.
#
# Chosen by running the six computable checks rather than by eye, against this
# page's own surface (INK_2), adjacent-pair mode:
#
#   CVD separation (min protan/deutan)   ΔE 12.9   (target >= 8)
#   Normal-vision floor                  ΔE 19.5   (hard floor 15)
#   Chroma floor                         both >= 0.10
#   Contrast vs INK_2                    8.4:1 and 5.1:1  (min 3:1)
#   Lightness band (dark)                SERIES_1 L=0.75 — 0.08 OVER the 0.67 ceiling
#
# That last one is a deliberate, recorded deviation, not an oversight. The dark
# band is calibrated for a conventional dark-grey chart surface (#1a1a19); this
# page's ground is near-black, and the band's UPPER bound exists to stop a mark
# blowing out against a lighter grey. Against INK_2 the contrast check — which
# is the functional test — passes with room to spare, and a mark dimmed into the
# band reads as muddy next to the surrounding cyan chrome. Every other check
# passes outright. If the HUD ground is ever lightened, re-run the checks.
SERIES_1 = "#3fc0e6"        # slot 1 — retrieval, volume, the primary measure
SERIES_2 = "#8f78e0"        # slot 2 — generation
SERIES = (SERIES_1, SERIES_2)


def chart_theme() -> dict:
    """Vega-Lite config that puts an Altair chart on the HUD's ground.

    Streamlit's built-in `st.line_chart`/`st.bar_chart` render with the stock
    theme, which on this page reads as a white report card dropped onto a status
    console. These charts are real Vega-Lite components rather than sanitised
    HTML, so — unlike everything routed through `ui/svg.py` — they can carry the
    hover layer a chart is supposed to have."""
    axis = {
        "labelFont": MONO, "titleFont": MONO,
        "labelFontSize": 10, "titleFontSize": 10,
        "labelColor": TEXT_DIM, "titleColor": TEXT_DIM,
        "domainColor": LINE, "tickColor": LINE,
        "gridColor": CYAN_GHOST, "gridOpacity": 0.55, "titlePadding": 8,
    }
    return {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "axis": axis,
        "axisX": dict(axis, grid=False),
        "legend": {
            "labelFont": MONO, "titleFont": MONO,
            "labelFontSize": 10, "titleFontSize": 10,
            "labelColor": TEXT, "titleColor": TEXT_DIM,
            "symbolStrokeWidth": 3, "orient": "top", "direction": "horizontal",
            "offset": 4,
        },
        "range": {"category": list(SERIES)},
        "title": {"font": MONO, "color": TEXT_DIM, "fontSize": 11,
                  "fontWeight": "normal", "anchor": "start"},
    }


def _c(state: str) -> str:
    """Accent colour for a state word."""
    return {"ok": CYAN, "good": GOOD, "warn": AMBER, "alert": ALERT}.get(state, CYAN)


def hud_css() -> str:
    """Global stylesheet. Injected once per page load.

    Restyles Streamlit's own containers as well as our components, because a
    HUD with default-styled dataframes and buttons sitting in it reads as a
    half-finished theme rather than a deliberate one."""
    return f"""
<style>
  .stApp {{
    background:
      radial-gradient(1200px 620px at 18% -10%, #123044 0%, transparent 62%),
      radial-gradient(900px 500px at 92% 0%, #10283a 0%, transparent 58%),
      {INK};
    color: {TEXT};
  }}
  /* Scanlines: a fixed overlay, pinned behind every widget so it can never
     sit on top of text and cost legibility. */
  .stApp::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: repeating-linear-gradient(
      to bottom, rgba(79,216,255,.030) 0 1px, transparent 1px 3px);
    mix-blend-mode: screen;
  }}
  .stApp > * {{ position: relative; z-index: 1; }}

  html, body, .stApp, [class*="st-"] {{ font-family: {MONO}; }}
  /* ...but NEVER the icon font. Streamlit draws every `:material/…:` icon as a
     LIGATURE in "Material Symbols Rounded" — the element's text content is the
     literal string `smart_toy`, and the font is what turns it into a glyph. The
     rule above matches those spans (their emotion classes start `st-`), so
     overriding the family printed the ligature names as words: `smart_toy` on
     every assistant avatar, `keyboard_double_arrow_right` on the sidebar
     toggle, `menu_book` on the sources expander. The avatars made it look like
     coloured blocks of text because Streamlit gives them a filled background.
     Restoring the family here is what keeps them glyphs. Any new selector that
     sets a font family broadly must be added to the exclusion below.

     Matched by SHAPE, not by enumeration. The icon component defaults to
     `data-testid="stIconMaterial"` but takes an override, and the overrides are
     scattered (`stExpanderIconError`, `stFileChipIconSpinner`,
     `stAlertDynamicIcon`, `stToastDynamicIcon`, `stElementToolbarButtonIcon`…).
     A hand-listed set missed the expander's chevron neighbours and left
     `keyboard_arrow_right` printing in the sidebar. `$="Icon"` plus the two
     prefixes below cover every icon testid in the shipped bundle.

     The monospace fallback after the symbol font is deliberate: font fallback
     is per-glyph, so a `stAlertDynamicIcon` carrying an EMOJI rather than a
     ligature still resolves — the symbol font simply has no glyph for it and
     the next family answers. */
  [data-testid="stIconMaterial"],
  [data-testid$="Icon"],
  [data-testid^="stExpanderIcon"],
  [data-testid^="stFileChipIcon"],
  [data-testid^="stChatMessageAvatar"],
  [data-testid^="stChatMessageAvatar"] *,
  span[class*="material-symbols"] {{
    font-family: "Material Symbols Rounded", {MONO} !important;
    font-feature-settings: "liga";
  }}
  h1, h2, h3, h4, h5, h6 {{
    font-family: {MONO} !important;
    text-transform: uppercase; letter-spacing: .16em;
    color: {TEXT} !important; font-weight: 600 !important;
  }}
  p, span, label, li, div {{ color: {TEXT}; }}
  hr {{ border-color: {LINE} !important; }}

  /* Streamlit widgets, pulled into the HUD */
  [data-testid="stMetricValue"] {{
    color: {CYAN} !important; font-family: {MONO};
    text-shadow: 0 0 14px rgba(79,216,255,.45);
  }}
  [data-testid="stMetricLabel"] {{
    color: {TEXT_DIM} !important; text-transform: uppercase;
    letter-spacing: .13em; font-size: .68rem !important;
  }}
  [data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid {LINE}; background: {INK_2};
  }}
  .stButton > button {{
    background: transparent; color: {CYAN};
    border: 1px solid {CYAN_DIM}; border-radius: 0;
    text-transform: uppercase; letter-spacing: .14em; font-size: .74rem;
    font-family: {MONO}; transition: background .18s, box-shadow .18s;
  }}
  .stButton > button:hover {{
    background: rgba(79,216,255,.10); border-color: {CYAN};
    box-shadow: 0 0 18px rgba(79,216,255,.30); color: {CYAN};
  }}
  [data-testid="stAlert"] {{
    background: {INK_2}; border: 1px solid {LINE}; border-radius: 0;
    color: {TEXT};
  }}

  /* --- HUD primitives --- */
  .hud-panel {{
    position: relative; border: 1px solid {LINE}; background:
      linear-gradient(180deg, rgba(20,52,68,.34), rgba(10,17,25,.22));
    padding: 13px 15px 14px; margin: 0 0 14px;
  }}
  /* Corner brackets — drawn with borders on two pseudo-elements rather than
     four absolutely-positioned divs, so a panel is one element in the DOM. */
  .hud-panel::before, .hud-panel::after {{
    content: ""; position: absolute; width: 13px; height: 13px;
    border-color: {CYAN}; border-style: solid; opacity: .85;
  }}
  .hud-panel::before {{ top: -1px; left: -1px; border-width: 2px 0 0 2px; }}
  .hud-panel::after  {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}
  .hud-title {{
    font-size: .66rem; letter-spacing: .24em; text-transform: uppercase;
    color: {CYAN}; margin: 0 0 11px; display: flex;
    align-items: center; gap: 9px;
  }}
  .hud-title::after {{
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, {CYAN_GHOST}, transparent);
  }}

  .hud-row {{
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px; padding: 4px 0; font-size: .8rem;
  }}
  .hud-row + .hud-row {{ border-top: 1px dashed rgba(22,57,74,.75); }}
  .hud-k {{ color: {TEXT_DIM}; letter-spacing: .08em; font-size: .72rem;
            text-transform: uppercase; white-space: nowrap; }}
  .hud-v {{ font-variant-numeric: tabular-nums; text-align: right; }}
  .hud-dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
              margin-right: 7px; vertical-align: middle; }}

  .hud-banner {{
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    border: 1px solid {LINE}; border-left-width: 3px;
    background: linear-gradient(90deg, rgba(20,52,68,.5), transparent 70%);
    padding: 12px 16px; margin-bottom: 16px;
  }}
  .hud-wordmark {{
    font-size: 1.18rem; letter-spacing: .52em; color: {TEXT};
    text-shadow: 0 0 22px rgba(79,216,255,.35);
  }}
  .hud-state {{ font-size: .78rem; letter-spacing: .2em; }}
  .hud-meta {{ color: {TEXT_DIM}; font-size: .68rem; letter-spacing: .1em;
               margin-left: auto; text-align: right; }}

  /* Segmented bar — discrete cells read as a gauge, a smooth fill reads as a
     progress bar (which would imply something is completing). */
  .hud-seg {{ display: flex; gap: 2px; margin-top: 7px; }}
  .hud-seg i {{ flex: 1; height: 7px; background: {CYAN_GHOST}; }}
  .hud-seg i.on {{ box-shadow: 0 0 9px rgba(79,216,255,.55); }}

  @keyframes hud-spin {{ to {{ transform: rotate(360deg); }} }}
  @keyframes hud-spin-rev {{ to {{ transform: rotate(-360deg); }} }}
  @keyframes hud-pulse {{
    0%,100% {{ opacity: .95; }} 50% {{ opacity: .55; }}
  }}
  @keyframes hud-sweep {{
    0% {{ transform: translateY(-100%); }} 100% {{ transform: translateY(2400%); }}
  }}
  .hud-core-outer {{ transform-origin: 100px 100px; animation: hud-spin 24s linear infinite; }}
  .hud-core-mid   {{ transform-origin: 100px 100px; animation: hud-spin-rev 15s linear infinite; }}
  .hud-core-glow  {{ animation: hud-pulse 3.4s ease-in-out infinite; }}
  .hud-scan {{
    position: absolute; left: 0; right: 0; height: 2px; top: 0;
    background: linear-gradient(90deg, transparent, rgba(79,216,255,.5), transparent);
    animation: hud-sweep 7s linear infinite; pointer-events: none;
  }}

  @media (prefers-reduced-motion: reduce) {{
    .hud-core-outer, .hud-core-mid, .hud-core-glow, .hud-scan {{
      animation: none;
    }}
    .hud-scan {{ display: none; }}
  }}
</style>
"""


def banner(state_label: str, state: str, meta: str) -> str:
    col = _c(state)
    return (
        f'<div class="hud-banner" style="border-left-color:{col}">'
        f'<span class="hud-wordmark">HAVEN</span>'
        f'<span class="hud-state" style="color:{col};'
        f'text-shadow:0 0 16px {col}66">&#9670; {html.escape(state_label)}</span>'
        f'<span class="hud-meta">{html.escape(meta)}</span>'
        f'</div>'
    )


def reactor(frac: float, big: str, small: str, caption: str,
            state: str = "ok") -> str:
    """The arc-reactor gauge: concentric rings with an arc filled to `frac`.

    `frac` is clamped, because a gauge that overshoots its own ring reads as a
    rendering bug rather than as an overload — and vLLM legitimately sits at
    ~96% VRAM here by design, close enough to 1.0 to matter.
    """
    f = max(0.0, min(1.0, frac))
    col = _c(state)
    r = 66
    circ = 2 * 3.141592653589793 * r
    dash = f"{circ * f:.2f} {circ:.2f}"
    ticks = "".join(
        f'<line x1="100" y1="18" x2="100" y2="26" stroke="{CYAN_DIM}" '
        f'stroke-width="1.6" transform="rotate({i * 15} 100 100)" '
        f'opacity="{0.9 if i % 2 == 0 else 0.4}"/>'
        for i in range(24)
    )
    # Self-contained: styles and keyframes live inside the document, because
    # page CSS cannot reach into an <img>. See svg_img().
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <style>
    .o {{ transform-origin:100px 100px; animation:spin 24s linear infinite; }}
    .m {{ transform-origin:100px 100px; animation:spinr 15s linear infinite; }}
    .g {{ animation:pulse 3.4s ease-in-out infinite; }}
    @keyframes spin  {{ to {{ transform:rotate(360deg); }} }}
    @keyframes spinr {{ to {{ transform:rotate(-360deg); }} }}
    @keyframes pulse {{ 0%,100% {{ opacity:.14; }} 50% {{ opacity:.05; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .o,.m,.g {{ animation:none; }}
    }}
  </style>
  <defs>
    <filter id="gl" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="4.2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <g class="o">{ticks}</g>
  <circle cx="100" cy="100" r="84" fill="none" stroke="{LINE}" stroke-width="1"/>
  <g class="m">
    <circle cx="100" cy="100" r="76" fill="none" stroke="{CYAN_DIM}"
            stroke-width="1.4" stroke-dasharray="16 10" opacity=".75"/>
  </g>
  <circle cx="100" cy="100" r="{r}" fill="none" stroke="{CYAN_GHOST}" stroke-width="9"/>
  <circle cx="100" cy="100" r="{r}" fill="none" stroke="{col}" stroke-width="9"
          stroke-linecap="round" stroke-dasharray="{dash}"
          transform="rotate(-90 100 100)" filter="url(#gl)"/>
  <circle class="g" cx="100" cy="100" r="47" fill="{col}" opacity=".10"/>
  <circle cx="100" cy="100" r="47" fill="none" stroke="{col}" stroke-width="1" opacity=".55"/>
  <text x="100" y="95" text-anchor="middle" fill="{TEXT}" font-family='{MONO}'
        font-size="27" font-weight="600">{html.escape(big)}</text>
  <text x="100" y="117" text-anchor="middle" fill="{TEXT_DIM}" font-family='{MONO}'
        font-size="10.5" letter-spacing="2.4">{html.escape(small)}</text>
</svg>"""
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'padding:2px 0 6px">'
        + svg_img(svg, "196px", f"{caption}: {big} {small}")
        + f'<div style="color:{TEXT_DIM};font-size:.64rem;letter-spacing:.22em;'
          f'text-transform:uppercase;margin-top:6px">{html.escape(caption)}</div>'
        '</div>'
    )


def nucleus(col: str, label: str) -> str:
    """An orbiting nucleus in `col`, captioned in words underneath.

    The drawing only. What it MEANS is the caller's business — `neutron()` uses
    it to carry the dashboard's overall verdict, and Haven uses it as the mark
    for "starting up" while the models load. Extracted rather than copied: two
    versions of an animation this fiddly is how the two surfaces drift apart.

    Whatever the caller uses it for, the caption is the carrier and the colour
    is not: an <img> has no text a screen reader can reach, and red/green is
    exactly the pair the commonest colour-vision deficiencies confuse.
    """
    rings = ""
    for rot, dur, cls in ((0, 7.5, "a"), (60, 9.5, "b"), (120, 11.5, "a")):
        # Nested groups: a CSS transform REPLACES the SVG presentation
        # attribute rather than composing with it, so the fixed offset and the
        # spin must live on different elements or all three rings collapse
        # onto each other at 0 degrees.
        rings += (
            f'<g transform="rotate({rot} 90 90)">'
            f'<g class="{cls}" style="animation-duration:{dur}s">'
            f'<ellipse cx="90" cy="90" rx="62" ry="24" fill="none" '
            f'stroke="{col}" stroke-width="1.3" opacity=".55"/>'
            f'<circle cx="152" cy="90" r="4.6" fill="{col}" filter="url(#g2)"/>'
            f'</g></g>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180">
  <style>
    .a,.b {{ transform-origin:90px 90px; }}
    .a {{ animation:orb 8s linear infinite; }}
    .b {{ animation:orbr 8s linear infinite; }}
    .n {{ animation:pul 2.6s ease-in-out infinite; }}
    @keyframes orb  {{ to {{ transform:rotate(360deg); }} }}
    @keyframes orbr {{ to {{ transform:rotate(-360deg); }} }}
    @keyframes pul  {{ 0%,100% {{ opacity:.30; }} 50% {{ opacity:.12; }} }}
    @media (prefers-reduced-motion: reduce) {{ .a,.b,.n {{ animation:none; }} }}
  </style>
  <defs>
    <filter id="g2" x="-70%" y="-70%" width="240%" height="240%">
      <feGaussianBlur stdDeviation="2.6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  {rings}
  <circle class="n" cx="90" cy="90" r="13" fill="{col}" opacity=".22"/>
  <circle cx="90" cy="90" r="7.5" fill="{col}" filter="url(#g2)"/>
</svg>"""
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'padding:6px 0 2px">'
        + svg_img(svg, "186px", label)
        + f'<div style="color:{col};font-size:.64rem;letter-spacing:.22em;'
          f'text-transform:uppercase;margin-top:6px;text-align:center">'
          f'{html.escape(label)}</div>'
        '</div>'
    )


def neutron(ok: bool, label: str) -> str:
    """The nucleus, carrying the dashboard's overall verdict by colour.

    Fills the column under the reactor, and earns the space: it is the element
    that answers "is anything wrong" from across the room, before a single
    label has been read. Green orbits mean every check passed, red means at
    least one did not.

    Colour is never the ONLY carrier. The caption under it says the verdict in
    words, the banner repeats it, and each failing check prints a plain-text
    error further down.
    """
    return nucleus(GOOD if ok else ALERT, label)


def rows(items: list[tuple[str, str, str]]) -> str:
    """Key / value / state triples as HUD readout lines."""
    out = []
    for k, v, state in items:
        col = _c(state)
        out.append(
            f'<div class="hud-row"><span class="hud-k">{html.escape(k)}</span>'
            f'<span class="hud-v" style="color:{col}">'
            f'<i class="hud-dot" style="background:{col};'
            f'box-shadow:0 0 8px {col}"></i>{html.escape(v)}</span></div>'
        )
    return "".join(out)


def segbar(frac: float, state: str = "ok", cells: int = 28) -> str:
    col = _c(state)
    on = int(round(max(0.0, min(1.0, frac)) * cells))
    return ('<div class="hud-seg">' + "".join(
        f'<i class="on" style="background:{col}"></i>' if i < on else "<i></i>"
        for i in range(cells)) + "</div>")


def panel(title: str, body: str, scan: bool = False) -> str:
    return (f'<div class="hud-panel">{"<div class=hud-scan></div>" if scan else ""}'
            f'<div class="hud-title">{html.escape(title)}</div>{body}</div>')


# A sparkline is drawn into a box far wider than it is tall and then stretched
# to the panel width, so the horizontal scale factor is several times the
# vertical one. Three consequences, all of which were visible in the first
# version and are handled below:
#
#  * `preserveAspectRatio="none"` scales the STROKE with the geometry, so a
#    nominally 1.4px line came out thin on the verticals and fat on the
#    horizontals. `vector-effect="non-scaling-stroke"` keeps the stroke at its
#    literal width whatever the box does to the path. The viewBox is also much
#    wider now, so the residual distortion of the shape itself is small.
#  * Min-max normalising to the series' OWN range gives a flat series full
#    amplitude: 41.2s → 41.4s over ten queries drew as a mountain range. The
#    flat guard below keeps a series whose spread is negligible against its own
#    magnitude drawn flat, which is what it is.
#  * The reading has no axis and cannot have a tooltip — a data-URI <img> takes
#    no hover, so the usual interactive layer is unavailable here by
#    construction (see ui/svg.py). Scale is therefore direct-labelled instead:
#    min and max sit at the ends and the latest sample carries a dot.
_SPARK_W, _SPARK_H = 320.0, 46.0
_SPARK_PAD = 7.0            # top/bottom room for the stroke and the end dot
_FLAT_REL = 0.02            # spread under 2% of |median| is noise, not shape


def sparkline(values: list[float], label: str, value: str,
              state: str = "ok", unit: str = "") -> str:
    """A trace of recent values, direct-labelled with its own vertical range.

    The number beside the label is the reading; this shows its shape over time.
    Stays legible when the series is too short or too flat to have one."""
    col = _c(state)
    vs = [float(v) for v in values if v is not None][-40:]

    if len(vs) < 2:
        body = (f'<div style="color:{TEXT_DIM};font-size:.66rem;height:{_SPARK_H:.0f}px;'
                f'display:flex;align-items:center">awaiting samples</div>')
        return (f'<div style="margin-bottom:12px">'
                f'<div class="hud-row" style="padding:0 0 2px">'
                f'<span class="hud-k">{html.escape(label)}</span>'
                f'<span class="hud-v" style="color:{col}">{html.escape(value)}</span>'
                f'</div>{body}</div>')

    lo, hi = min(vs), max(vs)
    mid = abs(sorted(vs)[len(vs) // 2]) or 1.0
    flat = (hi - lo) < _FLAT_REL * mid
    rng = (hi - lo) or 1.0
    usable = _SPARK_H - 2 * _SPARK_PAD
    step = _SPARK_W / (len(vs) - 1)

    def y(v: float) -> float:
        if flat:                      # centre it rather than amplify noise
            return _SPARK_H / 2
        return _SPARK_H - _SPARK_PAD - (v - lo) / rng * usable

    pts = " ".join(f"{i * step:.2f},{y(v):.2f}" for i, v in enumerate(vs))
    lastx, lasty = (len(vs) - 1) * step, y(vs[-1])
    # A faint fill under the trace reads as "volume of time" and keeps a
    # single-pixel line from disappearing on a dark ground.
    area = (f'{pts} {lastx:.2f},{_SPARK_H:.2f} 0,{_SPARK_H:.2f}')
    baseline = (f'<line x1="0" y1="{_SPARK_H - .6:.2f}" x2="{_SPARK_W:.2f}" '
                f'y2="{_SPARK_H - .6:.2f}" stroke="{LINE}" stroke-width="1" '
                f'vector-effect="non-scaling-stroke"/>')
    # Its own document: no page CSS reaches in here, so the gradient and every
    # colour are declared inline (ui/svg.py).
    spark = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_SPARK_W:.0f} {_SPARK_H:.0f}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{col}" stop-opacity=".26"/>'
        f'<stop offset="1" stop-color="{col}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'{baseline}'
        f'<polygon points="{area}" fill="url(#g)"/>'
        f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'vector-effect="non-scaling-stroke"/>'
        f'<circle cx="{lastx:.2f}" cy="{lasty:.2f}" r="2.6" fill="{col}" '
        f'vector-effect="non-scaling-stroke"/>'
        f'</svg>')
    # Explicit height, not "auto". With height:auto the <img> derives its height
    # from the viewBox ratio, so a full-width trace grew to ~140px in a 500px
    # column — the single biggest reason these read as lumpy area charts rather
    # than sparklines. See ui/svg.py.
    body = svg_img(spark, "100%", f"{label} trace, {len(vs)} samples",
                   height=f"{_SPARK_H:.0f}px")

    # Direct labels carry the vertical scale, because the graphic cannot.
    def fmt(v: float) -> str:
        return f"{v:,.0f}{unit}" if abs(v) >= 10 else f"{v:,.2f}{unit}"

    scale = ("flat" if flat else f"{fmt(lo)} – {fmt(hi)}")
    foot = (f'<div style="display:flex;justify-content:space-between;'
            f'font-size:.6rem;color:{TEXT_DIM};padding-top:1px">'
            f'<span>{html.escape(str(len(vs)))} samples</span>'
            f'<span>{html.escape(scale)}</span></div>')
    return (f'<div style="margin-bottom:12px">'
            f'<div class="hud-row" style="padding:0 0 2px">'
            f'<span class="hud-k">{html.escape(label)}</span>'
            f'<span class="hud-v" style="color:{col}">{html.escape(value)}</span>'
            f'</div>{body}{foot}</div>')
