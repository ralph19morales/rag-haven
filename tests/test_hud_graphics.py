"""Invariants for the HUD's graphics — the ones that fail SILENTLY in a browser.

Every check here encodes a defect that shipped. None of them can catch the whole
class, and that limitation is the point of `ui/svg.py`'s docstring: a graphic can
be emitted perfectly by the server and thrown away or mangled by the browser, and
no server-side assertion sees it. What CAN be pinned from here is the markup
contract that the browser behaviour depends on, so a later edit cannot quietly
break it again.

  * A bare <svg> passed to st.html is silently deleted by DOMPurify. It shipped
    three times (hero mark, waiting animation, the whole HUD). Every graphic must
    leave here as a data-URI <img>.
  * An <img> with height:auto takes its intrinsic ratio from the viewBox, so a
    full-width sparkline drawn into a short, wide box rendered ~140px tall in a
    500px column instead of ~46px — which is what made the traces read as lumpy
    area charts rather than sparklines.
  * Min-max normalising a series against its own range gave a flat series full
    amplitude: 41.2s to 41.4s over ten queries drew as a mountain range.

Run:  python tests/test_hud_graphics.py
"""
from __future__ import annotations

import base64
import re
import sys
import xml.dom.minidom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui import hud  # noqa: E402
from ui.svg import svg_img  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def decode(markup: str) -> str:
    """The SVG document inside a data-URI <img>."""
    m = re.search(r'base64,([^"]+)"', markup)
    return base64.b64decode(m.group(1)).decode("utf-8") if m else ""


def img_style(markup: str) -> str:
    img = re.search(r"<img[^>]+>", markup)
    return re.search(r'style="([^"]+)"', img.group(0)).group(1) if img else ""


def run() -> int:
    r = []

    rising = [50.1, 31.9, 28.4, 27.8, 29.9, 26.2, 27.1, 26.9]
    flat = [41.2, 41.25, 41.21, 41.4, 41.23, 41.26]

    spark = hud.sparkline(rising, "total", "28.4 s median", unit=" s")
    svg = decode(spark)

    # --- the three-times-shipped one ---------------------------------------
    # Whatever else is true, no raw <svg> element may appear in markup handed
    # to st.html: DOMPurify's html profile has no SVG namespace and deletes it.
    r.append(check("the graphic leaves as a data-URI <img>",
                   "<img src=\"data:image/svg+xml;base64," in spark))
    r.append(check("no bare <svg> element in the emitted HTML",
                   "<svg" not in spark.replace(svg, "")))
    for name, markup in (("empty series", hud.sparkline([], "kv", "—")),
                         ("single sample", hud.sparkline([5.0], "kv", "5"))):
        r.append(check(f"no bare <svg> in the {name} fallback",
                       "<svg" not in markup))

    r.append(check("the SVG is a well-formed document",
                   bool(svg) and _parses(svg)))

    # --- the oversizing one -------------------------------------------------
    style = img_style(spark)
    r.append(check("the trace is given an explicit height, not height:auto",
                   "height:auto" not in style and "height:46px" in style,
                   f"({style})"))
    r.append(check("the trace still spans the panel width",
                   "width:100%" in style))
    # An explicit height is only safe with a stretch fit and non-scaling strokes;
    # together they are what keep the line one weight in both directions.
    r.append(check("the drawing is set to fill that box",
                   'preserveAspectRatio="none"' in svg))
    r.append(check("every stroked mark opts out of stroke scaling",
                   svg.count("non-scaling-stroke") >= svg.count("stroke-width")))
    r.append(check("the empty state reserves the same height",
                   "height:46px" in hud.sparkline([], "kv", "—")))

    # --- the flat-series one ------------------------------------------------
    def ys(markup: str) -> list[float]:
        pts = re.search(r'<polyline points="([^"]+)"', decode(markup)).group(1)
        return [float(p.split(",")[1]) for p in pts.split()]

    fy, ry = ys(hud.sparkline(flat, "total", "41.2 s")), ys(spark)
    r.append(check("a flat series draws flat, not as full-amplitude noise",
                   max(fy) - min(fy) == 0, f"(spread {max(fy) - min(fy):.1f})"))
    r.append(check("a genuinely varying series still uses the box",
                   max(ry) - min(ry) > 20, f"(spread {max(ry) - min(ry):.1f})"))
    r.append(check("a flat series says so instead of showing a false range",
                   "flat" in hud.sparkline(flat, "total", "41.2 s")))

    # Scale is direct-labelled, because a data-URI <img> can carry no tooltip.
    r.append(check("the trace direct-labels its own vertical range",
                   "26 s" in spark and "50 s" in spark))
    r.append(check("the trace reports how many samples it drew",
                   f"{len(rising)} samples" in spark))

    # --- chart series colours ----------------------------------------------
    # Status hues are reserved: an amber or red line would read as a warning.
    reserved = {hud.AMBER, hud.ALERT, hud.GOOD}
    r.append(check("no chart series reuses a status colour",
                   not (set(hud.SERIES) & reserved),
                   f"({hud.SERIES})"))
    r.append(check("series slots are a fixed order, not generated",
                   list(hud.SERIES) == [hud.SERIES_1, hud.SERIES_2]))
    theme = hud.chart_theme()
    r.append(check("the chart theme carries the series range",
                   theme["range"]["category"] == list(hud.SERIES)))
    r.append(check("charts sit on a transparent ground, not stock white",
                   theme["background"] == "transparent"))

    # --- the icon-ligature one ----------------------------------------------
    # Streamlit draws every `:material/…:` icon as a LIGATURE: the element's
    # text content is the literal string `smart_toy`, and "Material Symbols
    # Rounded" is what turns it into a glyph. hud_css() sets a monospace family
    # on `[class*="st-"]`, which matches those spans — so the names printed as
    # words across both apps (`smart_toy` on every assistant avatar,
    # `keyboard_double_arrow_right` on the sidebar toggle), and the avatars,
    # which Streamlit gives a filled background, read as coloured blocks of
    # text. The restore rule below is the fix; these pin it.
    css = hud.hud_css()
    r.append(check("the icon font family is restored after the broad rule",
                   '"Material Symbols Rounded"' in css
                   and css.index('[class*="st-"]')
                   < css.index('"Material Symbols Rounded"')))

    # The restore must cover the icon testids by SHAPE. The component defaults
    # to `stIconMaterial` but accepts an override, and the overrides are
    # scattered — a hand-listed set of three selectors shipped, and missed the
    # expander's, leaving `keyboard_arrow_right` printing in the sidebar. The
    # list below is every icon testid in the shipped bundle:
    #   grep -rhao 'st[A-Za-z]*Icon[A-Za-z]*' streamlit/static | sort -u
    icon_testids = (
        "stIconMaterial", "stExpanderIcon", "stExpanderIconError",
        "stExpanderIconCheck", "stExpanderIconSpinner", "stFileChipIconError",
        "stFileChipIconSpinner", "stAlertDynamicIcon", "stToastDynamicIcon",
        "stElementToolbarButtonIcon", "stTooltipIcon", "stSpinnerIcon",
        "stMetricDeltaIcon", "stNumberInputIcon", "stTextInputIcon",
        "stDialogIcon", "stToolbarActionButtonIcon", "stImageIcon",
        "stChatMessageAvatarAssistant", "stChatMessageAvatarUser",
    )
    restore = css[css.index('[data-testid="stIconMaterial"]'):]
    selectors = restore[:restore.index("{")]
    patterns = re.findall(r'\[data-testid([\^$]?=)"([^"]+)"\]', selectors)

    def covered(testid: str) -> bool:
        for op, val in patterns:
            if (op == "=" and testid == val) \
                    or (op == "^=" and testid.startswith(val)) \
                    or (op == "$=" and testid.endswith(val)):
                return True
        return False

    missed = [t for t in icon_testids if not covered(t)]
    r.append(check("every icon testid in the bundle is covered",
                   not missed, f"(missed {missed})" if missed else ""))
    # Streamlit's own emotion classes are more specific than a bare attribute
    # selector, so the restore only lands with !important.
    r.append(check("the restore is marked !important",
                   "!important" in restore[:restore.index("}")]))
    # Per-glyph fallback: an icon slot carrying an emoji rather than a ligature
    # must still resolve, so a real family has to follow the symbol font.
    r.append(check("the symbol font falls back to a real family",
                   re.search(r'font-family:\s*"Material Symbols Rounded",\s*\S',
                             restore) is not None))

    # --- the nucleus is one drawing, not two --------------------------------
    # `neutron()` carries the dashboard's verdict and Haven uses the same figure
    # while it starts up. Two copies of an animation this fiddly is how the two
    # surfaces drift apart, so the verdict wrapper must delegate.
    r.append(check("neutron delegates to nucleus rather than redrawing",
                   hud.neutron(True, "x") == hud.nucleus(hud.GOOD, "x")
                   and hud.neutron(False, "x") == hud.nucleus(hud.ALERT, "x")))
    r.append(check("nucleus takes an arbitrary colour",
                   hud.CYAN in hud.nucleus(hud.CYAN, "x")))

    # svg_img's default must stay backwards-compatible: the fixed-width gauges
    # rely on height:auto to keep their square aspect.
    r.append(check("svg_img still defaults to height:auto",
                   "height:auto" in svg_img("<svg xmlns='http://www.w3.org/2000/svg'/>",
                                            "196px")))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


def _parses(svg: str) -> bool:
    try:
        xml.dom.minidom.parseString(svg)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(run())
