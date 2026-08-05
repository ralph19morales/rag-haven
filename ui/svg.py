"""The one supported way to get an SVG onto a Streamlit page.

READ THIS BEFORE ADDING ANY GRAPHIC. `st.html` sanitises its input with
DOMPurify configured `USE_PROFILES: {html: true}` — verified directly in the
shipped frontend bundle (`streamlit/static/static/js/Html.*.js`, which calls
`sanitize(body, {USE_PROFILES:{html:!0}, FORCE_BODY:!0, ADD_TAGS:['script',
'style'], ADD_ATTR:[...]})`). The HTML profile does not include the SVG
namespace, and `ADD_TAGS` restores only `script` and `style`.

So **every `<svg>` element passed to `st.html` is silently discarded.** The
surrounding `<div>`s and `<style>` survive, which is what makes this so hard to
catch: the page renders, the captions render, the layout is right, and there is
simply a hole where the drawing should be. Nothing errors. It is invisible to
`AppTest`, to the element tree, and to any server-side assertion, because the
server emits perfectly good markup and the browser throws the graphic away.

This project shipped that bug three times before it was found — the hero mark,
the waiting animation, and the whole ops HUD.

`<img>` IS in the html profile, `src` is in Streamlit's `ADD_ATTR`, and
DOMPurify permits `data:` URIs on image tags. So an SVG inlined as a data URI
survives, with no network fetch — the offline guarantee holds.

The trade, which every caller must respect: **the SVG becomes its own
document.** Page CSS cannot reach inside it. Every style, every `@keyframes`,
and every colour it needs must be written into a `<style>` element inside the
`<svg>` itself. Media queries still work in there, so
`prefers-reduced-motion` is still honoured against the reader's real setting.
"""
from __future__ import annotations

import base64
import html


def svg_img(svg: str, width: str, alt: str = "", height: str = "auto") -> str:
    """Inline a standalone SVG document as a data-URI <img>.

    `svg` must be a complete document — opening tag with `xmlns`, and any CSS
    it relies on inside it. `width` is a CSS length ("186px", "100%").

    `height` defaults to "auto", which is rarely what a wide, short graphic
    wants. An <img> with `height:auto` takes its INTRINSIC RATIO from the
    viewBox, so a sparkline drawn into `0 0 100 28` and given `width:100%`
    renders at 28% of the panel width — about 140px tall in a 500px column,
    roughly five times the intended ~30px, which is what made the traces look
    like area charts. Pass an explicit CSS length for anything whose height
    should not follow its width, and set `preserveAspectRatio="none"` on the
    SVG so the drawing fills the box you asked for. Give any stroke inside it
    `vector-effect="non-scaling-stroke"`, or the non-uniform scale will make
    horizontal and vertical strokes different weights.
    """
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{b64}" '
            f'alt="{html.escape(alt)}" '
            f'style="width:{width};height:{height};display:block">')
