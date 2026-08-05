"""What Haven is allowed to show a stranger.

`app.py` is the one surface a member of the public sees, and the rule it works
to is that nothing on it should be machinery. That is easy to state and easy to
erode: every one of the items checked below was on the page at some point, each
added for a good local reason, and together they turned a page meant for a
worried patient into a status readout — model identifiers in the sidebar, an
indexed-passage count in the hero, "retrieved passages · 6" over the sources,
the corpus filename under each one, and the raw transport error when the model
was unreachable.

None of it changes what a reader should do next. Worse, precision about the
apparatus invites someone to defer to the answer instead of checking it, which
is the opposite of what this page is for — and the same detail is a click away
in `cli.py status` and `dashboard.py`, where an operator will actually look.

A SECOND rule used to live here: that Haven and the ops dashboard must never
look alike, because a legal answer rendered in sci-fi chrome reads as a toy and
takes its disclaimer down with it. Haven is now the console — same palette, same
scanlines, same monospace — by explicit request, so those checks are gone and
the ones below took their place.

That was a real risk, not a stylistic preference, so it is now guarded
structurally instead of chromatically. Cyan is the colour of everything on a
console that is merely working, and the eye stops reading it within seconds; so
the not-legal-advice line is drawn in the ALERT hue, at panel weight, and its
sentence is never set in tracked-out capitals. Capitals are for micro-labels.
A legal caveat styled as a HUD label reads as decoration and gets skipped, which
is the failure the old rule existed to prevent. Those two properties are pinned
below — if the advisory ever drifts into the ambient accent, or into the label
voice, nothing else on the page is holding this up.

These are static and offline: they read the source and the chrome helpers, and
never start Streamlit.

Run:  python tests/test_public_surface.py
"""
from __future__ import annotations

import base64
import re
import sys
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import chrome  # noqa: E402

APP = (ROOT / "app.py").read_text(encoding="utf-8")
# Strip comments and docstrings: this file's own prose names the very things it
# forbids, and so does app.py's docstring explaining why they are gone.
_CODE = "\n".join(
    ln for ln in APP.splitlines() if not ln.lstrip().startswith("#"))
_CODE = re.sub(r'""".*?"""', "", _CODE, flags=re.S)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def run() -> int:
    r = []

    # --- model identifiers ---------------------------------------------------
    # "Language model: QuantTrio/Qwen3.6-27B-AWQ · Search: BAAI/bge-base-en-v1.5
    # + BM25 · Re-ranking: BAAI/bge-reranker-base" was the sidebar's last line.
    for name in ("LLM_MODEL", "EMBED_MODEL", "RERANK_MODEL"):
        r.append(check(f"the page does not name {name}",
                       f"config.{name}" not in _CODE))
    r.append(check("no bare model identifiers in the markup",
                   not re.search(r"BAAI/|Qwen|bge-|BM25", _CODE),
                   ""))

    # --- counts --------------------------------------------------------------
    # A passage count is a number the reader cannot act on and cannot check.
    r.append(check("the hero does not state an indexed-passage count",
                   "vectorstore.count" not in _CODE))
    meta = chrome.hero_meta()
    r.append(check("hero_meta carries no digits at all",
                   not re.search(r"\d", re.sub(r"<[^>]+>", "", meta)),
                   f"({re.sub(r'<[^>]+>', ' ', meta).split()})"))
    r.append(check("hero_meta takes no arguments to state one with",
                   chrome.hero_meta.__code__.co_argcount == 0))

    # --- the corpus filename -------------------------------------------------
    # "_fetched/doh/doh-ao-2008-0001-irr-of-ra-9439-…txt" under every card. A
    # reader cannot open it and cannot verify anything with it.
    card = chrome.source_card(1, "Republic Act No. 9439", "SEC. 2.")
    r.append(check("a source card shows the law", "Republic Act No. 9439" in card))
    r.append(check("a source card shows the section", "SEC. 2." in card))
    r.append(check("a source card shows no filename or path",
                   ".txt" not in card and "_fetched" not in card
                   and "/" not in re.sub(r"</?[^>]+>", "", card)))
    r.append(check("source_card cannot be passed a filename",
                   chrome.source_card.__code__.co_argcount == 3,
                   f"(takes {chrome.source_card.__code__.co_argcount})"))
    r.append(check("no monospace path style survives in the chrome",
                   "hv-file" not in chrome.chrome_css()))

    # --- jargon --------------------------------------------------------------
    r.append(check('no "retrieved passages" label over the sources',
                   "retrieved passages" not in _CODE.lower()))
    r.append(check("the waiting message does not count passages",
                   not re.search(r"Reading \{?len\(sources\)", _CODE)))

    # --- failures are explained, not dumped ----------------------------------
    # `msg` is llm.is_available()'s diagnostic — a transport error with a URL
    # and port in it. Fine in cli.py status; not an answer to "why isn't this
    # working" for someone who came here about a hospital bill.
    r.append(check("the LLM-down path does not print the raw diagnostic",
                   "st.caption(msg)" not in _CODE and "st.write(msg)" not in _CODE))
    r.append(check("the LLM-down path still tells the reader what to do",
                   "Try again" in APP))

    # --- what must NOT be stripped -------------------------------------------
    # De-cluttering must never reach the two things this page exists to make
    # unmissable.
    r.append(check("the not-legal-advice line survives in the hero",
                   "does not give legal advice" in APP))
    r.append(check("the per-answer disclaimer survives",
                   "not legal advice" in APP))
    r.append(check("the sources section survives",
                   "Where this comes from" in APP))
    # Matched on a fragment that survives wrapping: the sentence is split
    # across two adjacent string literals in the source, so the full phrase
    # never appears contiguously.
    r.append(check("the check-the-law instruction survives",
                   "law itself before you rely" in APP))
    # The sidebar's "It runs entirely on this computer. Nothing you type is
    # sent anywhere." was REMOVED by request, so this asserts its absence
    # rather than its presence. It is the one claim on the page that a change
    # of deployment can silently falsify: it is true of the documented local
    # setup and false the moment Haven is hosted for real users, and unlike a
    # wrong legal citation nobody reading it can tell.
    r.append(check("the local-only privacy claim is gone",
                   "Nothing you type is sent anywhere" not in APP
                   and "runs entirely on this computer" not in APP))
    # The hero's "PRIVATE BY DESIGN" chip made a weaker form of the same claim
    # and went with it. Checked against the rendered strip AND the page source,
    # since it lived in chrome.py rather than app.py.
    r.append(check("no privacy claim survives in the hero strip",
                   not re.search(r"privat|local|never leaves|on this computer",
                                 chrome.hero_meta(), re.I),
                   f"({re.sub(r'<[^>]+>', ' ', chrome.hero_meta()).split()})"))
    r.append(check("no privacy claim anywhere on the page",
                   not re.search(r"nothing (you|is) [a-z ]*sent|never leaves "
                                 r"(this|your) (computer|machine)", APP, re.I)))

    # --- one register, single-sourced ---------------------------------------
    # Haven is the console now. The thing worth pinning is not that it looks
    # like one but that it cannot drift out of step with the other surface: the
    # palette is imported, not restated, and the stylesheet is built ON hud_css()
    # rather than beside it. Two copies of a palette that must agree is exactly
    # how the two pages would diverge by accident.
    from ui import hud
    css = chrome.chrome_css()
    r.append(check("Haven shares the console's accent",
                   chrome.ACCENT.lower() == hud.CYAN.lower(),
                   f"({chrome.ACCENT})"))
    r.append(check("Haven's chrome is built on the console stylesheet, not a copy",
                   hud.hud_css() in css))

    # --- the caveat must not be swallowed by the chrome ----------------------
    # These two replace the old "must not look alike" rule and carry the same
    # burden. If either goes, the disclaimer is competing with the ambient
    # accent for attention and losing.
    adv = chrome.advisory("<b>Haven does not give legal advice.</b> Body.")
    adv_css = css[css.index(".hv-advisory {"):css.index(".hv-meta {")]
    r.append(check("the advisory is a panel of its own, not a caption",
                   'class="hv-advisory"' in adv and ".hv-advisory {" in css))
    r.append(check("the advisory is drawn in the alert hue, not the accent",
                   hud.ALERT.lower() in adv_css.lower()
                   and hud.CYAN.lower() not in adv_css.lower(),
                   f"(alert {hud.ALERT}, accent {hud.CYAN})"))
    # The label above it may be capitals; the sentence may not. `text-transform`
    # must appear in the label rule and nowhere in the body rule.
    body_rule = adv_css[adv_css.index(".hv-advisory p {"):]
    body_rule = body_rule[:body_rule.index("}")]
    r.append(check("the caveat sentence is not set in tracked-out capitals",
                   "text-transform" not in body_rule
                   and "letter-spacing: 0" in body_rule,
                   f"({' '.join(body_rule.split())})"))
    # And it reaches the page through advisory() rather than label() or a
    # caption — the panel is what gives it weight against the chrome.
    at = _CODE.find("chrome.advisory(")
    r.append(check("the caveat reaches the page through advisory()",
                   at != -1
                   and "does not give legal advice" in _CODE[at:at + 400]))

    # --- the mark ------------------------------------------------------------
    # Same trap as every other graphic in this repo: a bare <svg> handed to
    # st.html is silently deleted by DOMPurify, with no error and nothing wrong
    # in the element tree. See ui/svg.py. It shipped three times.
    from ui import robot
    for name, markup in (("hero", robot.robot_hero()),
                         ("hero, alert state", robot.robot_hero(alert=True)),
                         ("working", robot.robot_thinking("Searching…"))):
        svg = _decode(markup)
        r.append(check(f"the {name} mark leaves as a data-URI <img>",
                       'src="data:image/svg+xml;base64,' in markup
                       and "<svg" not in markup.replace(svg, "")))
        r.append(check(f"the {name} mark is a well-formed document",
                       bool(svg) and _parses(svg)))
        # Page CSS cannot reach into an <img>, so the reduced-motion query has
        # to be inside the drawing or it is not honoured at all.
        r.append(check(f"the {name} mark honours prefers-reduced-motion",
                       "prefers-reduced-motion" in svg))

    # --- avatars -------------------------------------------------------------
    # Streamlit fills the assistant avatar with the theme's ORANGE, which on
    # this palette is the amber WARNING hue — so every answer opened with a
    # caution block beside it. And the avatar's glyph is an icon ligature: set
    # a font-family on it and `smart_toy` prints as text (see hud_css()).
    av = css[css.index('[data-testid^="stChatMessageAvatar"] {'):]
    av = av[:av.index("}")]
    r.append(check("the assistant avatar is not left on the warning hue",
                   hud.INK_2.lower() in av.lower()
                   and hud.AMBER.lower() not in av.lower()))
    r.append(check("the avatar rule sets no font-family",
                   "font-family" not in av,
                   f"({' '.join(av.split())})"))

    # --- the boot panel ------------------------------------------------------
    # ~6s of embedder and reranker on first use used to be paid against a blank
    # page, which reads as broken rather than as busy.
    boot = chrome.booting()
    boot_svg = _decode(boot)
    r.append(check("the boot panel leaves as a data-URI <img>",
                   'src="data:image/svg+xml;base64,' in boot
                   and "<svg" not in boot.replace(boot_svg, "")))
    r.append(check("the boot panel honours prefers-reduced-motion",
                   "prefers-reduced-motion" in boot_svg))
    r.append(check("the boot panel names no machinery",
                   not re.search(r"BAAI/|Qwen|bge-|BM25|model|embed|rerank",
                                 re.sub(r"<[^>]+>", " ", boot), re.I)))
    r.append(check("the boot panel is shown BEFORE the slow work, not after",
                   _CODE.index("chrome.booting()") < _CODE.index("\nwarm_up()")))
    r.append(check("the boot panel is cleared once the work finishes",
                   "boot.empty()" in _CODE))
    r.append(check("the chrome is injected before the panel it styles",
                   _CODE.index("chrome.chrome_css()")
                   < _CODE.index("chrome.booting()")))

    # Colour is never the only carrier of a state. The mark turns red when the
    # model is unreachable; the sentence saying so is emitted BEFORE it.
    r.append(check("the alert mark differs from the resting one",
                   robot.robot_hero(alert=True) != robot.robot_hero()))
    r.append(check("the unreachable state is stated in words above the mark",
                   _CODE.index("isn't running on this computer")
                   < _CODE.index("robot_hero(")))

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


def _decode(markup: str) -> str:
    """The SVG document inside a data-URI <img>."""
    m = re.search(r'base64,([^"]+)"', markup)
    return base64.b64decode(m.group(1)).decode("utf-8") if m else ""


def _parses(svg: str) -> bool:
    try:
        xml.dom.minidom.parseString(svg)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(run())
