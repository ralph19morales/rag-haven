"""Computable checks on a chart palette. A design-time tool, not an eval.

Lives beside the constants it checks (`hud.SERIES`) rather than in `evals/`,
which is reserved for measurements of the RUNNING system. Nothing imports this;
run it by hand when you change a series colour.

    python ui/validate_palette.py "#3fc0e6,#8f78e0" --mode dark --surface "#0e1822"

A faithful port of the data-viz skill's `validate_palette.js` — same thresholds,
the same Machado-Oliveira-Fernandes (2009) severity-1.0 CVD matrices, the same
OKLab/OKLCH conversions — because there is no node on this box and eyeballing
whether two colours are far enough apart is exactly what the check exists to
replace. Verified against the reference light palette, which reproduces its
documented worst-pair figure of 19.6 here.

EXPECTED RESULT for the shipped pair: everything PASSes except the dark
lightness band, which slot 1 clears by 0.08. That deviation is deliberate and
argued where the constants are defined (`ui/hud.py`) — the band is calibrated
for a dark-GREY chart surface and this page is near-black, so the functional
test is the contrast check, which passes with room. Do not "fix" it by dimming
the series into the band without looking at the page: it goes muddy against the
cyan chrome. If the HUD ground is ever lightened, re-run and re-decide.
"""
from __future__ import annotations

import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [max(0.0, min(1.0, M[i][0] * r + M[i][1] * g + M[i][2] * b))
            for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or {"light": "#fcfcfb", "dark": "#1a1a19"}[mode]
    lo, hi = BAND[mode]
    ok = True
    rows = []

    off = [(c, round(oklch(c)[0], 3)) for c in palette
           if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    rows.append(("Lightness band", "PASS" if not off else "FAIL",
                 f"outside {lo}-{hi}: {off}" if off
                 else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette
            if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rows.append(("Chroma floor", "PASS" if not lowc else "FAIL",
                 f"below floor (reads gray): {lowc}" if lowc
                 else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]
    label = "all-pairs" if pairs == "all" else "adjacent"

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan")
               for i, j in pairlist), default=99)
    wd = worst[0] if worst else 99
    state = "PASS" if wd >= CVD_TARGET else ("FLOOR" if wd >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL"
    rows.append(("CVD separation", state,
                 f"worst {label} {worst[3]}<->{worst[2]} dE {wd:.1f} "
                 f"({worst[1]}) - tritan {tri:.1f}" if worst else "n/a"))

    nworst = None
    for i, j in pairlist:
        d = delta_e(palette[i], palette[j])
        if nworst is None or d < nworst[0]:
            nworst = (d, palette[i], palette[j])
    nd = nworst[0] if nworst else 99
    nstate = "PASS" if nd >= NORMAL_FLOOR else "FAIL"
    ok &= nstate == "PASS"
    rows.append(("Normal-vision floor", nstate,
                 f"worst {label} {nworst[2]}<->{nworst[1]} dE {nd:.1f} (normal)"
                 if nworst else "n/a"))

    weak = [(c, round(contrast(c, surface), 2)) for c in palette
            if contrast(c, surface) < CONTRAST_MIN]
    rows.append(("Contrast vs surface", "PASS" if not weak else "WARN",
                 f"below {CONTRAST_MIN}:1 vs {surface}: {weak}" if weak
                 else f"all >= {CONTRAST_MIN}:1 vs {surface}"))

    w = max(len(r[0]) for r in rows)
    for name, st, detail in rows:
        print(f"  {name:<{w}}  {st:<5}  {detail}")
    print(f"  => {'OK' if ok else 'FAILS'}")
    return ok


if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    mode = "dark"
    surface = None
    pairs = "adjacent"
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--mode":
            mode = args[i + 1]
        elif a == "--surface":
            surface = args[i + 1]
        elif a == "--pairs":
            pairs = args[i + 1]
    print(f"palette={pal} mode={mode} surface={surface or 'default'} pairs={pairs}")
    sys.exit(0 if validate(pal, mode, surface, pairs) else 1)
