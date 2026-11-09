"""Render the RigMark progress chart as light and dark SVGs: python scripts/rigmark_progress_chart.py docs/img."""
import sys
from pathlib import Path

LABELS = [("No drafter", "est."), ("Round 1", "Oct 15"), ("Round 2a", "Oct 16"), ("Round 2b", "Oct 16"),
          ("v79", "Oct 31"), ("v80", "Nov 1"), ("v81", "Nov 5")]
BASE = 28.4
SERIES = [  # name, key, marker, data, reference
    ("Structured", "s", "square", [BASE, 38.6, 43.0, 71.0, 74.3, 79.0, 84.2], 91.0),
    ("Code", "c", "triangle", [BASE, 38.5, 45.3, None, 59.9, 60.2, 63.3], 66.9),
    ("Prose", "p", "circle", [BASE, 27.2, 31.0, None, 35.8, 35.3, 36.0], 39.0),
]
THEMES = {
    "light": dict(s="#2a78d6", c="#eb6834", p="#1baf7a", ink="#0b0b0b", sec="#52514e", muted="#6f6d68",
                  grid="#e1e0d9", axis="#c3c2b7", surf="#ffffff"),
    "dark": dict(s="#3987e5", c="#d95926", p="#199e70", ink="#f0efec", sec="#c3c2b7", muted="#9a9890",
                 grid="#2c2c2a", axis="#4a4a46", surf="#0d1117"),
}
W, H = 960, 560
L, R, T, B = 64, 770, 124, 462  # plot area
YMIN, YMAX = 20, 100
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def x(i):
    return L + (R - L) * (i + 0.5) / len(LABELS)


def y(v):
    return B - (B - T) * (v - YMIN) / (YMAX - YMIN)


def marker(kind, cx, cy, fill, stroke, r=5.0):
    if kind == "square":
        return f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" rx="1.5" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    if kind == "triangle":
        pts = f"{cx:.1f},{cy - r * 1.2:.1f} {cx + r * 1.1:.1f},{cy + r * 0.8:.1f} {cx - r * 1.1:.1f},{cy + r * 0.8:.1f}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2" stroke-linejoin="round"/>'
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def render(t):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'font-family="{FONT}">',
         '<title>RigMark decode speed: Enntity\'s from-scratch GLM-5.3-Flash drafter, no drafter to v81, '
         'against the reference drafter</title>',
         f'<rect width="{W}" height="{H}" fill="{t["surf"]}"/>',
         f'<text x="{L - 40}" y="40" font-size="20" font-weight="600" fill="{t["ink"]}">'
         'From no drafter to v81: RigMark decode speed</text>',
         f'<text x="{L - 40}" y="64" font-size="14" fill="{t["sec"]}">DFlash2 drafter for GLM-5.3-Flash, trained '
         'from scratch on 2x DGX Spark · dashed lines: reference drafter</text>']
    # legend
    lx = L - 40
    for name, key, kind, _, _ in SERIES:
        o.append(f'<line x1="{lx}" y1="90" x2="{lx + 24}" y2="90" stroke="{t[key]}" stroke-width="2.5"/>')
        o.append(marker(kind, lx + 12, 90, t[key], t["surf"], 4.5))
        o.append(f'<text x="{lx + 32}" y="95" font-size="13" fill="{t["sec"]}">{name}</text>')
        lx += 32 + len(name) * 8 + 22
    o.append(f'<line x1="{lx}" y1="90" x2="{lx + 24}" y2="90" stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{lx + 32}" y="95" font-size="13" fill="{t["sec"]}">reference</text>')
    lx += 32 + 9 * 8 + 22
    o.append(marker("circle", lx + 6, 90, t["surf"], t["muted"], 4.5))
    o.append(f'<text x="{lx + 18}" y="95" font-size="13" fill="{t["sec"]}">estimate</text>')
    # grid and axes
    for v in range(YMIN, YMAX + 1, 20):
        o.append(f'<line x1="{L}" y1="{y(v):.1f}" x2="{R}" y2="{y(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{L - 10}" y="{y(v) + 4:.1f}" font-size="12" text-anchor="end" fill="{t["muted"]}">{v}</text>')
    o.append(f'<text transform="translate({L - 44},{(T + B) / 2:.0f}) rotate(-90)" font-size="12" text-anchor="middle" '
             f'fill="{t["muted"]}">decode tok/s</text>')
    o.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{t["axis"]}" stroke-width="1"/>')
    for i, (a, b) in enumerate(LABELS):
        o.append(f'<text x="{x(i):.1f}" y="{B + 22}" font-size="13" text-anchor="middle" fill="{t["sec"]}">{a}</text>')
        o.append(f'<text x="{x(i):.1f}" y="{B + 39}" font-size="12" text-anchor="middle" fill="{t["muted"]}">{b}</text>')
    # reference lines, then series
    for _, key, _, _, ref in SERIES:
        o.append(f'<line x1="{L}" y1="{y(ref):.1f}" x2="{R}" y2="{y(ref):.1f}" stroke="{t[key]}" stroke-width="1.5" '
                 'stroke-dasharray="6 5" opacity="0.85"/>')
    for name, key, kind, data, ref in SERIES:
        pts = [(x(i), y(v)) for i, v in enumerate(data) if v is not None]
        path = " ".join(f"{'M' if j == 0 else 'L'}{px:.1f},{py:.1f}" for j, (px, py) in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{t[key]}" stroke-width="2.5" stroke-linejoin="round" '
                 'stroke-linecap="round"/>')
        for i, v in enumerate(data):
            if v is None:
                continue
            first, last = i == 0, i == len(data) - 1
            o.append(marker(kind, x(i), y(v), t["surf"] if first else t[key], t[key] if first else t["surf"],
                            6.5 if last else 5.0))
    # end labels (structured, code, prose stack well apart at these values)
    ex = R + 12
    for name, key, _, data, ref in SERIES:
        v = data[-1]
        o.append(f'<text x="{L + 6}" y="{y(ref) - 6:.1f}" font-size="12" fill="{t[key]}">reference {ref:.1f}</text>')
        o.append(f'<text x="{ex}" y="{y(v) + 2:.1f}" font-size="13" font-weight="600" fill="{t["ink"]}">'
                 f'{name} {v:.1f}</text>')
        o.append(f'<text x="{ex}" y="{y(v) + 18:.1f}" font-size="12" fill="{t["sec"]}">'
                 f'{round(100 * v / ref)}% of reference</text>')
        o.append(f'<text x="{ex}" y="{y(v) + 33:.1f}" font-size="12" fill="{t["sec"]}">{v / BASE:.1f}x no drafter</text>')
    o.append(f'<text x="{L - 40}" y="{H - 30}" font-size="12" fill="{t["muted"]}">Atlas on two GB10 nodes, NVFP4 '
             'GLM-5.3-Flash, greedy, one request at a time; engine decode tok/s. Reference drafter: the best public</text>')
    o.append(f'<text x="{L - 40}" y="{H - 13}" font-size="12" fill="{t["muted"]}">DFlash2 drafter for GLM-5.3-Flash. '
             'No drafter is estimated from the engine\'s step-cost model; round 2b measured structured only.</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name, theme in THEMES.items():
    (out / f"rigmark-progress-{name}.svg").write_text(render(theme))
print("wrote", sorted(p.name for p in out.glob("rigmark-progress-*.svg")))
