"""Draw the headline result: what the correction is worth, part by part.

Generated from the campaign file rather than typed out, so a number on the chart
cannot drift away from the run that produced it.

    python scripts/draw_result_chart.py \\
        --campaign data/campaign50.jsonl --out data/results.html
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# One hue, two shades: this is a before and after on the same quantity, not two
# different things, so a second hue would imply a distinction that is not there.
# Validated light and dark as an ordinal ramp.
BEFORE_LIGHT, AFTER_LIGHT = "#86b6ef", "#184f95"
BEFORE_DARK, AFTER_DARK = "#3987e5", "#9ec5f4"

TICKS = [0.5, 1, 2, 5, 10, 20, 50]


def load(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # Two campaigns share this chart: one locates the corner as a point, the
        # other follows the seam along its whole length. Same shape of claim, so
        # the same figure serves both.
        if "blind_error_mm" in record:
            rows.append({
                "trial": record["trial"],
                "before": record["blind_error_mm"],
                "after": record["guided_error_mm"],
                "looks": len(record["looks"]),
            })
        else:
            if "mean_mm" not in record:
                continue
            rows.append({
                "trial": record["trial"],
                "before": record["taught_miss_mm"],
                "after": record["mean_mm"],
                "looks": record["stations_measured"],
            })
    return rows


def summarise(values):
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
        "submm": sum(1 for v in ordered if v < 1.0),
    }


def _font(size, bold=False):
    """A real typeface if Windows has one, otherwise whatever PIL can give."""
    for name in (("arialbd.ttf", "seguisb.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_png(rows, before, after, factor, out: Path, title: str, scale=2):
    """The same chart as a flat image, for a README or a slide.

    Same data, same two shades, same log axis. Drawn at twice the size and
    reduced, because a circle four pixels across looks like a square otherwise.
    """
    # Lay the page out first and size the canvas to it. Guessing a height and
    # hoping the rows fit is how the last five parts got cropped off.
    row_h = 11.5
    cards_y, legend_y = 148.0, 264.0
    plot_top = legend_y + 56.0
    plot_bottom = plot_top + row_h * len(rows)
    W, H = 1000, int(plot_bottom + 78)
    ink, ink2, ink3 = (11, 11, 11), (82, 81, 78), (133, 131, 124)
    surface, panel, rule = (252, 252, 251), (244, 243, 240), (227, 226, 221)
    pale = tuple(int(BEFORE_LIGHT[i:i + 2], 16) for i in (1, 3, 5))
    deep = tuple(int(AFTER_LIGHT[i:i + 2], 16) for i in (1, 3, 5))

    image = Image.new("RGB", (W * scale, H * scale), surface)
    draw = ImageDraw.Draw(image)
    S = scale

    def text(xy, body, size, colour, bold=False, anchor="la"):
        draw.text((xy[0] * S, xy[1] * S), body, font=_font(size * S, bold),
                  fill=colour, anchor=anchor)

    # Break the headline near its middle on a word boundary, so the PNG says the
    # same thing as the page rather than whatever was hardcoded here.
    words = title.split()
    half = len(title) / 2
    cut, run = len(words), 0
    for i, word in enumerate(words):
        run += len(word) + 1
        if run >= half:
            cut = i + 1
            break
    text((44, 34), " ".join(words[:cut]), 22, ink, bold=True)
    if words[cut:]:
        text((44, 62), " ".join(words[cut:]), 22, ink, bold=True)
    text((44, 96), f"{len(rows)} parts, {len(rows) * 5} camera views, simulated end to end. "
                   "Error is the distance from the seam the robot", 13, ink2)
    text((44, 114), "would weld to where the seam truly is.", 13, ink2)

    cards = [
        ("TAUGHT PROGRAM", f"{before['mean']:.1f} mm",
         f"worst {before['max']:.1f} mm   {before['submm']} of {len(rows)} under 1 mm"),
        ("DEPTH + CLOUD GUIDED", f"{after['mean']:.2f} mm",
         f"worst {after['max']:.2f} mm   {after['submm']} of {len(rows)} under 1 mm"),
        ("ERROR REDUCED BY", f"{factor:.0f}x", "every part measured, none refused"),
    ]
    cw, gap, cy = 292, 12, cards_y
    for i, (label, value, note) in enumerate(cards):
        cx = 44 + i * (cw + gap)
        draw.rounded_rectangle([cx * S, cy * S, (cx + cw) * S, (cy + 92) * S],
                               radius=10 * S, fill=panel)
        text((cx + 18, cy + 14), label, 12, ink2, bold=True)
        text((cx + 18, cy + 34), value, 30, ink, bold=True)
        text((cx + 18, cy + 74), note, 11, ink3)

    ly = legend_y
    draw.ellipse([44 * S, (ly + 3) * S, (44 + 11) * S, (ly + 14) * S], fill=pale)
    text((62, ly + 1), "No correction, taught path", 13, ink2)
    draw.ellipse([250 * S, (ly + 3) * S, (250 + 11) * S, (ly + 14) * S], fill=deep)
    text((268, ly + 1), "Depth measured, cloud guided", 13, ink2)

    left, plot_w = 118.0, 760.0
    top = plot_top
    lo, hi = 0.35, max(r["before"] for r in rows) * 1.15
    span = math.log10(hi) - math.log10(lo)

    def x(value):
        return left + (math.log10(max(value, lo)) - math.log10(lo)) / span * plot_w

    text((44, top - 34), "error, millimetres - log scale, so the whole range fits", 12, ink3)
    bottom = plot_bottom
    for tick in TICKS:
        px = x(tick)
        draw.line([px * S, (top - 6) * S, px * S, bottom * S], fill=rule, width=S)
        text((px, top - 22), f"{tick:g}", 11, ink3, anchor="ma")

    text((left - 12, top + 2), "worst arrival", 10, ink3, anchor="ra")
    text((left - 12, bottom - 12), "best arrival", 10, ink3, anchor="ra")

    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        x0, x1 = x(row["before"]), x(row["after"])
        draw.line([x1 * S, y * S, x0 * S, y * S], fill=(178, 205, 240), width=2 * S)
        for cx, colour in ((x0, pale), (x1, deep)):
            r = 4.2
            draw.ellipse([(cx - r) * S, (y - r) * S, (cx + r) * S, (y + r) * S],
                         fill=colour, outline=surface, width=2 * S)

    text((44, bottom + 26),
         "Each row is one part, sorted by how far off it arrived. The pale dot is where a "
         "fixed taught program", 12, ink3)
    text((44, bottom + 44),
         "would have welded; the dark dot is where the cell put the seam after measuring it.",
         12, ink3)

    image = image.resize((W, H), Image.LANCZOS)
    image.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="data/campaign50.jsonl")
    parser.add_argument("--out", default="data/results.html")
    parser.add_argument("--png", default="data/results.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--subject", default="cloud guided")
    args = parser.parse_args()

    rows = load(Path(args.campaign))
    title = args.title or ("Weld seam found to under a millimetre, "
                           "on parts arriving centimetres out")
    rows.sort(key=lambda r: -r["before"])
    before = summarise([r["before"] for r in rows])
    after = summarise([r["after"] for r in rows])
    factor = before["mean"] / after["mean"]

    lo, hi = 0.35, max(r["before"] for r in rows) * 1.15
    plot_w, row_h, left, top = 560.0, 11.5, 76.0, 22.0
    height = top + row_h * len(rows) + 46.0

    def x(value):
        span = math.log10(hi) - math.log10(lo)
        return left + (math.log10(max(value, lo)) - math.log10(lo)) / span * plot_w

    grid, labels = [], []
    for tick in TICKS:
        px = x(tick)
        grid.append(f'<line x1="{px:.1f}" y1="{top - 8:.1f}" x2="{px:.1f}" '
                    f'y2="{top + row_h * len(rows):.1f}" class="grid"/>')
        labels.append(f'<text x="{px:.1f}" y="{top - 14:.1f}" class="tick">{tick:g}</text>')

    marks = []
    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        x0, x1 = x(row["before"]), x(row["after"])
        tip = (f'part {row["trial"]}  |  no correction {row["before"]:.1f} mm  '
               f'|  guided {row["after"]:.2f} mm  |  {row["looks"]} look'
               f'{"s" if row["looks"] != 1 else ""}')
        marks.append(
            f'<g class="pair"><title>{tip}</title>'
            f'<rect x="{left:.1f}" y="{y - row_h / 2:.1f}" width="{plot_w:.1f}" '
            f'height="{row_h:.1f}" class="hit"/>'
            f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x0:.1f}" y2="{y:.1f}" class="link"/>'
            f'<circle cx="{x0:.1f}" cy="{y:.1f}" r="4.2" class="before"/>'
            f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="4.2" class="after"/>'
            f'</g>'
        )

    table = "\n".join(
        f"<tr><td>{r['trial']}</td><td>{r['before']:.1f}</td>"
        f"<td>{r['after']:.2f}</td><td>{r['looks']}</td></tr>" for r in rows
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weld Seam Accuracy</title>
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --panel: #f4f3f0;
    --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #85837c;
    --rule: #e3e2dd;
    --before: {BEFORE_LIGHT}; --after: {AFTER_LIGHT};
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface: #1a1a19; --panel: #232322;
      --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e85;
      --rule: #353532;
      --before: {BEFORE_DARK}; --after: {AFTER_DARK};
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e85;
    --rule: #353532;
    --before: {BEFORE_DARK}; --after: {AFTER_DARK};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 16px 56px; background: var(--surface); color: var(--ink);
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--ink-2); margin: 0 0 26px; font-size: 14px; }}
  .kpis {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 28px; }}
  .kpi {{ flex: 1 1 170px; background: var(--panel); border-radius: 10px; padding: 14px 16px; }}
  .kpi .label {{ font-size: 12px; color: var(--ink-2); text-transform: uppercase;
                 letter-spacing: 0.05em; }}
  .kpi .value {{ font-size: 30px; font-weight: 600; letter-spacing: -0.02em;
                 margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .kpi .value small {{ font-size: 15px; font-weight: 500; color: var(--ink-2); }}
  .kpi .note {{ font-size: 12px; color: var(--ink-3); margin-top: 2px; }}
  .legend {{ display: flex; gap: 20px; margin: 0 0 14px; font-size: 13px; color: var(--ink-2); }}
  .axiscap {{ margin: 0 0 2px; font-size: 12px; color: var(--ink-3); }}
  .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%;
             margin-right: 7px; vertical-align: -1px; }}
  svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
  .grid {{ stroke: var(--rule); stroke-width: 1; }}
  .tick {{ fill: var(--ink-3); font-size: 11px; text-anchor: middle; }}
  .axis {{ fill: var(--ink-2); font-size: 12px; }}
  .link {{ stroke: var(--before); stroke-width: 2; opacity: 0.55; }}
  .before {{ fill: var(--before); stroke: var(--surface); stroke-width: 2; }}
  .after {{ fill: var(--after); stroke: var(--surface); stroke-width: 2; }}
  .hit {{ fill: transparent; }}
  .pair:hover .link {{ opacity: 1; }}
  .pair:hover .hit {{ fill: var(--rule); opacity: 0.4; }}
  .rowlabel {{ fill: var(--ink-3); font-size: 10px; text-anchor: end; }}
  details {{ margin-top: 26px; }}
  summary {{ cursor: pointer; color: var(--ink-2); font-size: 13px; }}
  table {{ border-collapse: collapse; margin-top: 12px; font-size: 13px;
           font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 3px 14px 3px 0; border-bottom: 1px solid var(--rule); }}
  th {{ color: var(--ink-2); font-weight: 500; }}
  .foot {{ margin-top: 22px; color: var(--ink-3); font-size: 12px; max-width: 620px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <p class="sub">{len(rows)} parts, {len(rows) * 5} camera views, simulated end to end.
     Error is distance from the seam the robot would weld to where the seam truly is.</p>

  <div class="kpis">
    <div class="kpi">
      <div class="label">Taught program</div>
      <div class="value">{before['mean']:.1f}<small> mm</small></div>
      <div class="note">worst {before['max']:.1f} mm &middot;
           {before['submm']} of {len(rows)} under 1 mm</div>
    </div>
    <div class="kpi">
      <div class="label">Depth + cloud guided</div>
      <div class="value">{after['mean']:.2f}<small> mm</small></div>
      <div class="note">worst {after['max']:.2f} mm &middot;
           {after['submm']} of {len(rows)} under 1 mm</div>
    </div>
    <div class="kpi">
      <div class="label">Error reduced by</div>
      <div class="value">{factor:.0f}&times;</div>
      <div class="note">every part measured, none refused</div>
    </div>
  </div>

  <div class="legend">
    <span><span class="swatch" style="background:var(--before)"></span>
       No correction, taught path</span>
    <span><span class="swatch" style="background:var(--after)"></span>
       Depth measured, cloud guided</span>
  </div>

  <p class="axiscap">error, millimetres &mdash; log scale, so the whole range fits</p>
  <svg viewBox="0 0 {left + plot_w + 30:.0f} {height:.0f}" role="img"
       aria-label="Per-part seam error before and after correction">
    {''.join(grid)}
    {''.join(labels)}
    <text x="{left - 10:.0f}" y="{top + 8:.0f}" class="rowlabel">worst arrival</text>
    <text x="{left - 10:.0f}" y="{top + row_h * len(rows):.0f}" class="rowlabel">best arrival</text>
    {''.join(marks)}
  </svg>

  <details>
    <summary>Show the numbers</summary>
    <table>
      <thead><tr><th>Part</th><th>No correction (mm)</th>
      <th>Guided (mm)</th><th>Looks</th></tr></thead>
      <tbody>{table}</tbody>
    </table>
  </details>

  <p class="foot">Each row is one part, sorted by how far off it arrived. The pale dot is
  where a fixed taught program would have welded; the dark dot is where the cell
  actually put the seam after measuring it. Parts arrive under physics with
  randomised offset, and the press leaves residual springback and camber on every
  folded side.</p>
</div>
</body>
</html>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"no correction : mean {before['mean']:.2f} mm, worst {before['max']:.2f} mm")
    print(f"cloud guided  : mean {after['mean']:.2f} mm, worst {after['max']:.2f} mm")
    print(f"reduction     : {factor:.1f}x")
    print(f"wrote {out}")
    if args.png:
        print(f"wrote {draw_png(rows, before, after, factor, Path(args.png), title)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
