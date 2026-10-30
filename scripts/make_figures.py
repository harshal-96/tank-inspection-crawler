"""Redraw every figure the README shows.

A figure that is drawn once by hand and then pasted in drifts away from the run it
came from, and nobody notices because a picture does not fail a test. These are all
generated from the campaign files or from the capture that ships with the
repository, so a stale one is a command away from being correct.

The dashboard screenshots need the service running and a Chrome on the machine;
they are skipped, with a note, when either is missing. Everything else needs neither.
None of it calls a model or needs Blender.

    uvicorn service.app:app --reload      # only for the dashboard shots
    python scripts/make_figures.py
    python scripts/make_figures.py --only seam dashboard   # just those two
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"

# What --only can name, in the order they are drawn.
FIGURES = ("seam", "cell", "tracked", "ultrasound", "dashboard", "boundary")

CHROMES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def run(args: list[str]) -> bool:
    result = subprocess.run([sys.executable, *args], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-800:])
        print(result.stderr[-800:])
    return result.returncode == 0


def find_chrome() -> str | None:
    for path in CHROMES:
        if Path(path).exists():
            return path
    return shutil.which("google-chrome") or shutil.which("chromium")


def shoot(url: str, out: Path, width: int, height: int) -> bool:
    chrome = find_chrome()
    if not chrome:
        print(f"  skipped {out.name}: no Chrome found for a headless screenshot")
        return False
    subprocess.run([chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={width},{height}",
                    "--virtual-time-budget=11000",
                    f"--screenshot={out}", url],
                   capture_output=True, text=True)
    return out.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES),
                        help="redraw only these figures; by default, all of them")
    args = parser.parse_args()
    want = set(args.only)
    OUT.mkdir(parents=True, exist_ok=True)

    if "seam" in want:
        print("the headline result, from the fifty-part campaign")
        run(["scripts/draw_result_chart.py",
             "--campaign", "data/seam_campaign.jsonl",
             "--out", "docs/figures/seam_result.html",
             "--png", "docs/figures/seam_result.png",
             # Named, because the seam tracking is geometry and calling it cloud
             # guided would credit a model with work it did not do.
             "--subject", "depth measured, no model in the loop"])

    if "cell" in want:
        print("the cell, to scale, from the constants the simulation builds from")
        run(["scripts/draw_cell.py", "--out", "docs/figures/cell_layout.png"])

    if "tracked" in want:
        print("one part tracked from its depth frame, off the shipped capture")
        run(["scripts/track_on_model.py", "--manifest", "data/demo/manifest.json",
             "--part", "3", "--out", "docs/figures/seam_tracked.png"])

    if "ultrasound" in want:
        print("the simulated gauge through a bare wall and a lined one")
        run(["scripts/draw_ultrasound.py", "--out", "docs/figures/ultrasound_lining.png"])

    if not want & {"dashboard", "boundary"}:
        return 0

    try:
        with urllib.request.urlopen(f"{args.api}/healthz", timeout=3):
            live = True
    except (urllib.error.URLError, OSError):
        live = False
    if not live:
        print(f"the dashboard: skipped, nothing answering at {args.api}")
        print("  start it with: uvicorn service.app:app")
        return 0

    if "dashboard" in want:
        # The tank page, because it is the page the dashboard opens on and the
        # half of the project the README leads with. An earlier shot of the
        # welding summary page outlived two changes to that page.
        print("the dashboard's tank page")
        shoot(f"{args.api}/#/tank", OUT / "dashboard.png", 1440, 1150)

    if "boundary" not in want:
        return 0
    print("the boundary test, cropped out of the diagnosis page")
    full = OUT / "_diagnosis_full.png"
    if shoot(f"{args.api}/#/diagnosis", full, 1440, 3300):
        try:
            from PIL import Image
            # The card's position at a 1440 wide viewport, measured in the
            # browser rather than guessed. If the page grows another card above
            # it, this crop slides and the figure is visibly wrong, which is the
            # failure mode worth having: a silently stale figure is worse.
            Image.open(full).crop((268, 2129, 1401, 3278)).save(OUT / "boundary_test.png")
            print(f"  wrote {OUT / 'boundary_test.png'}")
        finally:
            full.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
