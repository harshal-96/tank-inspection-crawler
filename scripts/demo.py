"""Fill an empty cell from the capture that ships with this repository.

The point of this script is that it works on a fresh clone. The full captures are
about 900 MB and are not in git, so for a while the first command in the setup
instructions named a manifest that nobody else had, and anyone following the README
hit a missing file before they ever saw the cell run. data/demo is six parts of
real captured depth, packed small enough to carry, and this runs the actual
pipeline over it rather than replaying saved numbers.

It does as much as it can with what you have:

  seam tracking and weld scheduling   always, from the depth frames alone. No
                                      GPU, no network, no API key, a few seconds.
  look-and-judge routing              needs a Token Factory key, because the
                                      decision is a model call.
  post-weld inspection                needs a key and a vision model.
  cited corrective briefs             needs a key and a Tavily key, because the
                                      answer is in a welding code rather than in
                                      the measurement.

Results are written under data/demo_run/ rather than over the campaign files in
data/, which are the evidence for the numbers in the README and are not the
demo's to overwrite.

    uvicorn service.app:app --reload        # in another terminal
    python scripts/demo.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = "data/demo/manifest.json"
OUT = "data/demo_run"


def have_key() -> bool:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return bool(os.getenv("INFERENCE_API_KEY"))


def service_up(api: str) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{api}/healthz", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def run(label: str, args: list[str]) -> bool:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        print(f"  {label}: stopped with code {result.returncode}")
        return False
    print(f"  {label}: {time.time() - started:.1f} s")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--offline", action="store_true",
                        help="Geometry only. Skip everything that calls a model.")
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    if not manifest.exists():
        print(f"No capture at {manifest}.")
        print("This repository ships one at data/demo. If it is missing, the clone "
              "is incomplete.")
        return 1

    if not service_up(args.api):
        print(f"The service is not answering at {args.api}.")
        print("\nStart it in another terminal, then run this again:\n")
        print("    uvicorn service.app:app --reload\n")
        return 1

    keyed = (not args.offline) and have_key()
    print(f"capture   {args.manifest}")
    print(f"service   {args.api}")
    print(f"cloud     {'yes' if keyed else 'no, running the offline half only'}")

    ok = run("Tracking the seam on every part, and scheduling the torch",
             ["scripts/run_seam_campaign.py", "--manifest", args.manifest,
              "--api", args.api, "--out", f"{OUT}/seam_campaign.jsonl"])
    if not ok:
        return 1

    if keyed:
        run("Deciding whether each look can be trusted, through the view planner",
            ["scripts/run_campaign.py", "--manifest", args.manifest,
             "--api", args.api, "--routed", "--out", f"{OUT}/campaign.jsonl"])
        run("Judging the finished bead",
            ["scripts/run_inspection.py", "--manifest", args.manifest,
             "--api", args.api, "--out", f"{OUT}/inspections.jsonl"])
        if os.getenv("TAVILY_API_KEY"):
            run("Writing corrective briefs against published welding guidance",
                ["scripts/raise_briefs.py", "--api", args.api,
                 "--inspections", f"{OUT}/inspections.jsonl"])
        else:
            print("\nSkipping corrective briefs: they cite published guidance "
                  "and need TAVILY_API_KEY.")

    print(f"\n{'=' * 72}")
    print("The cell is populated. Open the dashboard:\n")
    print(f"    {args.api}\n")
    if not keyed:
        print("The seam pages and the studies are live. The routing, inspection and")
        print("supervisor pages need a Token Factory key in .env; see README Setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
