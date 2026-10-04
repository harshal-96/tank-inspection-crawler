"""Encode a rendered PNG sequence into an mp4.

    python scripts/encode_shot.py
    python scripts/encode_shot.py --frames tank/data/shots/seam_frames --fps 24

This Blender is built without video output, so shots render as numbered PNGs
and are encoded here. That is the better way round: a render that stops part
way can be resumed, and the frames survive a bad encode.

Refuses to encode a sequence with holes in it, because a missing frame does not
announce itself in the output. It just makes the motion stutter once, in a way
nobody notices until the video is submitted.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES = ROOT / "tank" / "data" / "shots" / "seam_frames"


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default=str(DEFAULT_FRAMES))
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf", type=int, default=16,
                        help="lower is better quality; 16 is visually lossless")
    parser.add_argument("--allow-holes", action="store_true")
    args = parser.parse_args()

    frames = Path(args.frames)
    found = sorted(frames.glob("f_*.png"))
    if not found:
        print(f"no frames in {frames}")
        return 1

    numbers = sorted(int(re.search(r"(\d+)", p.stem).group(1)) for p in found)
    first, last = numbers[0], numbers[-1]
    holes = [n for n in range(first, last + 1) if n not in set(numbers)]
    print(f"{len(numbers)} frames, {first} to {last}, {len(holes)} missing")
    if holes and not args.allow_holes:
        print(f"  missing: {holes[:12]}{' ...' if len(holes) > 12 else ''}")
        print("refusing to encode a sequence with holes; render them or pass "
              "--allow-holes")
        return 2

    out = Path(args.out) if args.out else frames.parent / f"{frames.name}.mp4"
    cmd = [
        ffmpeg_exe(), "-y",
        "-framerate", str(args.fps),
        "-start_number", str(first),
        "-i", str(frames / "f_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(args.crf),
        # Even dimensions and yuv420p, or half the players in the world and
        # every social upload will refuse it or show a green frame.
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart",
        str(out),
    ]
    print(" ".join(cmd[:1]) + f" ... -> {out}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-2500:])
        return result.returncode

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out}  {size_mb:.1f} MB  "
          f"{len(numbers) / args.fps:.1f} s at {args.fps} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
