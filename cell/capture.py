"""Reading depth frames, and packing them small enough to ship.

A campaign runs off depth buffers written by the simulation, one 640x640 float32
array per view, 1.6 MB each. Fifty parts at five views is nearly a gigabyte, which
is why the raw captures are not in this repository and why, for a long time, the
first command in the setup instructions pointed at a file nobody who cloned the
repo actually had.

So there is a second form. The near field is quantised to uint16 and gzipped, which
is 6 times smaller, and everything past the range limit is written as "no return"
because that is what a depth sensor does with it and because the tracker discards
it anyway. The cost is measured rather than assumed: across six parts it moves a
tracked station by at most 0.08 mm, against a tracked seam accuracy of 0.68 mm and
a sensor noise floor well above both. See scripts/make_demo_capture.py.

Both forms load through load_depth, so nothing downstream needs to know which one
it was handed.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Union

import numpy as np

# The part stands within a couple of metres of the camera. Past this is the cell
# floor and the far wall, which carry no joint and which every depth sensor has
# some equivalent limit for.
RANGE_M = 4.0
_NO_RETURN = 65535
SUFFIX = ".dz.npy.gz"


def pack_depth(depth: np.ndarray, out_path: Union[str, Path], range_m: float = RANGE_M) -> int:
    """Write a depth frame in the packed form. Returns the bytes written."""
    usable = np.isfinite(depth) & (depth >= 0.0) & (depth <= range_m)
    codes = np.where(
        usable,
        np.clip(depth / range_m, 0.0, 1.0) * (_NO_RETURN - 1),
        _NO_RETURN,
    ).astype(np.uint16)
    buffer = io.BytesIO()
    np.save(buffer, codes)
    blob = gzip.compress(buffer.getvalue(), 9)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return len(blob)


def unpack_depth(blob: bytes, range_m: float = RANGE_M) -> np.ndarray:
    codes = np.load(io.BytesIO(gzip.decompress(blob)))
    depth = codes.astype(np.float32) / (_NO_RETURN - 1) * range_m
    depth[codes == _NO_RETURN] = np.nan
    return depth


def load_depth(path: Union[str, Path]) -> np.ndarray:
    """Load a depth frame in either form.

    A manifest names the raw .npy. If only the packed frame is present beside it,
    that is loaded instead, so one manifest serves a full capture and a shipped
    demo without being rewritten.
    """
    p = Path(path)
    if p.suffix == ".gz" or p.name.endswith(SUFFIX):
        return unpack_depth(p.read_bytes())
    if p.exists():
        return np.load(p)
    packed = p.with_suffix(p.suffix + ".gz")
    if packed.exists():
        return unpack_depth(packed.read_bytes())
    raise FileNotFoundError(
        f"no depth frame at {p} or {packed}. A full capture is not in this "
        f"repository; run scripts/demo.py, which uses the shipped demo capture."
    )
