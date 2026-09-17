"""Forming error: what is left on a folded side after the shop has done its best.

Both of the errors here are already treated, at different stages, which is exactly
why what remains is small and why it cannot be driven to zero.

Camber is treated first and upstream. Coil comes off the mill with residual stress
and an edge that is not straight, and it is run through a roller or stretch
leveller before it is blanked: bent past yield and back so the fibre length comes
out even. That takes out most of it, not all, and what survives is still in the
sheet when the blank is cut.

Springback is treated later and at the machine. It cannot be pre-treated, because
it is the elastic part of the bend itself: the steel is bent past yield, and when
the ram lifts, the elastic strain recovers and the angle opens. The press brake
compensates for it as it bends, by overbending, by bottoming or coining, and on a
modern machine by measuring the angle in the die and pressing again until the
angle after recovery is inside tolerance.

What survives both is a residual. Adaptive angle measurement holds a couple of
tenths of a degree, and a tenth with dynamic deflection compensation on the ram,
and that is where it stops being worth chasing: the yield strength moves from coil
to coil, so the compensation that was right for the last lot is slightly wrong for
this one.

A couple of tenths of a degree sounds like nothing until it is multiplied by the
height of the side. A side 450 mm tall leaning out by two tenths puts its top edge
more than a millimetre off nominal, and a weld has to land within a few tenths of
the seam. So the press is in spec, the leveller did its job, and the seam is still
in the wrong place. That is the gap this cell exists to close, and it is a robot
problem rather than a press problem.

Plain arithmetic, no simulation dependency, so the Isaac process, the service and
the tests can all import it. Seeded, so a campaign replays identically.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

# Residual bend angle, after the brake has compensated. Adaptive angle measurement
# on a good machine holds a couple of tenths of a degree and a tenth at its best,
# so a tenth of a degree typical and three tenths at the limit is a press running
# properly, not a press that needs repairing.
LEAN_SIGMA_DEG = 0.11
LEAN_LIMIT_DEG = 0.30

# Residual edge bow between the base and the rim, from the camber the leveller did
# not take out. A levelled coil is most of the way flat, so a side this tall picks
# up well under a millimetre of it.
BOW_SIGMA_MM = 0.40
BOW_LIMIT_MM = 1.10


@dataclass(frozen=True)
class PanelForming:
    """How one folded side sits, against where the drawing puts it.

    lean_deg is how far the bend sprang open: positive leans the side outward, so
    the error grows with height and is zero at the base, where the fold holds it.

    bow_mm is how far the free edge bows out of straight between base and rim,
    worst at mid height and zero at both ends, which is the shape a cambered strip
    takes once it is folded.
    """

    lean_deg: float = 0.0
    bow_mm: float = 0.0

    def offset_m(self, z_m: float, height_m: float) -> float:
        """How far the panel face is from nominal at height z, in metres.

        Outward is positive. z_m is measured up from the base, height_m is how far
        the side stands above it.
        """
        if height_m <= 0.0:
            raise ValueError("height_m must be positive")
        z = min(max(z_m, 0.0), height_m)
        lean = math.tan(math.radians(self.lean_deg)) * z
        bow = (self.bow_mm / 1000.0) * math.sin(math.pi * z / height_m)
        return lean + bow

    def worst_mm(self, height_m: float) -> float:
        """The largest offset anywhere up the side, in millimetres."""
        samples = (self.offset_m(height_m * i / 40.0, height_m) for i in range(41))
        return 1000.0 * max(abs(s) for s in samples)


@dataclass(frozen=True)
class CornerForming:
    """The two sides that meet at one corner, and so the seam they make.

    Side A is the one lying in the plane of constant x, side B the one in constant
    y. Their offsets move the seam in x and y respectively, which is why the seam
    is neither straight nor vertical on a real part.
    """

    a: PanelForming = PanelForming()
    b: PanelForming = PanelForming()

    def seam_offset_m(self, z_m: float, height_m: float) -> tuple[float, float]:
        """Where the seam sits at height z, as (dx, dy) from nominal, in metres."""
        return (
            self.a.offset_m(z_m, height_m),
            -self.b.offset_m(z_m, height_m),
        )

    def seam_wander_mm(self, height_m: float, samples: int = 41) -> float:
        """How far the seam strays from the straight line between its own ends.

        This is the number a taught program cannot see. A seam that is merely
        shifted can be corrected by touching one point; a seam that wanders has to
        be followed along its length.
        """
        points = []
        for i in range(samples):
            z = height_m * i / (samples - 1)
            dx, dy = self.seam_offset_m(z, height_m)
            points.append((dx, dy, z))

        first, last = points[0], points[-1]
        axis = (last[0] - first[0], last[1] - first[1], last[2] - first[2])
        length = math.sqrt(sum(c * c for c in axis))
        if length == 0.0:
            return 0.0
        unit = tuple(c / length for c in axis)

        worst = 0.0
        for point in points:
            rel = tuple(point[i] - first[i] for i in range(3))
            along = sum(rel[i] * unit[i] for i in range(3))
            off = tuple(rel[i] - along * unit[i] for i in range(3))
            worst = max(worst, math.sqrt(sum(c * c for c in off)))
        return 1000.0 * worst

    def describe(self) -> str:
        """One line for logs."""
        return (
            f"lean {self.a.lean_deg:+.2f}/{self.b.lean_deg:+.2f} deg, "
            f"bow {self.a.bow_mm:+.1f}/{self.b.bow_mm:+.1f} mm"
        )


def _clip(value: float, limit: float) -> float:
    return min(max(value, -limit), limit)


def plan_panel(rng: random.Random, scale: float = 1.0) -> PanelForming:
    """Draw the forming error for one folded side.

    scale multiplies both terms, for running a batch of deliberately bad parts
    without changing what the numbers mean.
    """
    if scale < 0.0:
        raise ValueError("scale must not be negative")
    return PanelForming(
        lean_deg=_clip(rng.gauss(0.0, LEAN_SIGMA_DEG * scale), LEAN_LIMIT_DEG * max(scale, 1.0)),
        bow_mm=_clip(rng.gauss(0.0, BOW_SIGMA_MM * scale), BOW_LIMIT_MM * max(scale, 1.0)),
    )


def plan_corner(rng: random.Random, scale: float = 1.0) -> CornerForming:
    """Draw the forming error for both sides of one corner.

    The two sides are drawn independently: they were bent on separate strokes, so
    one can spring back further than the other. That is what tilts the seam out of
    vertical rather than merely shifting it.
    """
    return CornerForming(a=plan_panel(rng, scale), b=plan_panel(rng, scale))


def plan_campaign(n: int, seed: int = 0, scale: float = 1.0) -> list[CornerForming]:
    """A reproducible run of parts off the press."""
    rng = random.Random(seed)
    return [plan_corner(rng, scale) for _ in range(n)]


@dataclass(frozen=True)
class TrayForming:
    """The forming error on all four sides of one tray.

    Sides are named for the face they stand on: right and left lie in planes of
    constant x, front and back in planes of constant y. Every side gets its own
    error because every side was its own press stroke.
    """

    right: PanelForming = PanelForming()
    left: PanelForming = PanelForming()
    front: PanelForming = PanelForming()
    back: PanelForming = PanelForming()

    def welded_corner(self) -> CornerForming:
        """The corner the robot welds: where the right side meets the front side."""
        return CornerForming(a=self.right, b=self.front)

    def describe(self) -> str:
        return self.welded_corner().describe()


def plan_tray(rng: random.Random, scale: float = 1.0) -> TrayForming:
    """Draw the forming error for a whole tray, one side at a time."""
    return TrayForming(
        right=plan_panel(rng, scale),
        left=plan_panel(rng, scale),
        front=plan_panel(rng, scale),
        back=plan_panel(rng, scale),
    )


def plan_trays(n: int, seed: int = 0, scale: float = 1.0) -> list[TrayForming]:
    """A reproducible run of trays off the press."""
    rng = random.Random(seed)
    return [plan_tray(rng, scale) for _ in range(n)]


@dataclass(frozen=True)
class PressDrift:
    """A press that is going out of adjustment as the shift runs.

    Forming error is not only random part to part. Tooling warms up, a punch
    picks up material, a back gauge creeps, and the bend angle walks in one
    direction across a run. That is the failure worth catching, because it is
    the one that is fixable: random scatter is the process, a trend is a machine
    that needs attention.

    It also looks nothing like the other thing that moves a seam. A part sitting
    badly in the fixture moves the whole joint without bending it. A press going
    out moves the TOP of the joint and leaves the bottom alone, because it is an
    angle applied at the fold. Telling those two apart from the measurement
    stream is the diagnosis this cell is meant to make.
    """

    lean_deg_per_part: float = 0.0
    side: str = "right"
    starts_at: int = 0

    def extra_lean_deg(self, index: int) -> float:
        if index < self.starts_at:
            return 0.0
        return self.lean_deg_per_part * (index - self.starts_at)


def plan_shift(
    n: int,
    seed: int = 0,
    scale: float = 1.0,
    drift: Optional["PressDrift"] = None,
) -> list[TrayForming]:
    """A run of parts off a press that may be drifting as it goes.

    Same random scatter as plan_trays, with a trend laid on top of one side only.
    Returned in the order they were made, because the order is the evidence.
    """
    rng = random.Random(seed)
    trays = []
    for index in range(n):
        tray = plan_tray(rng, scale)
        if drift is not None and drift.lean_deg_per_part:
            extra = drift.extra_lean_deg(index)
            side = getattr(tray, drift.side)
            shifted = PanelForming(lean_deg=side.lean_deg + extra, bow_mm=side.bow_mm)
            tray = TrayForming(**{**tray.__dict__, drift.side: shifted})
        trays.append(tray)
    return trays
