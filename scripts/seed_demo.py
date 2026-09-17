"""Populate the database with a plausible shift so the dashboard can be developed
and filmed without running the simulation.

This is scaffolding for UI work. It is not a source of results: every number it
writes is synthetic, and nothing it produces should ever appear in a results table.

    python scripts/seed_demo.py --parts 40 --reset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlmodel import Session, delete  # noqa: E402

from service.db import engine, init_db  # noqa: E402
from service.models import (  # noqa: E402
    AgentMessage,
    Decision,
    DefectType,
    Inspection,
    Measurement,
    Part,
    PartStatus,
    ShiftReport,
    Station,
    Verdict,
)

NOMINAL_X, NOMINAL_Y = 4.398, -0.599  # the taught corner, metres
DEFECTS = [DefectType.UNDERCUT, DefectType.SKIP, DefectType.POROSITY, DefectType.SPATTER]


def reset(session: Session) -> None:
    for model in (AgentMessage, ShiftReport, Decision, Inspection, Measurement, Part):
        session.exec(delete(model))
    session.commit()


def seed(session: Session, n: int, seed_value: int) -> None:
    rng = random.Random(seed_value)

    for i in range(n):
        # Locator wear: the last third of the shift walks steadily in +y.
        drift_mm = 0.0 if i < n * 2 // 3 else (i - n * 2 // 3) * 0.22

        spawn_dy_mm = rng.gauss(0, 12)
        spawn_yaw = rng.gauss(0, 1.2)

        part = Part(
            serial=f"P-{i + 1:06d}",
            station=Station.SIM,
            status=PartStatus.INSPECTED,
            spawn_dy_mm=spawn_dy_mm,
            spawn_yaw_deg=spawn_yaw,
        )
        session.add(part)
        session.commit()
        session.refresh(part)

        true_x = NOMINAL_X + rng.gauss(0, 0.002)
        true_y = NOMINAL_Y + spawn_dy_mm / 1000.0

        # Detector error: a tight lognormal tail, plus the drift once it starts.
        err_m = rng.lognormvariate(-7.0, 0.55)
        theta = rng.uniform(0, 6.283185)
        det_x = true_x + err_m * rng.uniform(-1, 1)
        det_y = true_y + err_m * (1 if theta > 3.14 else -1) + drift_mm / 1000.0

        error_mm = 1000.0 * ((det_x - true_x) ** 2 + (det_y - true_y) ** 2) ** 0.5

        session.add(
            Measurement(
                part_id=part.id,
                station=Station.SIM,
                detected_x=det_x,
                detected_y=det_y,
                confidence=rng.uniform(0.86, 0.95),
                n_scans=8,
                spread_mm=rng.uniform(0.15, 0.45),
                latency_ms=rng.uniform(14.5, 18.0),
                true_x=true_x,
                true_y=true_y,
                error_mm=error_mm,
            )
        )

        # Defects get likelier as the path drifts off the joint.
        p_defect = 0.12 + min(0.5, drift_mm * 0.09)
        injected = rng.choice(DEFECTS) if rng.random() < p_defect else DefectType.NONE
        clean = injected == DefectType.NONE

        # An imperfect inspector, which is the point of scoring it.
        called_pass = clean if rng.random() < 0.9 else not clean
        verdict = Verdict.PASS if called_pass else Verdict.DEFECT
        called_defect = DefectType.NONE if called_pass else (
            injected if not clean else rng.choice(DEFECTS)
        )

        session.add(
            Inspection(
                part_id=part.id,
                image_path=f"data/inspections/{part.serial}.png",
                verdict=verdict,
                defect_type=called_defect,
                severity=None if called_pass else rng.uniform(0.3, 0.9),
                reasoning=(
                    "Bead is continuous with even toes and no visible undercut."
                    if called_pass
                    else f"Bead shows {called_defect.value.replace('_', ' ')} along the joint."
                ),
                model="seed",
                latency_ms=rng.uniform(900, 2200),
                injected_defect=injected,
                correct=(clean == called_pass),
            )
        )

        part.status = PartStatus.ACCEPTED if called_pass else PartStatus.REWORK
        session.add(part)

    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset", action="store_true", help="Clear existing rows first")
    args = parser.parse_args()

    init_db()
    with Session(engine) as session:
        if args.reset:
            reset(session)
        seed(session, args.parts, args.seed)

    print(f"Seeded {args.parts} synthetic parts. These are placeholders, not results.")


if __name__ == "__main__":
    main()
