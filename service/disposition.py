"""What happens to the part, and why.

A cell that measures and never decides is a dashboard. This is the gate: given
everything recorded against a part, it returns the status the part should carry
and the sentence that justifies it.

The order matters and is not arbitrary. Fit-up is judged before the bead, because
a joint the arc could not bridge is a pressing problem rather than a welding one,
and a bead laid across a gap that wide can look perfectly good in a photograph
while having nothing behind it. An inspection that cannot tell holds the part for
a person rather than guessing, because the two ways of being wrong here cost
different money: a false pass ships a bad part, a false alarm scraps a good one,
and neither is the machine's call to make on its own.
"""

from __future__ import annotations

from sqlmodel import Session, select

from service.models import Inspection, Part, PartStatus, Verdict, WeldSchedule


def decide(session: Session, part: Part) -> tuple[PartStatus, str]:
    """The status this part has earned, and the reason in one sentence."""
    schedule = session.exec(
        select(WeldSchedule)
        .where(WeldSchedule.part_id == part.id)
        .order_by(WeldSchedule.id.desc())
    ).first()
    inspection = session.exec(
        select(Inspection)
        .where(Inspection.part_id == part.id)
        .order_by(Inspection.id.desc())
    ).first()

    if schedule is not None and schedule.refused_stations:
        n = schedule.refused_stations
        return PartStatus.REWORK, (
            f"{n} station{'s' if n > 1 else ''} along the seam opened wider than one pass "
            f"can bridge, so the joint goes back for fit-up rather than more heat"
        )

    if inspection is None:
        if schedule is not None:
            return PartStatus.WELDED, "welded to schedule, not yet inspected"
        return part.status, "measured, not yet welded"

    if inspection.verdict == Verdict.DEFECT:
        return PartStatus.REWORK, (
            f"the bead was called {inspection.defect_type.value.replace('_', ' ')} "
            f"after welding"
        )

    if inspection.verdict == Verdict.UNCERTAIN:
        return PartStatus.INSPECTED, (
            "the image did not support a verdict either way, so the part is held "
            "for a person rather than guessed at"
        )

    return PartStatus.ACCEPTED, "welded to schedule and the bead came back clean"


def settle(session: Session, part: Part) -> tuple[PartStatus, str]:
    """Decide, and write the answer onto the part."""
    status, reason = decide(session, part)
    if part.status != status:
        part.status = status
        session.add(part)
    return status, reason


def settle_all(session: Session) -> list[dict]:
    """Re-decide every part. Used after a rule change, and to backfill a run."""
    out = []
    for part in session.exec(select(Part).order_by(Part.id)):
        status, reason = settle(session, part)
        out.append({
            "part_id": part.id,
            "serial": part.serial,
            "status": status.value,
            "reason": reason,
        })
    session.commit()
    return out
