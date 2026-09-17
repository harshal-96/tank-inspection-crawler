"""The gate that decides what happens to a part.

A cell that measures and never decides is a dashboard, so these cover the four
outcomes and, more importantly, the order they are judged in: fit-up before the
bead, because a joint the arc could not bridge can still photograph well.
"""

from __future__ import annotations


def _part(client) -> str:
    return client.post("/api/parts", json={}).json()["serial"]


def _schedule(client, serial: str, gaps: list[float]) -> None:
    """Post a schedule where any gap over 3.2 mm is refused, as the cell does."""
    stations = []
    for i, gap in enumerate(gaps):
        weldable = gap <= 3.2
        stations.append({
            "along_mm": 10.0 + i * 20.0,
            "gap_mm": gap,
            "current_a": 90.0 if weldable else 0.0,
            "voltage_v": 19.0 if weldable else 0.0,
            "travel_mm_s": 8.0 if weldable else 0.0,
            "heat_j_per_mm": 120.0 if weldable else 0.0,
            "weldable": weldable,
            "needs_attention": False,
            "reason": "test",
        })
    res = client.post("/api/weld-schedules",
                      json={"part_serial": serial, "stations": stations})
    assert res.status_code == 201, res.text


def _inspect(client, serial: str, verdict: str, defect: str = "none") -> None:
    res = client.post("/api/inspections", json={
        "part_serial": serial,
        "image_path": "x.png",
        "verdict": verdict,
        "defect_type": defect,
    })
    assert res.status_code == 201, res.text


def _status(client, serial: str) -> str:
    return client.get(f"/api/parts/{serial}").json()["status"]


def test_clean_bead_on_a_good_fit_is_accepted(client):
    serial = _part(client)
    _schedule(client, serial, [0.4, 0.8, 1.2])
    _inspect(client, serial, "pass")
    assert _status(client, serial) == "accepted"


def test_named_defect_sends_the_part_back(client):
    serial = _part(client)
    _schedule(client, serial, [0.4, 0.8])
    _inspect(client, serial, "defect", "undercut")
    assert _status(client, serial) == "rework"


def test_an_uncertain_verdict_holds_the_part_rather_than_guessing(client):
    serial = _part(client)
    _schedule(client, serial, [0.4, 0.8])
    _inspect(client, serial, "uncertain")
    assert _status(client, serial) == "inspected"


def test_a_refused_station_outranks_a_clean_bead(client):
    """The gap the arc could not bridge decides, not the photograph of it."""
    serial = _part(client)
    _schedule(client, serial, [0.4, 5.0, 0.8])
    _inspect(client, serial, "pass")
    assert _status(client, serial) == "rework"

    reason = next(d["reason"] for d in client.get("/api/dispositions").json()
                  if d["serial"] == serial)
    assert "fit-up" in reason


def test_disposition_reason_names_the_defect(client):
    serial = _part(client)
    _schedule(client, serial, [0.4])
    _inspect(client, serial, "defect", "burn_through")
    reason = next(d["reason"] for d in client.get("/api/dispositions").json()
                  if d["serial"] == serial)
    assert "burn through" in reason


def test_settle_backfills_a_run_recorded_before_the_gate(client):
    """Parts recorded by an older build get decided without being re-welded."""
    serial = _part(client)
    _schedule(client, serial, [0.4])
    _inspect(client, serial, "pass")

    # Put the part back into a pre-gate state, as the stored run was.
    from sqlmodel import Session, select

    from service.db import engine
    from service.models import Part, PartStatus

    with Session(engine) as session:
        part = session.exec(select(Part).where(Part.serial == serial)).one()
        part.status = PartStatus.INSPECTED
        session.add(part)
        session.commit()

    assert _status(client, serial) == "inspected"
    result = client.post("/api/parts/settle").json()
    assert result["counts"]["accepted"] >= 1
    assert _status(client, serial) == "accepted"


def test_a_part_that_has_only_been_welded_is_not_accepted_yet(client):
    serial = _part(client)
    _schedule(client, serial, [0.4, 0.9])
    assert _status(client, serial) == "welded"
