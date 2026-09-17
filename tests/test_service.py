"""Service smoke tests.

These cover the two columns the whole validation story rests on: the millimetre
error computed from ground truth, and the correctness flag scored against an
injected defect.
"""

from __future__ import annotations


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_part_serial_autoassigns(client):
    first = client.post("/api/parts", json={}).json()
    second = client.post("/api/parts", json={}).json()
    assert first["serial"] == "P-000001"
    assert second["serial"] == "P-000002"


def test_duplicate_serial_rejected(client):
    client.post("/api/parts", json={"serial": "P-fixed"})
    assert client.post("/api/parts", json={"serial": "P-fixed"}).status_code == 409


def test_measurement_computes_error_mm(client):
    serial = client.post("/api/parts", json={}).json()["serial"]

    # Detected 3 mm in x and 4 mm in y from truth, so the planar error is 5 mm.
    body = {
        "part_serial": serial,
        "detected_x": 0.003,
        "detected_y": 0.004,
        "confidence": 0.91,
        "true_x": 0.0,
        "true_y": 0.0,
    }
    measurement = client.post("/api/measurements", json=body).json()
    assert measurement["error_mm"] == 5.0

    assert client.get(f"/api/parts/{serial}").json()["status"] == "measured"


def test_measurement_without_truth_has_no_error(client):
    serial = client.post("/api/parts", json={"station": "bench"}).json()["serial"]
    body = {
        "part_serial": serial,
        "detected_x": 0.1,
        "detected_y": 0.2,
        "confidence": 0.8,
        "station": "bench",
    }
    assert client.post("/api/measurements", json=body).json()["error_mm"] is None


def test_measurement_unknown_part_404s(client):
    body = {"part_serial": "P-nope", "detected_x": 0.0, "detected_y": 0.0, "confidence": 0.5}
    assert client.post("/api/measurements", json=body).status_code == 404


def _inspect(client, verdict, injected):
    serial = client.post("/api/parts", json={}).json()["serial"]
    body = {
        "part_serial": serial,
        "image_path": f"data/inspections/{serial}.png",
        "verdict": verdict,
        "injected_defect": injected,
    }
    return client.post("/api/inspections", json=body).json()


def test_inspection_scores_against_injected_defect(client):
    assert _inspect(client, "pass", "none")["correct"] is True
    assert _inspect(client, "defect", "undercut")["correct"] is True
    assert _inspect(client, "pass", "porosity")["correct"] is False  # a miss
    assert _inspect(client, "defect", "none")["correct"] is False  # a false alarm


def test_inspection_without_truth_is_unscored(client):
    serial = client.post("/api/parts", json={}).json()["serial"]
    body = {"part_serial": serial, "image_path": "x.png", "verdict": "pass"}
    assert client.post("/api/inspections", json=body).json()["correct"] is None


def test_stats_builds_confusion_matrix(client):
    _inspect(client, "pass", "none")
    _inspect(client, "pass", "none")
    _inspect(client, "defect", "undercut")
    _inspect(client, "pass", "skip")  # missed defect
    _inspect(client, "defect", "none")  # false alarm

    stats = client.get("/api/stats").json()["inspection"]
    assert stats["n_scored"] == 5
    assert stats["true_pass"] == 2
    assert stats["true_defect"] == 1
    assert stats["false_pass"] == 1
    assert stats["false_defect"] == 1
    assert stats["accuracy"] == 3 / 5


def test_stats_correction_window(client):
    for i in range(10):
        serial = client.post("/api/parts", json={}).json()["serial"]
        client.post(
            "/api/measurements",
            json={
                "part_serial": serial,
                "detected_x": 0.0,
                "detected_y": i / 1000.0,  # error walks 0 mm to 9 mm
                "confidence": 0.9,
                "true_x": 0.0,
                "true_y": 0.0,
            },
        )

    correction = client.get("/api/stats").json()["correction"]
    assert correction["n"] == 10
    assert correction["mean_mm"] == 4.5
    assert correction["max_mm"] == 9.0
    assert correction["sub_mm_rate"] == 0.1  # only the 0 mm row is sub-millimetre


def test_stats_empty_cell_does_not_divide_by_zero(client):
    stats = client.get("/api/stats").json()
    assert stats["parts_total"] == 0
    assert stats["correction"]["n"] == 0
    assert stats["correction"]["mean_mm"] is None
    assert stats["inspection"]["accuracy"] is None
    assert stats["drift"]["trending"] is False


def test_decision_records_evidence(client):
    serial = client.post("/api/parts", json={}).json()["serial"]
    body = {
        "action": "recalibrate",
        "rationale": "Last 5 parts trend +y, consistent with locator wear.",
        "part_serial": serial,
        "evidence": {"shift_mm": 1.8, "window": 5},
    }
    decision = client.post("/api/decisions", json=body).json()
    assert decision["action"] == "recalibrate"
    assert decision["evidence"]["shift_mm"] == 1.8
