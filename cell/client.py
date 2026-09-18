"""HTTP client for the cell service.

Standard library only. This module is imported by the Isaac Sim process, which
runs under its own bundled interpreter with its own site-packages, so it cannot
depend on anything installed into the project venv.

If the service is unreachable the result is appended to a local JSONL file rather
than lost. A campaign that runs for an hour should not be thrown away because a
service restarted in the middle of it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_FALLBACK = Path("data/offline_queue.jsonl")


class CellServiceError(RuntimeError):
    """The service answered, but not with success."""


class CellClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
        fallback_path: Optional[Path] = DEFAULT_FALLBACK,
        verbose: bool = True,
    ) -> None:
        self.base_url = (base_url or os.getenv("CELL_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.fallback_path = Path(fallback_path) if fallback_path else None
        self.verbose = verbose
        self.offline = False

    # ------------------------------------------------------------------ transport

    def _request(self, method: str, path: str, body: Optional[dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload) if payload else None

    def _post(self, path: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            result = self._request("POST", path, body)
            if self.offline:
                self._log("service reachable again")
                self.offline = False
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise CellServiceError(f"{exc.code} from {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._queue_offline(path, body, reason=str(exc))
            return None

    def _queue_offline(self, path: str, body: dict[str, Any], reason: str) -> None:
        if not self.offline:
            self._log(f"service unreachable ({reason}), queueing to disk")
            self.offline = True
        if self.fallback_path is None:
            return
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"path": path, "body": body}) + "\n")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[cell-client] {message}", flush=True)

    # --------------------------------------------------------------------- health

    def healthy(self) -> bool:
        try:
            return self._request("GET", "/healthz").get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    # ----------------------------------------------------------------- production

    def create_part(
        self,
        station: str = "sim",
        spawn_dy_mm: Optional[float] = None,
        spawn_yaw_deg: Optional[float] = None,
        serial: Optional[str] = None,
    ) -> Optional[str]:
        """Register a part. Returns its serial, or None if the service is offline."""
        body = {
            "station": station,
            "spawn_dy_mm": spawn_dy_mm,
            "spawn_yaw_deg": spawn_yaw_deg,
        }
        if serial:
            body["serial"] = serial
        try:
            result = self._post("/api/parts", body)
        except CellServiceError as exc:
            # Re-running a campaign over the same parts is an ordinary thing to
            # do, and a serial that already exists is not a failure: it is the
            # part being registered a second time. Anything else still raises.
            if serial and "already exists" in str(exc):
                self._log(f"part {serial} already registered, reusing it")
                return serial
            raise
        return result["serial"] if result else None

    def post_measurement(
        self,
        part_serial: str,
        detected: tuple[float, float],
        confidence: float,
        truth: Optional[tuple[float, float]] = None,
        n_scans: int = 1,
        spread_mm: Optional[float] = None,
        latency_ms: Optional[float] = None,
        station: str = "sim",
    ) -> Optional[dict[str, Any]]:
        body: dict[str, Any] = {
            "part_serial": part_serial,
            "detected_x": detected[0],
            "detected_y": detected[1],
            "confidence": confidence,
            "n_scans": n_scans,
            "spread_mm": spread_mm,
            "latency_ms": latency_ms,
            "station": station,
        }
        if truth is not None:
            body["true_x"], body["true_y"] = truth
        return self._post("/api/measurements", body)

    def post_seam_track(
        self,
        part_serial: str,
        path: list[dict[str, Any]],
        stations_asked: int,
        wander_mm: Optional[float] = None,
        taught_miss_mm: Optional[float] = None,
        gap_mm_mean: Optional[float] = None,
        gap_mm_max: Optional[float] = None,
        seam_bow_mm: Optional[float] = None,
        error_mm: Optional[float] = None,
        worst_station_mm: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Send a whole tracked seam, not a located point.

        The path is the claim. Summary numbers are carried alongside it so the
        dashboard does not have to recompute them on every poll, but the stations
        are what an operator looks at when a part is argued about.
        """
        return self._post(
            "/api/seam-tracks",
            {
                "part_serial": part_serial,
                "path": path,
                "stations_measured": len(path),
                "stations_asked": stations_asked,
                "wander_mm": wander_mm,
                "taught_miss_mm": taught_miss_mm,
                "gap_mm_mean": gap_mm_mean,
                "gap_mm_max": gap_mm_max,
                "seam_bow_mm": seam_bow_mm,
                "error_mm": error_mm,
                "worst_station_mm": worst_station_mm,
                "latency_ms": latency_ms,
            },
        )

    def post_weld_schedule(
        self,
        part_serial: str,
        stations: list[dict[str, Any]],
        predicted_drift_mm: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Send what the torch was told to do at each station, refusals included."""
        return self._post(
            "/api/weld-schedules",
            {
                "part_serial": part_serial,
                "stations": stations,
                "predicted_drift_mm": predicted_drift_mm,
            },
        )

    def get_weld_schedules(self, limit: int = 200) -> list[dict[str, Any]]:
        """Read back what was scheduled, so a later pass can find the refusals."""
        try:
            return self._request("GET", f"/api/weld-schedules?limit={limit}") or []
        except Exception:  # noqa: BLE001
            return []

    def post_brief(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Store a corrective action brief with the sources it was written from."""
        return self._post("/api/briefs", payload)

    def post_view_decision(
        self, part_serial: str, **fields: Any
    ) -> Optional[dict[str, Any]]:
        """Send one look-and-judge decision, and which model size made it.

        Which model answered is part of the answer: the sizes disagree, and an
        operator asking why a part took four looks deserves to see the route.
        """
        return self._post("/api/view-decisions", {"part_serial": part_serial, **fields})

    def post_inspection(
        self,
        part_serial: str,
        image_path: str,
        verdict: str,
        defect_type: str = "none",
        severity: Optional[float] = None,
        reasoning: Optional[str] = None,
        model: Optional[str] = None,
        latency_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        injected_defect: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        return self._post(
            "/api/inspections",
            {
                "part_serial": part_serial,
                "image_path": image_path,
                "verdict": verdict,
                "defect_type": defect_type,
                "severity": severity,
                "reasoning": reasoning,
                "model": model,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "injected_defect": injected_defect,
            },
        )


def replay_offline_queue(
    path: Path = DEFAULT_FALLBACK, base_url: Optional[str] = None
) -> tuple[int, int]:
    """Push anything queued while the service was down. Returns (sent, failed).

    The queue file is only cleared when every row lands, so a partial failure can
    be retried without losing the rest.
    """
    path = Path(path)
    if not path.exists():
        return (0, 0)

    client = CellClient(base_url=base_url, fallback_path=None, verbose=False)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    sent, failed = 0, []
    for row in rows:
        try:
            client._request("POST", row["path"], row["body"])
            sent += 1
        except Exception:  # noqa: BLE001
            failed.append(row)

    if failed:
        with path.open("w", encoding="utf-8") as handle:
            for row in failed:
                handle.write(json.dumps(row) + "\n")
    else:
        path.unlink()

    return (sent, len(failed))
