"""The millimetre limits every part of the cell agrees on.

They were typed into three places before this module existed: a study, the
supervisor's prompt and the dashboard. One number in three places is three
numbers. They are read from the environment so a site can set its own without
touching code.

    CELL_WELDABLE_MM   a joint located within this of the truth is welded on
    CELL_PROBLEM_MM    beyond this the part is held and a person is called
"""

from __future__ import annotations

import os


def _mm(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


WELDABLE_MM = _mm("CELL_WELDABLE_MM", 1.0)
PROBLEM_MM = _mm("CELL_PROBLEM_MM", 3.0)
