"""Strict numeric boundary helpers shared by offline safety/decision layers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def finite_float(value: Any) -> float | None:
    """Return a finite real number; reject booleans, strings, NaN, and infinity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric
