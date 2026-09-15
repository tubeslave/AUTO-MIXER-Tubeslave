"""Versioned, context-specific target intervals fitted only to approved mixes.

Features must be measured with the same extractor. A corridor is an objective,
not a perceptual quality certificate. No training takes place on live decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FEATURE_VERSION = "shadow-mix-v1"


@dataclass(frozen=True)
class TargetCorridor:
    context: str
    bands: dict[str, tuple[float, float, float]]
    sample_count: int
    rejected_count: int = 0
    feature_version: str = FEATURE_VERSION
    source: str = "approved_mixes"

    def __post_init__(self) -> None:
        if not self.context or self.feature_version != FEATURE_VERSION or not self.bands:
            raise ValueError("Missing context/features or incompatible feature version")
        if self.sample_count < 0 or self.rejected_count < 0:
            raise ValueError("Negative sample count")
        for name, interval in self.bands.items():
            if not isinstance(name, str) or len(interval) != 3:
                raise ValueError("Invalid interval")
            lo, mid, hi = map(float, interval)
            if not np.isfinite([lo, mid, hi]).all() or not lo <= mid <= hi or hi <= lo:
                raise ValueError(f"Invalid corridor for {name}")

    @classmethod
    def fit(
        cls, records: Iterable[Mapping[str, Any]], *, context: str,
        features: Iterable[str], min_samples: int = 5, min_half_width: float = 0.5,
    ) -> "TargetCorridor":
        """Median/MAD envelope after row-wise outlier rejection; one vote per mix.

        All features are in dB/LU. Invalid, unapproved, duplicate, wrong-context
        and gross-outlier rows are excluded and counted, never silently imputed.
        """
        names = tuple(features)
        if not context or not names or len(set(names)) != len(names):
            raise ValueError("Distinct feature names and a context are required")
        if min_samples < 5 or not np.isfinite(min_half_width) or min_half_width <= 0:
            raise ValueError("At least five mixes and a positive width are required")
        values, seen, rejected = [], set(), 0
        for record in records:
            mix_id = record.get("mix_id")
            valid = (record.get("approved") is True and record.get("context") == context
                     and record.get("feature_version") == FEATURE_VERSION
                     and isinstance(mix_id, str) and bool(mix_id) and mix_id not in seen)
            try:
                row = [float(record["features"][name]) for name in names]
                valid = valid and np.isfinite(row).all()
            except (KeyError, TypeError, ValueError, OverflowError):
                valid, row = False, []
            if not valid:
                rejected += 1
                continue
            seen.add(mix_id)
            values.append(row)
        if len(values) < min_samples:
            raise ValueError("Not enough distinct approved complete mixes")
        x = np.asarray(values, dtype=np.float64)
        median = np.median(x, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(x - median), axis=0), min_half_width)
        keep = np.all(np.abs(x - median) <= 4.5 * scale, axis=1)
        rejected += int(np.count_nonzero(~keep))
        x = x[keep]
        if len(x) < min_samples:
            raise ValueError("Too few mixes after robust outlier rejection")
        median = np.median(x, axis=0)
        width = np.maximum(2.5 * 1.4826 * np.median(np.abs(x - median), axis=0),
                           min_half_width)
        bands = {name: (float(mid - w), float(mid), float(mid + w))
                 for name, mid, w in zip(names, median, width)}
        return cls(context, bands, len(x), rejected)

    def loss(self, features: Mapping[str, float], *, context: str) -> float:
        """Mean squared, width-normalized distance outside intervals only."""
        if context != self.context:
            raise ValueError("Corridor context mismatch")
        distances = []
        for name, (lo, mid, hi) in self.bands.items():
            value = float(features[name])
            if not np.isfinite(value):
                raise ValueError("Non-finite objective feature")
            distances.append(max(lo - value, 0.0, value - hi) / max((hi - lo) / 2, 1e-6))
        result = float(np.mean(np.square(distances)))
        if not np.isfinite(result):
            raise ValueError("Objective overflow")
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TargetCorridor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["bands"] = {key: tuple(value) for key, value in data["bands"].items()}
        return cls(**data)
