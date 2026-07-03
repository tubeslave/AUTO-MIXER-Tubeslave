"""Shared analyzer/decision contract for AutoGain recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class AutoGainRecommendation:
    """Clamp-ready per-channel recommendation produced by an AutoGain analyzer."""

    channel: int
    source: str
    current_trim_db: float
    recommended_target_trim_db: float
    delta_db: float
    reason: str
    confidence: float
    limited_by: str = "none"
    blocked_reason: str = ""
    role: str = "unknown"
    analysis_state: str = "idle"
    live_apply_safe: bool = False
    dry_run_only: bool = True
    level_source: str = "unknown"
    audio_channel: int | None = None
    mixer_channel: int | None = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp_target_trim(
    current_trim_db: float,
    requested_delta_db: float,
    min_trim_db: float,
    max_trim_db: float,
) -> tuple[float, float]:
    """Return clamp-ready target trim and the resulting delta."""

    unclamped_target = float(current_trim_db) + float(requested_delta_db)
    clamped_target = max(float(min_trim_db), min(float(max_trim_db), unclamped_target))
    return clamped_target, clamped_target - float(current_trim_db)
