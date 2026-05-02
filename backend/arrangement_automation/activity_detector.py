"""Hysteretic channel activity detector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from .models import ActivityState


@dataclass
class _ActivityMemory:
    active: bool = False
    candidate_since: Optional[float] = None
    inactive_since: Optional[float] = None
    active_since: Optional[float] = None


class ActivityDetector:
    """Detect meaningful channel activity without triggering on noise alone."""

    def __init__(
        self,
        min_active_db: float = -50.0,
        noise_margin_db: float = 10.0,
        attack_hold_sec: float = 0.3,
        release_hold_sec: float = 1.0,
    ):
        self.min_active_db = float(min_active_db)
        self.noise_margin_db = float(noise_margin_db)
        self.attack_hold_sec = float(attack_hold_sec)
        self.release_hold_sec = float(release_hold_sec)
        self._memory: Dict[int, _ActivityMemory] = {}

    def update(
        self,
        channel_id: int,
        rms_db: float,
        peak_db: Optional[float] = None,
        short_term_loudness_db: Optional[float] = None,
        spectral_energy_db: Optional[float] = None,
        noise_floor_db: float = -80.0,
        timestamp: Optional[float] = None,
    ) -> ActivityState:
        """Update and return activity state for one channel."""
        now = time.time() if timestamp is None else float(timestamp)
        peak = float(peak_db if peak_db is not None else rms_db)
        loudness = float(short_term_loudness_db if short_term_loudness_db is not None else rms_db)
        spectral = float(spectral_energy_db if spectral_energy_db is not None else rms_db)
        signal_db = max(float(rms_db), loudness, spectral)
        threshold = max(self.min_active_db, float(noise_floor_db) + self.noise_margin_db)
        above = signal_db >= threshold and peak >= threshold - 6.0

        memory = self._memory.setdefault(channel_id, _ActivityMemory())
        if above:
            memory.inactive_since = None
            if memory.candidate_since is None:
                memory.candidate_since = now
            if not memory.active and now - memory.candidate_since >= self.attack_hold_sec:
                memory.active = True
                memory.active_since = memory.candidate_since
        else:
            memory.candidate_since = None
            if memory.inactive_since is None:
                memory.inactive_since = now
            if memory.active and now - memory.inactive_since >= self.release_hold_sec:
                memory.active = False
                memory.active_since = None

        duration = 0.0
        if memory.active and memory.active_since is not None:
            duration = max(0.0, now - memory.active_since)

        confidence = self._confidence(signal_db, threshold, memory.active)
        return ActivityState(
            channel_id=channel_id,
            active=memory.active,
            activity_confidence=confidence,
            rms_db=float(rms_db),
            peak_db=peak,
            activity_duration_sec=duration,
            noise_floor_db=float(noise_floor_db),
        )

    @staticmethod
    def _confidence(signal_db: float, threshold_db: float, active: bool) -> float:
        margin = signal_db - threshold_db
        if active:
            return max(0.0, min(1.0, 0.5 + margin / 20.0))
        return max(0.0, min(0.6, 0.3 + margin / 20.0))
