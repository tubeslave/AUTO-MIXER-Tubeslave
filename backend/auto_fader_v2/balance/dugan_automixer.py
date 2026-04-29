"""
Dugan-style NOM automix controller.

This module implements deterministic control values only. It does not send
mixer commands directly. Callers map the returned attenuation values onto their
own fader baseline and safety limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


MIN_LEVEL_DB = -120.0


@dataclass
class DuganAutomixSettings:
    """Settings for music-oriented Dugan gain sharing."""

    active_threshold_db: float = -50.0
    auto_mix_depth_db: float = 24.0
    max_full_gain_mics: Optional[int] = None
    last_hold_enabled: bool = True
    smoothing_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.auto_mix_depth_db = max(0.0, float(self.auto_mix_depth_db))
        self.smoothing_alpha = max(0.01, min(1.0, float(self.smoothing_alpha)))
        if self.max_full_gain_mics is not None:
            self.max_full_gain_mics = max(1, int(self.max_full_gain_mics))


class DuganAutomixer:
    """
    Calculate Dugan-style attenuation from active channel levels.

    For each active channel:

        gain_db = channel_level_db - sum_level_db

    Equal uncorrelated sources therefore get the expected NOM attenuation:
    1 mic = 0 dB, 2 mics = -3.01 dB, 4 mics = -6.02 dB.
    """

    def __init__(self, settings: Optional[DuganAutomixSettings] = None):
        self.settings = settings or DuganAutomixSettings()
        self.last_active_channel: Optional[int] = None
        self.current_targets: Dict[int, float] = {}
        self.last_nom: int = 0

    def reset_all(self) -> None:
        """Clear smoothing and Last Hold state."""
        self.last_active_channel = None
        self.current_targets.clear()
        self.last_nom = 0

    @staticmethod
    def nom_attenuation_db(nom: int) -> float:
        """Return nominal attenuation for N equally loud open microphones."""
        if nom <= 1:
            return 0.0
        return 10.0 * math.log10(float(nom))

    @staticmethod
    def _db_to_power(level_db: float) -> float:
        return 10.0 ** (max(level_db, MIN_LEVEL_DB) / 10.0)

    def calculate_target_gains(
        self,
        current_levels: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Return target attenuation per channel in dB.

        Values are always <= 0 dB. Channels below the active threshold are held
        at the configured automix depth, except when Last Hold keeps the most
        recent active channel open.
        """
        if not current_levels:
            self.last_nom = 0
            return {}

        active = {
            ch: float(level)
            for ch, level in current_levels.items()
            if math.isfinite(float(level)) and float(level) > self.settings.active_threshold_db
        }
        self.last_nom = len(active)

        if not active:
            raw_targets = {ch: -self.settings.auto_mix_depth_db for ch in current_levels}
            if (
                self.settings.last_hold_enabled
                and self.last_active_channel in current_levels
            ):
                raw_targets[self.last_active_channel] = 0.0
            return self._smooth(raw_targets)

        self.last_active_channel = max(active, key=active.get)
        total_power = sum(self._db_to_power(level_db) for level_db in active.values())
        if total_power <= 0.0:
            return {}

        gain_limit_offset = 0.0
        if self.settings.max_full_gain_mics:
            effective_nom = min(len(active), self.settings.max_full_gain_mics)
            gain_limit_offset = self.nom_attenuation_db(effective_nom)

        raw_targets: Dict[int, float] = {}
        for ch, level_db in current_levels.items():
            if ch not in active:
                raw_targets[ch] = -self.settings.auto_mix_depth_db
                continue

            sum_level_db = 10.0 * math.log10(total_power)
            dugan_gain = float(level_db) - sum_level_db + gain_limit_offset
            dugan_gain = min(0.0, dugan_gain)
            raw_targets[ch] = max(-self.settings.auto_mix_depth_db, dugan_gain)

        return self._smooth(raw_targets)

    def get_state(self) -> Dict[str, object]:
        """Return current state for status payloads/tests."""
        return {
            "last_active_channel": self.last_active_channel,
            "last_nom": self.last_nom,
            "current_targets": dict(self.current_targets),
            "settings": {
                "active_threshold_db": self.settings.active_threshold_db,
                "auto_mix_depth_db": self.settings.auto_mix_depth_db,
                "max_full_gain_mics": self.settings.max_full_gain_mics,
                "last_hold_enabled": self.settings.last_hold_enabled,
                "smoothing_alpha": self.settings.smoothing_alpha,
            },
        }

    def _smooth(self, raw_targets: Dict[int, float]) -> Dict[int, float]:
        alpha = self.settings.smoothing_alpha
        smoothed: Dict[int, float] = {}

        for ch, target in raw_targets.items():
            previous = self.current_targets.get(ch, target)
            value = (alpha * target) + ((1.0 - alpha) * previous)
            value = min(0.0, max(-self.settings.auto_mix_depth_db, value))
            self.current_targets[ch] = value
            smoothed[ch] = value

        stale_channels = set(self.current_targets) - set(raw_targets)
        for ch in stale_channels:
            self.current_targets.pop(ch, None)

        return smoothed
