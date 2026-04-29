"""Bus-level balance heuristics for offline and live-safe mix decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrumBusBalanceDecision:
    """Decision for aligning drum bus loudness against the music bed."""

    drum_lufs: float
    music_lufs: float
    current_delta_lu: float
    target_delta_lu: float
    gain_db: float
    min_gain_db: float
    max_gain_db: float


def calculate_drum_bus_gain(
    drum_lufs: float,
    music_lufs: float,
    target_delta_lu: float = 1.5,
    min_gain_db: float = -2.0,
    max_gain_db: float = 4.0,
) -> DrumBusBalanceDecision:
    """
    Calculate bounded drum-bus correction from drum-vs-music loudness.

    Rule:

        target_drum_vs_music = +1..+2 LU
        drum_bus_gain = clamp(target_delta - current_delta, -2 dB, +4 dB)

    LU and dB are treated as equivalent for this bus-level gain correction.
    """
    values = [drum_lufs, music_lufs, target_delta_lu, min_gain_db, max_gain_db]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("drum bus balance inputs must be finite")
    if min_gain_db > max_gain_db:
        raise ValueError("min_gain_db must be <= max_gain_db")

    current_delta_lu = float(drum_lufs) - float(music_lufs)
    raw_gain_db = float(target_delta_lu) - current_delta_lu
    gain_db = max(float(min_gain_db), min(float(max_gain_db), raw_gain_db))

    return DrumBusBalanceDecision(
        drum_lufs=float(drum_lufs),
        music_lufs=float(music_lufs),
        current_delta_lu=current_delta_lu,
        target_delta_lu=float(target_delta_lu),
        gain_db=gain_db,
        min_gain_db=float(min_gain_db),
        max_gain_db=float(max_gain_db),
    )
