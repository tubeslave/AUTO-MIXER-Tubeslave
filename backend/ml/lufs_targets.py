"""
Genre-aware LUFS target levels for automatic gain staging.
Based on EBU R128, AES recommendations, and live sound engineering practices.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class LUFSTarget:
    """LUFS target for a specific context."""
    integrated: float  # Target integrated LUFS
    short_term_max: float  # Maximum short-term LUFS
    momentary_max: float  # Maximum momentary LUFS
    true_peak_max: float  # Maximum true peak dBTP
    loudness_range: float  # Target LRA (loudness range)
    tolerance: float = 1.0  # Acceptable deviation in LU


GENRE_TARGETS: Dict[str, LUFSTarget] = {
    'rock': LUFSTarget(-18.0, -14.0, -10.0, -1.0, 8.0),
    'pop': LUFSTarget(-16.0, -12.0, -8.0, -1.0, 6.0),
    'jazz': LUFSTarget(-20.0, -16.0, -12.0, -1.0, 12.0),
    'classical': LUFSTarget(-23.0, -18.0, -14.0, -1.0, 15.0),
    'edm': LUFSTarget(-14.0, -10.0, -6.0, -1.0, 4.0),
    'folk': LUFSTarget(-20.0, -16.0, -12.0, -1.0, 10.0),
    'metal': LUFSTarget(-16.0, -12.0, -8.0, -1.0, 5.0),
    'hip_hop': LUFSTarget(-14.0, -10.0, -6.0, -1.0, 5.0),
    'country': LUFSTarget(-18.0, -14.0, -10.0, -1.0, 8.0),
    'r_and_b': LUFSTarget(-16.0, -12.0, -8.0, -1.0, 7.0),
    'worship': LUFSTarget(-18.0, -14.0, -10.0, -1.0, 10.0),
    'spoken_word': LUFSTarget(-24.0, -20.0, -16.0, -1.0, 8.0, tolerance=0.5),
    'theater': LUFSTarget(-20.0, -16.0, -12.0, -1.0, 12.0),
    'broadcast': LUFSTarget(-24.0, -20.0, -16.0, -2.0, 8.0, tolerance=0.5),
    'default': LUFSTarget(-18.0, -14.0, -10.0, -1.0, 8.0),
}

VENUE_OFFSETS: Dict[str, float] = {
    'small_club': 2.0,
    'medium_venue': 0.0,
    'large_arena': -2.0,
    'outdoor_festival': -3.0,
    'theater': 3.0,
    'church': 2.0,
    'conference': 4.0,
    'broadcast_studio': 0.0,
}


@dataclass
class InstrumentLUFSProfile:
    """Per-instrument target level relative to mix bus."""
    relative_db: float  # Target level relative to mix bus LUFS
    priority: int  # 1=highest (lead vocal), 5=lowest (ambient)
    duck_amount: float = 0.0  # Auto-duck amount when higher priority plays


INSTRUMENT_PROFILES: Dict[str, InstrumentLUFSProfile] = {
    'lead_vocal': InstrumentLUFSProfile(-0.0, 1),
    'backing_vocal': InstrumentLUFSProfile(-6.0, 2),
    'kick': InstrumentLUFSProfile(-6.0, 2),
    'snare': InstrumentLUFSProfile(-8.0, 2),
    'bass_guitar': InstrumentLUFSProfile(-6.0, 2),
    'electric_guitar': InstrumentLUFSProfile(-8.0, 3),
    'acoustic_guitar': InstrumentLUFSProfile(-8.0, 3),
    'keys_piano': InstrumentLUFSProfile(-10.0, 3),
    'overheads': InstrumentLUFSProfile(-12.0, 4),
    'hi_hat': InstrumentLUFSProfile(-14.0, 4),
    'toms': InstrumentLUFSProfile(-10.0, 3),
    'room_mics': InstrumentLUFSProfile(-18.0, 5, duck_amount=3.0),
    'ambient_mic': InstrumentLUFSProfile(-20.0, 5, duck_amount=6.0),
    'audience': InstrumentLUFSProfile(-24.0, 5, duck_amount=6.0),
    'brass': InstrumentLUFSProfile(-8.0, 3),
    'woodwind': InstrumentLUFSProfile(-10.0, 3),
    'strings': InstrumentLUFSProfile(-10.0, 3),
    'synth': InstrumentLUFSProfile(-8.0, 3),
    'organ': InstrumentLUFSProfile(-10.0, 3),
    'percussion': InstrumentLUFSProfile(-10.0, 3),
    'dj_playback': InstrumentLUFSProfile(-4.0, 1),
    'click_track': InstrumentLUFSProfile(-60.0, 5),
    'choir': InstrumentLUFSProfile(-6.0, 2),
    'unknown': InstrumentLUFSProfile(-12.0, 3),
}


class LUFSTargetManager:
    """Manages LUFS targets based on genre, venue, and instrument context."""

    def __init__(self, genre: str = 'default', venue: str = 'medium_venue'):
        self.genre = genre
        self.venue = venue
        self._custom_targets: Dict[str, LUFSTarget] = {}

    @property
    def target(self) -> LUFSTarget:
        if self.genre in self._custom_targets:
            return self._custom_targets[self.genre]
        base = GENRE_TARGETS.get(self.genre, GENRE_TARGETS['default'])
        offset = VENUE_OFFSETS.get(self.venue, 0.0)
        return LUFSTarget(
            integrated=base.integrated + offset,
            short_term_max=base.short_term_max + offset,
            momentary_max=base.momentary_max + offset,
            true_peak_max=base.true_peak_max,
            loudness_range=base.loudness_range,
            tolerance=base.tolerance,
        )

    def get_channel_target(self, instrument: str) -> float:
        profile = INSTRUMENT_PROFILES.get(instrument, INSTRUMENT_PROFILES['unknown'])
        return self.target.integrated + profile.relative_db

    def get_gain_adjustment(self, instrument: str, current_lufs: float) -> float:
        target = self.get_channel_target(instrument)
        if current_lufs < -70:
            return 0.0
        diff = target - current_lufs
        return max(-12.0, min(12.0, diff))

    def set_custom_target(self, genre: str, target: LUFSTarget):
        self._custom_targets[genre] = target

    def get_duck_amount(self, instrument: str) -> float:
        return INSTRUMENT_PROFILES.get(
            instrument, INSTRUMENT_PROFILES['unknown']
        ).duck_amount

    def get_priority(self, instrument: str) -> int:
        return INSTRUMENT_PROFILES.get(
            instrument, INSTRUMENT_PROFILES['unknown']
        ).priority
