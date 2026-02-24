"""
Genre profiles with instrument balance offsets.

Based on IMP 7.1.1 (Intelligent Music Production, De Man, Reiss & Stables):

- "the vocal should be approximately as loud as everything else, i.e.,
  the vocal should be 3 LU below the total mix loudness" [12, 74, 123, 267].
- Genre-dependent differences in balance preference [132].
- Equal loudness assumption disproved [12, 64, 74, 120, 123, 267, 268].

Offsets are relative to target_lufs (genre-specific overall target).
A channel target = target_lufs + instrument_offset.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class GenreType(Enum):
    """Genre types."""
    CUSTOM = "custom"
    ROCK = "rock"
    POP = "pop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"


@dataclass
class GenreProfile:
    """Genre profile with target LUFS, ratio and per-instrument offsets."""
    name: str
    instrument_offsets: Dict[str, float]
    target_lufs: float = -18.0
    ratio: float = 2.0


# Per-genre offsets (LU relative to genre target_lufs)
#
# Key references from the book:
#   [12] Pestana & Reiss (2014) - vocal ~3 LU below total mix loudness
#   [132] King et al. - genre-dependent balance differences
#   [74] De Man, Reiss & Stables - analysis of 600 mixes
#   [120] Jillings & Stables - online DAW study

POP_OFFSETS = {
    "leadVocal": -3.0,
    "backingVocal": -8.0,
    "kick": -6.0,
    "snare": -6.0,
    "bass": -5.0,
    "drums": -7.0,
    "tom": -9.0,
    "electricGuitar": -8.0,
    "acousticGuitar": -8.0,
    "keys": -9.0,
    "piano": -9.0,
    "synth": -8.0,
    "pads": -11.0,
    "hihat": -12.0,
    "ride": -12.0,
    "overhead": -12.0,
    "room": -16.0,
    "percussion": -10.0,
    "brass": -9.0,
    "sax": -9.0,
    "woodwinds": -10.0,
    "strings": -10.0,
    "fx": -12.0,
    "playback": -7.0,
    "accordion": -9.0,
    "unknown": -7.0,
}

ROCK_OFFSETS = {
    "leadVocal": -3.0,
    "backingVocal": -8.0,
    "kick": -5.0,
    "snare": -5.0,
    "bass": -5.0,
    "drums": -6.0,
    "tom": -8.0,
    "electricGuitar": -6.0,
    "acousticGuitar": -8.0,
    "keys": -10.0,
    "piano": -10.0,
    "synth": -10.0,
    "pads": -12.0,
    "hihat": -11.0,
    "ride": -11.0,
    "overhead": -11.0,
    "room": -14.0,
    "percussion": -10.0,
    "brass": -9.0,
    "sax": -9.0,
    "woodwinds": -10.0,
    "strings": -10.0,
    "fx": -12.0,
    "playback": -7.0,
    "accordion": -9.0,
    "unknown": -7.0,
}

JAZZ_OFFSETS = {
    "leadVocal": -5.0,
    "backingVocal": -9.0,
    "kick": -7.0,
    "snare": -7.0,
    "bass": -5.0,
    "drums": -7.0,
    "tom": -9.0,
    "electricGuitar": -7.0,
    "acousticGuitar": -6.0,
    "keys": -6.0,
    "piano": -5.0,
    "synth": -8.0,
    "pads": -11.0,
    "hihat": -10.0,
    "ride": -9.0,
    "overhead": -10.0,
    "room": -12.0,
    "percussion": -9.0,
    "brass": -6.0,
    "sax": -5.0,
    "woodwinds": -6.0,
    "strings": -8.0,
    "fx": -14.0,
    "playback": -7.0,
    "accordion": -8.0,
    "unknown": -7.0,
}

CLASSICAL_OFFSETS = {
    "leadVocal": -5.0,
    "backingVocal": -7.0,
    "kick": -8.0,
    "snare": -8.0,
    "bass": -6.0,
    "drums": -8.0,
    "tom": -9.0,
    "electricGuitar": -8.0,
    "acousticGuitar": -6.0,
    "keys": -6.0,
    "piano": -5.0,
    "synth": -10.0,
    "pads": -10.0,
    "hihat": -11.0,
    "ride": -11.0,
    "overhead": -9.0,
    "room": -10.0,
    "percussion": -8.0,
    "brass": -6.0,
    "sax": -7.0,
    "woodwinds": -5.0,
    "strings": -5.0,
    "fx": -15.0,
    "playback": -7.0,
    "accordion": -8.0,
    "unknown": -7.0,
}

ELECTRONIC_OFFSETS = {
    "leadVocal": -5.0,
    "backingVocal": -9.0,
    "kick": -3.0,
    "snare": -5.0,
    "bass": -3.0,
    "drums": -5.0,
    "tom": -8.0,
    "electricGuitar": -9.0,
    "acousticGuitar": -10.0,
    "keys": -7.0,
    "piano": -8.0,
    "synth": -5.0,
    "pads": -8.0,
    "hihat": -11.0,
    "ride": -12.0,
    "overhead": -12.0,
    "room": -16.0,
    "percussion": -9.0,
    "brass": -10.0,
    "sax": -10.0,
    "woodwinds": -12.0,
    "strings": -10.0,
    "fx": -8.0,
    "playback": -5.0,
    "accordion": -10.0,
    "unknown": -7.0,
}

CUSTOM_OFFSETS = {
    "leadVocal": -3.0,
    "backingVocal": -8.0,
    "kick": -6.0,
    "snare": -6.0,
    "bass": -5.0,
    "drums": -7.0,
    "tom": -9.0,
    "electricGuitar": -8.0,
    "acousticGuitar": -8.0,
    "keys": -9.0,
    "piano": -8.0,
    "synth": -8.0,
    "pads": -11.0,
    "hihat": -12.0,
    "ride": -12.0,
    "overhead": -12.0,
    "room": -15.0,
    "percussion": -10.0,
    "brass": -9.0,
    "sax": -9.0,
    "woodwinds": -10.0,
    "strings": -10.0,
    "fx": -12.0,
    "playback": -7.0,
    "accordion": -9.0,
    "unknown": -7.0,
}


GENRE_PROFILES = {
    GenreType.CUSTOM: GenreProfile(
        "Custom", CUSTOM_OFFSETS, target_lufs=-18.0, ratio=2.0,
    ),
    GenreType.POP: GenreProfile(
        "Pop", POP_OFFSETS, target_lufs=-18.0, ratio=2.5,
    ),
    GenreType.ROCK: GenreProfile(
        "Rock", ROCK_OFFSETS, target_lufs=-18.0, ratio=3.0,
    ),
    GenreType.JAZZ: GenreProfile(
        "Jazz", JAZZ_OFFSETS, target_lufs=-20.0, ratio=1.5,
    ),
    GenreType.CLASSICAL: GenreProfile(
        "Classical", CLASSICAL_OFFSETS, target_lufs=-23.0, ratio=1.2,
    ),
    GenreType.ELECTRONIC: GenreProfile(
        "Electronic", ELECTRONIC_OFFSETS, target_lufs=-16.0, ratio=4.0,
    ),
}
