"""Genre profiles with instrument balance offsets."""

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
    """Genre profile with name and instrument offsets."""
    name: str
    instrument_offsets: Dict[str, float]


DEFAULT_OFFSETS = {
    "kick": -7.0,
    "snare": -7.0,
    "tom": -9.0,
    "hihat": -12.0,
    "ride": -12.0,
    "overhead": -12.0,
    "room": -15.0,
    "bass": -7.0,
    "drums": -8.0,
    "leadVocal": -5.0,
    "backingVocal": -9.0,
    "electricGuitar": -8.0,
    "acousticGuitar": -9.0,
    "unknown": -7.0,
}

GENRE_PROFILES = {
    GenreType.CUSTOM: GenreProfile("Custom", DEFAULT_OFFSETS),
    GenreType.ROCK: GenreProfile("Rock", {**DEFAULT_OFFSETS}),
    GenreType.POP: GenreProfile("Pop", {**DEFAULT_OFFSETS}),
    GenreType.JAZZ: GenreProfile("Jazz", {**DEFAULT_OFFSETS}),
    GenreType.CLASSICAL: GenreProfile("Classical", {**DEFAULT_OFFSETS}),
    GenreType.ELECTRONIC: GenreProfile("Electronic", {**DEFAULT_OFFSETS}),
}
