"""
Per-instrument LUFS targets for automatic gain staging.

Provides target loudness levels for each instrument type relative
to a base reference level (default -18 LUFS), with genre-specific
modifiers for different musical styles.
"""

# Instrument LUFS offsets from base reference (-18 LUFS by default)
# Positive = louder than reference, negative = quieter
INSTRUMENT_LUFS_OFFSETS = {
    # Drums
    "kick": 3.0,        # Kick drum sits prominently in most mixes
    "snare": 2.0,       # Snare is key rhythmic element
    "hihat": -4.0,      # Hi-hats should sit back, avoid harshness
    "toms": 0.0,        # Toms at reference level, fill hits
    "overheads": -2.0,  # Overheads provide ambience, not dominant
    "percussion": -1.0, # Percussion adds texture, slightly below ref

    # Bass
    "bass_guitar": 1.0,     # Bass foundation, slightly above ref

    # Guitars
    "electric_guitar": -1.0,  # Electric guitar sits in the mid range
    "acoustic_guitar": -2.0,  # Acoustic often accompaniment, slightly lower

    # Keys
    "keys": -1.5,       # Keys fill harmonic space, slightly below

    # Vocals
    "vocals": 4.0,      # Lead vocals are the loudest element in most mixes

    # Orchestral
    "brass": 0.5,       # Brass can cut through, slight boost
    "strings": -2.0,    # Strings provide pad/texture, sit back
}

# Genre-specific modifiers applied on top of base offsets
# Values are additive dB adjustments per instrument for each genre
GENRE_MODIFIERS = {
    "rock": {
        "kick": 1.0,
        "snare": 1.5,
        "hihat": -1.0,
        "toms": 1.0,
        "overheads": 0.0,
        "bass_guitar": 1.0,
        "electric_guitar": 2.0,    # Guitar-forward in rock
        "acoustic_guitar": -1.0,
        "keys": -1.0,
        "vocals": 1.0,
        "brass": 0.0,
        "strings": -1.0,
        "percussion": 0.0,
    },
    "jazz": {
        "kick": -2.0,              # Softer drums in jazz
        "snare": -1.0,
        "hihat": 1.0,              # Hi-hat brushwork more prominent
        "toms": -1.0,
        "overheads": 1.0,          # Room/cymbals important in jazz
        "bass_guitar": 2.0,        # Upright/electric bass featured
        "electric_guitar": -1.0,
        "acoustic_guitar": 1.0,
        "keys": 2.0,               # Piano/keys often featured
        "vocals": 2.0,
        "brass": 3.0,              # Brass is central to jazz
        "strings": 1.0,
        "percussion": 1.0,
    },
    "pop": {
        "kick": 2.0,
        "snare": 2.0,
        "hihat": -1.0,
        "toms": 0.0,
        "overheads": -1.0,
        "bass_guitar": 1.0,
        "electric_guitar": 0.0,
        "acoustic_guitar": 0.0,
        "keys": 1.0,
        "vocals": 3.0,             # Vocals king in pop
        "brass": -1.0,
        "strings": 0.0,
        "percussion": 0.0,
    },
    "electronic": {
        "kick": 3.0,               # Heavy kick in electronic
        "snare": 1.0,
        "hihat": 0.0,
        "toms": -1.0,
        "overheads": -3.0,
        "bass_guitar": 3.0,        # Sub-bass dominant
        "electric_guitar": -2.0,
        "acoustic_guitar": -3.0,
        "keys": 2.0,               # Synths featured
        "vocals": 1.0,
        "brass": -2.0,
        "strings": 0.0,
        "percussion": 1.0,
    },
    "classical": {
        "kick": -3.0,
        "snare": -3.0,
        "hihat": -3.0,
        "toms": -2.0,
        "overheads": 2.0,          # Natural room sound
        "bass_guitar": -2.0,
        "electric_guitar": -3.0,
        "acoustic_guitar": 1.0,
        "keys": 2.0,               # Piano prominent
        "vocals": 3.0,             # Operatic/choral vocals
        "brass": 2.0,
        "strings": 3.0,            # Strings are central
        "percussion": 0.0,         # Timpani etc.
    },
    "acoustic": {
        "kick": -1.0,              # Cajon/light kick
        "snare": -1.0,
        "hihat": -2.0,
        "toms": -1.0,
        "overheads": 1.0,
        "bass_guitar": 1.0,
        "electric_guitar": -2.0,
        "acoustic_guitar": 3.0,    # Acoustic guitar is central
        "keys": 1.0,
        "vocals": 3.0,
        "brass": 0.0,
        "strings": 2.0,
        "percussion": 1.0,         # Shaker/tambourine
    },
    "metal": {
        "kick": 3.0,               # Double kick prominent
        "snare": 2.0,
        "hihat": -1.0,
        "toms": 1.0,
        "overheads": 0.0,
        "bass_guitar": 2.0,
        "electric_guitar": 3.0,    # Wall of guitars
        "acoustic_guitar": -3.0,
        "keys": -2.0,
        "vocals": 2.0,
        "brass": -2.0,
        "strings": -1.0,
        "percussion": 0.0,
    },
}


def get_target_lufs(instrument, genre=None, base=-18.0):
    """
    Compute the target LUFS level for an instrument.

    Args:
        instrument: string instrument type (must be key in INSTRUMENT_LUFS_OFFSETS)
        genre: optional genre string for genre-specific adjustment
        base: base reference LUFS level (default -18 LUFS)

    Returns:
        target_lufs: float target loudness in LUFS

    Examples:
        >>> get_target_lufs("vocals")
        -14.0
        >>> get_target_lufs("kick", genre="electronic")
        -12.0
        >>> get_target_lufs("electric_guitar", genre="rock", base=-16)
        -15.0
    """
    # Get base offset for instrument
    offset = INSTRUMENT_LUFS_OFFSETS.get(instrument, 0.0)

    # Apply genre modifier if available
    genre_offset = 0.0
    if genre and genre.lower() in GENRE_MODIFIERS:
        genre_mods = GENRE_MODIFIERS[genre.lower()]
        genre_offset = genre_mods.get(instrument, 0.0)

    target = base + offset + genre_offset
    return float(target)


def get_all_targets(genre=None, base=-18.0):
    """
    Get target LUFS for all instrument types.

    Args:
        genre: optional genre string
        base: base reference LUFS level

    Returns:
        dict mapping instrument name to target LUFS
    """
    return {
        inst: get_target_lufs(inst, genre=genre, base=base)
        for inst in INSTRUMENT_LUFS_OFFSETS
    }


def get_relative_balance(genre=None):
    """
    Get relative dB balance between all instruments (independent of base).

    Args:
        genre: optional genre string

    Returns:
        dict mapping instrument name to relative dB offset
    """
    targets = get_all_targets(genre=genre, base=0.0)
    # Normalize so vocals = 0 dB reference
    vocal_target = targets.get("vocals", 0.0)
    return {inst: level - vocal_target for inst, level in targets.items()}
