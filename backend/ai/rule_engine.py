"""
Deterministic rule engine for live sound mixing.

Provides fast, predictable mixing decisions based on professional audio
engineering literature and best practices. No ML inference required.

All frequency values in Hz, gain in dB, times in ms unless noted.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EQ band representation
# ---------------------------------------------------------------------------

class EQBand:
    """Single parametric EQ band."""

    __slots__ = ("freq", "gain", "q", "band_type")

    def __init__(
        self,
        freq: float,
        gain: float,
        q: float = 1.0,
        band_type: str = "peaking",
    ):
        """
        Args:
            freq: Center frequency in Hz.
            gain: Gain in dB (negative = cut, positive = boost).
            q: Q factor (bandwidth).
            band_type: 'peaking', 'lowshelf', 'highshelf', 'highpass', 'lowpass'.
        """
        self.freq = freq
        self.gain = gain
        self.q = q
        self.band_type = band_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freq": self.freq,
            "gain": self.gain,
            "q": self.q,
            "type": self.band_type,
        }


# ---------------------------------------------------------------------------
# HPF (High-Pass Filter) frequencies per instrument
# ---------------------------------------------------------------------------

# Based on Yamaha CL/TF, Digico, and general live-sound best practices.
# These are starting points — room acoustics and mic placement matter.
HPF_FREQUENCIES: Dict[str, float] = {
    # Drums
    "kick": 30.0,       # Allow sub-bass fundamental, remove rumble
    "snare": 80.0,      # Remove low-end bleed from kick
    "tom": 60.0,        # Preserve body, cut sub rumble
    "hihat": 200.0,     # Aggressive HPF to remove bleed
    "ride": 200.0,      # Similar to hihat
    "cymbals": 200.0,   # Remove all low-end bleed
    "overheads": 100.0, # Keep some body of kit
    "room": 80.0,       # Keep room character

    # Bass
    "bass": 30.0,       # Preserve low fundamentals
    "sub": 20.0,        # Sub bass — minimal filtering

    # Guitars
    "electricGuitar": 100.0,   # Fundamental ~82 Hz (low E), safe cut below
    "acousticGuitar": 80.0,    # Preserve some body
    "guitar": 100.0,           # Generic guitar

    # Keys / Synth
    "synth": 30.0,      # Depends on patch; conservative default
    "piano": 40.0,      # Lowest note ~27.5 Hz
    "organ": 30.0,      # Can go very low
    "keys": 40.0,       # Generic keys

    # Vocals
    "leadVocal": 80.0,    # Standard vocal HPF
    "backVocal": 100.0,   # Slightly higher for backing vox
    "vocal": 80.0,        # Generic vocal

    # Brass / Woodwinds
    "trumpet": 120.0,
    "saxophone": 80.0,
    "trombone": 60.0,
    "flute": 200.0,
    "clarinet": 120.0,

    # Strings
    "violin": 150.0,
    "cello": 60.0,
    "contrabass": 30.0,

    # Other
    "accordion": 60.0,
    "harmonica": 150.0,
    "playback": 20.0,  # Full range backing track
    "djTrack": 20.0,
}


# ---------------------------------------------------------------------------
# Default EQ presets per instrument
# ---------------------------------------------------------------------------

DEFAULT_EQ: Dict[str, List[Dict[str, Any]]] = {
    "kick": [
        {"freq": 60, "gain": 3.0, "q": 1.5, "type": "peaking"},      # Sub weight
        {"freq": 400, "gain": -4.0, "q": 2.0, "type": "peaking"},     # Remove mud/boxiness
        {"freq": 2500, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Beater attack/click
        {"freq": 8000, "gain": 1.5, "q": 0.7, "type": "highshelf"},   # Air / presence
    ],
    "snare": [
        {"freq": 200, "gain": 2.0, "q": 1.5, "type": "peaking"},      # Body / fatness
        {"freq": 800, "gain": -3.0, "q": 2.0, "type": "peaking"},     # Remove boxiness
        {"freq": 3500, "gain": 2.5, "q": 1.5, "type": "peaking"},     # Crack / attack
        {"freq": 10000, "gain": 2.0, "q": 0.7, "type": "highshelf"},  # Snare wire sizzle
    ],
    "tom": [
        {"freq": 100, "gain": 2.5, "q": 1.5, "type": "peaking"},      # Fundamental body
        {"freq": 400, "gain": -3.5, "q": 2.0, "type": "peaking"},     # Remove boxiness
        {"freq": 3000, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Attack definition
    ],
    "hihat": [
        {"freq": 400, "gain": -4.0, "q": 1.0, "type": "peaking"},     # Remove bleed mud
        {"freq": 6000, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Brightness
        {"freq": 10000, "gain": 1.5, "q": 0.7, "type": "highshelf"},  # Air / shimmer
    ],
    "overheads": [
        {"freq": 250, "gain": -2.0, "q": 1.0, "type": "peaking"},     # Reduce mud
        {"freq": 3000, "gain": 1.5, "q": 1.0, "type": "peaking"},     # Cymbal definition
        {"freq": 12000, "gain": 2.0, "q": 0.7, "type": "highshelf"},  # Air / sparkle
    ],
    "room": [
        {"freq": 300, "gain": -3.0, "q": 1.0, "type": "peaking"},     # Control mud
        {"freq": 2000, "gain": 1.0, "q": 1.0, "type": "peaking"},     # Presence
        {"freq": 8000, "gain": 1.5, "q": 0.7, "type": "highshelf"},   # Openness
    ],
    "bass": [
        {"freq": 80, "gain": 2.0, "q": 1.5, "type": "peaking"},       # Low-end weight
        {"freq": 250, "gain": -3.0, "q": 2.0, "type": "peaking"},     # Clean up mud
        {"freq": 700, "gain": 1.5, "q": 1.5, "type": "peaking"},      # Growl / midrange definition
        {"freq": 2500, "gain": 2.0, "q": 1.5, "type": "peaking"},     # String attack / fret noise
    ],
    "electricGuitar": [
        {"freq": 200, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Reduce boom
        {"freq": 800, "gain": 1.5, "q": 1.5, "type": "peaking"},      # Body / crunch
        {"freq": 3000, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Bite / presence
        {"freq": 6000, "gain": -1.5, "q": 2.0, "type": "peaking"},    # Tame harshness
    ],
    "acousticGuitar": [
        {"freq": 100, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Reduce boominess
        {"freq": 250, "gain": -1.5, "q": 2.0, "type": "peaking"},     # Control mud
        {"freq": 3000, "gain": 2.0, "q": 1.5, "type": "peaking"},     # String clarity
        {"freq": 10000, "gain": 2.0, "q": 0.7, "type": "highshelf"},  # Sparkle / air
    ],
    "leadVocal": [
        {"freq": 200, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Proximity effect / mud
        {"freq": 800, "gain": -1.5, "q": 2.0, "type": "peaking"},     # Nasal honk reduction
        {"freq": 3000, "gain": 2.5, "q": 1.5, "type": "peaking"},     # Presence / intelligibility
        {"freq": 10000, "gain": 2.0, "q": 0.7, "type": "highshelf"},  # Air / brightness
    ],
    "backVocal": [
        {"freq": 200, "gain": -3.0, "q": 1.5, "type": "peaking"},     # Clean up proximity effect
        {"freq": 800, "gain": -2.0, "q": 2.0, "type": "peaking"},     # Reduce nasal
        {"freq": 3500, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Presence
        {"freq": 8000, "gain": 1.5, "q": 0.7, "type": "highshelf"},   # Air
    ],
    "synth": [
        {"freq": 300, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Reduce mud
        {"freq": 2000, "gain": 1.5, "q": 1.0, "type": "peaking"},     # Presence
        {"freq": 8000, "gain": 1.0, "q": 0.7, "type": "highshelf"},   # Brightness
    ],
    "piano": [
        {"freq": 250, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Reduce mud
        {"freq": 1000, "gain": 1.0, "q": 1.5, "type": "peaking"},     # Body
        {"freq": 5000, "gain": 2.0, "q": 1.0, "type": "peaking"},     # Clarity
        {"freq": 10000, "gain": 1.5, "q": 0.7, "type": "highshelf"},  # Sparkle
    ],
    "accordion": [
        {"freq": 200, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Reduce boom
        {"freq": 1000, "gain": 1.0, "q": 1.5, "type": "peaking"},     # Body
        {"freq": 3500, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Reeds clarity
        {"freq": 8000, "gain": -1.0, "q": 1.0, "type": "peaking"},    # Tame harsh reeds
    ],
    "trumpet": [
        {"freq": 300, "gain": -1.5, "q": 1.5, "type": "peaking"},     # Reduce boxiness
        {"freq": 1500, "gain": 1.5, "q": 1.5, "type": "peaking"},     # Body
        {"freq": 5000, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Brilliance
        {"freq": 8000, "gain": -1.0, "q": 2.0, "type": "peaking"},    # Tame harshness
    ],
    "saxophone": [
        {"freq": 250, "gain": -2.0, "q": 1.5, "type": "peaking"},     # Reduce honk
        {"freq": 800, "gain": 1.5, "q": 1.5, "type": "peaking"},      # Body
        {"freq": 3000, "gain": 2.0, "q": 1.5, "type": "peaking"},     # Presence
        {"freq": 8000, "gain": 1.0, "q": 0.7, "type": "highshelf"},   # Air
    ],
    "playback": [
        # Minimal EQ for pre-mixed tracks
        {"freq": 200, "gain": -1.0, "q": 0.7, "type": "peaking"},     # Slight low-mid cleanup
    ],
}


# ---------------------------------------------------------------------------
# Compressor presets per instrument
# ---------------------------------------------------------------------------

DEFAULT_COMPRESSOR: Dict[str, Dict[str, Any]] = {
    "kick": {
        "threshold": -12.0, "ratio": 4.0,
        "attack": 20.0, "release": 80.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "snare": {
        "threshold": -10.0, "ratio": 4.5,
        "attack": 5.0, "release": 100.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "tom": {
        "threshold": -12.0, "ratio": 4.0,
        "attack": 10.0, "release": 100.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "hihat": {
        "threshold": -18.0, "ratio": 3.0,
        "attack": 5.0, "release": 80.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "overheads": {
        "threshold": -20.0, "ratio": 3.0,
        "attack": 10.0, "release": 150.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "room": {
        "threshold": -24.0, "ratio": 2.5,
        "attack": 15.0, "release": 200.0,
        "knee": 1.0, "makeup_gain": 0.0,
    },
    "bass": {
        "threshold": -15.0, "ratio": 4.0,
        "attack": 25.0, "release": 200.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "electricGuitar": {
        "threshold": -14.0, "ratio": 3.0,
        "attack": 15.0, "release": 180.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "acousticGuitar": {
        "threshold": -18.0, "ratio": 3.0,
        "attack": 15.0, "release": 150.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "leadVocal": {
        "threshold": -18.0, "ratio": 3.0,
        "attack": 10.0, "release": 120.0,
        "knee": 3.0, "makeup_gain": 0.0,
    },
    "backVocal": {
        "threshold": -20.0, "ratio": 3.5,
        "attack": 8.0, "release": 100.0,
        "knee": 3.0, "makeup_gain": 0.0,
    },
    "synth": {
        "threshold": -16.0, "ratio": 2.5,
        "attack": 15.0, "release": 150.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "piano": {
        "threshold": -20.0, "ratio": 2.5,
        "attack": 20.0, "release": 200.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "accordion": {
        "threshold": -16.0, "ratio": 3.0,
        "attack": 10.0, "release": 150.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "trumpet": {
        "threshold": -14.0, "ratio": 3.0,
        "attack": 5.0, "release": 120.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "saxophone": {
        "threshold": -16.0, "ratio": 3.0,
        "attack": 8.0, "release": 140.0,
        "knee": 2.0, "makeup_gain": 0.0,
    },
    "playback": {
        "threshold": -10.0, "ratio": 2.0,
        "attack": 20.0, "release": 200.0,
        "knee": 3.0, "makeup_gain": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Gate presets per instrument
# ---------------------------------------------------------------------------

DEFAULT_GATE: Dict[str, Dict[str, Any]] = {
    "kick": {
        "threshold": -30.0, "attack": 0.5, "hold": 50.0, "release": 200.0,
        "range": 40.0,
    },
    "snare": {
        "threshold": -28.0, "attack": 0.5, "hold": 40.0, "release": 150.0,
        "range": 30.0,
    },
    "tom": {
        "threshold": -32.0, "attack": 1.0, "hold": 60.0, "release": 250.0,
        "range": 40.0,
    },
    "hihat": {
        "threshold": -36.0, "attack": 0.3, "hold": 30.0, "release": 100.0,
        "range": 20.0,
    },
    "overheads": {
        # Overheads typically don't need gating, but provide safe defaults
        "threshold": -60.0, "attack": 5.0, "hold": 100.0, "release": 300.0,
        "range": 6.0,
    },
    "leadVocal": {
        "threshold": -40.0, "attack": 1.0, "hold": 80.0, "release": 300.0,
        "range": 10.0,
    },
    "backVocal": {
        "threshold": -36.0, "attack": 1.0, "hold": 60.0, "release": 250.0,
        "range": 15.0,
    },
    "bass": {
        "threshold": -40.0, "attack": 2.0, "hold": 80.0, "release": 300.0,
        "range": 15.0,
    },
    "electricGuitar": {
        "threshold": -45.0, "attack": 2.0, "hold": 60.0, "release": 250.0,
        "range": 10.0,
    },
    "acousticGuitar": {
        "threshold": -42.0, "attack": 2.0, "hold": 80.0, "release": 300.0,
        "range": 10.0,
    },
    "synth": {
        # Synths rarely need gating
        "threshold": -60.0, "attack": 5.0, "hold": 100.0, "release": 300.0,
        "range": 6.0,
    },
    "piano": {
        "threshold": -50.0, "attack": 3.0, "hold": 100.0, "release": 400.0,
        "range": 6.0,
    },
    "accordion": {
        "threshold": -42.0, "attack": 2.0, "hold": 80.0, "release": 300.0,
        "range": 10.0,
    },
    "trumpet": {
        "threshold": -38.0, "attack": 1.0, "hold": 60.0, "release": 200.0,
        "range": 15.0,
    },
    "saxophone": {
        "threshold": -40.0, "attack": 1.0, "hold": 70.0, "release": 250.0,
        "range": 12.0,
    },
    "playback": {
        # No gating for playback
        "threshold": -80.0, "attack": 10.0, "hold": 200.0, "release": 500.0,
        "range": 3.0,
    },
}


# ---------------------------------------------------------------------------
# Panning defaults
# ---------------------------------------------------------------------------

DEFAULT_PAN: Dict[str, float] = {
    # Center (0.0)
    "kick": 0.0,
    "snare": 0.0,
    "bass": 0.0,
    "leadVocal": 0.0,
    "playback": 0.0,
    "sub": 0.0,

    # Slight off-center
    "hihat": -30.0,     # Drummer perspective: hihat slightly left
    "ride": 30.0,       # Drummer perspective: ride slightly right

    # Moderate pan
    "tom": 0.0,          # Floor tom panned; this is a default; per-tom varies
    "overheads": 0.0,    # Typically stereo pair; individual L/R set separately

    # Wider stereo field
    "electricGuitar": -40.0,    # Stage left typical
    "acousticGuitar": 40.0,     # Stage right typical
    "synth": 25.0,
    "piano": -25.0,
    "accordion": 20.0,

    # Backing vocals spread
    "backVocal": 0.0,  # Individual BGV panned per arrangement

    # Brass
    "trumpet": -30.0,
    "saxophone": 30.0,
    "trombone": -20.0,

    # Room / ambience center
    "room": 0.0,
}


# ---------------------------------------------------------------------------
# Feedback frequency bands commonly problematic in live sound
# ---------------------------------------------------------------------------

FEEDBACK_BANDS: List[Dict[str, Any]] = [
    {"freq": 250, "label": "low_mid_room_mode"},
    {"freq": 400, "label": "boxiness"},
    {"freq": 630, "label": "nasal"},
    {"freq": 800, "label": "honk"},
    {"freq": 1000, "label": "mid_honk"},
    {"freq": 1600, "label": "nasal_upper"},
    {"freq": 2000, "label": "harshness_low"},
    {"freq": 2500, "label": "presence_feedback"},
    {"freq": 3150, "label": "presence_ring"},
    {"freq": 4000, "label": "harshness"},
    {"freq": 5000, "label": "sibilance_ring"},
    {"freq": 6300, "label": "sibilance"},
    {"freq": 8000, "label": "high_ring"},
]


# ---------------------------------------------------------------------------
# RuleEngine class
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Deterministic mixing rule engine.

    Provides instant, predictable mixing parameters based on professional
    live-sound engineering practices. No inference latency.
    """

    def __init__(self) -> None:
        logger.info("RuleEngine initialized")

    # ---- HPF ----

    def apply_hpf(self, instrument_type: str) -> float:
        """
        Get recommended high-pass filter frequency for an instrument.

        Args:
            instrument_type: Instrument type identifier (e.g. 'kick', 'leadVocal').

        Returns:
            HPF frequency in Hz.
        """
        freq = HPF_FREQUENCIES.get(instrument_type)
        if freq is None:
            # Try case-insensitive match
            lower_map = {k.lower(): v for k, v in HPF_FREQUENCIES.items()}
            freq = lower_map.get(instrument_type.lower(), 100.0)
            logger.debug(
                f"No exact HPF match for '{instrument_type}', "
                f"using {'matched' if freq != 100.0 else 'default'} {freq} Hz"
            )
        return freq

    # ---- EQ ----

    def get_default_eq(self, instrument_type: str) -> List[Dict[str, Any]]:
        """
        Get default EQ bands for an instrument type.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            List of EQ band dicts with keys: freq, gain, q, type.
        """
        eq = DEFAULT_EQ.get(instrument_type)
        if eq is None:
            lower_map = {k.lower(): v for k, v in DEFAULT_EQ.items()}
            eq = lower_map.get(instrument_type.lower())
        if eq is None:
            logger.debug(f"No EQ preset for '{instrument_type}', returning flat")
            return []
        # Return copies to prevent mutation
        return [dict(band) for band in eq]

    # ---- Compressor ----

    def get_default_compressor(self, instrument_type: str) -> Dict[str, Any]:
        """
        Get default compressor settings for an instrument.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            Dict with keys: threshold, ratio, attack, release, knee, makeup_gain.
        """
        comp = DEFAULT_COMPRESSOR.get(instrument_type)
        if comp is None:
            lower_map = {k.lower(): v for k, v in DEFAULT_COMPRESSOR.items()}
            comp = lower_map.get(instrument_type.lower())
        if comp is None:
            logger.debug(
                f"No compressor preset for '{instrument_type}', "
                "returning gentle defaults"
            )
            return {
                "threshold": -20.0, "ratio": 2.0,
                "attack": 15.0, "release": 150.0,
                "knee": 2.0, "makeup_gain": 0.0,
            }
        return dict(comp)

    # ---- Gate ----

    def get_default_gate(self, instrument_type: str) -> Dict[str, Any]:
        """
        Get default gate settings for an instrument.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            Dict with keys: threshold, attack, hold, release, range.
        """
        gate = DEFAULT_GATE.get(instrument_type)
        if gate is None:
            lower_map = {k.lower(): v for k, v in DEFAULT_GATE.items()}
            gate = lower_map.get(instrument_type.lower())
        if gate is None:
            logger.debug(
                f"No gate preset for '{instrument_type}', "
                "returning open defaults"
            )
            return {
                "threshold": -60.0, "attack": 5.0,
                "hold": 100.0, "release": 300.0,
                "range": 6.0,
            }
        return dict(gate)

    # ---- Pan ----

    def get_default_pan(self, instrument_type: str) -> float:
        """
        Get default pan position for an instrument.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            Pan value from -100.0 (hard left) to 100.0 (hard right).
        """
        pan = DEFAULT_PAN.get(instrument_type)
        if pan is None:
            lower_map = {k.lower(): v for k, v in DEFAULT_PAN.items()}
            pan = lower_map.get(instrument_type.lower(), 0.0)
        return pan

    # ---- Gain Staging ----

    def gain_stage_to_target(
        self,
        current_lufs: float,
        target_lufs: float = -20.0,
    ) -> float:
        """
        Calculate gain adjustment to reach target loudness.

        Uses K-20 metering standard by default (-20 LUFS for music mixing).
        Broadcast standard uses -18 LUFS (set target_lufs=-18).

        Args:
            current_lufs: Current integrated loudness in LUFS.
            target_lufs: Target loudness in LUFS (default -20 for K-20).

        Returns:
            Gain adjustment in dB (positive = boost, negative = cut).
            Clamped to safe range of -24 to +12 dB.
        """
        if not math.isfinite(current_lufs):
            logger.warning(
                f"Non-finite current_lufs ({current_lufs}), returning 0 adjustment"
            )
            return 0.0

        adjustment = target_lufs - current_lufs

        # Clamp to safe range to prevent destructive gain changes
        max_boost = 12.0
        max_cut = -24.0
        clamped = max(max_cut, min(max_boost, adjustment))

        if clamped != adjustment:
            logger.warning(
                f"Gain adjustment clamped from {adjustment:.1f} dB to "
                f"{clamped:.1f} dB (safety limit)"
            )

        return round(clamped, 1)

    # ---- Feedback Handling ----

    def handle_feedback(self, frequency_hz: float) -> Dict[str, Any]:
        """
        Generate notch EQ parameters to suppress feedback at a given frequency.

        Creates a narrow notch filter centered on the feedback frequency.
        Starts with a moderate cut and narrow Q to minimize tonal impact.

        Args:
            frequency_hz: Detected feedback frequency in Hz.

        Returns:
            Dict with keys: freq, gain, q, type, severity.
        """
        # Validate frequency range (human hearing)
        freq = max(20.0, min(20000.0, frequency_hz))

        # Start with moderate cut; severe feedback needs deeper cut
        initial_gain = -6.0
        initial_q = 8.0  # Narrow notch to preserve surrounding frequencies

        # Find nearest known problem band for context
        nearest_band = min(
            FEEDBACK_BANDS,
            key=lambda b: abs(b["freq"] - freq),
        )
        dist = abs(nearest_band["freq"] - freq) / nearest_band["freq"]

        severity = "moderate"
        if dist < 0.05:
            # Very close to a known problem frequency — may need deeper cut
            initial_gain = -9.0
            initial_q = 10.0
            severity = "high"

        return {
            "freq": round(freq, 1),
            "gain": initial_gain,
            "q": initial_q,
            "type": "peaking",
            "severity": severity,
            "nearest_band": nearest_band["label"],
        }

    # ---- Utility: full channel preset ----

    def get_full_channel_preset(
        self, instrument_type: str
    ) -> Dict[str, Any]:
        """
        Get a complete channel preset for an instrument type.

        Combines HPF, EQ, compressor, gate, and pan into one dict.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            Dict with keys: hpf, eq, compressor, gate, pan, instrument.
        """
        return {
            "instrument": instrument_type,
            "hpf": self.apply_hpf(instrument_type),
            "eq": self.get_default_eq(instrument_type),
            "compressor": self.get_default_compressor(instrument_type),
            "gate": self.get_default_gate(instrument_type),
            "pan": self.get_default_pan(instrument_type),
        }

    # ---- Utility: should gate be active? ----

    @staticmethod
    def should_enable_gate(instrument_type: str) -> bool:
        """
        Determine whether gating is recommended for this instrument type.

        Gates are generally useful for drums and close-miked sources,
        but should be off for sustained/ambient sources.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            True if gating is recommended.
        """
        gate_recommended = {
            "kick", "snare", "tom", "hihat",
            "leadVocal", "backVocal",
        }
        return instrument_type in gate_recommended

    # ---- Utility: should enable compressor? ----

    @staticmethod
    def should_enable_compressor(instrument_type: str) -> bool:
        """
        Determine whether compression is recommended for this instrument type.

        Almost all live sources benefit from some compression.

        Args:
            instrument_type: Instrument type identifier.

        Returns:
            True if compression is recommended.
        """
        # Playback tracks are already mastered, skip compression
        no_compress = {"playback", "djTrack"}
        return instrument_type not in no_compress
