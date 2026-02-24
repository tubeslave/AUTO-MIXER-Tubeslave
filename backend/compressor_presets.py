"""
Compressor presets knowledge base for Auto Compressor.

Per-instrument presets and task variants (base, punch, control, gentle, aggressive, broadcast).
All levels in dB, times in ms. Wing ratio is applied as string ("2.0", "4.0", etc.).
"""

import json
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


PEAK_DETECTOR_INSTRUMENTS = {
    "kick",
    "snare",
    "tom",
    "hihat",
    "ride",
    "cymbals",
    "drums",
}

# Default presets (instrument type -> task -> params)
DEFAULT_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "kick": {
        "base": {"threshold": -12, "ratio": 4.0, "attack_ms": 20, "release_ms": 80, "knee": 2, "makeup_gain": 0},
        "punch": {"threshold": -10, "ratio": 5.0, "attack_ms": 25, "release_ms": 70, "knee": 2, "makeup_gain": 1},
        "control": {"threshold": -14, "ratio": 6.0, "attack_ms": 15, "release_ms": 90, "knee": 3, "makeup_gain": 0},
        "gentle": {"threshold": -8, "ratio": 3.0, "attack_ms": 30, "release_ms": 100, "knee": 1, "makeup_gain": 0},
        "aggressive": {"threshold": -15, "ratio": 6.0, "attack_ms": 10, "release_ms": 60, "knee": 4, "makeup_gain": 2},
    },
    "snare": {
        "base": {"threshold": -10, "ratio": 4.5, "attack_ms": 5, "release_ms": 100, "knee": 2, "makeup_gain": 0},
        "punch": {"threshold": -8, "ratio": 4.0, "attack_ms": 8, "release_ms": 80, "knee": 2, "makeup_gain": 1},
        "control": {"threshold": -12, "ratio": 5.0, "attack_ms": 3, "release_ms": 120, "knee": 3, "makeup_gain": 0},
        "gentle": {"threshold": -6, "ratio": 3.0, "attack_ms": 15, "release_ms": 150, "knee": 1, "makeup_gain": 0},
        "aggressive": {"threshold": -12, "ratio": 5.0, "attack_ms": 0.5, "release_ms": 80, "knee": 4, "makeup_gain": 1},
    },
    "tom": {
        "base": {"threshold": -12, "ratio": 4.0, "attack_ms": 10, "release_ms": 100, "knee": 2, "makeup_gain": 0},
        "punch": {"threshold": -10, "ratio": 4.0, "attack_ms": 15, "release_ms": 90, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -14, "ratio": 5.0, "attack_ms": 8, "release_ms": 110, "knee": 3, "makeup_gain": 0},
        "gentle": {"threshold": -8, "ratio": 3.0, "attack_ms": 20, "release_ms": 120, "knee": 1, "makeup_gain": 0},
        "aggressive": {"threshold": -14, "ratio": 5.0, "attack_ms": 5, "release_ms": 80, "knee": 3, "makeup_gain": 1},
    },
    "hihat": {
        "base": {"threshold": -18, "ratio": 3.0, "attack_ms": 5, "release_ms": 80, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -20, "ratio": 4.0, "attack_ms": 3, "release_ms": 100, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -14, "ratio": 2.0, "attack_ms": 10, "release_ms": 120, "knee": 1, "makeup_gain": 0},
    },
    "overheads": {
        "base": {"threshold": -20, "ratio": 3.0, "attack_ms": 10, "release_ms": 150, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -22, "ratio": 4.0, "attack_ms": 8, "release_ms": 180, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -16, "ratio": 2.0, "attack_ms": 15, "release_ms": 200, "knee": 1, "makeup_gain": 0},
    },
    "room": {
        "base": {"threshold": -24, "ratio": 2.5, "attack_ms": 15, "release_ms": 200, "knee": 1, "makeup_gain": 0},
        "gentle": {"threshold": -20, "ratio": 2.0, "attack_ms": 20, "release_ms": 250, "knee": 1, "makeup_gain": 0},
    },
    "bass": {
        "base": {"threshold": -15, "ratio": 4.0, "attack_ms": 25, "release_ms": 200, "knee": 2, "makeup_gain": 0},
        "punch": {"threshold": -12, "ratio": 4.0, "attack_ms": 30, "release_ms": 180, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -18, "ratio": 5.0, "attack_ms": 20, "release_ms": 250, "knee": 3, "makeup_gain": 0},
        "gentle": {"threshold": -10, "ratio": 3.0, "attack_ms": 40, "release_ms": 300, "knee": 1, "makeup_gain": 0},
        "aggressive": {"threshold": -18, "ratio": 6.0, "attack_ms": 15, "release_ms": 150, "knee": 4, "makeup_gain": 1},
    },
    "electricGuitar": {
        "base": {"threshold": -14, "ratio": 3.0, "attack_ms": 15, "release_ms": 180, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -16, "ratio": 4.0, "attack_ms": 12, "release_ms": 200, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -10, "ratio": 2.0, "attack_ms": 25, "release_ms": 250, "knee": 1, "makeup_gain": 0},
        "aggressive": {"threshold": -16, "ratio": 4.0, "attack_ms": 10, "release_ms": 150, "knee": 3, "makeup_gain": 0},
    },
    "acousticGuitar": {
        "base": {"threshold": -18, "ratio": 3.0, "attack_ms": 15, "release_ms": 150, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -14, "ratio": 2.0, "attack_ms": 20, "release_ms": 200, "knee": 1, "makeup_gain": 0},
        "control": {"threshold": -20, "ratio": 4.0, "attack_ms": 10, "release_ms": 180, "knee": 2, "makeup_gain": 0},
    },
    "leadVocal": {
        "base": {"threshold": -18, "ratio": 3.0, "attack_ms": 12, "release_ms": 150, "knee": 2, "makeup_gain": 0},
        "broadcast": {"threshold": -20, "ratio": 4.0, "attack_ms": 8, "release_ms": 180, "knee": 3, "makeup_gain": 0},
        "control": {"threshold": -22, "ratio": 4.0, "attack_ms": 10, "release_ms": 200, "knee": 3, "makeup_gain": 0},
        "gentle": {"threshold": -14, "ratio": 2.0, "attack_ms": 20, "release_ms": 200, "knee": 1, "makeup_gain": 0},
        "aggressive": {"threshold": -22, "ratio": 5.0, "attack_ms": 5, "release_ms": 120, "knee": 4, "makeup_gain": 1},
    },
    "backVocal": {
        "base": {"threshold": -20, "ratio": 3.0, "attack_ms": 15, "release_ms": 180, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -22, "ratio": 4.0, "attack_ms": 12, "release_ms": 200, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -16, "ratio": 2.0, "attack_ms": 25, "release_ms": 220, "knee": 1, "makeup_gain": 0},
    },
    "synth": {
        "base": {"threshold": -16, "ratio": 3.0, "attack_ms": 15, "release_ms": 150, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -18, "ratio": 4.0, "attack_ms": 12, "release_ms": 180, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -12, "ratio": 2.0, "attack_ms": 25, "release_ms": 200, "knee": 1, "makeup_gain": 0},
    },
    "playback": {
        "base": {"threshold": -18, "ratio": 2.5, "attack_ms": 20, "release_ms": 200, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -20, "ratio": 3.0, "attack_ms": 15, "release_ms": 220, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -14, "ratio": 2.0, "attack_ms": 30, "release_ms": 250, "knee": 1, "makeup_gain": 0},
    },
    "accordion": {
        "base": {"threshold": -16, "ratio": 3.0, "attack_ms": 15, "release_ms": 160, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -12, "ratio": 2.0, "attack_ms": 25, "release_ms": 200, "knee": 1, "makeup_gain": 0},
    },
    "cymbals": {
        "base": {"threshold": -20, "ratio": 3.0, "attack_ms": 8, "release_ms": 120, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -16, "ratio": 2.0, "attack_ms": 12, "release_ms": 150, "knee": 1, "makeup_gain": 0},
    },
    "ride": {
        "base": {"threshold": -18, "ratio": 3.0, "attack_ms": 8, "release_ms": 100, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -14, "ratio": 2.0, "attack_ms": 12, "release_ms": 130, "knee": 1, "makeup_gain": 0},
    },
    "custom": {
        "base": {"threshold": -15, "ratio": 3.0, "attack_ms": 15, "release_ms": 150, "knee": 2, "makeup_gain": 0},
        "control": {"threshold": -18, "ratio": 4.0, "attack_ms": 10, "release_ms": 180, "knee": 2, "makeup_gain": 0},
        "gentle": {"threshold": -12, "ratio": 2.0, "attack_ms": 25, "release_ms": 200, "knee": 1, "makeup_gain": 0},
    },
}


def get_preset(instrument_type: str, task: str = "base") -> Dict[str, Any]:
    """Get preset for instrument and task. Falls back to custom/base if missing."""
    it = instrument_type if instrument_type in DEFAULT_PRESETS else "custom"
    presets = DEFAULT_PRESETS.get(it, DEFAULT_PRESETS["custom"])
    out = presets.get(task) or presets.get("base")
    if not out:
        out = DEFAULT_PRESETS["custom"]["base"]
    preset = dict(out)
    # Ensure detector type is always present for GR estimation.
    if "detector" not in preset:
        preset["detector"] = "peak" if it in PEAK_DETECTOR_INSTRUMENTS else "rms"
    return preset


def get_available_tasks(instrument_type: str) -> list:
    """Return list of task keys available for this instrument."""
    presets = DEFAULT_PRESETS.get(instrument_type) or DEFAULT_PRESETS["custom"]
    return list(presets.keys())


def load_presets_from_file(path: str) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Load presets from JSON. Returns None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Could not load compressor presets from {path}: {e}")
    return None


def save_presets_to_file(presets: Dict[str, Dict[str, Dict[str, Any]]], path: str) -> bool:
    """Save presets to JSON."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Could not save compressor presets to {path}: {e}")
    return False
