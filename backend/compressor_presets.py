"""
Compressor presets knowledge base for Auto Compressor.

Per-instrument presets and task variants (base, punch, control, gentle, aggressive, broadcast).
All levels in dB, times in ms. Wing ratio is applied as string ("2.0", "4.0", etc.).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from method_preset_loader import load_method_preset_base, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_COMPRESSOR_PRESET_BASE_PATH = (
    Path(__file__).resolve().parents[1] / "presets" / "method_presets_compressor.json"
)
_ACTIVE_PRESETS_FILE = str(DEFAULT_COMPRESSOR_PRESET_BASE_PATH)
_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "presets": None, "aliases": None, "defaults": None}

INSTRUMENT_ALIASES = {
    "leadvocal": "leadVocal",
    "backvocal": "backVocal",
    "electricguitar": "electricGuitar",
    "acousticguitar": "acousticGuitar",
    "overheads": "overheads",
    "hihat": "hihat",
    "kick_in": "kick",
    "kick_out": "kick",
    "kick_sub": "kick",
    "snare_top": "snare",
    "snare_bottom": "snare",
    "tom_hi": "tom",
    "tom_mid": "tom",
    "tom_floor": "tom",
    "hi_hat": "hihat",
    "overhead": "overheads",
    "electric_guitar": "electricGuitar",
    "acoustic_guitar": "acousticGuitar",
    "lead_vocal": "leadVocal",
    "back_vocal": "backVocal",
    "vocal": "leadVocal",
}


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


def set_presets_file(path: Optional[str]) -> None:
    """Set active presets JSON file path (unified schema)."""
    global _ACTIVE_PRESETS_FILE
    if path:
        _ACTIVE_PRESETS_FILE = str(path)


def _load_external_presets() -> tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, str], Dict[str, Any]]:
    resolved = resolve_path(_ACTIVE_PRESETS_FILE, DEFAULT_COMPRESSOR_PRESET_BASE_PATH)
    mtime = resolved.stat().st_mtime if resolved.exists() else None
    if (
        _CACHE["path"] == str(resolved)
        and _CACHE["mtime"] == mtime
        and _CACHE["presets"] is not None
    ):
        return _CACHE["presets"], _CACHE["aliases"], _CACHE["defaults"]

    base_data = load_method_preset_base(resolved, expected_method="compressor")
    presets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    aliases: Dict[str, str] = {}
    defaults = base_data.get("defaults", {}) if isinstance(base_data.get("defaults"), dict) else {}
    default_tasks = defaults.get("tasks", {}) if isinstance(defaults.get("tasks"), dict) else {}

    for item in base_data.get("instruments", []):
        if not isinstance(item, dict):
            continue
        inst_id = str(item.get("id", "")).strip()
        if not inst_id:
            continue
        params = item.get("params", {}) if isinstance(item.get("params"), dict) else {}
        tasks = params.get("tasks", {}) if isinstance(params.get("tasks"), dict) else {}
        merged_tasks: Dict[str, Dict[str, Any]] = {}
        for task_name, task_cfg in default_tasks.items():
            if isinstance(task_cfg, dict):
                merged_tasks[task_name] = dict(task_cfg)
        for task_name, task_cfg in tasks.items():
            if isinstance(task_cfg, dict):
                merged_tasks[task_name] = dict(task_cfg)
        if merged_tasks:
            detector = params.get("detector", defaults.get("detector"))
            if detector:
                for task_cfg in merged_tasks.values():
                    task_cfg.setdefault("detector", detector)
            presets[inst_id] = merged_tasks
            for alias in item.get("aliases", []) or []:
                alias_key = str(alias).strip().lower().replace("-", "_").replace(" ", "_")
                if alias_key:
                    aliases[alias_key] = inst_id

    if not presets:
        presets = DEFAULT_PRESETS
        aliases = INSTRUMENT_ALIASES

    _CACHE["path"] = str(resolved)
    _CACHE["mtime"] = mtime
    _CACHE["presets"] = presets
    _CACHE["aliases"] = aliases
    _CACHE["defaults"] = defaults
    return presets, aliases, defaults


def get_preset(instrument_type: str, task: str = "base") -> Dict[str, Any]:
    """Get preset for instrument and task. Falls back to custom/base if missing."""
    presets_source, aliases_source, defaults = _load_external_presets()
    builtin_defaults = DEFAULT_PRESETS
    normalized = str(instrument_type or "custom").strip().lower().replace("-", "_").replace(" ", "_")
    it = aliases_source.get(normalized, normalized)
    if it not in presets_source:
        it = "custom"
    presets = presets_source.get(it, presets_source.get("custom", {}))
    out = presets.get(task) or presets.get("base")
    if not out:
        out = (
            presets_source.get("custom", {}).get("base")
            or builtin_defaults["custom"]["base"]
        )
    preset = dict(out)
    # Ensure detector type is always present for GR estimation.
    if "detector" not in preset:
        preset["detector"] = "peak" if it in PEAK_DETECTOR_INSTRUMENTS else "rms"
    return preset


def get_available_tasks(instrument_type: str) -> list:
    """Return list of task keys available for this instrument."""
    presets_source, aliases_source, _defaults = _load_external_presets()
    normalized = str(instrument_type or "custom").strip().lower().replace("-", "_").replace(" ", "_")
    it = aliases_source.get(normalized, normalized)
    presets = presets_source.get(it) or presets_source.get("custom") or DEFAULT_PRESETS["custom"]
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
