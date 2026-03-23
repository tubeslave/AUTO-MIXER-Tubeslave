"""
Инструмент-специфичные профили для Gain Staging (AUTO-MIXER-Tubeslave).

Содержит target_lufs, peak_limit, gate_threshold, hpf, comp-параметры
для каждого типа инструмента. Используется SafeGainCalibrator, AutoFader.
"""

from enum import Enum
from typing import Dict, Any, Optional

logger = __import__("logging").getLogger(__name__)


class InstrumentType(Enum):
    """Типы инструментов для профилей."""
    KICK = "kick"
    SNARE = "snare"
    TOM = "tom"
    HI_HAT = "hihat"
    OVERHEAD = "overhead"
    BASS = "bass"
    GUITAR_AMP = "guitar_amp"
    GUITAR_DI = "guitar_di"
    KEYS = "keys"
    VOCALS = "vocals"
    PLAYBACK = "playback"
    LINE = "line"


# Профили инструментов (target LUFS, peak limit, gate, HPF, comp)
INSTRUMENT_PROFILES: Dict[InstrumentType, Dict[str, Any]] = {
    InstrumentType.KICK: {
        "target_lufs": -15,
        "peak_limit": -3,
        "gate_threshold": -40,
        "hpf": 40,
        "comp_threshold": -12,
        "comp_ratio": 4,
        "comp_attack": 2,
        "comp_release": 80,
    },
    InstrumentType.SNARE: {
        "target_lufs": -16,
        "peak_limit": -3,
        "gate_threshold": -40,
        "hpf": 100,
        "comp_threshold": -15,
        "comp_ratio": 4,
        "comp_attack": 3,
        "comp_release": 100,
    },
    InstrumentType.TOM: {
        "target_lufs": -18,
        "peak_limit": -3,
        "gate_threshold": -40,
        "hpf": 80,
        "comp_threshold": -15,
        "comp_ratio": 4,
        "comp_attack": 3,
        "comp_release": 100,
    },
    InstrumentType.HI_HAT: {
        "target_lufs": -22,
        "peak_limit": -6,
        "gate_threshold": -45,
        "hpf": 200,
        "comp_threshold": -20,
        "comp_ratio": 3,
        "comp_attack": 2,
        "comp_release": 80,
    },
    InstrumentType.OVERHEAD: {
        "target_lufs": -20,
        "peak_limit": -6,
        "gate_threshold": -45,
        "hpf": 200,
        "comp_threshold": -20,
        "comp_ratio": 3,
        "comp_attack": 5,
        "comp_release": 150,
    },
    InstrumentType.BASS: {
        "target_lufs": -15,
        "peak_limit": -3,
        "gate_threshold": -40,
        "hpf": 30,
        "comp_threshold": -15,
        "comp_ratio": 6,
        "comp_attack": 10,
        "comp_release": 100,
    },
    InstrumentType.VOCALS: {
        "target_lufs": -18,
        "peak_limit": -3,
        "gate_threshold": -45,
        "hpf": 80,
        "comp_threshold": -18,
        "comp_ratio": 3,
        "comp_attack": 5,
        "comp_release": 150,
    },
    InstrumentType.GUITAR_AMP: {
        "target_lufs": -20,
        "peak_limit": -6,
        "gate_threshold": -45,
        "hpf": 80,
        "comp_threshold": -20,
        "comp_ratio": 4,
        "comp_attack": 10,
        "comp_release": 200,
    },
    InstrumentType.GUITAR_DI: {
        "target_lufs": -18,
        "peak_limit": -6,
        "gate_threshold": -50,
        "hpf": 60,
        "comp_threshold": -18,
        "comp_ratio": 3,
        "comp_attack": 10,
        "comp_release": 200,
    },
    InstrumentType.KEYS: {
        "target_lufs": -18,
        "peak_limit": -6,
        "gate_threshold": -45,
        "hpf": 40,
        "comp_threshold": -20,
        "comp_ratio": 3,
        "comp_attack": 15,
        "comp_release": 250,
    },
    InstrumentType.PLAYBACK: {
        "target_lufs": -16,
        "peak_limit": -3,
        "gate_threshold": -60,
        "hpf": 20,
        "comp_threshold": -18,
        "comp_ratio": 3,
        "comp_attack": 10,
        "comp_release": 200,
    },
    InstrumentType.LINE: {
        "target_lufs": -18,
        "peak_limit": -3,
        "gate_threshold": -45,
        "hpf": 40,
        "comp_threshold": -20,
        "comp_ratio": 3,
        "comp_attack": 10,
        "comp_release": 200,
    },
}

# Маппинг channel_recognizer preset_id -> InstrumentType
PRESET_TO_INSTRUMENT: Dict[str, InstrumentType] = {
    "kick": InstrumentType.KICK,
    "snare": InstrumentType.SNARE,
    "tom": InstrumentType.TOM,
    "toms": InstrumentType.TOM,
    "hihat": InstrumentType.HI_HAT,
    "ride": InstrumentType.OVERHEAD,
    "cymbals": InstrumentType.OVERHEAD,
    "overheads": InstrumentType.OVERHEAD,
    "overhead": InstrumentType.OVERHEAD,
    "room": InstrumentType.OVERHEAD,
    "bass": InstrumentType.BASS,
    "electricGuitar": InstrumentType.GUITAR_AMP,
    "acousticGuitar": InstrumentType.GUITAR_DI,
    "leadVocal": InstrumentType.VOCALS,
    "lead_vocal": InstrumentType.VOCALS,
    "backVocal": InstrumentType.VOCALS,
    "backing_vocal": InstrumentType.VOCALS,
    "back_vocal": InstrumentType.VOCALS,
    "synth": InstrumentType.KEYS,
    "playback": InstrumentType.PLAYBACK,
    "accordion": InstrumentType.KEYS,
    "drums_bus": InstrumentType.LINE,
    "vocal_bus": InstrumentType.LINE,
    "instrument_bus": InstrumentType.LINE,
}


def get_profile_for_preset(preset_id: Optional[str]) -> Dict[str, Any]:
    """
    Получить профиль для preset_id из channel_recognizer.

    Args:
        preset_id: 'kick', 'snare', 'leadVocal', etc. или None

    Returns:
        dict с target_lufs, peak_limit, gate_threshold, hpf, comp_*...
    """
    if not preset_id:
        return INSTRUMENT_PROFILES[InstrumentType.LINE].copy()
    preset_lower = str(preset_id).lower()
    inst_type = PRESET_TO_INSTRUMENT.get(preset_lower, InstrumentType.LINE)
    return INSTRUMENT_PROFILES[inst_type].copy()
