"""
Automation modules — лучшие алгоритмы из разных сборок.

- INSTRUMENT_PROFILES: инструмент-специфичные target LUFS, gate, comp (AUTO-MIXER-Tubeslave)
- get_profile_for_preset: маппинг channel_recognizer preset -> профиль
"""

from .instrument_profiles import (
    InstrumentType,
    INSTRUMENT_PROFILES,
    PRESET_TO_INSTRUMENT,
    get_profile_for_preset,
)

__all__ = [
    "InstrumentType",
    "INSTRUMENT_PROFILES",
    "PRESET_TO_INSTRUMENT",
    "get_profile_for_preset",
]
