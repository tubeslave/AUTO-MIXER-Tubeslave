"""Конверт источника: identity-маска, crest-цель, зоны касания (Ф8, DEC-0008).

Конверт — цель на инструмент из двух половин (бриф
objective-mix-theory-solver-validation-20260730 §2):
- спектральная: identity-маска (ядро голоса: полосы в пределах
  CORE_WINDOW_DB от максимума долгосрочного спектра по окнам СОБСТВЕННОЙ
  активности) и зоны касания (полосы, где источник обязан прикасаться к
  сумме — слоты из методики оператора);
- динамическая: целевой crest-фактор по классу инструмента.

Ядро неприкосновенно: авто-EQ не режет в ядре глубже CORE_MAX_CUT_DB
(канон CLAUDE.md «маскинг-срезы не трогают тембровое ядро»). Все цели —
СТАРТОВЫЕ профили: выучиваются из референсов и вердиктов оператора,
blessed mix переписывает их (DEC-0008 §5.C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

THIRD_OCT_CENTERS = (
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0,
    315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
    2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0,
)
CORE_WINDOW_DB = 6.0        # ядро = полосы в 6 дБ от максимума (канон)
CORE_MAX_CUT_DB = -3.0      # глубже в ядре авто-EQ не режет (канон)
ACTIVITY_RMS_GATE_DB = -45.0
MIN_ACTIVITY_FRACTION = 0.05   # меньше — маска не заслуживает доверия
FRAME_MS = 50.0

# Целевые crest по классам (дБ, peak−RMS). СТАРТОВЫЕ прикидки DEC-0008 —
# подлежат выучиванию; транзиентным широкий, сустейновым узкий.
CREST_TARGETS_DB: Dict[str, Tuple[float, float]] = {
    # class: (min, max)
    "kick": (14.0, 24.0),
    "snare": (14.0, 24.0),
    "rack_tom": (12.0, 22.0),
    "floor_tom": (12.0, 22.0),
    "overhead": (10.0, 20.0),
    "bass_guitar": (6.0, 12.0),
    "electric_guitar": (6.0, 14.0),
    "accordion": (6.0, 14.0),
    "keys": (8.0, 16.0),
    "lead_vocal": (8.0, 14.0),
    "backing_vocal": (6.0, 12.0),
    "playback": (8.0, 16.0),
}
DEFAULT_CREST_TARGET_DB = (8.0, 18.0)

# Зоны касания (Гц-диапазоны) — слоты из методики оператора (бриф
# spectral-corridor §2, «касание кривой» DEC-0008): где источник обязан
# прикасаться к сумме. Не identity (то из замера) — а роль в миксе.
TOUCH_ZONES_HZ: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "kick": ((50.0, 70.0), (2000.0, 5000.0)),
    "snare": ((150.0, 250.0), (1000.0, 3000.0)),
    "rack_tom": ((80.0, 160.0), (2000.0, 4000.0)),
    "floor_tom": ((60.0, 120.0), (2000.0, 4000.0)),
    "overhead": ((6000.0, 12000.0),),
    "bass_guitar": ((40.0, 120.0), (700.0, 1500.0)),
    "electric_guitar": ((200.0, 1200.0), (1500.0, 3500.0)),
    "accordion": ((250.0, 2000.0),),
    "keys": ((200.0, 3000.0),),
    "lead_vocal": ((1000.0, 5000.0), (200.0, 600.0)),
    "backing_vocal": ((500.0, 3000.0),),
    "playback": ((100.0, 8000.0),),
}


@dataclass
class IdentityMask:
    core_bands_hz: List[float] = field(default_factory=list)
    band_levels_db: Dict[float, float] = field(default_factory=dict)
    confidence: float = 0.0
    activity_fraction: float = 0.0

    def is_core(self, freq_hz: float) -> bool:
        """Частота внутри ядра, если ближайший третьоктавный центр — ядро."""
        if not self.core_bands_hz:
            return False
        centers = np.asarray(THIRD_OCT_CENTERS)
        nearest = float(centers[np.argmin(np.abs(np.log2(centers / max(freq_hz, 1.0))))])
        return nearest in self.core_bands_hz


@dataclass
class SourceEnvelope:
    instrument: str
    identity: IdentityMask
    crest_target_db: Tuple[float, float]
    touch_zones_hz: Tuple[Tuple[float, float], ...]
    measured_crest_db: float = 0.0

    @property
    def crest_ok(self) -> bool:
        lo, hi = self.crest_target_db
        return lo <= self.measured_crest_db <= hi


def activity_mask(x: np.ndarray, sr: int, gate_db: float = ACTIVITY_RMS_GATE_DB) -> np.ndarray:
    """Кадры собственной активности по RMS-гейту (окно FRAME_MS).

    Прокси этапа 1: полный proximity-гейт (30-мс окна онсетов против
    соседних микрофонов, канон bleed-доминантности) подключается позже —
    интерфейс не изменится: маска кадров булева.
    """
    mono = np.asarray(x, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    frame = max(1, int(FRAME_MS * sr / 1000.0))
    n_frames = len(mono) // frame
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    frames = mono[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12)) > gate_db


def band_spectrum_db(x: np.ndarray, sr: int, act: Optional[np.ndarray] = None) -> Dict[float, float]:
    """Третьоктавный долгосрочный спектр по активным кадрам (дБ)."""
    mono = np.asarray(x, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if act is not None and act.size:
        frame = max(1, int(FRAME_MS * sr / 1000.0))
        keep = np.repeat(act, frame)[: len(mono)]
        mono = mono[keep[: len(mono)]]
    if mono.size < sr // 10:
        return {}
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    out: Dict[float, float] = {}
    for center in THIRD_OCT_CENTERS:
        lo, hi = center / 2 ** (1 / 6), center * 2 ** (1 / 6)
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            out[center] = float(10.0 * np.log10(spectrum[sel].mean() + 1e-20))
    return out


def compute_identity_mask(x: np.ndarray, sr: int) -> IdentityMask:
    act = activity_mask(x, sr)
    fraction = float(act.mean()) if act.size else 0.0
    mask = IdentityMask(activity_fraction=fraction)
    if fraction < MIN_ACTIVITY_FRACTION:
        return mask  # мало собственной активности — маске веры нет (bleed)
    levels = band_spectrum_db(x, sr, act)
    if not levels:
        return mask
    peak = max(levels.values())
    mask.band_levels_db = levels
    mask.core_bands_hz = sorted(
        c for c, db in levels.items() if db >= peak - CORE_WINDOW_DB
    )
    # уверенность растёт с долей активности, насыщается к 30%
    mask.confidence = float(min(1.0, fraction / 0.3))
    return mask


def measure_crest_db(x: np.ndarray, act: Optional[np.ndarray] = None, sr: int = 48000) -> float:
    mono = np.asarray(x, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if act is not None and act.size:
        frame = max(1, int(FRAME_MS * sr / 1000.0))
        keep = np.repeat(act, frame)[: len(mono)]
        mono = mono[keep[: len(mono)]]
    if mono.size == 0:
        return 0.0
    peak = float(np.abs(mono).max())
    rms = float(np.sqrt((mono ** 2).mean()))
    return 20.0 * np.log10(max(peak, 1e-12) / max(rms, 1e-12))


def build_envelope(instrument: str, x: np.ndarray, sr: int) -> SourceEnvelope:
    act = activity_mask(x, sr)
    return SourceEnvelope(
        instrument=instrument,
        identity=compute_identity_mask(x, sr),
        crest_target_db=CREST_TARGETS_DB.get(instrument, DEFAULT_CREST_TARGET_DB),
        touch_zones_hz=TOUCH_ZONES_HZ.get(instrument, ()),
        measured_crest_db=measure_crest_db(x, act, sr),
    )


DUAL_FRAME = 4096
DUAL_HOP = 2048
STATIC_BAND_CREST_DB = 6.0
# Полосный crest (пик−RMS по кадрам ~85 мс): стационарный тон ≈ 0–1 дБ,
# перкуссия ≈ 11–13 дБ (окно размазывает удар) — порог 6 дБ посередине.


def dual_band_spectrum(
    x: np.ndarray,
    sr: int,
    frame: int = DUAL_FRAME,
    hop: int = DUAL_HOP,
) -> Dict[float, Tuple[float, float]]:
    """Двухспектровый замер (DEC-0008 §3): по полосе (RMS_дБ, пик_дБ).

    RMS-спектр — что звучит постоянно; пик-спектр — что звучит в ударах.
    Один и тот же горб лечится по-разному: избыток в RMS — статический EQ;
    избыток только в пиках — полосная динамика (статический рез сделает
    тело тонким).
    """
    mono = np.asarray(x, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if len(mono) < frame:
        return {}
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    band_sel = {
        c: (freqs >= c / 2 ** (1 / 6)) & (freqs < c * 2 ** (1 / 6))
        for c in THIRD_OCT_CENTERS
    }
    frame_energies: Dict[float, List[float]] = {c: [] for c in THIRD_OCT_CENTERS}
    for start in range(0, len(mono) - frame + 1, hop):
        spec = np.abs(np.fft.rfft(mono[start:start + frame] * window)) ** 2
        for c, sel in band_sel.items():
            if sel.any():
                frame_energies[c].append(float(spec[sel].mean()))
    out: Dict[float, Tuple[float, float]] = {}
    for c, energies in frame_energies.items():
        if not energies:
            continue
        arr = np.asarray(energies)
        rms_db = float(10.0 * np.log10(arr.mean() + 1e-20))
        peak_db = float(10.0 * np.log10(arr.max() + 1e-20))
        out[c] = (rms_db, peak_db)
    return out


def band_treatment_hint(
    dual: Dict[float, Tuple[float, float]],
    center_hz: float,
    static_crest_db: float = STATIC_BAND_CREST_DB,
) -> str:
    """Для полосы с избытком: 'static_eq' (постоянный) или 'dynamic' (в ударах)."""
    if center_hz not in dual:
        return "unknown"
    rms_db, peak_db = dual[center_hz]
    return "static_eq" if (peak_db - rms_db) <= static_crest_db else "dynamic"


def eq_cut_allowed(mask: IdentityMask, freq_hz: float, gain_db: float) -> float:
    """Кламп EQ-реза защитой ядра: в ядре не глубже CORE_MAX_CUT_DB.

    Возвращает разрешённый gain (буст не трогаем). При низкой уверенности
    маски (<0.5) консервативно считаем ЛЮБУЮ полосу ядром — сомневаешься,
    не режь глубоко (канон «при сомнениях — снижай, не повышай» про
    громкость; про EQ наоборот: сомневаешься — не углубляй рез).
    """
    if gain_db >= 0:
        return float(gain_db)
    protected = mask.confidence < 0.5 or mask.is_core(freq_hz)
    if protected:
        return float(max(gain_db, CORE_MAX_CUT_DB))
    return float(gain_db)
