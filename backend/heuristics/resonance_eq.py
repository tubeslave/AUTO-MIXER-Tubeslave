"""EQ v3, слой 1: детекция резонансов + HPF по фундаментальной.

Валидированные правила Pestana & Reiss (AES 53, 2014): #18 «подавлять
резонансы» и #22 «резать узко (высокий Q), поднимать широко». Порог
prominence 6 дБ — JND узкой полосы (Caswell-Midwinter 2019).

Отличие от «чистого» метода документа: перед резом кандидат проверяется
гармоническим фильтром проекта (compute_phpr_db / compute_subharmonic_ratio_db
из feedback_detector) — устойчивая НОТА инструмента (гармонический ряд)
резонансом не считается и не режется. Режутся негармонические звоны:
корпус барабана, стойка, резонанс комнаты в микрофоне.

HPF по фундаментальной: срез 0.75·f0 сохраняет фундаментальную ноту,
границы роли не дают срезу уехать (kick/bass/floor_tom — HPF выключен).
Proposal-only: модуль ничего не пишет на пульт.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from feedback_detector import (  # noqa: E402
    compute_phpr_db,
    compute_subharmonic_ratio_db,
)

RESONANCE_PROMINENCE_THRESHOLD_DB = 6.0   # JND узкой полосы
RESONANCE_CUT_FACTOR = 0.7                # cut = -prominence * K
MAX_CUT_SOUNDCHECK_DB = 9.0
MAX_CUT_LIVE_DB = 3.0
Q_MIN, Q_MAX = 2.0, 8.0                   # валидировано #22: резать узко
HARMONIC_SERIES_THRESHOLD_DB = 15.0       # гармоники ближе -> это нота
FFT_SIZE = 8192
LOCAL_WINDOW_OCT = 1.0 / 3.0

# Ring-тест (замечание оператора 2026-07-06 «EQ просто давит громкое»):
# проблемный резонанс ЗВЕНИТ — после спада широкополосной огибающей его
# полоса затухает медленнее остального спектра. Формант/тембровое ядро
# умирает вместе с нотой. Без подтверждения звоном срез ограничивается
# гигиеническим кэпом.
RING_BROADBAND_DROP_DB = 10.0        # спад огибающей, где меряем затухание
RING_DECAY_RATIO_MAX = 0.6           # band_decay/broadband_decay ниже -> звон
RING_MIN_GAPS = 2                    # минимум окон спада для вердикта
UNCONFIRMED_CUT_CAP_DB = 3.0         # кэп среза без ring-подтверждения

# Границы HPF по ролям (min, max); None = HPF выключен для роли
ROLE_HPF_BOUNDS: Dict[str, Optional[Tuple[float, float]]] = {
    "kick": None,
    "bass": None,
    "floor_tom": None,
    "tom": (40.0, 80.0),
    "snare": (80.0, 140.0),
    "overheads": (150.0, 400.0),
    "hihat": (250.0, 500.0),
    "lead_vocal": (75.0, 120.0),
    "backing_vocal": (80.0, 140.0),
    "guitar": (80.0, 140.0),
    "acoustic_guitar": (70.0, 110.0),
    "keys": (40.0, 80.0),
    "playback": None,
    "line": (40.0, 120.0),
}
F0_SEARCH_HZ = (40.0, 1200.0)


@dataclass
class ResonanceCut:
    """Предложенный рез резонанса (proposal-only)."""
    freq_hz: float
    gain_db: float          # < 0
    q: float
    prominence_db: float
    bandwidth_hz: float
    harmonic_check: Dict[str, Any] = field(default_factory=dict)
    ring_confirmed: Optional[bool] = None   # None = проверить было нечем
    ring_decay_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freq_hz": round(self.freq_hz, 1),
            "gain_db": round(self.gain_db, 2),
            "q": round(self.q, 2),
            "prominence_db": round(self.prominence_db, 2),
            "bandwidth_hz": round(self.bandwidth_hz, 1),
            "harmonic_check": dict(self.harmonic_check),
            "ring_confirmed": self.ring_confirmed,
            "ring_decay_ratio": (
                None if self.ring_decay_ratio is None
                else round(self.ring_decay_ratio, 2)
            ),
            "rule": "resonance_cut_v3",
        }


def ring_decay_ratio(
    x: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    bandwidth_hz: float,
) -> Optional[float]:
    """Отношение затухания полосы кандидата к широкополосному в паузах.

    Ищем окна, где широкополосная огибающая падает на RING_BROADBAND_DROP_DB
    от локального пика; в них сравниваем спад уровня полосы кандидата со
    спадом всего спектра. Резонанс звенит: ratio < RING_DECAY_RATIO_MAX.
    Формант умирает с нотой: ratio ~ 1. None — пауз для замера нет.
    """
    x = np.asarray(x, dtype=np.float64)
    hop = max(int(0.01 * sample_rate), 64)
    n_bins = (len(x) - hop) // hop
    if n_bins < 20:
        return None
    # огибающие: широкополосная и полосовая (биквад-полоса через FFT-маску
    # слишком дорога поэлементно; берём короткие DFT-пробы на f кандидата)
    t = np.arange(hop) / sample_rate
    probe = np.exp(-2j * np.pi * freq_hz * t) * np.hanning(hop)
    broad_db = np.empty(n_bins)
    band_db = np.empty(n_bins)
    for i in range(n_bins):
        seg = x[i * hop:(i + 1) * hop]
        broad_db[i] = 10.0 * np.log10(float(np.mean(seg ** 2)) + 1e-12)
        band_db[i] = 20.0 * np.log10(abs(complex(np.dot(seg, probe))) + 1e-12)
    ratios: List[float] = []
    i = 1
    while i < n_bins - 3:
        # локальный пик широкополосной огибающей
        if broad_db[i] >= broad_db[i - 1] and broad_db[i] >= broad_db[i + 1]:
            # ищем спад на RING_BROADBAND_DROP_DB в ближайших 400 мс
            for j in range(i + 1, min(i + 40, n_bins)):
                broad_drop = broad_db[i] - broad_db[j]
                if broad_drop >= RING_BROADBAND_DROP_DB:
                    band_drop = band_db[i] - band_db[j]
                    ratios.append(float(band_drop / max(broad_drop, 1e-6)))
                    i = j  # не пересекать окна спада
                    break
        i += 1
    if len(ratios) < RING_MIN_GAPS:
        return None
    return float(np.median(ratios))


def estimate_fundamental_hz(
    x: np.ndarray,
    sample_rate: int,
    search_hz: Tuple[float, float] = F0_SEARCH_HZ,
) -> Optional[float]:
    """Оценка фундаментальной по автокорреляции (параболическое уточнение).

    None — периодичность не выражена (шум/перкуссия без тона).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < int(sample_rate / search_hz[0]) * 2:
        return None
    x = x - np.mean(x)
    energy = float(np.sum(x ** 2))
    if energy <= 0.0:
        return None
    # нормированная автокорреляция через FFT
    n = int(2 ** np.ceil(np.log2(len(x) * 2)))
    spec = np.fft.rfft(x, n)
    ac = np.fft.irfft(spec * np.conj(spec))[: len(x)]
    ac = ac / (ac[0] + 1e-12)
    lag_min = int(sample_rate / search_hz[1])
    lag_max = min(int(sample_rate / search_hz[0]), len(ac) - 2)
    if lag_max <= lag_min + 2:
        return None
    window = ac[lag_min:lag_max]
    peak_rel = int(np.argmax(window))
    peak = lag_min + peak_rel
    if ac[peak] < 0.4:  # слабая периодичность — не тональный сигнал
        return None
    # параболическое уточнение вершины
    y0, y1, y2 = ac[peak - 1], ac[peak], ac[peak + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    lag = peak + float(np.clip(shift, -1.0, 1.0))
    return float(sample_rate / lag)


def hpf_from_fundamental(
    role: str,
    f0_hz: Optional[float],
) -> Optional[float]:
    """HPF-частота: clamp(0.75·f0, границы роли). None — HPF не ставить."""
    bounds = ROLE_HPF_BOUNDS.get(role, (40.0, 120.0))
    if bounds is None:
        return None
    lo, hi = bounds
    if f0_hz is None or f0_hz <= 0:
        return float(lo)
    return float(np.clip(0.75 * f0_hz, lo, hi))


def _smoothed_spectrum_db(
    x: np.ndarray, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = min(len(x), FFT_SIZE)
    if n < 1024:
        return np.array([]), np.array([])
    seg = x[-n:] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return freqs, 20.0 * np.log10(mag + 1e-10)


def detect_resonances(
    x: np.ndarray,
    sample_rate: int,
    *,
    max_cut_db: float = MAX_CUT_SOUNDCHECK_DB,
    prominence_threshold_db: float = RESONANCE_PROMINENCE_THRESHOLD_DB,
    max_cuts: int = 3,
    freq_range_hz: Tuple[float, float] = (80.0, 8000.0),
) -> List[ResonanceCut]:
    """Резонансы: локальный максимум > threshold над медианой ±1/3 окт.

    Q = f/bw по ширине пика на уровне prominence−3 дБ (кламп 2..8),
    cut = −min(prominence·0.7, max_cut). Члены гармонического ряда
    (нота) отбрасываются.
    """
    freqs, mag_db = _smoothed_spectrum_db(x, sample_rate)
    if freqs.size == 0:
        return []
    freq_res = freqs[1] - freqs[0]

    cuts: List[ResonanceCut] = []
    for i in range(2, len(freqs) - 2):
        f = freqs[i]
        if not (freq_range_hz[0] <= f <= freq_range_hz[1]):
            continue
        if not (mag_db[i] > mag_db[i - 1] and mag_db[i] >= mag_db[i + 1]
                and mag_db[i] > mag_db[i - 2] and mag_db[i] >= mag_db[i + 2]):
            continue
        lo_i = int(np.searchsorted(freqs, f / (2 ** LOCAL_WINDOW_OCT)))
        hi_i = int(np.searchsorted(freqs, f * (2 ** LOCAL_WINDOW_OCT)))
        local = float(np.median(mag_db[max(0, lo_i):max(hi_i, lo_i + 1)]))
        prominence = float(mag_db[i]) - local
        if prominence < prominence_threshold_db:
            continue

        # гармонический фильтр: нота не режется как резонанс
        phpr = compute_phpr_db(mag_db, i, freq_res)
        sub = compute_subharmonic_ratio_db(mag_db, i)
        is_note = (
            (phpr is not None and phpr < HARMONIC_SERIES_THRESHOLD_DB)
            or (sub is not None and sub < HARMONIC_SERIES_THRESHOLD_DB)
        )
        if is_note:
            continue

        # ширина пика на prominence-3 дБ
        level = mag_db[i] - min(3.0, prominence / 2.0)
        left = i
        while left > lo_i and mag_db[left] > level:
            left -= 1
        right = i
        while right < hi_i - 1 and mag_db[right] > level:
            right += 1
        bw = max((right - left) * freq_res, freq_res)
        q = float(np.clip(f / bw, Q_MIN, Q_MAX))
        cut = -min(prominence * RESONANCE_CUT_FACTOR, max_cut_db)

        # ring-тест: без подтверждения звоном (формант/тембровое ядро —
        # затухает вместе с нотой) срез ограничивается гигиеническим кэпом
        ratio = ring_decay_ratio(x, sample_rate, float(f), float(bw))
        ring_confirmed = None if ratio is None else bool(ratio < RING_DECAY_RATIO_MAX)
        if ring_confirmed is not True:
            cut = max(cut, -UNCONFIRMED_CUT_CAP_DB)

        cuts.append(ResonanceCut(
            freq_hz=float(f), gain_db=float(cut), q=q,
            prominence_db=prominence, bandwidth_hz=float(bw),
            ring_confirmed=ring_confirmed, ring_decay_ratio=ratio,
            harmonic_check={
                "phpr_db": None if phpr is None else round(phpr, 1),
                "subharmonic_ratio_db": None if sub is None else round(sub, 1),
                "classified_as_note": False,
            },
        ))

    cuts.sort(key=lambda c: -c.prominence_db)
    # дедуп близких пиков (в пределах 1/6 октавы держим сильнейший)
    deduped: List[ResonanceCut] = []
    for c in cuts:
        if all(abs(np.log2(c.freq_hz / d.freq_hz)) > 1.0 / 6.0 for d in deduped):
            deduped.append(c)
    return deduped[:max_cuts]
