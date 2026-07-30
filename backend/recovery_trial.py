"""Испытание восстановлением (Ф8, DEC-0008 §6): ground truth по построению.

Берём хороший микс, ломаем ИЗВЕСТНЫМ образом, даём системе чинить и
меряем дистанцию восстановления до оригинала. Сотни объективных тестов
слуха без слушателя; ухо оператора тратится только там, где правильного
ответа не существует.

recovery_ratio = 1 − dist(repaired, original) / dist(broken, original)
  1.0 — идеальное восстановление; 0.0 — ничего не исправлено;
  <0 — сделал хуже. Ворота лестницы 82→88: средний ratio ≥ 0.9 по корпусу.

Ремонтная функция инжектируется зависимостью: v1 стенда не привязан к
конкретному корректору (подключается smart-DSP слой, затем решатель Ф8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

from source_envelope import THIRD_OCT_CENTERS

DIST_BAND_LO_HZ = 60.0
DIST_BAND_HI_HZ = 12500.0


@dataclass
class Defect:
    """Известная поломка: параметрический EQ-сдвиг (peaking biquad)."""

    name: str
    freq_hz: float
    gain_db: float
    q: float = 1.0

    def inverse(self) -> "Defect":
        return Defect(f"inv_{self.name}", self.freq_hz, -self.gain_db, self.q)


# Стартовый корпус дефектов из брифа (§6): расширяется без изменения стенда.
DEFAULT_CORPUS: Tuple[Defect, ...] = (
    Defect("mud_300", 300.0, +4.0, 0.9),
    Defect("harsh_3k", 3150.0, +3.0, 1.4),
    Defect("boxy_500", 500.0, +3.5, 1.1),
    Defect("scooped_mids", 1000.0, -4.0, 0.8),
    Defect("boomy_80", 80.0, +4.0, 1.0),
    Defect("dull_top", 8000.0, -4.0, 0.7),
)


def _peaking_coeffs(sr: int, freq_hz: float, gain_db: float, q: float):
    """Peaking EQ по Audio EQ Cookbook (RBJ)."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * a
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a
    return (
        np.array([b0, b1, b2]) / a0,
        np.array([1.0, a1 / a0, a2 / a0]),
    )


def apply_defect(x: np.ndarray, sr: int, defect: Defect) -> np.ndarray:
    """Применить поломку (или её inverse — это тот же оператор)."""
    b, a = _peaking_coeffs(sr, defect.freq_hz, defect.gain_db, defect.q)
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        return np.column_stack([lfilter(b, a, y[:, c]) for c in range(y.shape[1])])
    return lfilter(b, a, y)


def _band_levels_db(x: np.ndarray, sr: int) -> Dict[float, float]:
    mono = np.asarray(x, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    out: Dict[float, float] = {}
    for center in THIRD_OCT_CENTERS:
        if not (DIST_BAND_LO_HZ <= center <= DIST_BAND_HI_HZ):
            continue
        lo, hi = center / 2 ** (1 / 6), center * 2 ** (1 / 6)
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            out[center] = float(10.0 * np.log10(spectrum[sel].mean() + 1e-20))
    return out


def spectral_distance_db(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Средний |Δ| третьоктавных уровней 60 Гц–12.5 кГц, дБ."""
    la, lb = _band_levels_db(a, sr), _band_levels_db(b, sr)
    common = sorted(set(la) & set(lb))
    if not common:
        return 0.0
    return float(np.mean([abs(la[c] - lb[c]) for c in common]))


RepairFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class TrialResult:
    defect: str
    broken_distance_db: float
    repaired_distance_db: float

    @property
    def recovery_ratio(self) -> float:
        if self.broken_distance_db <= 1e-9:
            return 1.0
        return 1.0 - self.repaired_distance_db / self.broken_distance_db


@dataclass
class TrialReport:
    results: List[TrialResult] = field(default_factory=list)

    @property
    def mean_recovery_ratio(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.recovery_ratio for r in self.results]))

    def gate_passed(self, threshold: float = 0.9) -> bool:
        return bool(self.results) and self.mean_recovery_ratio >= threshold

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "automixer.recovery_trial.v1",
            "mean_recovery_ratio": round(self.mean_recovery_ratio, 4),
            "gate_0p9_passed": self.gate_passed(),
            "results": [
                {
                    "defect": r.defect,
                    "broken_distance_db": round(r.broken_distance_db, 3),
                    "repaired_distance_db": round(r.repaired_distance_db, 3),
                    "recovery_ratio": round(r.recovery_ratio, 4),
                }
                for r in self.results
            ],
        }


def run_trial(
    original: np.ndarray,
    sr: int,
    repair: RepairFn,
    corpus: Sequence[Defect] = DEFAULT_CORPUS,
) -> TrialReport:
    """Полный проход корпуса: ломаем → чиним → меряем."""
    report = TrialReport()
    for defect in corpus:
        broken = apply_defect(original, sr, defect)
        repaired = repair(broken, sr)
        report.results.append(
            TrialResult(
                defect=defect.name,
                broken_distance_db=spectral_distance_db(broken, original, sr),
                repaired_distance_db=spectral_distance_db(repaired, original, sr),
            )
        )
    return report
