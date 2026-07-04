"""EQ v3, слой 3: предсказательная A/B-проверка с K-weighting (BS.1770).

Правило документа: любая автоматическая оценка «лучше/хуже» — только при
равной громкости, иначе «громче = лучше» искажает решение (loudness bias).

Специфика проекта: USB-tap снимается PRE-trim (pre-EQ), поэтому замер
ПОСЛЕ применения EQ не увидит — классический A/B невозможен до появления
post-fader tap. Здесь A/B ПРЕДСКАЗАТЕЛЬНЫЙ: кривая предложенного EQ
применяется к захваченному спектру математически, K-взвешенная громкость
до/после сравнивается аналитически. Это гейт для proposal-пути, не
замена слуховой проверки оператора.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.signal import freqz

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from lufs_gain_staging import KWeightingFilter  # noqa: E402

MAX_LOUDNESS_BIAS_LU = 0.5
MAX_FADER_COMPENSATION_DB = 6.0   # больший сдвиг = вероятная ошибка анализа


@dataclass
class ABCheckResult:
    loudness_before_db: float
    loudness_after_db: float
    loudness_delta_lu: float
    compensation_db: float
    allowed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loudness_before_db": round(self.loudness_before_db, 2),
            "loudness_after_db": round(self.loudness_after_db, 2),
            "loudness_delta_lu": round(self.loudness_delta_lu, 2),
            "compensation_db": round(self.compensation_db, 2),
            "allowed": self.allowed,
            "reason": self.reason,
            "kind": "predictive_ab_bs1770",
            "note": "pre-trim tap: A/B предсказательный, не пост-замер",
        }


def k_weighting_curve_db(freqs_hz: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """|H| K-weighting (shelf + HPF из BS.1770) на сетке частот, дБ."""
    kw = KWeightingFilter(sample_rate=sample_rate)
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / sample_rate
    w = np.clip(w, 0.0, np.pi - 1e-9)
    _, h_shelf = freqz(kw.shelf_b, kw.shelf_a, worN=w)
    _, h_hp = freqz(kw.hp_b, kw.hp_a, worN=w)
    mag = np.abs(h_shelf) * np.abs(h_hp)
    return 20.0 * np.log10(mag + 1e-12)


def bell_response_db(
    freqs_hz: np.ndarray, center_hz: float, gain_db: float, q: float
) -> np.ndarray:
    """Магнитудный отклик bell-фильтра (аналитическая аппроксимация RBJ)."""
    f = np.maximum(np.asarray(freqs_hz, dtype=np.float64), 1e-6)
    # безразмерная расстройка: x = Q * (f/fc - fc/f)
    x = q * (f / center_hz - center_hz / f)
    return gain_db / (1.0 + x ** 2)


def eq_curve_db(
    freqs_hz: np.ndarray,
    bands: Sequence[Mapping[str, Any]],
    hpf_hz: float | None = None,
) -> np.ndarray:
    """Суммарная кривая EQ: bells + HPF 12 дБ/окт."""
    curve = np.zeros_like(np.asarray(freqs_hz, dtype=np.float64))
    for band in bands:
        curve += bell_response_db(
            freqs_hz,
            float(band["freq_hz"]),
            float(band["gain_db"]),
            float(band.get("q", 1.0)),
        )
    if hpf_hz and hpf_hz > 0:
        f = np.maximum(np.asarray(freqs_hz, dtype=np.float64), 1e-6)
        below = f < hpf_hz
        curve[below] += -12.0 * np.log2(hpf_hz / f[below])
    return curve


def _k_weighted_loudness_db(
    freqs_hz: np.ndarray, spectrum_db: np.ndarray, sample_rate: int
) -> float:
    kw = k_weighting_curve_db(freqs_hz, sample_rate)
    power = 10.0 ** ((spectrum_db + kw) / 10.0)
    return 10.0 * np.log10(float(np.sum(power)) + 1e-12)


def predictive_ab_check(
    freqs_hz: np.ndarray,
    spectrum_db: np.ndarray,
    bands: Sequence[Mapping[str, Any]],
    *,
    hpf_hz: float | None = None,
    sample_rate: int = 48000,
    max_bias_lu: float = MAX_LOUDNESS_BIAS_LU,
    max_compensation_db: float = MAX_FADER_COMPENSATION_DB,
) -> ABCheckResult:
    """Гейт loudness-bias: |ΔLUFS(K-weighted)| после EQ должна быть ≤ 0.5 LU.

    Если предложение меняет громкость сильнее — оно либо маскирует
    тональную оценку громкостью, либо требует фейдерной компенсации:
    возвращается compensation_db (равная −Δ), и предложение допускается
    только вместе с ней.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    spectrum_db = np.asarray(spectrum_db, dtype=np.float64)
    if freqs_hz.size == 0 or freqs_hz.size != spectrum_db.size:
        return ABCheckResult(0.0, 0.0, 0.0, 0.0, False, "empty_or_mismatched_spectrum")

    before = _k_weighted_loudness_db(freqs_hz, spectrum_db, sample_rate)
    curve = eq_curve_db(freqs_hz, bands, hpf_hz)
    after = _k_weighted_loudness_db(freqs_hz, spectrum_db + curve, sample_rate)
    delta = after - before

    if abs(delta) <= max_bias_lu:
        return ABCheckResult(before, after, delta, 0.0, True, "loudness_matched")
    return ABCheckResult(
        before, after, delta, -delta, True,
        "requires_fader_compensation",
    ) if abs(delta) <= max_compensation_db else ABCheckResult(
        before, after, delta, -delta, False,
        "loudness_bias_exceeds_safe_compensation",
    )


def masking_improvement_db(
    mask_before: np.ndarray, mask_after: np.ndarray
) -> float:
    """Суммарное снижение перекрытия в конфликтных полосах (>0 = лучше)."""
    before = np.maximum(np.asarray(mask_before, dtype=np.float64), 0.0)
    after = np.maximum(np.asarray(mask_after, dtype=np.float64), 0.0)
    return float(np.sum(before) - np.sum(after))
