"""Автопараметризация компрессора по Ma, De Man, Pestana, Black, Reiss (2015).

«Intelligent Multitrack Dynamic Range Compression», JAES 63(6):412-426:
регрессии ratio/threshold от percussivity weighting (w_p) и low-frequency
weighting (w_f), выведенные из субъективных данных. Баллистика attack/release
масштабируется по crest factor и spectral flux (Giannoulis, Massberg &
Reiss 2012): транзиентнее сигнал — короче константы времени.

w_p и w_f здесь — документированные прокси признаков статьи (точные
определения фич за paywall): w_p — доля энергии выпрямленного spectral flux
(транзиентность), w_f — доля мощности ниже 200 Гц. Обе в [0, 1].

Коэффициенты регрессий выведены на конкретной топологии компрессора —
при живом использовании клампуются к безопасным границам проекта
(ratio <= 6, makeup bounded по правилу CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

LF_CUTOFF_HZ = 200.0
FLUX_FRAME = 1024
FLUX_HOP = 512

RATIO_MIN, RATIO_MAX = 1.0, 6.0
THRESHOLD_MIN_DB, THRESHOLD_MAX_DB = -40.0, -8.0
ATTACK_MIN_MS, ATTACK_MAX_MS = 2.0, 40.0
RELEASE_MIN_MS, RELEASE_MAX_MS = 60.0, 400.0
MAX_MAKEUP_DB = 2.0


@dataclass
class Ma2015Params:
    """Параметры компрессора по регрессиям Ma et al. 2015."""
    w_p: float
    w_f: float
    x_rms_db: float
    crest_db: float
    ratio: float
    threshold_db: float
    attack_ms: float
    release_ms: float
    makeup_db: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_p": round(self.w_p, 4),
            "w_f": round(self.w_f, 4),
            "x_rms_db": round(self.x_rms_db, 2),
            "crest_db": round(self.crest_db, 2),
            "ratio": round(self.ratio, 3),
            "threshold_db": round(self.threshold_db, 2),
            "attack_ms": round(self.attack_ms, 1),
            "release_ms": round(self.release_ms, 1),
            "makeup_db": round(self.makeup_db, 2),
            "source": "ma2015_regression",
        }


def percussivity_weighting(x: np.ndarray, sample_rate: int) -> float:
    """w_p ∈ [0,1]: транзиентность через выпрямленный spectral flux.

    Доля положительного межкадрового прироста магнитуд относительно полной
    спектральной энергии. Перкуссия -> ~0.4-1.0, устойчивый тон -> ~0.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < FLUX_FRAME * 2:
        return 0.0
    window = np.hanning(FLUX_FRAME)
    mags = []
    for start in range(0, len(x) - FLUX_FRAME + 1, FLUX_HOP):
        frame = x[start:start + FLUX_FRAME] * window
        mags.append(np.abs(np.fft.rfft(frame)))
    mags = np.asarray(mags)
    if mags.shape[0] < 2:
        return 0.0
    flux = np.maximum(np.diff(mags, axis=0), 0.0).sum()
    total = mags[1:].sum()
    if total <= 0.0:
        return 0.0
    # Нормировка: у полностью импульсного сигнала flux/total -> ~0.5
    return float(np.clip(2.0 * flux / total, 0.0, 1.0))


def low_frequency_weighting(x: np.ndarray, sample_rate: int) -> float:
    """w_f ∈ [0,1]: доля мощности ниже LF_CUTOFF_HZ (200 Гц)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    return float(np.clip(spectrum[freqs < LF_CUTOFF_HZ].sum() / total, 0.0, 1.0))


def ratio_ma2015(w_p: float, w_f: float) -> float:
    """R(m) = 0.54*w_p + 0.764*w_f + 1 (Ma et al. 2015, R^2=0.69)."""
    return float(np.clip(0.54 * w_p + 0.764 * w_f + 1.0, RATIO_MIN, RATIO_MAX))


def threshold_ma2015(x_rms_db: float, w_p: float) -> float:
    """T(m) = -11.03 + 0.441*x_RMS - 4.987*w_p (Ma et al. 2015, R^2=0.56)."""
    t = -11.03 + 0.441 * x_rms_db - 4.987 * w_p
    return float(np.clip(t, THRESHOLD_MIN_DB, THRESHOLD_MAX_DB))


# ── Кросс-адаптивные признаки статьи (мультитрек) ────────────────
#
# Оригинальные w_p/w_f Ma et al. считаются ОТНОСИТЕЛЬНО ВСЕХ каналов микса:
# перкуссивность — гауссово отклонение crest factor канала от среднего по
# трекам (w_p ∈ (0,2)), НЧ-вес — отношение НЧ-доли канала к средней по
# трекам. Одноканальные функции выше — прокси для путей без мультитрек-
# контекста (например, realtime CF-контроллер).

PERCUSSIVITY_SIGMA = 2.0


def percussivity_weighting_multitrack(
    crest_db_by_channel: "dict[object, float]",
) -> "dict[object, float]":
    """w_p(m) ∈ (0, 2) относительно среднего crest по всем трекам (σ=2)."""
    if not crest_db_by_channel:
        return {}
    values = np.asarray(list(crest_db_by_channel.values()), dtype=np.float64)
    mean_crest = float(np.mean(values))
    out = {}
    for ch, crest in crest_db_by_channel.items():
        dev = (float(crest) - mean_crest) ** 2 / (2.0 * PERCUSSIVITY_SIGMA ** 2)
        g = float(np.exp(-dev))  # ограничиваем exp: гаусс, не разлетается
        if float(crest) <= mean_crest:
            out[ch] = g
        else:
            out[ch] = 2.0 - g
    return out


def low_frequency_weighting_multitrack(
    lf_ratio_by_channel: "dict[object, float]",
) -> "dict[object, float]":
    """w_f(m): НЧ-доля канала, нормированная на среднюю по трекам."""
    if not lf_ratio_by_channel:
        return {}
    values = np.asarray(list(lf_ratio_by_channel.values()), dtype=np.float64)
    mean_lf = float(np.mean(values))
    if mean_lf <= 0.0:
        return {ch: 0.0 for ch in lf_ratio_by_channel}
    return {ch: float(np.clip(v / mean_lf, 0.0, 2.0))
            for ch, v in lf_ratio_by_channel.items()}


def ballistics_giannoulis2013(crest_db: float) -> tuple[float, float]:
    """tau_A = 2*tau_A_max/x_crest^2, tau_R = 2*tau_R_max/x_crest^2.

    Giannoulis et al. 2013 (Parameter Automation in a DRC): x_crest —
    ЛИНЕЙНЫЙ crest factor; tau_A_max=80 мс, tau_R_max=1000 мс. Чем
    транзиентнее сигнал — тем короче константы времени.
    """
    x_crest = 10.0 ** (max(float(crest_db), 0.0) / 20.0)
    x2 = max(x_crest ** 2, 1.0)
    attack_ms = float(np.clip(2.0 * 80.0 / x2, ATTACK_MIN_MS, ATTACK_MAX_MS))
    release_ms = float(np.clip(2.0 * 1000.0 / x2, RELEASE_MIN_MS, RELEASE_MAX_MS))
    return attack_ms, release_ms


def knee_width_ma2015(threshold_db: float) -> float:
    """W(m) = |T|/2 — ниже порог -> шире (мягче) колено."""
    return float(abs(threshold_db) / 2.0)


def ballistics_giannoulis(crest_db: float, w_p: float) -> tuple[float, float]:
    """Attack/release от транзиентности (Giannoulis et al. 2012).

    Чем перкуссивнее (высокий crest / w_p) — тем короче константы времени.
    """
    t_norm = float(np.clip(0.5 * np.clip(crest_db / 20.0, 0.0, 1.0) + 0.5 * w_p, 0.0, 1.0))
    attack_ms = ATTACK_MAX_MS - (ATTACK_MAX_MS - ATTACK_MIN_MS) * t_norm
    release_ms = RELEASE_MAX_MS - (RELEASE_MAX_MS - RELEASE_MIN_MS) * t_norm
    return float(attack_ms), float(release_ms)


def bounded_makeup_db(
    x_rms_db: float,
    threshold_db: float,
    ratio: float,
    true_peak_dbtp: Optional[float] = None,
    true_peak_ceiling_dbtp: float = -1.0,
) -> float:
    """Bounded makeup: 70% оценённого GR, не более MAX_MAKEUP_DB, и не
    в обход true-peak headroom (правило CLAUDE.md)."""
    if x_rms_db <= threshold_db or ratio <= 1.0:
        return 0.0
    gr = (x_rms_db - threshold_db) * (ratio - 1.0) / ratio
    makeup = min(0.7 * gr, MAX_MAKEUP_DB)
    if true_peak_dbtp is not None:
        headroom = true_peak_ceiling_dbtp - true_peak_dbtp
        makeup = min(makeup, max(0.0, headroom))
    return float(max(0.0, makeup))


def ma2015_params_from_audio(
    x: np.ndarray,
    sample_rate: int,
    *,
    true_peak_dbtp: Optional[float] = None,
) -> Ma2015Params:
    """Полный расчёт параметров компрессора от блока аудио."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    x_rms_db = 20.0 * np.log10(rms + 1e-10)
    peak_db = 20.0 * np.log10(peak + 1e-10)
    crest_db = peak_db - x_rms_db

    w_p = percussivity_weighting(x, sample_rate)
    w_f = low_frequency_weighting(x, sample_rate)
    ratio = ratio_ma2015(w_p, w_f)
    threshold_db = threshold_ma2015(x_rms_db, w_p)
    attack_ms, release_ms = ballistics_giannoulis(crest_db, w_p)
    makeup_db = bounded_makeup_db(
        x_rms_db, threshold_db, ratio,
        true_peak_dbtp=true_peak_dbtp if true_peak_dbtp is not None else peak_db,
    )
    return Ma2015Params(
        w_p=w_p, w_f=w_f, x_rms_db=x_rms_db, crest_db=crest_db,
        ratio=ratio, threshold_db=threshold_db,
        attack_ms=attack_ms, release_ms=release_ms, makeup_db=makeup_db,
    )
