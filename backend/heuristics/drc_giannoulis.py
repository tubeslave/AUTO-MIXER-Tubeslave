"""Comp v3, ядро: компрессор Giannoulis, Massberg & Reiss (JAES 60(6), 2012).

Валидированная, воспроизводимая математика DRC: feed-forward топология,
soft-knee gain computer в лог-домене и smooth decoupled peak detector
(рекомендованное в статье размещение — лог-домен после gain computer:
самая плавная огибающая и полное использование времени релиза).

Назначение в проекте: НЕ live-обработка (компрессором на концерте является
сам WING) — это симуляционный примитив для предсказательной верификации
параметров на захваченном буфере (level-match, метрика пампинга, bleed-lift)
и офлайн-пайплайнов. Детектор считается на декимированной огибающей
(hop-рейт ~3 кГц): для верификации и метрик модуляции 2-8 Гц этого
достаточно, а полный Python-цикл по сэмплам не нужен.

Определение времён: 63% (1 − 1/e), т.е. alpha = exp(−1/(tau·fs)) — как в
статье и открытой реализации CTAGDRC; фиксируем это соглашение.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

EPS = 1e-9
ENV_HOP_SAMPLES = 16          # огибающая ~3 кГц при 48 кГц
PUMPING_BAND_HZ = (2.0, 8.0)  # амплитудная модуляция максимально слышима ~4 Гц


def ballistics_alpha(time_ms: float, rate_hz: float) -> float:
    """alpha = exp(-1/(tau*fs)) — соглашение 63% (Giannoulis/CTAGDRC)."""
    tau_s = max(float(time_ms), 1e-3) / 1000.0
    return float(np.exp(-1.0 / (tau_s * float(rate_hz))))


def soft_knee_gain_db(
    x_db: np.ndarray | float,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> np.ndarray | float:
    """Gain computer (статическая характеристика): требуемое ослабление x_L >= 0.

    Уравнения статьи; при knee_db=0 вырождается в жёсткое колено.
    """
    x = np.asarray(x_db, dtype=np.float64)
    T, R, W = float(threshold_db), max(float(ratio), 1.0), max(float(knee_db), 0.0)
    d = x - T
    if W <= 0.0:
        y = np.where(d < 0.0, x, T + d / R)
    else:
        below = 2.0 * d < -W
        inside = np.abs(2.0 * d) <= W
        y = np.where(
            below,
            x,
            np.where(
                inside,
                x + (1.0 / R - 1.0) * (d + W / 2.0) ** 2 / (2.0 * W),
                T + d / R,
            ),
        )
    out = x - y  # x_L: требуемое ослабление, дБ (>= 0)
    return float(out) if np.isscalar(x_db) else out


def smooth_decoupled_detector(
    x_l_db: np.ndarray,
    attack_ms: float,
    release_ms: float,
    env_rate_hz: float,
) -> np.ndarray:
    """Smooth decoupled peak detector (рекомендация статьи), лог-домен.

    y1[n] = max(xL[n], aR*y1[n-1] + (1-aR)*xL[n])
    yL[n] = aA*yL[n-1] + (1-aA)*y1[n]
    """
    a_a = ballistics_alpha(attack_ms, env_rate_hz)
    a_r = ballistics_alpha(release_ms, env_rate_hz)
    y1 = 0.0
    y_l = 0.0
    out = np.empty_like(x_l_db)
    for n in range(len(x_l_db)):
        y1 = max(float(x_l_db[n]), a_r * y1 + (1.0 - a_r) * float(x_l_db[n]))
        y_l = a_a * y_l + (1.0 - a_a) * y1
        out[n] = y_l
    return out


@dataclass
class DRCSimulation:
    output: np.ndarray
    gain_db_series: np.ndarray     # применённое усиление (makeup − GR), дБ
    gr_db_series: np.ndarray       # сглаженное ослабление yL, дБ (>= 0)
    env_rate_hz: float

    @property
    def mean_gr_db(self) -> float:
        return float(np.mean(self.gr_db_series))

    @property
    def max_gr_db(self) -> float:
        return float(np.max(self.gr_db_series)) if len(self.gr_db_series) else 0.0


def simulate_compressor(
    x: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float,
    ratio: float,
    knee_db: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float = 0.0,
) -> DRCSimulation:
    """Симуляция feed-forward компрессора на буфере (контрольный рейт ~3 кГц)."""
    x = np.asarray(x, dtype=np.float64)
    hop = ENV_HOP_SAMPLES
    n_frames = max(1, len(x) // hop)
    frames = x[: n_frames * hop].reshape(n_frames, hop)
    level_db = 20.0 * np.log10(np.max(np.abs(frames), axis=1) + EPS)
    env_rate = sample_rate / hop

    x_l = soft_knee_gain_db(level_db, threshold_db, ratio, knee_db)
    gr = smooth_decoupled_detector(np.asarray(x_l), attack_ms, release_ms, env_rate)
    gain_db = makeup_db - gr

    gain_lin = 10.0 ** (np.repeat(gain_db, hop) / 20.0)
    out = x.copy()
    out[: n_frames * hop] *= gain_lin
    return DRCSimulation(output=out, gain_db_series=gain_db,
                         gr_db_series=gr, env_rate_hz=env_rate)


def pumping_metric_db(gain_db_series: np.ndarray, env_rate_hz: float) -> float:
    """Энергия модуляции огибающей усиления в полосе 2-8 Гц (RMS, дБ).

    Пампинг/«дыхание» максимально слышимы около 4 Гц — метрика напрямую
    меряет то, что слышит зал.
    """
    g = np.asarray(gain_db_series, dtype=np.float64)
    if len(g) < 16:
        return 0.0
    g = g - float(np.mean(g))
    spec = np.fft.rfft(g * np.hanning(len(g)))
    freqs = np.fft.rfftfreq(len(g), 1.0 / env_rate_hz)
    band = (freqs >= PUMPING_BAND_HZ[0]) & (freqs <= PUMPING_BAND_HZ[1])
    if not band.any():
        return 0.0
    # RMS модуляции в полосе, приведённая к дБ-домену усиления
    return float(np.sqrt(np.sum(np.abs(spec[band]) ** 2) * 2.0) / len(g))


def lf_release_guard_ms(release_ms: float, f_low_hz: float) -> float:
    """Правило «релиз > 1/f_low»: короче периода нижней частоты компрессор
    модулирует форму волны внутри периода (НЧ-искажения)."""
    period_ms = 1000.0 / max(float(f_low_hz), 20.0)
    return float(max(release_ms, period_ms))
