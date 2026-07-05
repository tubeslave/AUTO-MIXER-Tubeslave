"""Balance v3, слой измерений: устойчивость к проникновению (bleed).

Фундаментальная проблема живого поканального измерения громкости:
вокальный микрофон содержит громкие барабаны — equal-loudness алгоритм
ошибочно ОПУСТИТ фейдер вокала (Moffat & Sandler, AES 146, 2019: модель
«один трек = один источник» неверна для живой записи).

Три инструмента (документ FOH, §3.3):
  (a) bleed-пол канала: уровень микрофона, когда его инструмент молчит
      (оценивается в паузах/при чужих соло на саундчеке) — измерение
      громкости гейтируется относительно этого пола;
  (b) матрица утечек L[i][j] (уровень в микрофоне i при соло источника j)
      + неотрицательное решение x = L·s (NNLS с регуляризацией Тихонова)
      даёт истинные поканальные уровни из загрязнённых измерений;
  (c) измерительный микрофон FOH — только как глобальная проверка
      (вне этого модуля).

Proposal-only математика: модуль ничего не пишет на пульт.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import nnls

EPS = 1e-12
DEFAULT_BLEED_MARGIN_DB = 6.0
DEFAULT_TIKHONOV_LAMBDA = 1e-2


def bleed_floor_db(
    quiet_block_levels_db: Sequence[float],
    percentile: float = 75.0,
) -> float:
    """Bleed-пол: типичный уровень канала, когда его инструмент МОЛЧИТ.

    Берётся перцентиль (не максимум) уровней блоков из пауз/чужих соло —
    устойчиво к единичным выбросам.
    """
    values = [float(v) for v in quiet_block_levels_db if np.isfinite(v)]
    if not values:
        return -120.0
    return float(np.percentile(values, percentile))


def gated_channel_loudness_db(
    block_levels_db: Sequence[float],
    bleed_floor: float,
    margin_db: float = DEFAULT_BLEED_MARGIN_DB,
) -> Optional[float]:
    """Громкость канала только по блокам, где он выше bleed-пола + запас.

    None — канал не поднимался над своим bleed-полом (инструмент молчит,
    измерение было бы чистым bleed'ом; решения по нему принимать нельзя).
    """
    gate = bleed_floor + margin_db
    active = [float(v) for v in block_levels_db
              if np.isfinite(v) and float(v) > gate]
    if not active:
        return None
    power = np.mean([10.0 ** (v / 10.0) for v in active])
    return float(10.0 * np.log10(power + EPS))


@dataclass
class LeakageMatrix:
    """Матрица утечек: L[i][j] — линейная мощность в микрофоне i при
    единичном соло источника j (снимается на поканальных соло саундчека)."""

    channels: List[Any]
    matrix: np.ndarray  # (n_mics, n_sources), линейная мощность

    @classmethod
    def from_solo_measurements(
        cls,
        solo_levels_db: Mapping[Any, Mapping[Any, float]],
    ) -> "LeakageMatrix":
        """solo_levels_db[j][i] — уровень (дБ) в микрофоне i при соло j.

        Диагональ нормируется к 1 (собственный сигнал = единичная мощность):
        матрица описывает ОТНОСИТЕЛЬНЫЕ утечки.
        """
        channels = sorted(solo_levels_db.keys(), key=str)
        n = len(channels)
        m = np.zeros((n, n), dtype=np.float64)
        for j_idx, j in enumerate(channels):
            row = solo_levels_db[j]
            own_db = float(row.get(j, 0.0))
            for i_idx, i in enumerate(channels):
                level_db = row.get(i)
                if level_db is None:
                    m[i_idx, j_idx] = 1.0 if i_idx == j_idx else 0.0
                    continue
                m[i_idx, j_idx] = 10.0 ** ((float(level_db) - own_db) / 10.0)
        return cls(channels=channels, matrix=m)

    def estimate_true_levels_db(
        self,
        measured_db: Mapping[Any, float],
        lam: float = DEFAULT_TIKHONOV_LAMBDA,
    ) -> Dict[Any, float]:
        """Истинные поканальные уровни из загрязнённых измерений.

        Решает x = L·s неотрицательно (NNLS) с Тихоновской регуляризацией:
        расширенная система [L; sqrt(lam)·I] s = [x; 0].
        """
        x = np.array(
            [10.0 ** (float(measured_db.get(ch, -120.0)) / 10.0)
             for ch in self.channels],
            dtype=np.float64,
        )
        n = len(self.channels)
        a_aug = np.vstack([self.matrix, np.sqrt(lam) * np.eye(n)])
        b_aug = np.concatenate([x, np.zeros(n)])
        s, _ = nnls(a_aug, b_aug)
        return {
            ch: float(10.0 * np.log10(s[idx] + EPS))
            for idx, ch in enumerate(self.channels)
        }
