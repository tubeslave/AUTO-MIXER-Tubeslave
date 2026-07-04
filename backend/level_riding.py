"""Кросс-адаптивный райдинг уровней по R128 (Perez-Gonzalez & Reiss 2009;
Mansbridge, Finn & Reiss 2012).

Каналы приводятся к целевой громкости с приоритетными офсетами (лид-вокал
на несколько дБ громче среднего). Ключевые safety-элементы документа:
deadband (анти-дребезг), лимит скорости в дБ/с и hysteresis loudness gate —
намеренную динамику песни (тихий куплет, громкий припев) фейдером НЕ гоняем.

Модуль чистых функций: решения о применении идут через SafetyGate v2 и
supervised-путь. Помнить правило проекта: непрерывные коррекции — только
идемпотентные абсолютные цели (не относительные приращения от замера,
который не видит применённое изменение).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

DEFAULT_DEADBAND_LU = 1.0
DEFAULT_MAX_STEP_DB = 1.5
DEFAULT_MAX_RATE_DB_PER_SEC = 3.0
DEFAULT_SILENCE_LUFS = -45.0
DEFAULT_INTENTIONAL_DYNAMICS_LU = 6.0

# Приоритетные офсеты к средней громкости микса (роль -> LU)
PRIORITY_OFFSET_LU = {
    "lead_vocal": 3.0,
    "solo": 2.5,
    "snare": 0.0,
    "kick": 0.0,
    "bass": -0.5,
    "guitar": -1.0,
    "keys": -2.0,
    "backing_vocal": -2.0,
    "playback": -1.0,
    "overheads": -3.0,
    "line": -1.5,
}


def priority_target_lu(mix_avg_lufs: float, role: str) -> float:
    """L_target = среднее микса + приоритетный офсет роли."""
    return float(mix_avg_lufs) + float(PRIORITY_OFFSET_LU.get(role, -1.5))


def riding_step_db(
    measured_lufs: float,
    target_lufs: float,
    dt_s: float,
    *,
    deadband_lu: float = DEFAULT_DEADBAND_LU,
    max_step_db: float = DEFAULT_MAX_STEP_DB,
    max_rate_db_per_sec: float = DEFAULT_MAX_RATE_DB_PER_SEC,
) -> float:
    """Шаг райдинга: clamp(error, deadband, шаг, скорость). 0.0 = не трогать."""
    error = float(target_lufs) - float(measured_lufs)
    if abs(error) < deadband_lu:
        return 0.0
    step = float(np.clip(error, -max_step_db, max_step_db))
    rate_cap = max_rate_db_per_sec * max(float(dt_s), 0.0)
    return float(np.clip(step, -rate_cap, rate_cap))


def loudness_hysteresis_gate(
    short_term_lufs: Sequence[float],
    *,
    silence_threshold_lufs: float = DEFAULT_SILENCE_LUFS,
    intentional_dynamics_lu: float = DEFAULT_INTENTIONAL_DYNAMICS_LU,
) -> bool:
    """True — райдинг разрешён; False — канал держим (не гоняем динамику).

    Логика hysteresis loudness gate (Mansbridge et al.): райдим только когда
    канал в «обычном» рабочем диапазоне. Блокируем, если:
    - канал молчит/пауза (ниже silence_threshold), или
    - текущая громкость намеренно далеко от типичной (медианы активных
      блоков) — тихий куплет/брейкдаун, который выравнивать нельзя.
    """
    series = [v for v in short_term_lufs if np.isfinite(v)]
    if not series:
        return False
    current = float(series[-1])
    if current < silence_threshold_lufs:
        return False
    active = [v for v in series if v >= silence_threshold_lufs]
    if len(active) < 4:
        return False  # мало статистики — не трогаем
    typical = float(np.median(active))
    if abs(current - typical) > intentional_dynamics_lu:
        return False
    return True
