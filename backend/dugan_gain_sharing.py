"""Dugan gain sharing (Dugan 1975, JAES 23) — автомикс открытых микрофонов.

Каждый канал ослабляется на разницу между его уровнем и уровнем суммы всех
каналов (по мощности): суммарный гейн системы остаётся 0 дБ. Сильнейший
микрофон получает наибольший гейн, слабые приглушаются — меньше реверберации,
шума и риска заводки, чем у гейтов.

Область применения (правило документа): только речевые/многомикрофонные
сегменты (ведущий, несколько вокалистов, паузы между песнями) — НЕ
музыкальный ансамбль. Активация — оператором или детектором речевого
сегмента; модуль сам решения о включении не принимает.

Live-ограничения проекта: сглаживание с лимитом скорости в дБ/с и шаг
обновления не чаще OSC-троттлера WING (~10 Гц на адрес). Только
отрицательные гейны — потолок 0 дБ соблюдается по построению.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

MIN_LEVEL_DB = -80.0
DEFAULT_FLOOR_DB = -40.0          # ниже не приглушаем (естественный фон)
DEFAULT_MAX_RATE_DB_PER_SEC = 12.0  # речь требует быстрее фейдер-райдинга
DEFAULT_MIN_UPDATE_INTERVAL_S = 0.12  # > интервала WING-троттлера (100 мс)


def dugan_gains_db(
    levels_db: Mapping[str, float],
    weights_db: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Чистая формула Dugan: gain_i = level_i − 10·log10(Σ 10^(level_j/10)).

    weights_db — относительная чувствительность канала (приоритет): вес
    прибавляется к уровню ПРИ РАСЧЁТЕ ДОЛИ, но не меняет суммарный гейн.
    Все гейны <= 0; сумма мощностных долей равна 1 (суммарный гейн 0 дБ).
    """
    if not levels_db:
        return {}
    weights_db = weights_db or {}
    weighted = {
        ch: max(float(lvl), MIN_LEVEL_DB) + float(weights_db.get(ch, 0.0))
        for ch, lvl in levels_db.items()
    }
    powers = {ch: 10.0 ** (lvl / 10.0) for ch, lvl in weighted.items()}
    total = sum(powers.values())
    if total <= 0.0:
        return {ch: 0.0 for ch in levels_db}
    return {ch: 10.0 * np.log10(powers[ch] / total) for ch in powers}


@dataclass
class _ChannelShareState:
    smoothed_gain_db: float = 0.0
    last_update_t: float = 0.0


@dataclass
class DuganGainSharing:
    """Stateful gain-sharing с live-сглаживанием.

    update() принимает измеренные уровни (дБ) открытых микрофонов и
    возвращает целевые ОТНОСИТЕЛЬНЫЕ гейны (дБ, <= 0) вокруг baseline
    фейдеров. Значения уже сглажены и ограничены по скорости; пустой dict
    — обновлять нечего (интервал троттлера не прошёл).
    """

    weights_db: Dict[str, float] = field(default_factory=dict)
    floor_db: float = DEFAULT_FLOOR_DB
    max_rate_db_per_sec: float = DEFAULT_MAX_RATE_DB_PER_SEC
    min_update_interval_s: float = DEFAULT_MIN_UPDATE_INTERVAL_S
    time_provider: object = time.monotonic
    _states: Dict[str, _ChannelShareState] = field(default_factory=dict)
    _last_batch_t: float = field(default=0.0)

    def update(
        self,
        levels_db: Mapping[str, float],
        *,
        segment_type: str = "speech",
        enabled_channels: "frozenset | set | None" = None,
    ) -> Dict[str, float]:
        """segment_type — тип текущего сегмента программы:
        - "speech": штатный режим (речь/объявления) — gain-sharing работает;
        - "vocal"/"music": ПЕНИЕ или музыка — слой выключен и возвращает {}
          (кейс дуэта: Dugan не давит тихого вокалиста на вокальных
          сегментах; балансом поющих занимается guard по суммарной шине).
        enabled_channels — фильтр (инструментальное шоу: только
        announce-микрофоны; пустой набор = слой выключен).
        """
        if segment_type != "speech":
            return {}
        if enabled_channels is not None:
            levels_db = {ch: v for ch, v in levels_db.items()
                         if ch in enabled_channels}
            if not levels_db:
                return {}
        now = float(self.time_provider())
        if self._last_batch_t and (now - self._last_batch_t) < self.min_update_interval_s:
            return {}
        dt = (now - self._last_batch_t) if self._last_batch_t else self.min_update_interval_s
        self._last_batch_t = now

        raw = dugan_gains_db(levels_db, self.weights_db)
        max_step = self.max_rate_db_per_sec * dt
        out: Dict[str, float] = {}
        for ch, target in raw.items():
            target = max(target, self.floor_db)
            state = self._states.setdefault(ch, _ChannelShareState())
            delta = float(np.clip(target - state.smoothed_gain_db, -max_step, max_step))
            state.smoothed_gain_db += delta
            state.last_update_t = now
            out[ch] = round(state.smoothed_gain_db, 2)
        return out

    def reset(self):
        self._states.clear()
        self._last_batch_t = 0.0


def is_speech_segment(
    active_channel_count: int,
    music_activity_score: float,
    *,
    max_music_score: float = 0.3,
    min_channels: int = 2,
) -> bool:
    """Грубый гейт применимости: несколько открытых микрофонов и нет
    музыкальной активности (score из ScenarioDetector/классификатора).
    Финальное слово — за оператором (режим включается явно)."""
    return active_channel_count >= min_channels and music_activity_score <= max_music_score
