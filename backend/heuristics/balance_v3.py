"""Balance v3 — двухрежимный метод баланса громкостей (синтез FOH-документа).

САУНДЧЕК-контур:
  Вектор целей строится от ведущего вокала (единственная строго измеренная
  эвристика: вокал ≈ −3 LU относительно суммы микса, De Man et al. ISMIR
  2014, SD ±1.3 LU — самая согласованная величина у профессионалов) с
  жанровыми пресетами, и якорится на АБСОЛЮТНОЙ плоскости Ayaic проекта
  (lead −22 LUFS — правило CLAUDE.md): относительная иерархия документа и
  Ayaic-плоскости согласованы, пресет только модулирует смещения.
  Итеративный решатель фейдеров: fader_i += target_i − measured_i,
  2–3 прохода, сходимость ±0.5 LU, потолок 0 dBFS (правило проекта —
  нормализация только вниз).

LIVE-контур (слои, по возрастанию риска):
  1. Страж разборчивости вокала (по умолчанию ВКЛ): вокал < бэнд + 2 LU
     дольше 3 с → подъём ≤ 0.5 дБ/с, максимум +3 дБ, гистерезис 1 LU.
  2. Медленная дрейф-коррекция к снапшоту: мёртвая зона ±1.75 LU (JND в
     сложной музыке ≈ 3 дБ), ≤ 0.3 дБ/с, кламп ±3 дБ, freeze при
     намеренной динамике. Цели АБСОЛЮТНЫЕ (правило проекта против
     интеграторного храповика: якорь — снапшот, не текущее значение).
  3. Dugan gain-sharing — только речевые сегменты (dugan_gain_sharing.py).
  Детектор намеренной динамики: коррелированный спад большинства каналов
  = музыка (тихий бридж), автоматика замирает.

Правила проекта, наследуемые контрактом: fader ≤ 0 dBFS; предусиление в
лайве НЕ трогается (только фейдеры/DCA); bleed-робастные измерения — из
heuristics/bleed_leakage. Proposal-only: модуль ничего не пишет на пульт.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

FADER_CEILING_DB = 0.0
SOLVER_TOLERANCE_LU = 0.5
SOLVER_MAX_PASSES = 3
SOUNDCHECK_MAX_MOVE_DB = 12.0

VOCAL_GUARD_MARGIN_LU = 2.0
VOCAL_GUARD_HOLD_S = 3.0
VOCAL_GUARD_RATE_DB_S = 0.5
VOCAL_GUARD_CAP_DB = 3.0
VOCAL_GUARD_HYSTERESIS_LU = 1.0

DRIFT_DEADBAND_LU = 1.75          # JND в сложной музыке ≈ 3 дБ
DRIFT_RATE_DB_S = 0.3
DRIFT_CAP_DB = 3.0

MUSICAL_DROP_LU = -1.5
MUSICAL_DROP_FRACTION = 0.7

AYAIC_LEAD_VOCAL_LUFS = -22.0     # абсолютный якорь плоскостей проекта

# Жанровые пресеты: смещения LU относительно ведущего вокала (= 0).
# pop_rock — гибрид измеренной иерархии De Man и Ayaic-плоскостей;
# выбирается оператором НА САУНДЧЕКЕ, не угадывается в лайве.
GENRE_PRESETS: Dict[str, Dict[str, float]] = {
    "pop_rock": {
        "lead_vocal": 0.0, "backing_vocal": -4.0, "bass": -3.0,
        "kick": -4.0, "snare": -5.0, "tom": -6.0, "floor_tom": -6.0,
        "overheads": -4.0, "hihat": -6.0, "guitar": -5.0,
        "acoustic_guitar": -5.0, "keys": -6.0, "playback": -3.0,
        "line": -5.0,
    },
    "bass_forward": {
        "lead_vocal": 0.0, "backing_vocal": -5.0, "bass": -1.0,
        "kick": -0.5, "snare": -4.0, "tom": -6.0, "floor_tom": -6.0,
        "overheads": -5.0, "hihat": -7.0, "guitar": -6.0,
        "acoustic_guitar": -6.0, "keys": -6.0, "playback": -2.0,
        "line": -5.0,
    },
    "vocal_forward": {
        "lead_vocal": 0.0, "backing_vocal": -5.0, "bass": -7.0,
        "kick": -8.0, "snare": -9.0, "tom": -10.0, "floor_tom": -10.0,
        "overheads": -8.0, "hihat": -10.0, "guitar": -8.0,
        "acoustic_guitar": -7.0, "keys": -8.0, "playback": -6.0,
        "line": -8.0,
    },
}


def target_vector_lufs(
    roles_by_channel: Mapping[Any, str],
    genre: str = "pop_rock",
    *,
    anchor_lufs: float = AYAIC_LEAD_VOCAL_LUFS,
) -> Dict[Any, float]:
    """Абсолютные LUFS-цели: вокал-относительный пресет на Ayaic-якоре."""
    offsets = GENRE_PRESETS.get(genre, GENRE_PRESETS["pop_rock"])
    return {
        ch: float(anchor_lufs) + float(offsets.get(role, offsets["line"]))
        for ch, role in roles_by_channel.items()
    }


@dataclass
class FaderSolution:
    faders_db: Dict[Any, float]
    passes: int
    converged: bool
    residual_lu: Dict[Any, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faders_db": {str(k): round(v, 2) for k, v in self.faders_db.items()},
            "passes": self.passes,
            "converged": self.converged,
            "residual_lu": {str(k): round(v, 2) for k, v in self.residual_lu.items()},
            "notes": list(self.notes),
            "method": "balance_v3_soundcheck",
        }


def solve_faders(
    measured_lufs: Mapping[Any, float],
    targets_lufs: Mapping[Any, float],
    *,
    current_faders_db: Optional[Mapping[Any, float]] = None,
    measure_fn=None,
    tolerance_lu: float = SOLVER_TOLERANCE_LU,
    max_passes: int = SOLVER_MAX_PASSES,
    max_move_db: float = SOUNDCHECK_MAX_MOVE_DB,
) -> FaderSolution:
    """Итеративный решатель фейдеров саундчека.

    fader_i += target_i − measured_i, 2–3 прохода (шины/маскировка меняют
    сумму), сходимость ±0.5 LU. Потолок 0 dBFS соблюдается НОРМАЛИЗАЦИЕЙ
    ВНИЗ: если целям нужен фейдер > 0, весь вектор сдвигается вниз (баланс
    относителен — правило проекта).

    measure_fn(faders_db) -> measured_lufs — пересчёт измерений с учётом
    применённых фейдеров (в офлайне/симуляции); по умолчанию линейная
    модель measured + delta.
    """
    faders = {ch: float((current_faders_db or {}).get(ch, 0.0))
              for ch in measured_lufs}
    start = dict(faders)
    measured = {ch: float(v) for ch, v in measured_lufs.items()}
    targets = {ch: float(v) for ch, v in targets_lufs.items()}
    notes: List[str] = []

    passes = 0
    converged = False
    residual: Dict[Any, float] = {}
    for passes in range(1, max_passes + 1):
        residual = {ch: targets[ch] - measured[ch]
                    for ch in measured if ch in targets}
        if all(abs(e) <= tolerance_lu for e in residual.values()):
            converged = True
            break
        for ch, err in residual.items():
            faders[ch] += err
            # bounded move от стартовой позиции
            faders[ch] = float(np.clip(
                faders[ch], start[ch] - max_move_db, start[ch] + max_move_db))
        # Потолок 0 dBFS: баланс ОТНОСИТЕЛЕН — при переполнении вниз сдвигается
        # и вектор фейдеров, и вектор целей (иначе абсолютные цели становятся
        # недостижимыми и решатель осциллирует).
        overshoot = max(faders.values()) - FADER_CEILING_DB
        if overshoot > 0.0:
            faders = {ch: v - overshoot for ch, v in faders.items()}
            targets = {ch: v - overshoot for ch, v in targets.items()}
            notes.append(
                f"pass {passes}: вектор сдвинут на {-overshoot:.1f} dB "
                "(потолок 0 dBFS, нормализация вниз)")
        if measure_fn is not None:
            measured = {ch: float(v) for ch, v in measure_fn(faders).items()}
        else:
            # линейная модель: измерение следует за суммарным смещением фейдера
            measured = {ch: float(measured_lufs[ch]) + (faders[ch] - start[ch])
                        for ch in measured}
    else:
        residual = {ch: targets[ch] - measured[ch]
                    for ch in measured if ch in targets}
        converged = all(abs(e) <= tolerance_lu for e in residual.values())

    return FaderSolution(faders_db=faders, passes=passes,
                         converged=converged, residual_lu=residual, notes=notes)


# ── LIVE слой 1: страж разборчивости вокала ──────────────────────

@dataclass
class VocalGuardState:
    boost_db: float = 0.0
    deficit_seconds: float = 0.0
    margin_lu: float = VOCAL_GUARD_MARGIN_LU
    hold_s: float = VOCAL_GUARD_HOLD_S
    rate_db_s: float = VOCAL_GUARD_RATE_DB_S
    cap_db: float = VOCAL_GUARD_CAP_DB
    hysteresis_lu: float = VOCAL_GUARD_HYSTERESIS_LU


def vocal_guard_step(
    state: VocalGuardState,
    vocal_lu: float,
    band_bus_lu: float,
    dt_s: float,
    *,
    vocal_active: bool = True,
) -> VocalGuardState:
    """Слой 1: вокал не должен тонуть под бэндом (самая ценная и наименее
    рискованная live-функция). Дефицит держится hold_s секунд прежде чем
    гвард начинает поднимать — защита от коротких музыкальных перекрытий."""
    if not vocal_active:
        state.deficit_seconds = 0.0
        return state
    deficit = (band_bus_lu + state.margin_lu) - vocal_lu
    if deficit > 0.0:
        state.deficit_seconds += dt_s
        if state.deficit_seconds >= state.hold_s:
            step = min(state.rate_db_s * dt_s, deficit)
            state.boost_db = min(state.boost_db + step, state.cap_db)
    else:
        state.deficit_seconds = 0.0
        if deficit < -state.hysteresis_lu and state.boost_db > 0.0:
            state.boost_db = max(state.boost_db - 0.5 * state.rate_db_s * dt_s, 0.0)
    return state


# ── LIVE слой 2: дрейф-коррекция + детектор намеренной динамики ──

def musical_dynamics_detected(
    delta_lu_by_channel: Mapping[Any, float],
    *,
    drop_threshold_lu: float = MUSICAL_DROP_LU,
    fraction: float = MUSICAL_DROP_FRACTION,
) -> bool:
    """Коррелированный спад большинства каналов = намеренная музыкальная
    динамика (тихий бридж) — автоматика обязана замереть, не «выравнивать»."""
    if not delta_lu_by_channel:
        return False
    drops = [d for d in delta_lu_by_channel.values() if d < drop_threshold_lu]
    return len(drops) >= fraction * len(delta_lu_by_channel)


def drift_correction_step(
    measured_lu: float,
    snapshot_target_lu: float,
    accumulated_correction_db: float,
    dt_s: float,
    *,
    deadband_lu: float = DRIFT_DEADBAND_LU,
    rate_db_s: float = DRIFT_RATE_DB_S,
    cap_db: float = DRIFT_CAP_DB,
    channel_active: bool = True,
    musical_dynamics: bool = False,
) -> float:
    """Слой 2: шаг медленной коррекции дрейфа к снапшоту (дБ, может быть 0).

    Цель АБСОЛЮТНАЯ (снапшот), накопленная коррекция клампится ±cap_db —
    интеграторный храповик исключён по построению."""
    if not channel_active or musical_dynamics:
        return 0.0
    error = float(snapshot_target_lu) - float(measured_lu)
    if abs(error) <= deadband_lu:
        return 0.0
    step = float(np.clip(error, -rate_db_s * dt_s, rate_db_s * dt_s))
    new_total = float(np.clip(accumulated_correction_db + step, -cap_db, cap_db))
    return new_total - accumulated_correction_db
