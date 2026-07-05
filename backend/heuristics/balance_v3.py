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

# Абсолютная Ayaic-плоскость по роли якоря (правило CLAUDE.md): якорем может
# быть не только вокал — инструментальный сет якорится на своей плоскости.
ROLE_ANCHOR_LUFS: Dict[str, float] = {
    "lead_vocal": -22.0, "lead_instrument": -22.0,
    "kick": -25.0, "snare": -25.0, "bass": -25.0, "guitar": -25.0,
    "acoustic_guitar": -25.0, "playback": -25.0, "backing_vocal": -27.0,
    "overheads": -28.0, "tom": -28.0, "floor_tom": -28.0, "hihat": -28.0,
    "keys": -30.0, "line": -27.0,
}

ANCHOR_ACTIVE_HOLD_S = 2.0        # непрерывное подтверждение активности якоря
ANCHOR_BLEED_MARGIN_DB = 6.0      # над bleed-полом из снапшота
INSTRUMENTAL_GUARD_MARGIN_LU = 1.5
DRIFT_REF_STABLE_LU = 1.0         # |ΔLU_drift_ref| ниже — референс стабилен

# Жанровые пресеты: смещения LU относительно ведущего вокала (= 0).
# pop_rock — гибрид измеренной иерархии De Man и Ayaic-плоскостей;
# выбирается оператором НА САУНДЧЕКЕ, не угадывается в лайве.
# backing_vocal −4…−5 LU — НЕкалиброванный практический prior (внутри
# вокальной шины, относительно лид-вокала); требует позальной калибровки.
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


@dataclass
class BalanceConfig:
    """Конфиг баланса. Якорь назначается НА САУНДЧЕКЕ и фиксируется в
    снапшоте; автоматическое переназначение якоря в LIVE запрещено
    (защита от скачка системы координат при ошибке классификатора)."""
    anchor_channel: str = "lead_vocal"      # роль опорного канала
    anchor_fallback: List[str] = field(default_factory=lambda: [
        "lead_instrument", "kick", "snare"])
    targets_LU: Optional[Dict[str, float]] = None  # офсеты отн. anchor = 0
    genre: str = "pop_rock"
    guard_enabled: bool = True
    guard_margin_LU: float = 2.5            # вокал 2-3 LU; инструмент 1-2 LU
    guard_bus: str = "vocal_bus"            # guard читает/двигает ШИНУ, не каналы
    guard_reassign_on_instrumental: bool = True  # False -> guard выключается
    drift_reference: Optional[str] = None   # ручное переопределение
    announce_channels: List[str] = field(default_factory=list)  # Dugan в инстр. шоу


def resolve_anchor(
    config: BalanceConfig,
    active_roles: "set[str] | frozenset[str]",
) -> "tuple[str, Optional[str]]":
    """Выбор якоря: сконфигурированный, иначе первый доступный из fallback.

    Возвращает (anchor_role, warning). Warning != None — якорь заменён.
    """
    if config.anchor_channel in active_roles:
        return config.anchor_channel, None
    for candidate in config.anchor_fallback:
        if candidate in active_roles:
            warning = (
                f"anchor '{config.anchor_channel}' отсутствует среди активных "
                f"каналов саундчека — выбран fallback '{candidate}'")
            logger = __import__("logging").getLogger(__name__)
            logger.warning(warning)
            return candidate, warning
    fallback = sorted(active_roles)[0] if active_roles else config.anchor_channel
    return fallback, f"ни один fallback-якорь не активен — взят '{fallback}'"


def rebase_targets(
    targets_lu: Mapping[str, float],
    new_anchor: str,
) -> Dict[str, float]:
    """Смена системы координат: target_i_new = target_i_old − target_anchor_old.

    Абсолютные позиции фейдеров НЕ меняются — только точка отсчёта офсетов.
    """
    if new_anchor not in targets_lu:
        raise KeyError(f"новый якорь '{new_anchor}' отсутствует в таблице целей")
    shift = float(targets_lu[new_anchor])
    return {role: float(v) - shift for role, v in targets_lu.items()}


def target_vector_lufs(
    roles_by_channel: Mapping[Any, str],
    genre: str = "pop_rock",
    *,
    anchor_lufs: float = AYAIC_LEAD_VOCAL_LUFS,
    anchor_role: str = "lead_vocal",
    offsets_lu: Optional[Mapping[str, float]] = None,
) -> Dict[Any, float]:
    """Абсолютные LUFS-цели: якорь-относительные офсеты на Ayaic-плоскости.

    Офсеты (пресет жанра или config.targets_LU) ребейзятся так, чтобы
    anchor_role = 0, затем ложатся на абсолютную плоскость якоря. Кейс
    дуэта покрыт по построению: оба канала с ролью якоря получают 0 LU.
    """
    offsets = dict(offsets_lu or GENRE_PRESETS.get(genre, GENRE_PRESETS["pop_rock"]))
    if anchor_role not in offsets:
        offsets[anchor_role] = 0.0
    offsets = rebase_targets(offsets, anchor_role)
    fallback_offset = offsets.get("line", -5.0)
    return {
        ch: float(anchor_lufs) + float(offsets.get(role, fallback_offset))
        for ch, role in roles_by_channel.items()
    }


def build_target_vector(
    config: BalanceConfig,
    roles_by_channel: Mapping[Any, str],
) -> "tuple[Dict[Any, float], str, Optional[str]]":
    """Полный путь: выбор якоря (с fallback) → ребейз офсетов → абсолютные цели.

    Возвращает (targets, anchor_role, warning).
    """
    anchor_role, warning = resolve_anchor(
        config, set(roles_by_channel.values()))
    anchor_lufs = ROLE_ANCHOR_LUFS.get(anchor_role, -25.0)
    targets = target_vector_lufs(
        roles_by_channel, config.genre,
        anchor_lufs=anchor_lufs, anchor_role=anchor_role,
        offsets_lu=config.targets_LU,
    )
    return targets, anchor_role, warning


@dataclass
class FaderSolution:
    faders_db: Dict[Any, float]
    passes: int
    converged: bool
    residual_lu: Dict[Any, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # Якорь фиксируется при сохранении снапшота саундчека; автоматическое
    # переназначение в LIVE запрещено (скачок системы координат).
    anchor_role: str = "lead_vocal"
    anchor_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faders_db": {str(k): round(v, 2) for k, v in self.faders_db.items()},
            "passes": self.passes,
            "converged": self.converged,
            "residual_lu": {str(k): round(v, 2) for k, v in self.residual_lu.items()},
            "notes": list(self.notes),
            "anchor_role": self.anchor_role,
            "anchor_warning": self.anchor_warning,
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


@dataclass
class AnchorActivityGate:
    """Gated-детекция активности якоря для guard-слоя.

    Активен = short-term LUFS якоря выше его bleed-пола ИЗ СНАПШОТА
    (+ запас) И подтверждение классификатора — непрерывно ≥ hold_s.
    При anchor_active == False guard спит и не корректирует ничего.
    """
    bleed_floor_snapshot_db: float
    margin_db: float = ANCHOR_BLEED_MARGIN_DB
    hold_s: float = ANCHOR_ACTIVE_HOLD_S
    confirmed_seconds: float = 0.0
    active: bool = False

    def step(
        self,
        short_term_lufs: float,
        classifier_confirmed: bool,
        dt_s: float,
    ) -> bool:
        above_bleed = short_term_lufs > (self.bleed_floor_snapshot_db + self.margin_db)
        if above_bleed and classifier_confirmed:
            self.confirmed_seconds += dt_s
        else:
            self.confirmed_seconds = 0.0
        self.active = self.confirmed_seconds >= self.hold_s
        return self.active


def make_guard_state(
    config: BalanceConfig,
    *,
    instrumental: bool = False,
) -> Optional["VocalGuardState"]:
    """Guard-состояние по конфигу: вокальный margin 2.5, инструментальный 1.5.

    None — guard выключен (guard_enabled=False, либо инструментальный сет
    с guard_reassign_on_instrumental=False). Guard читает и двигает
    config.guard_bus (шину/DCA), не отдельные микрофонные каналы.
    """
    if not config.guard_enabled:
        return None
    if instrumental and not config.guard_reassign_on_instrumental:
        return None
    margin = (INSTRUMENTAL_GUARD_MARGIN_LU if instrumental
              else config.guard_margin_LU)
    return VocalGuardState(margin_lu=margin)


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


def select_drift_reference(
    short_term_lufs_by_channel: Mapping[Any, "list[float]"],
    manual_override: Optional[Any] = None,
    *,
    silence_lufs: float = -60.0,
    min_samples: int = 4,
) -> Optional[Any]:
    """Выбор drift-референса на саундчеке: канал с минимальной дисперсией
    short-term LUFS по активным сегментам (argmin σ) — обычно kick.
    manual_override из конфига имеет приоритет."""
    if manual_override is not None:
        return manual_override
    best_ch, best_sigma = None, None
    for ch, series in short_term_lufs_by_channel.items():
        active = [float(v) for v in series
                  if np.isfinite(v) and float(v) > silence_lufs]
        if len(active) < min_samples:
            continue
        sigma = float(np.std(active))
        if best_sigma is None or sigma < best_sigma:
            best_ch, best_sigma = ch, sigma
    return best_ch


def drift_precheck(
    delta_ref_lu: float,
    delta_lu_by_channel: Mapping[Any, float],
    *,
    ref_stable_lu: float = DRIFT_REF_STABLE_LU,
    moved_threshold_lu: float = DRIFT_DEADBAND_LU,
    correlated_fraction: float = MUSICAL_DROP_FRACTION,
) -> str:
    """Дешёвая предварительная проверка ПЕРЕД полной корреляционной матрицей.

    Возвращает:
      "correct_individual_drift" — drift_reference стабилен (|Δ| < 1 LU),
        а отдельные каналы ушли → это их дрейф, корректировать их;
      "freeze_musical" — референс и большинство каналов ушли коррелированно
        (одного знака) → музыкальная динамика/смена секции → freeze;
      "full_check" — картина неоднозначна → полная корреляционная проверка
        (musical_dynamics_detected) как подтверждающий этап.
    """
    others = {ch: float(d) for ch, d in delta_lu_by_channel.items()}
    if abs(float(delta_ref_lu)) < ref_stable_lu:
        moved = [d for d in others.values() if abs(d) > moved_threshold_lu]
        return "correct_individual_drift" if moved else "full_check"
    ref_sign = 1.0 if delta_ref_lu > 0 else -1.0
    same_sign_moved = [
        d for d in others.values()
        if abs(d) > moved_threshold_lu and (1.0 if d > 0 else -1.0) == ref_sign
    ]
    if others and len(same_sign_moved) >= correlated_fraction * len(others):
        return "freeze_musical"
    return "full_check"


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
