"""Gain v3 — двухрежимный метод входного усиления (синтез FOH-документа).

SOUNDCHECK (gain-set, разовая калибровка):
  Класс-зависимые цели пиков (−18…−12 dBFS) = show margin (эффект шоу
  6–12 дБ, музыканты играют громче под адреналином) + true-peak allowance
  +3 дБ (межсэмповые пики, ITU-R BS.1770). Квантование к шагу консоли
  ВНИЗ (в сторону headroom). Pad — только при line-hot источнике или
  когда нужный gain ниже минимума преампа, и только ДО финального gain.
  SNR-верификация: шумовой пол сцены/зала через микрофоны (−50…−60 dBFS)
  доминирует над шумом преампа на 40+ дБ — «прожаривать» преамп ради SNR
  бессмысленно, консервативный запас почти ничего не стоит.

GAIN LOCK: gain замораживается снапшотом; граф зависимостей
  gain → {порог гейта, порог компрессора, мониторные посылы, метры}.
  Изменение gain после lock — либо каскад ре-деривации, либо
  trim-компенсация по закону trim += −Δgain (Gain Tracking/Compensation:
  DiGiCo/Yamaha/A&H). Trim стоит ДО динамики → сдвигает её эффективный
  порог на −Δtrim.

LIVE (только мониторинг + узкая аварийная полоса):
  Непрерывный AGC на входном усилении ЗАПРЕЩЁН. Авто-преамп и авто-pad
  на шоу запрещены (мониторные посылы post-gain pre-fader → фидбэк-петля
  мониторов, сдвиг порогов, доверие артиста). Разрешён только цифровой
  trim ±6 дБ, rate-limited (1 дБ/с; быстрая полоса ВНИЗ 3 дБ/с при
  устойчивом клипе — клиппинг хуже провала). Правило проекта: trim-boost
  в live только по явному opt-in оператора — детектор too-quiet даёт
  ТОЛЬКО алерт. shared_preamps=True (общий стейджрэк) → hard lock преампа.

Специфика WING: USB-tap снимается pre-trim (post-ADC) — клип-детектор
видит именно клиппинг АЦП, а применённый trim в захвате не виден
(открытый контур, правило CLAUDE.md). Proposal-only: модуль ничего не
пишет на пульт.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

EPS = 1e-12
TRUE_PEAK_ALLOWANCE_DB = 3.0
PAD_DB = 20.0
NOISE_FLOOR_OK_DBFS = -65.0
NOISE_FLOOR_FLAG_DBFS = -50.0     # выше — проблема кабеля/источника, не gain
TOO_LOW_PEAK_DBFS = -30.0         # пики активного канала ниже — gain мал

LIVE_TRIM_BOUND_DB = 6.0
TRIM_RATE_DB_S = 1.0
TRIM_RATE_FAST_DB_S = 3.0         # устойчивый клип: вниз быстрее

CLIP_FULLSCALE_RATIO = 0.995
CLIP_CONSECUTIVE_SAMPLES = 3
CREST_COLLAPSE_DB = 6.0
CREST_COLLAPSE_HOLD_S = 5.0
TOO_QUIET_MARGIN_DB = 12.0
TOO_QUIET_HOLD_S = 60.0
FLOOR_JUMP_DB = 10.0
DC_OFFSET_LIMIT = 0.02

MOMENTARY_BLOCK_S = 0.4
RMS_BLOCK_S = 0.3

# Таблица (a): класс-зависимые цели. target_peak_dbfs уже включает show
# margin и true-peak allowance; show_margin_db — приор «эффекта шоу»
# (фольклор, калибруется под группу).
CLASS_GAIN_PROFILE: Dict[str, Dict[str, Any]] = {
    "lead_vocal": {"target_peak_dbfs": -13.0, "show_margin_db": 10.0,
                   "expected_crest_db": 15.0, "line_hot": False},
    "backing_vocal": {"target_peak_dbfs": -13.0, "show_margin_db": 10.0,
                      "expected_crest_db": 14.0, "line_hot": False},
    "snare": {"target_peak_dbfs": -16.0, "show_margin_db": 8.0,
              "expected_crest_db": 21.0, "line_hot": False},
    "tom": {"target_peak_dbfs": -16.0, "show_margin_db": 8.0,
            "expected_crest_db": 21.0, "line_hot": False},
    "floor_tom": {"target_peak_dbfs": -16.0, "show_margin_db": 8.0,
                  "expected_crest_db": 21.0, "line_hot": False},
    "kick": {"target_peak_dbfs": -16.0, "show_margin_db": 8.0,
             "expected_crest_db": 21.0, "line_hot": False},
    "overheads": {"target_peak_dbfs": -16.0, "show_margin_db": 8.0,
                  "expected_crest_db": 17.0, "line_hot": False},
    "hihat": {"target_peak_dbfs": -16.0, "show_margin_db": 8.0,
              "expected_crest_db": 17.0, "line_hot": False},
    "bass": {"target_peak_dbfs": -13.0, "show_margin_db": 6.0,
             "expected_crest_db": 12.0, "line_hot": True},
    "keys": {"target_peak_dbfs": -13.0, "show_margin_db": 11.0,
             "expected_crest_db": 11.0, "line_hot": True},   # ±10 дБ между патчами
    "guitar": {"target_peak_dbfs": -13.0, "show_margin_db": 6.0,
               "expected_crest_db": 8.0, "line_hot": False},
    "acoustic_guitar": {"target_peak_dbfs": -13.0, "show_margin_db": 6.0,
                        "expected_crest_db": 14.0, "line_hot": False},
    "playback": {"target_peak_dbfs": -13.0, "show_margin_db": 6.0,
                 "expected_crest_db": 8.0, "line_hot": True},
    "line": {"target_peak_dbfs": -14.0, "show_margin_db": 8.0,
             "expected_crest_db": 12.0, "line_hot": False},
}


@dataclass
class ChannelGainStats:
    sample_peak_db: float
    true_peak_est_db: float       # sample peak + 3 дБ allowance
    rms_db: float
    crest_db: float
    loud_l5_db: float             # 5-й сверху перцентиль momentary (громкое место)
    noise_floor_db: float         # L95 тихих блоков
    signal_present: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(float(v), 2) if not isinstance(v, bool) else v)
                for k, v in self.__dict__.items()}


def channel_gain_stats(x: np.ndarray, sample_rate: int) -> ChannelGainStats:
    """Статистики уровня «громкого места» + шумовой пол канала."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    sample_peak_db = 20.0 * np.log10(peak + EPS)
    rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) + EPS)

    block = max(int(MOMENTARY_BLOCK_S * sample_rate), 512)
    levels = []
    for i in range(0, len(x) - block + 1, block):
        seg = x[i:i + block]
        levels.append(20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + EPS))
    levels_arr = np.asarray(levels or [rms_db])

    noise_floor = float(np.percentile(levels_arr, 5))
    active = levels_arr[levels_arr > noise_floor + 10.0]
    loud_l5 = float(np.percentile(active, 95)) if active.size else float(np.max(levels_arr))
    signal_present = bool(active.size >= 2)

    return ChannelGainStats(
        sample_peak_db=sample_peak_db,
        true_peak_est_db=sample_peak_db + TRUE_PEAK_ALLOWANCE_DB,
        rms_db=rms_db,
        crest_db=sample_peak_db - rms_db,
        loud_l5_db=loud_l5,
        noise_floor_db=noise_floor,
        signal_present=signal_present,
    )


def quantize_toward_headroom(gain_delta_db: float, step_db: float = 0.5) -> float:
    """Округление ВНИЗ к шагу консоли — всегда в сторону большего запаса."""
    if step_db <= 0:
        return float(gain_delta_db)
    return float(np.floor(gain_delta_db / step_db) * step_db)


@dataclass
class GainSetResult:
    channel: Any
    role: str
    gain_db: float                # абсолютный целевой gain
    gain_delta_db: float
    pad_engaged: bool
    target_peak_dbfs: float
    snr_flags: List[str] = field(default_factory=list)
    stats: Optional[ChannelGainStats] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel, "role": self.role,
            "gain_db": round(self.gain_db, 2),
            "gain_delta_db": round(self.gain_delta_db, 2),
            "pad_engaged": self.pad_engaged,
            "target_peak_dbfs": round(self.target_peak_dbfs, 2),
            "snr_flags": list(self.snr_flags),
            "stats": None if self.stats is None else self.stats.to_dict(),
            "method": "gain_v3_soundcheck",
        }


def derive_gain_set(
    stats: ChannelGainStats,
    role: str,
    *,
    channel: Any = None,
    current_gain_db: float = 0.0,
    console_step_db: float = 0.5,
    gain_min_db: float = -20.0,
    pad_engaged: bool = False,
) -> Optional[GainSetResult]:
    """Шаг 5 пайплайна: gain_delta = target_peak − measured_true_peak.

    None — сигнала нет (решение по каналу не принимается).
    """
    if not stats.signal_present:
        return None
    profile = CLASS_GAIN_PROFILE.get(role, CLASS_GAIN_PROFILE["line"])
    target_peak = float(profile["target_peak_dbfs"])
    gain_delta = target_peak - stats.true_peak_est_db

    pad = bool(pad_engaged)
    if not pad and (current_gain_db + gain_delta < gain_min_db
                    or bool(profile.get("line_hot"))
                    and stats.true_peak_est_db > -6.0):
        # pad ДО финального gain: только line-hot источник или нужный gain
        # ниже минимума преампа. НИКОГДА не авто-pad в live.
        pad = True
        gain_delta += PAD_DB

    gain_delta = quantize_toward_headroom(gain_delta, console_step_db)

    flags: List[str] = []
    post_floor = stats.noise_floor_db + gain_delta
    if post_floor > NOISE_FLOOR_FLAG_DBFS:
        flags.append("noise_floor_above_-50dBFS_check_cable_or_source_not_gain")
    elif post_floor > NOISE_FLOOR_OK_DBFS:
        flags.append("noise_floor_marginal")
    if stats.sample_peak_db < TOO_LOW_PEAK_DBFS and stats.signal_present:
        flags.append("peaks_below_-30dBFS_gain_too_low_before_set")

    return GainSetResult(
        channel=channel, role=role,
        gain_db=float(current_gain_db + gain_delta),
        gain_delta_db=float(gain_delta),
        pad_engaged=pad,
        target_peak_dbfs=target_peak,
        snr_flags=flags,
        stats=stats,
    )


# ── GAIN LOCK и граф зависимостей ────────────────────────────────

@dataclass
class GainLock:
    """Снапшот залоченного gain. Downstream-деривации (гейт/комп/EQ/баланс)
    запускаются ТОЛЬКО после lock; позднее изменение gain — каскад
    ре-деривации либо trim-компенсация (с логированием)."""
    channel: Any
    gain_db: float
    pad_engaged: bool
    stats: ChannelGainStats
    locked_at: float = field(default_factory=time.time)
    shared_preamps: bool = False   # общий стейджрэк -> преамп read-only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "gain_db": round(self.gain_db, 2),
            "pad_engaged": self.pad_engaged,
            "stats": self.stats.to_dict(),
            "locked_at": self.locked_at,
            "shared_preamps": self.shared_preamps,
        }


def dependency_shifts(gain_delta_db: float) -> Dict[str, float]:
    """Граф зависимостей gain → {gate, comp, мониторные посылы, метры}.

    +Δ gain = пороги гейта/компа эффективно НИЖЕ на Δ (нужна ре-деривация
    на +Δ), мониторные посылы и метры выше на Δ.
    """
    d = float(gain_delta_db)
    return {
        "gate_threshold_rederive_db": d,
        "comp_threshold_rederive_db": d,
        "monitor_send_level_shift_db": d,
        "meter_shift_db": d,
    }


def trim_compensation(gain_delta_db: float) -> Dict[str, float]:
    """Закон Gain Tracking/Compensation: trim += −Δgain.

    Пост-trim уровень остаётся постоянным (downstream и вторая консоль не
    видят изменения), но trim стоит ДО динамики → её эффективный порог
    сдвигается на −Δtrim = +Δgain; слой динамики уведомляется.
    """
    d = float(gain_delta_db)
    return {
        "trim_delta_db": -d,
        "dynamics_effective_threshold_shift_db": d,
    }


# ── LIVE-детекторы ───────────────────────────────────────────────

def detect_clip(x: np.ndarray, *,
                fullscale_ratio: float = CLIP_FULLSCALE_RATIO,
                consecutive: int = CLIP_CONSECUTIVE_SAMPLES) -> bool:
    """Клип АЦП: ≥3 последовательных full-scale сэмпла (USB-tap post-ADC)."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    over = x >= fullscale_ratio
    if not over.any():
        return False
    run = 0
    for flag in over:
        run = run + 1 if flag else 0
        if run >= consecutive:
            return True
    return False


def dc_offset_alert(x: np.ndarray, limit: float = DC_OFFSET_LIMIT) -> bool:
    return abs(float(np.mean(np.asarray(x, dtype=np.float64)))) > limit


@dataclass
class GainSupervisorConfig:
    auto_trim_enabled: bool = False    # дефолт: только лог + алерт человеку
    shared_preamps: bool = False       # hard lock преампа (общий стейджрэк)
    trim_bound_db: float = LIVE_TRIM_BOUND_DB
    trim_rate_db_s: float = TRIM_RATE_DB_S
    trim_rate_fast_db_s: float = TRIM_RATE_FAST_DB_S


@dataclass
class GainSupervisor:
    """LIVE-супервизор: детекторы аномалий + лестница реакции.

    Реакция ограничена по построению: log → alert → (opt-in) авто-TRIM
    только ВНИЗ при устойчивом клипе, в границах ±6 дБ. Преамп и pad не
    трогаются никогда; too-quiet — только алерт (trim-boost в live требует
    явного opt-in оператора — правило проекта).
    """
    snapshot: GainLock
    config: GainSupervisorConfig = field(default_factory=GainSupervisorConfig)
    trim_db: float = 0.0
    _crest_low_s: float = 0.0
    _quiet_s: float = 0.0
    _clip_s: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def step(
        self,
        x: np.ndarray,
        sample_rate: int,
        dt_s: float,
    ) -> List[Dict[str, Any]]:
        stats = channel_gain_stats(x, sample_rate)
        snap = self.snapshot.stats
        alerts: List[Dict[str, Any]] = []

        clipping = detect_clip(x)
        self._clip_s = self._clip_s + dt_s if clipping else 0.0
        if clipping:
            alerts.append({"name": "CLIP", "severity": "fast",
                           "suggestion": "снизить trim (авто при opt-in); проверить источник"})

        if stats.signal_present:
            if (snap.crest_db - stats.crest_db) > CREST_COLLAPSE_DB:
                self._crest_low_s += dt_s
                if self._crest_low_s >= CREST_COLLAPSE_HOLD_S:
                    alerts.append({"name": "CREST_COLLAPSE", "severity": "alert",
                                   "suggestion": "лимит/клип выше по тракту"})
            else:
                self._crest_low_s = 0.0

            if stats.sample_peak_db < (snap.loud_l5_db - TOO_QUIET_MARGIN_DB):
                self._quiet_s += dt_s
                if self._quiet_s >= TOO_QUIET_HOLD_S:
                    alerts.append({"name": "TOO_QUIET", "severity": "alert",
                                   "suggestion": "смена патча/обрыв? trim-boost только по opt-in оператора"})
            else:
                self._quiet_s = 0.0
        else:
            self._crest_low_s = 0.0

        if (stats.noise_floor_db - snap.noise_floor_db) > FLOOR_JUMP_DB:
            alerts.append({"name": "FLOOR_JUMP", "severity": "alert",
                           "suggestion": "кабель/фантом/RF-дропаут"})
        if dc_offset_alert(x):
            alerts.append({"name": "DC_OFFSET", "severity": "alert",
                           "suggestion": "фантом/кабель"})

        trim_step = 0.0
        if (self.config.auto_trim_enabled
                and clipping and self._clip_s >= 1.0):
            # быстрая полоса ВНИЗ: клиппинг хуже провала
            rate = self.config.trim_rate_fast_db_s
            trim_step = -min(rate * dt_s,
                             self.config.trim_bound_db + self.trim_db)
            new_trim = float(np.clip(self.trim_db + trim_step,
                                     -self.config.trim_bound_db,
                                     self.config.trim_bound_db))
            trim_step = new_trim - self.trim_db
            if trim_step != 0.0:
                self.trim_db = new_trim
                event = {"name": "AUTO_TRIM_DOWN", "delta_db": round(trim_step, 2),
                         "trim_db": round(self.trim_db, 2),
                         "revertable": True}
                self.events.append(event)
                alerts.append({**event, "severity": "action",
                               "suggestion": "авто-trim вниз (клип); revert доступен"})

        for a in alerts:
            if a.get("severity") != "action":
                self.events.append({"name": a["name"], "stats": stats.to_dict()})
        return alerts

    def revert_trim(self) -> float:
        """Откат всех авто-trim движений (кнопка revert)."""
        delta = -self.trim_db
        self.trim_db = 0.0
        self.events.append({"name": "TRIM_REVERTED", "delta_db": round(delta, 2)})
        return delta
