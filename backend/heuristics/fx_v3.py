"""FX v3 — управление эффектами для лайва (синтез FOH-документа).

Два режима:
  САУНДЧЕК — полный вывод: профиль зала (грубый RT60 по Шрёдеру,
  ±0.2–0.4 с), clamp «RT зала съедает бюджет реверберации» (оптимум
  зала для pop/rock ≈ 0.6 с — Adelman-Larsen, JASA 2010/Springer 2021),
  темпо-синхро дилея и pre-delay, калибровка посылов к целевому
  возврат/сухой −10±3 дБ по КОНСОЛЬНЫМ метрам (не акустической оценке),
  ducking-конфиг, одна ре-проверка маскировки.
  ЛАЙВ — только «гигиена»: SAFE-полоса уровней посылов (≤1 дБ/с, ±3 дБ
  от снапшота), duck/mute возвратов по детекту речи (confidence>0.8,
  фейд 0.5–1 с, НЕ жёсткий мьют — мьютятся посылы, хвост доживает),
  ре-синхро темпа только в паузах ≥2 с; охрана: runaway-guard дилея,
  возвраты в scope feedback-детектора. Творческие throws/свипы —
  FORBIDDEN (исполнительское искусство, не автоматика).

WING-доктрины (правила проекта + документ): `/fx/N/on` не существует
(слот активен с моделью); имена моделей — официальные FW 3.1 (ST-DL);
fxmix держится 100% на send-return шине, управление — уровнем посыла;
смена `mdl` под аудио запрещена, пока глитч-тест на реальном пульте не
доказал безопасность (порог: клик > −60 dBFS → только в тишине);
sidechain «шина→шина» у WING нет — ducking = компрессор на возвратном
канале с external key от вокальной шины; единственная OSC-подписка —
монопольное владение (проект уже продлевает /xremote).

Классификация чисел: RT↔STI и оптимум 0.6 с — валидировано; все
RT/pre-delay/feedback/damping — craft-priors; clamp-таблица и пороги
полос — инженерные предложения, калибруются. Proposal-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12

VENUE_RT_OPTIMUM_S = 0.6          # pop/rock оптимум (Adelman-Larsen)
RT_CLAMP_S = (0.8, 3.0)
SEND_TARGET_RETURN_VS_DRY_DB = -10.0
SEND_TARGET_TOL_DB = 3.0
SEND_RATE_DB_S = 1.0
SEND_LIVE_CAP_DB = 3.0
SPEECH_CONFIDENCE_MIN = 0.8
UNMUTE_HYSTERESIS_S = 1.5
DUCK_DEPTH_DB = 8.0               # 6–12 дБ
DUCK_FADE_RATE_DB_S = 6.0         # фейд 0.5–1 с, не жёсткий мьют
DELAY_FEEDBACK_CAP_PCT = 70.0
TEMPO_DRIFT_RESYNC_PCT = 3.0
RESYNC_MIN_PAUSE_S = 2.0
MDL_GLITCH_THRESHOLD_DBFS = -60.0
STI_MIN = 0.50                    # IEC 60849 / EN 54-16
MASKING_RECHECK_LIMIT_DB = 1.0    # рост маскировки вокала 1–4 кГц с FX

# Live-изменяемость токенов WING (поверять по firmware)
FX_TOKEN_LIVE_POLICY: Dict[str, str] = {
    "mdl": "silence_only",
    "dcy": "silence_gated",
    "pdel": "silence_gated",
    "damp": "silence_gated",
    "time": "silence_only",
    "feed": "safe_with_cap",
    "flc": "static_from_soundcheck",
    "fhc": "static_from_soundcheck",
    "fxmix": "keep_100_percent",
    "send_level": "safe_rate_limited",
}

# Стартовые RT по жанру и роли (craft-priors)
GENRE_REVERB_RT_S: Dict[str, Dict[str, float]] = {
    "pop_rock": {"vocal": 1.6, "drums": 0.9},
    "ballad": {"vocal": 2.4, "drums": 1.2},
    "edm": {"vocal": 1.8, "drums": 0.6},
    "jazz_acoustic": {"vocal": 1.2, "drums": 0.8},
}
ROLE_REVERB_MODEL = {"vocal": "PLATE", "drums": "ROOM"}
ROLE_PREDELAY_MS = {"vocal": 30.0, "drums": 10.0}


# ── Профиль зала ─────────────────────────────────────────────────

def estimate_rt60_schroeder(
    decay_segment: np.ndarray,
    sample_rate: int,
) -> Dict[str, Any]:
    """Грубая оценка RT60 по обратному интегралу Шрёдера (T20-стиль).

    Вход — сегмент затухания («стоп-аккорд»/прерванный шум с
    измерительного микрофона). Точность ±0.2–0.4 с — маркируется
    'rough', не для сертификации (ISO 3382 — формальный метод).
    """
    x = np.asarray(decay_segment, dtype=np.float64)
    if len(x) < sample_rate // 10:
        return {"rt60_s": None, "quality": "insufficient_data"}
    energy = x ** 2
    sch = np.cumsum(energy[::-1])[::-1]
    sch_db = 10.0 * np.log10(sch / (np.max(sch) + EPS) + EPS)
    idx5 = int(np.argmax(sch_db <= -5.0))
    idx25 = int(np.argmax(sch_db <= -25.0))
    if idx25 <= idx5:
        return {"rt60_s": None, "quality": "no_clean_decay"}
    slope_db_per_s = (sch_db[idx25] - sch_db[idx5]) / ((idx25 - idx5) / sample_rate)
    if slope_db_per_s >= 0:
        return {"rt60_s": None, "quality": "no_clean_decay"}
    rt60 = float(-60.0 / slope_db_per_s)
    return {"rt60_s": round(rt60, 2), "quality": "rough_pm_0.2_0.4s"}


def sti_proxy(rt60_s: float, snr_db: float) -> float:
    """STI-прокси Houtgast–Steeneken: НЕ для формальной сертификации.

    STI = 0.15 + 0.85·(0.65·R_T + 0.35·R_S);
    R_T = 1/(1+(RT60/0.8)^1.6), R_S = 1/(1+10^(−(SNR−15)/10)).
    """
    r_t = 1.0 / (1.0 + (max(rt60_s, 0.05) / 0.8) ** 1.6)
    r_s = 1.0 / (1.0 + 10.0 ** (-(snr_db - 15.0) / 10.0))
    return float(0.15 + 0.85 * (0.65 * r_t + 0.35 * r_s))


def venue_class_from_rt(rt60_s: float) -> str:
    if rt60_s < 0.8:
        return "dry"
    if rt60_s <= 1.3:
        return "medium"
    return "reverberant"


def select_reverb_params(
    role: str,
    genre: str,
    venue_rt60_s: float,
    *,
    tempo_bpm: float = 120.0,
) -> Dict[str, Any]:
    """Выбор реверба: жанр × зал. RT зала съедает бюджет реверберации.

    rt = clamp(genre_base − 0.5·max(0, RT_зала − 0.6), 0.8, 3.0);
    в гулком зале — темнее (damping вниз). Pre-delay — темпо-фит
    (~1/64..1/32 ноты, кламп 15–40 мс).
    """
    base = GENRE_REVERB_RT_S.get(genre, GENRE_REVERB_RT_S["pop_rock"])
    genre_rt = float(base.get(role, base["vocal"]))
    excess = max(0.0, float(venue_rt60_s) - VENUE_RT_OPTIMUM_S)
    rt = float(np.clip(genre_rt - 0.5 * excess, *RT_CLAMP_S))

    vclass = venue_class_from_rt(venue_rt60_s)
    damp_hz = {"dry": 7000.0, "medium": 5500.0, "reverberant": 4500.0}[vclass]

    quarter_ms = 60000.0 / max(float(tempo_bpm), 40.0)
    predelay = float(np.clip(quarter_ms / 16.0, 15.0, 40.0))

    return {
        "mdl": ROLE_REVERB_MODEL.get(role, "PLATE"),
        "dcy": round(rt, 2),
        "pdel": round(predelay, 1),
        "damp": damp_hz,
        "lc": 200.0,
        "hc": 8000.0,
        "venue_class": vclass,
        "confidence": "craft_prior_clamped_by_validated_rt_budget",
    }


# ── Темпо-синхро ─────────────────────────────────────────────────

NOTE_MULTIPLIERS = {
    "1/4": 1.0, "1/8": 0.5, "1/8d": 0.75, "1/8t": 1.0 / 3.0,
    "1/16": 0.25, "1/16d": 0.375, "1/2": 2.0,
}


def tempo_delay_ms(bpm: float, note: str = "1/8d") -> Dict[str, float]:
    """Дилей и pre-delay от темпа. Дефолт — dotted-1/8 («The Edge»)."""
    quarter = 60000.0 / max(float(bpm), 40.0)
    mult = NOTE_MULTIPLIERS.get(note, 0.75)
    return {
        "delay_ms": round(quarter * mult, 1),
        "predelay_ms": round(float(np.clip(quarter / 16.0, 15.0, 40.0)), 1),
    }


# ── Калибровка посылов (по консольным метрам) ────────────────────

def calibrate_send_step(
    send_db: float,
    return_meter_db: float,
    dry_meter_db: float,
    *,
    target_db: float = SEND_TARGET_RETURN_VS_DRY_DB,
    tol_db: float = SEND_TARGET_TOL_DB,
    max_step_db: float = 1.0,
    send_bounds_db: Tuple[float, float] = (-40.0, 6.0),
) -> float:
    """Один шаг калибровки посыла к цели возврат/сухой (−10±3 дБ).

    Опирается на КОНСОЛЬНЫЕ метры возврата и сухой шины — акустическая
    оценка wet/dry с измерительного микрофона ненадёжна.
    """
    ratio = float(return_meter_db) - float(dry_meter_db)
    err = ratio - float(target_db)
    if abs(err) <= tol_db:
        return float(send_db)
    step = float(np.clip(-err, -max_step_db, max_step_db))
    return float(np.clip(send_db + step, *send_bounds_db))


def ducking_config(role: str = "vocal") -> Dict[str, Any]:
    """Ducking время-FX: у WING нет sidechain «шина→шина» — компрессор
    на ВОЗВРАТНОМ канале с external key от вокальной шины."""
    return {
        "implementation": "compressor_on_return_channel_keyed_from_vocal_bus",
        "depth_db": DUCK_DEPTH_DB,
        "release_ms": 300.0,
        "attack_ms": 5.0,
        "note": "wing_has_no_bus_to_bus_sidechain_ducker",
    }


def masking_recheck_adjustment(masking_delta_db: float) -> Dict[str, Any]:
    """Одна ре-проверка маскировки с FX (однонаправленный конвейер).

    Рост маскировки вокала в 1–4 кГц > 1 дБ → снизить посылы / поднять
    damping. Осциллирующего цикла FX↔маскировка нет по построению.
    """
    if masking_delta_db <= MASKING_RECHECK_LIMIT_DB:
        return {"action": "none", "delta_db": round(masking_delta_db, 2)}
    return {
        "action": "reduce_sends_or_darken",
        "send_reduction_db": round(min(masking_delta_db, 3.0), 1),
        "damp_shift_hz": -1000.0,
        "delta_db": round(masking_delta_db, 2),
    }


# ── LIVE: state-machine речи и охранные полосы ───────────────────

@dataclass
class FXSpeechState:
    """Duck/mute время-FX по детекту речи. Фейд, не жёсткий мьют:
    мьютятся ПОСЫЛЫ, хвост возврата доживает естественно."""
    snapshot_send_db: float
    current_db: float = None  # type: ignore[assignment]
    singing_hold_s: float = 0.0
    state: str = "music"

    def __post_init__(self):
        if self.current_db is None:
            self.current_db = self.snapshot_send_db

    def step(
        self,
        dt_s: float,
        *,
        speech_confidence: float = 0.0,
        singing_confidence: float = 0.0,
        silence_between_songs: bool = False,
    ) -> Dict[str, Any]:
        if silence_between_songs:
            target = -60.0
            self.state = "muted_between_songs"
            self.singing_hold_s = 0.0
        elif speech_confidence > SPEECH_CONFIDENCE_MIN:
            target = self.snapshot_send_db - DUCK_DEPTH_DB
            self.state = "speech_ducked"
            self.singing_hold_s = 0.0
        elif singing_confidence > SPEECH_CONFIDENCE_MIN:
            self.singing_hold_s += dt_s
            if self.singing_hold_s >= UNMUTE_HYSTERESIS_S:
                target = self.snapshot_send_db
                self.state = "music"
            else:
                target = self.current_db     # гистерезис: ждём подтверждения
        else:
            target = self.current_db          # неуверенность → держим
        step = float(np.clip(target - self.current_db,
                             -DUCK_FADE_RATE_DB_S * dt_s,
                             DUCK_FADE_RATE_DB_S * dt_s))
        self.current_db += step
        return {"send_db": round(self.current_db, 2), "state": self.state,
                "queued_changes_allowed": silence_between_songs}


def send_level_lane_step(
    current_send_db: float,
    desired_send_db: float,
    snapshot_send_db: float,
    dt_s: float,
) -> float:
    """SAFE-полоса уровня посыла: ≤1 дБ/с, кламп ±3 дБ от снапшота."""
    desired = float(np.clip(desired_send_db,
                            snapshot_send_db - SEND_LIVE_CAP_DB,
                            snapshot_send_db + SEND_LIVE_CAP_DB))
    step = float(np.clip(desired - current_send_db,
                         -SEND_RATE_DB_S * dt_s, SEND_RATE_DB_S * dt_s))
    return float(current_send_db + step)


@dataclass
class DelayRunawayGuard:
    """Runaway дилея / регенеративная петля FX↔PA: уровень возврата
    растёт при стабильном посыле → кэп feedback + alert."""
    window_s: float = 1.0
    rise_db_threshold: float = 3.0
    _history: List[Tuple[float, float]] = field(default_factory=list)

    def step(
        self,
        t_s: float,
        return_meter_db: float,
        send_stable: bool,
        current_feedback_pct: float,
    ) -> Optional[Dict[str, Any]]:
        self._history.append((t_s, float(return_meter_db)))
        self._history = [(t, v) for t, v in self._history
                         if t >= t_s - self.window_s]
        if len(self._history) < 3 or not send_stable:
            return None
        rise = self._history[-1][1] - self._history[0][1]
        if rise > self.rise_db_threshold:
            capped = min(float(current_feedback_pct) * 0.5,
                         DELAY_FEEDBACK_CAP_PCT)
            return {
                "name": "DELAY_RUNAWAY",
                "rise_db": round(rise, 1),
                "action": {"token": "feed", "value_pct": round(capped, 1)},
                "alert": "delay runaway / regenerative loop",
            }
        return None


def tempo_resync_decision(
    live_bpm: float,
    synced_bpm: float,
    *,
    beat_confidence: float,
    pause_duration_s: float,
) -> Dict[str, Any]:
    """Полоса дрейфа темпа: >3% при высокой уверенности → предложение;
    авто-применение ТОЛЬКО в паузе ≥2 с (или между песнями)."""
    if beat_confidence < 0.7:
        return {"action": "none",
                "reason": "beat_confidence_low_fallback_setlist_or_tap"}
    drift_pct = abs(live_bpm - synced_bpm) / max(synced_bpm, EPS) * 100.0
    if drift_pct <= TEMPO_DRIFT_RESYNC_PCT:
        return {"action": "none", "drift_pct": round(drift_pct, 1)}
    if pause_duration_s >= RESYNC_MIN_PAUSE_S:
        return {"action": "resync_now",
                "new_delay": tempo_delay_ms(live_bpm),
                "drift_pct": round(drift_pct, 1)}
    return {"action": "propose_resync_wait_for_pause",
            "drift_pct": round(drift_pct, 1)}


def mdl_change_allowed(
    *,
    silence_detected: bool,
    glitch_test_passed: bool = False,
) -> bool:
    """Смена типа эффекта (`mdl`): под аудио запрещена, пока глитч-тест
    на реальном WING не доказал клик < −60 dBFS; до этого — только в
    тишине. Creative throws — forbidden всегда (не автоматика)."""
    return bool(silence_detected or glitch_test_passed)
