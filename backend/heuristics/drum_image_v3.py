"""Drum image v3 — операторская схема панорамы и фазы ударных (2026-07-06).

Панорама:
  kick / snare T / snare B / bass / lead vox — центр;
  стерео-пары L/R — жёстко влево/вправо, КРОМЕ оверхедов;
  OH L/R — панорамируются по РАВНОЙ громкости снейра в каждом
  (снейр в картинке OH встаёт в центр за счёт панорамы);
  томы/хэт/райд — в ту точку панорамы, где они звучат в
  ПАНОРАМИРОВАННЫХ оверхедах (образ близкого микрофона подкрепляет
  образ OH, а не расщепляет его).

Фаза/задержка (порядок жёсткий, опора — оверхеды):
  1. OH L ↔ OH R по снейру в них: задержка на ОДИН оверхед,
     второй остаётся 0 мс и становится опорой (oh_ref);
  2. oh_ref ↔ Snare T: близкий микрофон задерживается на время
     полёта до оверхеда;
  3. Snare T ↔ Snare B: полярность (низ обычно перевёрнут) +
     микрозадержка (физический потолок пары одного барабана — 2 мс);
  4. Томы к oh_ref по СНЕЙРУ, звучащему в них (bleed — сигнал
     выравнивания: гасится снейр-гребёнка суммы);
  5. Хэт/райд — так же, по снейру, относительно оверхедов.

Все замеры onset-gated по изолированным ударам снейра (isolated_hits,
Clifford & Reiss) с MAD/IQR-гейтами estimate_pair_delay; задержки —
АБСОЛЮТНЫЕ цели (pre-trim USB tap не слышит применённых задержек,
инкремент от текущего = храповик).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from heuristics.phase_v3 import (
    PairDelayEstimate,
    detect_onsets,
    estimate_pair_delay,
    isolated_hits,
    polarity_decision,
)

# физический потолок внутренней задержки пары OH: при большем замере
# пара стоит криво/замер неоднозначен — не трогаем
OH_INTERNAL_DELAY_MAX_MS = 2.0
# близкий микрофон -> оверхед: до ~3.4 м полёта
CLOSE_TO_OH_DELAY_MAX_MS = 10.0
# пара одного барабана (snare T/B) — сантиметры
SAME_SOURCE_DELAY_MAX_MS = 2.0
# кэп учитываемого дисбаланса снейра в OH (дальше — не панорама, а брак)
ILD_CAP_DB = 12.0
HIT_WINDOW_MS = 30.0
MIN_HITS_FOR_ILD = 3


def hit_levels_db(sig: np.ndarray, hits: Sequence[int], sample_rate: int,
                  win_ms: float = HIT_WINDOW_MS) -> List[float]:
    """RMS (дБ) в окнах ударов."""
    win = max(int(win_ms / 1000.0 * sample_rate), 64)
    out = []
    sig = np.asarray(sig, dtype=np.float64)
    for t in hits:
        seg = sig[t:t + win]
        if seg.size < win // 2:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms > 1e-9:
            out.append(20.0 * math.log10(rms))
    return out


def pair_ild_db(sig_a: np.ndarray, sig_b: np.ndarray,
                hits: Sequence[int], sample_rate: int) -> Optional[float]:
    """Медианная разница уровней a−b (дБ) в окнах ударов; None если мало."""
    la = hit_levels_db(sig_a, hits, sample_rate)
    lb = hit_levels_db(sig_b, hits, sample_rate)
    n = min(len(la), len(lb))
    if n < MIN_HITS_FOR_ILD:
        return None
    diffs = np.asarray(la[:n]) - np.asarray(lb[:n])
    return float(np.median(diffs))


def equal_power_gains(pan: float) -> Tuple[float, float]:
    """Гейны (в L-шину, в R-шину) равномощностной панорамы, pan −100..100."""
    theta = (float(pan) + 100.0) / 200.0 * (math.pi / 2.0)
    return math.cos(theta), math.sin(theta)


def oh_pans_for_center(snare_ild_db: float) -> Tuple[float, float]:
    """Панорамы (OH L, OH R), выравнивающие громкость снейра в шинах.

    Более тихая по снейру сторона остаётся жёсткой; более громкий OH
    заводится внутрь ровно настолько, чтобы снейр встал в центр:
      cos(θ) − sin(θ) = 10^(−d/20), θ ∈ [0°, 45°] от жёсткого края.
    """
    d = float(np.clip(snare_ild_db, -ILD_CAP_DB, ILD_CAP_DB))
    ratio = 10.0 ** (-abs(d) / 20.0)
    # cosθ − sinθ = √2·cos(θ+45°) = ratio
    theta = math.acos(ratio / math.sqrt(2.0)) - math.pi / 4.0
    theta = max(0.0, min(math.pi / 4.0, theta))
    inward = math.degrees(theta) / 90.0 * 200.0  # 0..100 к центру
    if d >= 0:   # снейр громче в OH L — L внутрь
        return (-100.0 + inward, 100.0)
    return (-100.0, 100.0 - inward)


def image_position_pan(level_in_l_db: float, level_in_r_db: float,
                       pan_l: float, pan_r: float) -> float:
    """Позиция источника в ПАНОРАМИРОВАННОЙ паре OH -> пан близкого микрофона.

    Источник слышен в OH L на level_l и в OH R на level_r; после панорам
    OH его энергия в шинах L/R даёт угол равномощностной панорамы.
    """
    al = 10.0 ** (float(level_in_l_db) / 20.0)
    ar = 10.0 ** (float(level_in_r_db) / 20.0)
    gll, glr = equal_power_gains(pan_l)
    grl, grr = equal_power_gains(pan_r)
    bus_l = al * gll + ar * grl
    bus_r = al * glr + ar * grr
    theta = math.atan2(bus_r, bus_l)  # 0..π/2
    pan = theta / (math.pi / 2.0) * 200.0 - 100.0
    return float(np.clip(pan, -100.0, 100.0))


@dataclass
class DrumImagePlan:
    pans: Dict[int, float] = field(default_factory=dict)
    delays_ms: Dict[int, float] = field(default_factory=dict)
    polarity_flip: Dict[int, bool] = field(default_factory=dict)
    oh_ref: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pans": {int(k): round(v, 1) for k, v in self.pans.items()},
            "delays_ms": {int(k): round(v, 2)
                          for k, v in self.delays_ms.items()},
            "polarity_flip": {int(k): v
                              for k, v in self.polarity_flip.items()},
            "oh_ref": self.oh_ref,
            "notes": list(self.notes),
            "measurements": self.measurements,
        }


def plan_drum_image(
    buffers: Dict[int, np.ndarray],
    sample_rate: int,
    *,
    oh_pair: Optional[Tuple[int, int]] = None,   # (OH L, OH R)
    snare_t: Optional[int] = None,
    snare_b: Optional[int] = None,
    kick: Optional[int] = None,
    toms: Sequence[int] = (),
    cymbals: Sequence[int] = (),
) -> DrumImagePlan:
    """План схемы по сырым (pre-trim) буферам каналов."""
    plan = DrumImagePlan()

    def buf(ch: Optional[int]) -> Optional[np.ndarray]:
        if ch is None:
            return None
        x = buffers.get(ch)
        if x is None or len(x) < sample_rate:
            return None
        return np.asarray(x, dtype=np.float64).reshape(-1)

    snare = buf(snare_t)
    if snare is None:
        plan.notes.append("нет сигнала Snare T — схема требует снейр")
        return plan
    kick_buf = buf(kick)
    if kick_buf is not None:
        snare_hits = isolated_hits(snare, kick_buf, sample_rate)
    else:
        snare_hits = detect_onsets(snare, sample_rate)
    plan.measurements["snare_hits"] = len(snare_hits)
    if len(snare_hits) < MIN_HITS_FOR_ILD:
        plan.notes.append(
            f"мало изолированных ударов снейра ({len(snare_hits)}) — "
            "фаза/OH-панорама отложены")
        snare_hits = []

    # центр всегда: snare T/B и kick (bass/lead vox добавляет движок)
    for ch in (snare_t, snare_b, kick):
        if ch is not None:
            plan.pans[ch] = 0.0

    oh_l_buf = buf(oh_pair[0]) if oh_pair else None
    oh_r_buf = buf(oh_pair[1]) if oh_pair else None
    have_oh = oh_l_buf is not None and oh_r_buf is not None
    pan_l, pan_r = -100.0, 100.0
    oh_ref_buf = None

    if have_oh and snare_hits:
        # ── 1. OH L ↔ OH R по снейру: задержка на ОДИН, второй — опора ──
        est = estimate_pair_delay(oh_r_buf, oh_l_buf, sample_rate,
                                  snare_hits)
        plan.measurements["oh_internal"] = est.to_dict()
        if est.accepted and abs(est.delay_ms) > 0.1:
            if abs(est.delay_ms) <= OH_INTERNAL_DELAY_MAX_MS:
                if est.delay_ms > 0:      # OH R позже -> задержать OH L
                    plan.delays_ms[oh_pair[0]] = abs(est.delay_ms)
                    plan.oh_ref = oh_pair[1]
                else:                      # OH L позже -> задержать OH R
                    plan.delays_ms[oh_pair[1]] = abs(est.delay_ms)
                    plan.oh_ref = oh_pair[0]
            else:
                plan.notes.append(
                    f"внутренняя задержка OH {est.delay_ms:.2f} мс > "
                    f"{OH_INTERNAL_DELAY_MAX_MS} мс — пара стоит криво, "
                    "не трогаем")
        if plan.oh_ref is None:
            plan.oh_ref = oh_pair[0]
        oh_ref_buf = oh_l_buf if plan.oh_ref == oh_pair[0] else oh_r_buf

        # ── панорама OH по равной громкости снейра ──
        ild = pair_ild_db(oh_l_buf, oh_r_buf, snare_hits, sample_rate)
        plan.measurements["oh_snare_ild_db"] = (
            round(ild, 2) if ild is not None else None)
        if ild is not None:
            pan_l, pan_r = oh_pans_for_center(ild)
            plan.pans[oh_pair[0]] = pan_l
            plan.pans[oh_pair[1]] = pan_r

        # ── 2. oh_ref ↔ Snare T ──
        est2 = estimate_pair_delay(oh_ref_buf, snare, sample_rate,
                                   snare_hits)
        plan.measurements["snare_t_to_oh"] = est2.to_dict()
        snare_t_delay = 0.0
        if est2.accepted and 0.1 < est2.delay_ms <= CLOSE_TO_OH_DELAY_MAX_MS:
            snare_t_delay = float(est2.delay_ms)
            plan.delays_ms[snare_t] = snare_t_delay
        elif est2.accepted and est2.delay_ms < -0.1:
            plan.notes.append(
                f"OH раньше близкого снейра ({est2.delay_ms:.2f} мс) — "
                "физически неправдоподобно, шаг 2 пропущен")

        # ── 3. Snare T ↔ Snare B: правдоподобие лага, ПОТОМ полярность ──
        snare_b_buf = buf(snare_b)
        if snare_b_buf is not None and snare_hits:
            est3 = estimate_pair_delay(snare, snare_b_buf, sample_rate,
                                       snare_hits)
            plan.measurements["snare_t_to_b"] = est3.to_dict()
            if est3.accepted and abs(est3.delay_ms) > SAME_SOURCE_DELAY_MAX_MS:
                # Правило CLAUDE.md («9.90 мс» live 2026-07-06): лаг за
                # пределами физики пары сантиметровых микрофонов =
                # периодическая неоднозначность GCC-PHAT; знак r на таком
                # лаге — мусор. Пара пропускается ЦЕЛИКОМ, включая
                # полярность (ревью C3: флип уезжал ДО этой проверки).
                plan.notes.append(
                    f"T↔B {est3.delay_ms:.2f} мс > "
                    f"{SAME_SOURCE_DELAY_MAX_MS} мс — периодическая "
                    "неоднозначность GCC-PHAT, пара пропущена целиком "
                    "(включая полярность)")
            else:
                lag_for_pol = est3.delay_ms if est3.accepted else 0.0
                pol = polarity_decision(snare, snare_b_buf, sample_rate,
                                        lag_ms=lag_for_pol)
                plan.measurements["snare_b_polarity"] = pol
                if pol.get("flip") is True:
                    plan.polarity_flip[snare_b] = True
                if est3.accepted:
                    # est3>0: T позже B -> B догоняет задержкой; абсолютная цель
                    b_delay = snare_t_delay + float(est3.delay_ms)
                    if b_delay < 0:
                        plan.notes.append(
                            f"Snare B позже T на {-est3.delay_ms:.2f} мс — "
                            "опережение невозможно, задержка B оставлена 0")
                        b_delay = 0.0
                    if b_delay > 0.1:
                        plan.delays_ms[snare_b] = b_delay

        # ── 4/5. Томы и тарелки к oh_ref по СНЕЙРУ в них ──
        for ch in list(toms) + list(cymbals):
            x = buf(ch)
            if x is None:
                continue
            est_c = estimate_pair_delay(oh_ref_buf, x, sample_rate,
                                        snare_hits)
            plan.measurements[f"ch{ch}_to_oh"] = est_c.to_dict()
            if est_c.accepted and 0.1 < est_c.delay_ms <= CLOSE_TO_OH_DELAY_MAX_MS:
                plan.delays_ms[ch] = float(est_c.delay_ms)

            # панорама — в позицию образа СОБСТВЕННЫХ ударов в OH
            own_hits = isolated_hits(x, snare, sample_rate)
            if len(own_hits) >= MIN_HITS_FOR_ILD:
                ll = hit_levels_db(oh_l_buf, own_hits, sample_rate)
                lr = hit_levels_db(oh_r_buf, own_hits, sample_rate)
                n = min(len(ll), len(lr))
                if n >= MIN_HITS_FOR_ILD:
                    plan.pans[ch] = image_position_pan(
                        float(np.median(ll[:n])), float(np.median(lr[:n])),
                        pan_l, pan_r)
            else:
                plan.notes.append(
                    f"ch{ch}: нет собственных ударов — панорама не трогается "
                    "(bleed-правило)")
    elif not have_oh:
        plan.notes.append("нет пары оверхедов — фазовые шаги 1-5 пропущены")

    return plan
