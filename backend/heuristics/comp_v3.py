"""Comp v3 — двухконтурный метод компрессии (синтез FOH-документа и проекта).

SOUNDCHECK-контур (агрессивный, однократный → снапшот):
  1. Профилирование динамики: crest (окна 200 мс), перцентили громкости
     L10/L50/L90 (блоки 400 мс), sigma_level, onset rate, доминирующая
     нижняя частота.
  2. Гейт «по умолчанию НЕ сжимать»: канал компрессируется только при
     sigma_level или crest выше порогов класса (философия Senior:
     «компрессируй по причине»; стабильному исполнителю компрессия не нужна).
  3. Параметры: ratio — кросс-адаптивная регрессия Ma 2015 с live-клампами;
     threshold из целевого GR на громких секциях: T = L90 − GR·R/(R−1);
     attack — приор класса; release — от темпа с защитой от НЧ-искажений
     (релиз > 1/f_low); колено W=|T|/2; makeup — bounded, headroom-aware.
  4. Верификация симуляцией ядра Giannoulis (drc_giannoulis): level-match
     (|ΔLUFS| ≤ 0.5 LU после makeup), метрика пампинга 2-8 Гц (при
     превышении релиз удлиняется шагами +20%), bleed-lift (подъём пола в
     тихих блоках не больше лимита — иначе GR снижается).

LIVE-контур (консервативный, поверх снапшота):
  Супервизор GR: медленный трекинг порога к цели (дрейф ≤ ±4 дБ), при
  GR > 2× цели > 5 с — поднимается ПОРОГ (ratio в лайве не меняется
  никогда); makeup rate-limited 0.5 дБ/с и замораживается при запасе до
  самовозбуждения < 6 дБ (X дБ GR + X дБ makeup = X дБ потери запаса в
  паузах); при низкой уверенности классификатора — safe-generic (2:1).

Proposal-only: модуль ничего не пишет на пульт.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from comp_params_ma2015 import (  # noqa: E402
    knee_width_ma2015,
    low_frequency_weighting_multitrack,
    percussivity_weighting_multitrack,
    ratio_ma2015,
)
from heuristics.drc_giannoulis import (  # noqa: E402
    lf_release_guard_ms,
    pumping_metric_db,
    simulate_compressor,
)

EPS = 1e-9
CREST_WIN_S = 0.2
LOUDNESS_BLOCK_S = 0.4
SILENCE_DB = -70.0

SIGMA_COMPRESS_THRESHOLD_DB = 3.5     # X ~ 3-4 дБ (калибруется)
CREST_EXCESS_THRESHOLD_DB = 2.5       # Y ~ 2-3 дБ (калибруется)
MAX_LOUDNESS_BIAS_LU = 0.5
PUMPING_LIMIT = 0.35                  # метрика 2-8 Гц (калибруется на зале)
BLEED_LIFT_LIMIT_DB = 3.0
RELEASE_STEP_FACTOR = 1.2
MAX_VERIFY_ITERATIONS = 4

RATIO_LIVE_MIN, RATIO_LIVE_MAX = 1.5, 4.0
RATIO_VOCAL_MAX = 6.0

# Целевой GR (дБ) и ожидаемый crest (дБ) по классам; клампы GR — safety
CLASS_PROFILE: Dict[str, Dict[str, float]] = {
    "lead_vocal": {"gr_target": 4.0, "expected_crest": 12.0, "gr_clamp": 8.0, "attack_ms": 8.0},
    "backing_vocal": {"gr_target": 4.0, "expected_crest": 12.0, "gr_clamp": 8.0, "attack_ms": 10.0},
    "bass": {"gr_target": 6.0, "expected_crest": 8.0, "gr_clamp": 10.0, "attack_ms": 10.0},
    "kick": {"gr_target": 4.0, "expected_crest": 20.0, "gr_clamp": 6.0, "attack_ms": 20.0},
    "snare": {"gr_target": 4.0, "expected_crest": 20.0, "gr_clamp": 6.0, "attack_ms": 10.0},
    "tom": {"gr_target": 4.0, "expected_crest": 20.0, "gr_clamp": 6.0, "attack_ms": 10.0},
    "floor_tom": {"gr_target": 4.0, "expected_crest": 20.0, "gr_clamp": 6.0, "attack_ms": 10.0},
    "overheads": {"gr_target": 2.0, "expected_crest": 18.0, "gr_clamp": 3.0, "attack_ms": 20.0},
    "hihat": {"gr_target": 2.0, "expected_crest": 18.0, "gr_clamp": 3.0, "attack_ms": 20.0},
    "guitar": {"gr_target": 3.0, "expected_crest": 12.0, "gr_clamp": 6.0, "attack_ms": 15.0},
    "acoustic_guitar": {"gr_target": 3.0, "expected_crest": 14.0, "gr_clamp": 6.0, "attack_ms": 20.0},
    "keys": {"gr_target": 2.0, "expected_crest": 12.0, "gr_clamp": 3.0, "attack_ms": 20.0},
    "strings": {"gr_target": 2.0, "expected_crest": 14.0, "gr_clamp": 3.0, "attack_ms": 20.0},
    "playback": {"gr_target": 2.0, "expected_crest": 10.0, "gr_clamp": 3.0, "attack_ms": 20.0},
    "line": {"gr_target": 3.0, "expected_crest": 12.0, "gr_clamp": 6.0, "attack_ms": 10.0},
}
SAFE_GENERIC = {"ratio": 2.0, "attack_ms": 10.0, "release_ms": 200.0,
                "gr_target": 3.0, "gr_clamp": 6.0}


@dataclass
class ChannelDynamicsStats:
    crest_median_db: float
    l10_db: float
    l50_db: float
    l90_db: float
    sigma_level_db: float
    onset_rate_hz: float
    f_low_hz: float
    lf_ratio: float
    rms_db: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(float(v), 2) for k, v in self.__dict__.items()}


def channel_dynamics_stats(x: np.ndarray, sample_rate: int) -> ChannelDynamicsStats:
    """Профиль динамики канала за окно саундчека."""
    x = np.asarray(x, dtype=np.float64)
    crest_win = max(int(CREST_WIN_S * sample_rate), 256)
    block = max(int(LOUDNESS_BLOCK_S * sample_rate), 512)

    crests: List[float] = []
    for i in range(0, len(x) - crest_win + 1, crest_win):
        seg = x[i:i + crest_win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        peak = float(np.max(np.abs(seg)))
        if rms > 1e-6:
            crests.append(20.0 * np.log10(peak / rms + EPS))
    crest_median = float(np.median(crests)) if crests else 0.0

    levels: List[float] = []
    for i in range(0, len(x) - block + 1, block):
        seg = x[i:i + block]
        levels.append(20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + EPS))
    levels_arr = np.asarray([v for v in levels if v > SILENCE_DB])
    if levels_arr.size == 0:
        levels_arr = np.asarray(levels or [SILENCE_DB])
    # Lxx в терминах шумомера: L10 — уровень, превышаемый 10% времени (громко)
    l10 = float(np.percentile(levels_arr, 90))
    l50 = float(np.percentile(levels_arr, 50))
    l90 = float(np.percentile(levels_arr, 10))
    sigma = float(np.std(levels_arr))

    # onset rate: скачки огибающей > 6 дБ
    hop = max(int(0.01 * sample_rate), 64)
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    env_db = 20.0 * np.log10(env + EPS)
    onsets = int(np.sum(np.diff(env_db) > 6.0))
    onset_rate = onsets / max(len(x) / sample_rate, EPS)

    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    total = float(spectrum.sum()) + EPS
    lf_ratio = float(spectrum[freqs < 1000.0].sum() / total)  # LPF 1 кГц (Ma 2015)
    # доминирующая нижняя частота: пик спектра в 20..300 Гц
    low_mask = (freqs >= 20.0) & (freqs <= 300.0)
    f_low = float(freqs[low_mask][np.argmax(spectrum[low_mask])]) if low_mask.any() else 40.0

    rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) + EPS)
    return ChannelDynamicsStats(
        crest_median_db=crest_median, l10_db=l10, l50_db=l50, l90_db=l90,
        sigma_level_db=sigma, onset_rate_hz=float(onset_rate),
        f_low_hz=f_low, lf_ratio=lf_ratio, rms_db=rms_db,
    )


def should_compress(stats: ChannelDynamicsStats, role: str) -> Dict[str, Any]:
    """Гейт «по умолчанию не сжимать»: стабильный канал остаётся нетронутым."""
    profile = CLASS_PROFILE.get(role, SAFE_GENERIC | {"expected_crest": 12.0})
    sigma_trigger = stats.sigma_level_db > SIGMA_COMPRESS_THRESHOLD_DB
    crest_trigger = stats.crest_median_db > (
        float(profile.get("expected_crest", 12.0)) + CREST_EXCESS_THRESHOLD_DB
    )
    return {
        "compress": bool(sigma_trigger or crest_trigger),
        "sigma_trigger": sigma_trigger,
        "crest_trigger": crest_trigger,
        "reason": (
            "sigma_level_above_threshold" if sigma_trigger
            else "crest_above_class_expectation" if crest_trigger
            else "stable_channel_bypass"
        ),
    }


@dataclass
class CompV3Params:
    threshold_db: float
    ratio: float
    knee_db: float
    attack_ms: float
    release_ms: float
    makeup_db: float
    gr_target_db: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(float(v), 2) for k, v in self.__dict__.items()}


@dataclass
class CompV3ChannelProposal:
    channel: Any
    role: str
    compress: bool
    decision: Dict[str, Any] = field(default_factory=dict)
    stats: Optional[ChannelDynamicsStats] = None
    params: Optional[CompV3Params] = None
    w_p: float = 0.0
    w_f: float = 0.0
    verification: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "role": self.role,
            "compress": self.compress,
            "decision": dict(self.decision),
            "stats": None if self.stats is None else self.stats.to_dict(),
            "params": None if self.params is None else self.params.to_dict(),
            "w_p": round(self.w_p, 3),
            "w_f": round(self.w_f, 3),
            "verification": dict(self.verification),
            "warnings": list(self.warnings),
            "method": "comp_v3_soundcheck",
        }


def derive_params(
    stats: ChannelDynamicsStats,
    role: str,
    w_p: float,
    w_f: float,
    *,
    tempo_bpm: float = 120.0,
    true_peak_dbtp: Optional[float] = None,
) -> CompV3Params:
    profile = CLASS_PROFILE.get(role, dict(SAFE_GENERIC))
    ratio_cap = RATIO_VOCAL_MAX if (
        role == "lead_vocal" and stats.sigma_level_db > 4.0
    ) else RATIO_LIVE_MAX
    ratio = float(np.clip(ratio_ma2015(w_p, w_f), RATIO_LIVE_MIN, ratio_cap))

    gr_target = float(profile.get("gr_target", 3.0))
    # T = L90(громкие секции) − GR·R/(R−1): целевой GR достигается на громком
    threshold = stats.l10_db - gr_target * ratio / max(ratio - 1.0, EPS)
    threshold = float(np.clip(threshold, -60.0, -4.0))

    attack_ms = float(profile.get("attack_ms", 10.0))
    release_ms = float(np.clip(60000.0 / max(tempo_bpm, 40.0) / 2.0, 50.0, 500.0))
    release_ms = lf_release_guard_ms(release_ms, stats.f_low_hz)

    knee_db = float(np.clip(knee_width_ma2015(threshold), 6.0, 12.0))

    # makeup: компенсировать ожидаемый средний GR (70%), bounded по headroom
    expected_gr = max((stats.l10_db - threshold) * (1.0 - 1.0 / ratio), 0.0)
    makeup = min(0.7 * expected_gr, 2.0)
    if true_peak_dbtp is not None:
        makeup = min(makeup, max(0.0, -1.0 - true_peak_dbtp))
    return CompV3Params(
        threshold_db=threshold, ratio=ratio, knee_db=knee_db,
        attack_ms=attack_ms, release_ms=release_ms,
        makeup_db=float(max(makeup, 0.0)), gr_target_db=gr_target,
    )


def _loudness_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(x) ** 2))) + EPS)


def verify_params(
    x: np.ndarray,
    sample_rate: int,
    params: CompV3Params,
    *,
    quiet_percentile: float = 20.0,
) -> Dict[str, Any]:
    """Верификация симуляцией: level-match, пампинг, bleed-lift.

    Мягкая деградация: пампинг лечится удлинением релиза (+20% шаги),
    bleed-lift — снижением makeup; неустранимое помечается в вердикте.
    """
    x = np.asarray(x, dtype=np.float64)
    current = params
    result: Dict[str, Any] = {"iterations": 0, "adjustments": []}
    for _ in range(MAX_VERIFY_ITERATIONS):
        result["iterations"] += 1
        sim = simulate_compressor(
            x, sample_rate,
            threshold_db=current.threshold_db, ratio=current.ratio,
            knee_db=current.knee_db, attack_ms=current.attack_ms,
            release_ms=current.release_ms, makeup_db=current.makeup_db,
        )
        loud_delta = _loudness_db(sim.output) - _loudness_db(x)
        pumping = pumping_metric_db(sim.gain_db_series, sim.env_rate_hz)

        # bleed-lift: подъём тихих блоков (пауз) после makeup
        block = max(int(LOUDNESS_BLOCK_S * sample_rate), 512)
        n_blocks = max(1, len(x) // block)
        in_lv = np.array([_loudness_db(x[i * block:(i + 1) * block]) for i in range(n_blocks)])
        out_lv = np.array([_loudness_db(sim.output[i * block:(i + 1) * block]) for i in range(n_blocks)])
        active = in_lv > SILENCE_DB
        if active.any():
            quiet_thr = np.percentile(in_lv[active], quiet_percentile)
            quiet = active & (in_lv <= quiet_thr)
            bleed_lift = float(np.mean(out_lv[quiet] - in_lv[quiet])) if quiet.any() else 0.0
        else:
            bleed_lift = 0.0

        result.update({
            "loudness_delta_lu": round(float(loud_delta), 2),
            "pumping_metric": round(float(pumping), 3),
            "bleed_lift_db": round(float(bleed_lift), 2),
            "mean_gr_db": round(sim.mean_gr_db, 2),
            "max_gr_db": round(sim.max_gr_db, 2),
        })

        if pumping > PUMPING_LIMIT:
            current = CompV3Params(**{**current.__dict__,
                                      "release_ms": current.release_ms * RELEASE_STEP_FACTOR})
            result["adjustments"].append(
                f"release +20% -> {current.release_ms:.0f} ms (pumping)")
            continue
        if bleed_lift > BLEED_LIFT_LIMIT_DB and current.makeup_db > 0.0:
            current = CompV3Params(**{**current.__dict__,
                                      "makeup_db": max(current.makeup_db - 0.5, 0.0)})
            result["adjustments"].append(
                f"makeup -0.5 -> {current.makeup_db:.1f} dB (bleed lift)")
            continue
        break

    # Loudness-дельта — НЕ блокирующий критерий: снижение громкости и есть
    # работа компрессора, а makeup в проекте bounded (правило CLAUDE.md).
    # Остаток уровня компенсируется фейдером (как в EQ v3): предложение
    # допускается вместе с fader_compensation_db, если она в безопасных
    # пределах; артефакты (пампинг, bleed-lift) — блокирующие.
    residual = float(result["loudness_delta_lu"])
    fader_comp = -residual if abs(residual) > MAX_LOUDNESS_BIAS_LU else 0.0
    result["fader_compensation_db"] = round(fader_comp, 2)
    result["passed"] = bool(
        result["pumping_metric"] <= PUMPING_LIMIT
        and result["bleed_lift_db"] <= BLEED_LIFT_LIMIT_DB
        and abs(fader_comp) <= 6.0
    )
    result["final_params"] = current.to_dict()
    return result | {"params_obj": current}


def soundcheck_comp_v3(
    channels: Mapping[Any, Mapping[str, Any]],
    sample_rate: int,
    *,
    tempo_bpm: float = 120.0,
) -> Dict[Any, CompV3ChannelProposal]:
    """Саундчек-контур Comp v3 (мультиканальный: w_p/w_f кросс-адаптивны)."""
    stats_by_ch: Dict[Any, ChannelDynamicsStats] = {}
    roles: Dict[Any, str] = {}
    for ch, payload in channels.items():
        stats_by_ch[ch] = channel_dynamics_stats(
            np.asarray(payload["audio"], dtype=np.float64), sample_rate)
        roles[ch] = str(payload.get("role", "line"))

    w_p_map = percussivity_weighting_multitrack(
        {ch: s.crest_median_db for ch, s in stats_by_ch.items()})
    w_f_map = low_frequency_weighting_multitrack(
        {ch: s.lf_ratio for ch, s in stats_by_ch.items()})

    proposals: Dict[Any, CompV3ChannelProposal] = {}
    for ch, stats in stats_by_ch.items():
        role = roles[ch]
        decision = should_compress(stats, role)
        proposal = CompV3ChannelProposal(
            channel=ch, role=role, compress=decision["compress"],
            decision=decision, stats=stats,
            w_p=float(w_p_map.get(ch, 1.0)), w_f=float(w_f_map.get(ch, 1.0)),
        )
        if decision["compress"]:
            params = derive_params(
                stats, role, proposal.w_p, proposal.w_f,
                tempo_bpm=tempo_bpm,
                true_peak_dbtp=float(channels[ch].get("true_peak_dbtp"))
                if channels[ch].get("true_peak_dbtp") is not None else None,
            )
            verification = verify_params(
                np.asarray(channels[ch]["audio"], dtype=np.float64),
                sample_rate, params)
            proposal.params = verification.pop("params_obj")
            proposal.verification = verification
            if not verification["passed"]:
                proposal.compress = False
                proposal.warnings.append(
                    "верификация не пройдена после мягкой деградации — bypass")
        proposals[ch] = proposal
    return proposals


# ── LIVE-контур: супервизор GR ───────────────────────────────────

@dataclass
class LiveGRSupervisorState:
    threshold_db: float
    threshold_snapshot_db: float
    gr_target_db: float
    makeup_db: float = 0.0
    makeup_target_db: float = 0.0
    over_gr_seconds: float = 0.0
    max_drift_db: float = 4.0
    track_tau_s: float = 15.0
    makeup_rate_db_per_s: float = 0.5
    feedback_margin_freeze_db: float = 6.0


def live_gr_supervise(
    state: LiveGRSupervisorState,
    gr_measured_db: float,
    dt_s: float,
    *,
    feedback_margin_db: float = 12.0,
    signal_active: bool = True,
) -> LiveGRSupervisorState:
    """Один шаг live-супервизора. Ratio НЕ меняется никогда — только порог.

    - Трекинг порога к целевому GR (медленно, дрейф ≤ ±4 дБ от снапшота);
    - GR > 2× цели дольше 5 с — порог поднимается активнее;
    - makeup rate-limited и замораживается при запасе до фидбэка < 6 дБ;
    - в тишине (signal_active=False) — freeze всего.
    """
    if not signal_active:
        return state

    err = float(gr_measured_db) - state.gr_target_db
    if abs(err) > 1.5:
        step = float(np.clip(0.5 * err, -0.5, 0.5)) * (dt_s / state.track_tau_s) * state.track_tau_s / 10.0
        state.threshold_db += step
    if gr_measured_db > 2.0 * state.gr_target_db:
        state.over_gr_seconds += dt_s
        if state.over_gr_seconds > 5.0:
            state.threshold_db += 0.5 * dt_s
    else:
        state.over_gr_seconds = 0.0
    state.threshold_db = float(np.clip(
        state.threshold_db,
        state.threshold_snapshot_db - state.max_drift_db,
        state.threshold_snapshot_db + state.max_drift_db,
    ))

    if feedback_margin_db >= state.feedback_margin_freeze_db:
        max_step = state.makeup_rate_db_per_s * dt_s
        state.makeup_db += float(np.clip(
            state.makeup_target_db - state.makeup_db, -max_step, max_step))
    # margin < 6 дБ: рост makeup заморожен (снижение всегда разрешено)
    elif state.makeup_target_db < state.makeup_db:
        max_step = state.makeup_rate_db_per_s * dt_s
        state.makeup_db += float(np.clip(
            state.makeup_target_db - state.makeup_db, -max_step, 0.0))
    return state


def safe_generic_params(l10_db: float) -> CompV3Params:
    """Безопасный дженерик при низкой уверенности классификатора (< 0.8)."""
    ratio = SAFE_GENERIC["ratio"]
    gr = SAFE_GENERIC["gr_target"]
    threshold = float(np.clip(l10_db - gr * ratio / (ratio - 1.0), -60.0, -4.0))
    return CompV3Params(
        threshold_db=threshold, ratio=ratio,
        knee_db=float(np.clip(knee_width_ma2015(threshold), 6.0, 12.0)),
        attack_ms=SAFE_GENERIC["attack_ms"], release_ms=SAFE_GENERIC["release_ms"],
        makeup_db=0.0, gr_target_db=gr,
    )
