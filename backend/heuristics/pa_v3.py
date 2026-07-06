"""PA v3 — системный слой настройки PA измерительным микрофоном.

Работает в фазе SETUP (пустой зал, ДО канального саундчека) и владеет
выходными секциями WING (main/mtx): 8-полосный PEQ, нативный кроссовер
(leq/heq = LR/BW/Bessel до 48 дБ/окт), задержка до 500 мс, лимитер до
100:1, полярность — всё по OSC. Системный снапшот — источник истины;
выходные параметры НИКОГДА не попадают в песенные сцены. Гейт передачи:
канальные пайплайны стартуют только после блокировки системного слоя.

Измерительный движок (валидировано): H1 = Sxy/Sxx (Bendat & Piersol,
оптимальна при шуме на микрофоне), когерентность γ² = |Sxy|²/(Sxx·Syy)
как порог доверия (0.7 setup / 0.6 live-freeze); ESS-деконволюция ИХ
(Farina): h = IFFT(FFT(y)/FFT(x)); RT60 — Шрёдер (переиспользуется из
fx_v3); пространственное усреднение — взвешенное по когерентности, с
маской устойчивости (вариантность позиций < 3 дБ).

Целевые кривые — ремесленный консенсус, единственно верной НЕТ
(McCarthy: «±12 дБ от one-size-fits-all») → три пресета + настройка.
EQ: «тренда, а не ряби» — только когерентное и мультипозиционно
устойчивое; prefer-cut, кламп −6/+3 дБ, ≤8 фильтров.

LIVE — консервативно: из коробки только алерты (наклон, лимитер GR) и
SPL-надзор по юрисдикции; единственная live-EQ автоматизация — opt-in
ВЧ-полка компенсации поглощения публикой (+1 дБ шаги, ≤0.5 дБ/мин,
кап +3 дБ); γ² < 0.6 → полная заморозка + откат. Смена задержек/
полярности/кроссовера в лайве запрещена. Proposal-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from heuristics.eq_ab_check import bell_response_db  # noqa: E402

EPS = 1e-12

SETUP_COHERENCE_MIN = 0.7
LIVE_COHERENCE_FREEZE = 0.6
SPATIAL_CONSISTENCY_DB = 3.0
MAX_SYSTEM_EQ_FILTERS = 8
EQ_GAIN_CLAMP_DB = (-6.0, 3.0)
EQ_Q_RANGE = (0.7, 4.0)
SUBTOP_SCAN_MS = 15.0
SUBTOP_STEP_MS = 0.1
SUBTOP_BAND_HZ = (60.0, 140.0)
CROSSOVER_DIP_MAX_DB = 3.0
FILL_LEVEL_CEILING_VS_MAINS_DB = -6.0
FILL_PRECEDENCE_MAX_MS = 5.0
LIMITER_MARGIN_DEFAULT_DB = 3.0

HF_DRIFT_BAND_HZ = (4000.0, 10000.0)
HF_DRIFT_TRIGGER_DB = -2.0
HF_DRIFT_HOLD_S = 600.0            # 10 минут устойчиво
HF_SHELF_STEP_DB = 1.0
HF_SHELF_CAP_DB = 3.0
HF_SHELF_RATE_DB_PER_MIN = 0.5
SPL_PREDICTIVE_MARGIN_DB = 1.5

# Юридические SPL-лимиты (монография ВОЗ; юрисдикция выбирается конфигом)
SPL_LIMITS: Dict[str, Dict[str, Any]] = {
    "germany_din_15905_5": {"laeq_db": 99.0, "window_min": 30, "peak": "LCpeak<135"},
    "switzerland": {"laeq_db": 100.0, "window_min": 60, "peak": "LAF<125"},
    "netherlands_vnpf": {"laeq_db": 103.0, "window_min": 15, "peak": "LCpeak<140"},
    "france_2017": {"laeq_db": 102.0, "window_min": 15, "peak": "LCeq15<118"},
    "belgium_flanders": {"laeq_db": 100.0, "window_min": 60, "peak": None},
    "norway_2011": {"laeq_db": 99.0, "window_min": 30, "peak": "LCpeak<130"},
    "austria": {"laeq_db": 100.0, "window_min": 1, "peak": None},
}

# Целевые кривые — предложения отчёта (настраиваемые), НЕ «истина»
TARGET_CURVE_PRESETS = ("flat_tilt", "live_standard", "edm", "pink")

# Карта выходных OSC-путей WING. Подтверждена на реальном пульте
# 2026-07-05 (read-only опрос /main/1 и /mtx/1 + write/rollback leq:
# LR24/LR48 принимаются и откатываются к SHV).
# Доктрина: системные параметры исключаются из песенных сцен.
WING_OUTPUT_OSC = {
    "eq_on": "/{root}/{n}/eq/on",
    "eq_band_freq": "/{root}/{n}/eq/{band}f",
    "eq_band_gain": "/{root}/{n}/eq/{band}g",
    "eq_band_q": "/{root}/{n}/eq/{band}q",
    "eq_low_type": "/{root}/{n}/eq/leq",     # PEQ,SHV,CUT,BW..,LR.. (кроссовер!)
    "eq_high_type": "/{root}/{n}/eq/heq",
    "delay_on": "/{root}/{n}/dly/on",
    "delay_ms": "/{root}/{n}/dly/dly",       # до ~500 мс, шаг 0.1 мс
    "dyn_on": "/{root}/{n}/dyn/on",
    "dyn_threshold": "/{root}/{n}/dyn/thr",
    "dyn_ratio": "/{root}/{n}/dyn/ratio",    # до 100 = лимитер
    "polarity": "/{root}/{n}/in/set/inv",
    "fader": "/{root}/{n}/fdr",
    "verify_per_firmware": False,            # verified 2026-07-05, live console
}


# ── Измерительный движок ─────────────────────────────────────────

def transfer_function_h1(
    reference: np.ndarray,
    measurement: np.ndarray,
    sample_rate: int,
    *,
    nfft: int = 8192,
    overlap: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Двухканальная TF: H1 = Sxy/Sxx + когерентность γ².

    H1 оптимальна при шуме на выходе (микрофоне) — живые/шумные условия.
    """
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(measurement, dtype=np.float64)
    n = min(len(x), len(y))
    hop = max(int(nfft * (1.0 - overlap)), 1)
    window = np.hanning(nfft)
    sxx = np.zeros(nfft // 2 + 1)
    syy = np.zeros(nfft // 2 + 1)
    sxy = np.zeros(nfft // 2 + 1, dtype=np.complex128)
    frames = 0
    for start in range(0, n - nfft + 1, hop):
        xf = np.fft.rfft(window * x[start:start + nfft])
        yf = np.fft.rfft(window * y[start:start + nfft])
        sxx += (xf.conj() * xf).real
        syy += (yf.conj() * yf).real
        sxy += xf.conj() * yf
        frames += 1
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    if frames == 0:
        return {"freqs": freqs, "h1": np.zeros_like(sxy),
                "magnitude_db": np.full_like(sxx, -120.0),
                "coherence": np.zeros_like(sxx)}
    h1 = sxy / (sxx + EPS)
    coherence = (np.abs(sxy) ** 2) / (sxx * syy + EPS)
    return {
        "freqs": freqs,
        "h1": h1,
        "magnitude_db": 20.0 * np.log10(np.abs(h1) + EPS),
        "coherence": np.clip(coherence, 0.0, 1.0),
    }


def ess_sweep(
    f1_hz: float, f2_hz: float, duration_s: float, sample_rate: int,
) -> np.ndarray:
    """Экспоненциальный синус-свип (Farina)."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    ln_ratio = np.log(f2_hz / f1_hz)
    k = 2.0 * np.pi * f1_hz * duration_s / ln_ratio
    return np.sin(k * (np.exp(t * ln_ratio / duration_s) - 1.0))


def ess_deconvolve(
    sweep: np.ndarray, recording: np.ndarray, sample_rate: int,
) -> Dict[str, Any]:
    """ИХ деконволюцией h = IFFT(FFT(y)/FFT(x)); пик ИХ = задержка."""
    n = 1 << int(np.ceil(np.log2(len(sweep) + len(recording))))
    spec_x = np.fft.rfft(sweep, n)
    spec_y = np.fft.rfft(recording, n)
    ir = np.fft.irfft(spec_y / (spec_x + EPS))
    peak = int(np.argmax(np.abs(ir)))
    return {"ir": ir, "delay_s": peak / sample_rate, "peak_index": peak}


def load_cal_file(lines: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузка .frd/.txt калибровки: «частота дБ», */; — комментарии."""
    freqs: List[float] = []
    dbs: List[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped[0] in "*;":
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            freqs.append(float(parts[0]))
            dbs.append(float(parts[1]))
        except ValueError:
            continue
    return np.asarray(freqs), np.asarray(dbs)


def apply_cal(
    magnitude_db: np.ndarray,
    fft_freqs: np.ndarray,
    cal_freqs: np.ndarray,
    cal_db: np.ndarray,
) -> np.ndarray:
    """Линейная интерполяция калибровки к бинам FFT и вычитание."""
    if len(cal_freqs) == 0:
        return np.asarray(magnitude_db)
    correction = np.interp(fft_freqs, cal_freqs, cal_db)
    return np.asarray(magnitude_db) - correction


def spatial_average(
    magnitudes_db: Sequence[np.ndarray],
    coherences: Sequence[np.ndarray],
    *,
    consistency_db: float = SPATIAL_CONSISTENCY_DB,
) -> Dict[str, np.ndarray]:
    """Мультипозиционное усреднение, взвешенное по когерентности.

    consistent-маска: вариантность позиций < 3 дБ — EQ только там
    («тренда, а не ряби»; один микрофон ≠ истина зала).
    """
    mags = np.asarray(magnitudes_db, dtype=np.float64)
    cohs = np.asarray(coherences, dtype=np.float64)
    weights = cohs + EPS
    avg = np.sum(mags * weights, axis=0) / np.sum(weights, axis=0)
    consistent = np.std(mags, axis=0) < consistency_db
    return {"average_db": avg, "consistent": consistent,
            "mean_coherence": np.mean(cohs, axis=0)}


# ── Целевые кривые и системный EQ ────────────────────────────────

def target_curve_db(freqs_hz: np.ndarray, preset: str = "live_standard") -> np.ndarray:
    """Пресеты целевых кривых (предложения; единственно верной нет)."""
    f = np.maximum(np.asarray(freqs_hz, dtype=np.float64), 1.0)
    curve = np.zeros_like(f)
    if preset == "flat_tilt":
        above = f > 1000.0
        curve[above] = -1.0 * np.log2(f[above] / 1000.0)
    elif preset == "edm":
        curve[f < 90.0] = 9.0
        above = f > 1500.0
        curve[above] = -1.0 * np.log2(f[above] / 1500.0)
    elif preset == "pink":
        # розовый шум: равная энергия на октаву = −3 дБ/окт по бинам
        # (директива оператора 2026-07-06: балансировка общего микса
        # по розовому шуму)
        curve = -3.0 * np.log2(f / 1000.0)
    else:  # live_standard
        curve[f < 100.0] = 6.0
        above = f > 2000.0
        curve[above] = -1.5 * np.log2(f[above] / 2000.0)
    return curve


def solve_system_eq(
    freqs_hz: np.ndarray,
    averaged_db: np.ndarray,
    target_db: np.ndarray,
    coherence: np.ndarray,
    consistent: np.ndarray,
    *,
    max_filters: int = MAX_SYSTEM_EQ_FILTERS,
    coherence_min: float = SETUP_COHERENCE_MIN,
) -> Dict[str, Any]:
    """Жадный peak-fit системного EQ: prefer-cut, кламп −6/+3, Q 0.7–4.

    Ошибка обнуляется вне когерентных/устойчивых зон — гребёнка и
    позиционная рябь не EQ-ится.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    error = np.asarray(averaged_db) - np.asarray(target_db)
    reliable = (np.asarray(coherence) >= coherence_min) & np.asarray(consistent)
    error = np.where(reliable, error, 0.0)

    filters: List[Dict[str, float]] = []
    residual = error.copy()
    audible = (freqs >= 40.0) & (freqs <= 16000.0)
    selectable = audible.copy()
    for _ in range(max_filters):
        masked = np.abs(residual) * selectable
        if not masked.any():
            break
        idx = int(np.argmax(masked))
        peak_err = float(residual[idx])
        if abs(peak_err) < 1.5:      # меньше — не трогаем (экономия фильтров)
            break
        gain = float(np.clip(-peak_err, EQ_GAIN_CLAMP_DB[0], EQ_GAIN_CLAMP_DB[1]))
        q = float(np.clip(2.0 if peak_err > 0 else 1.0, *EQ_Q_RANGE))
        filt = {"freq_hz": float(freqs[idx]), "gain_db": round(gain, 2), "q": q}
        filters.append(filt)
        residual = residual + bell_response_db(freqs, filt["freq_hz"], gain, q)
        # минимальный разнос 1/3 октавы: жадный fit не должен класть два
        # фильтра на одну частоту, когда пик глубже клампа (live 2026-07-06:
        # полосы 1 и 4 легли на 56.8 Гц по −6 = −12 суммарно)
        spacing = 2.0 ** (1.0 / 3.0)
        selectable &= ~((freqs >= filt["freq_hz"] / spacing)
                        & (freqs <= filt["freq_hz"] * spacing))
    cuts = sum(1 for f in filters if f["gain_db"] < 0)
    return {
        "filters": filters,
        "residual_max_db": round(float(np.max(np.abs(residual * audible))), 2),
        "prefer_cut_ratio": round(cuts / len(filters), 2) if filters else 1.0,
        "skipped_unreliable_fraction": round(float(1.0 - np.mean(reliable)), 3),
    }


# ── Стыковка саб/топ и филлы ─────────────────────────────────────

def align_sub_top(
    sub: np.ndarray,
    top: np.ndarray,
    sample_rate: int,
    *,
    band_hz: Tuple[float, float] = SUBTOP_BAND_HZ,
    scan_ms: float = SUBTOP_SCAN_MS,
    step_ms: float = SUBTOP_STEP_MS,
) -> Dict[str, Any]:
    """Скан задержки ±15 мс (шаг 0.1 мс) × обе полярности: максимум
    энергии суммы в полосе кроссовера. Смена в LIVE запрещена."""
    sub = np.asarray(sub, dtype=np.float64)
    top = np.asarray(top, dtype=np.float64)
    n = min(len(sub), len(top))

    def band_energy(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
        m = (freqs >= band_hz[0]) & (freqs < band_hz[1])
        return float(spec[m].sum())

    solo_max = max(band_energy(sub[:n]), band_energy(top[:n]))
    best = {"score": -np.inf, "delay_ms": 0.0, "polarity": 1}
    for polarity in (1, -1):
        for delay_ms in np.arange(0.0, scan_ms + step_ms, step_ms):
            shift = int(round(delay_ms / 1000.0 * sample_rate))
            delayed = np.roll(sub, shift)[:n]
            score = band_energy(polarity * delayed + top[:n])
            if score > best["score"]:
                best = {"score": score, "delay_ms": round(float(delay_ms), 1),
                        "polarity": polarity}
    summation_db = 10.0 * np.log10(best["score"] / (solo_max + EPS))
    return {
        "delay_ms": best["delay_ms"],
        "polarity_invert": best["polarity"] < 0,
        "summation_gain_db": round(summation_db, 2),
        "verified": bool(summation_db > -CROSSOVER_DIP_MAX_DB),
        "live_changeable": False,
    }


def solve_fill(
    mains_arrival_s: float,
    fill_arrival_s: float,
    *,
    precedence_offset_ms: float = 0.0,
    mains_level_db: float = 0.0,
) -> Dict[str, Any]:
    """Задержка филла = Δt прихода (+прецедентность ≤5 мс); уровень —
    потолок мейны −6 дБ на шве."""
    delta_ms = max(0.0, (mains_arrival_s - fill_arrival_s) * 1000.0)
    offset = float(np.clip(precedence_offset_ms, 0.0, FILL_PRECEDENCE_MAX_MS))
    return {
        "delay_ms": round(delta_ms + offset, 1),
        "level_ceiling_db": mains_level_db + FILL_LEVEL_CEILING_VS_MAINS_DB,
        "hf_target_improvement_db": (2.0, 4.0),
    }


def limiter_threshold_policy(
    observed_show_max_dbfs: float,
    *,
    margin_db: float = LIMITER_MARGIN_DEFAULT_DB,
) -> Dict[str, Any]:
    """Порог выходного лимитера: наблюдаемый максимум − запас.

    Лимитер ≠ SPL-лимит: защита оборудования и регуляторный лимит —
    разные механизмы (регуляторный — в SPLSupervisor)."""
    return {
        "threshold_db": round(float(observed_show_max_dbfs) - margin_db, 1),
        "ratio": 100.0,               # = лимитер
        "detector": "PEAK",
        "note": "equipment_protection_not_regulatory_spl_limit",
    }


# ── Системный снапшот и гейт передачи ────────────────────────────

@dataclass
class SystemSnapshot:
    """Источник истины системного слоя. Выходные параметры исключаются
    из песенных сцен; канальные пайплайны стартуют только после lock."""
    eq_filters_by_output: Dict[str, List[Dict[str, float]]]
    sub_top_alignment: Dict[str, Any]
    fill_delays: Dict[str, Dict[str, Any]]
    limiter_by_output: Dict[str, Dict[str, Any]]
    target_curve: str
    rt60_s: Optional[float] = None
    spl_calibrated: bool = False
    locked: bool = False
    hf_reference_db: Optional[float] = None   # снимок 4–10 кГц для drift lane

    def lock(self) -> "SystemSnapshot":
        self.locked = True
        return self

    def channel_pipelines_may_start(self) -> bool:
        return self.locked


# ── LIVE: полосы надзора ─────────────────────────────────────────

@dataclass
class HFDriftLane:
    """Единственная live-EQ автоматизация системы (opt-in): компенсация
    ВЧ-поглощения публикой полкой +1 дБ шагами, ≤0.5 дБ/мин, кап +3 дБ.
    γ² < 0.6 → freeze. Всё логируется и откатываемо."""
    opt_in: bool = False
    applied_db: float = 0.0
    droop_seconds: float = 0.0
    last_step_minutes_ago: float = 999.0
    frozen: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)

    def step(
        self,
        hf_now_db: float,
        hf_snapshot_db: float,
        mean_coherence: float,
        dt_s: float,
    ) -> Optional[Dict[str, Any]]:
        if mean_coherence < LIVE_COHERENCE_FREEZE:
            self.frozen = True
            self.events.append({"name": "SYSTEM_AUTOMATION_FROZEN",
                                "coherence": round(mean_coherence, 2)})
            return {"name": "SYSTEM_AUTOMATION_FROZEN"}
        self.frozen = False
        self.last_step_minutes_ago += dt_s / 60.0
        droop = hf_now_db - hf_snapshot_db
        if droop <= HF_DRIFT_TRIGGER_DB:
            self.droop_seconds += dt_s
        else:
            self.droop_seconds = 0.0
            return None
        if (self.opt_in
                and self.droop_seconds >= HF_DRIFT_HOLD_S
                and self.applied_db < HF_SHELF_CAP_DB
                and self.last_step_minutes_ago >= HF_SHELF_STEP_DB / HF_SHELF_RATE_DB_PER_MIN):
            step_db = min(HF_SHELF_STEP_DB, HF_SHELF_CAP_DB - self.applied_db)
            self.applied_db += step_db
            self.last_step_minutes_ago = 0.0
            event = {"name": "HF_SHELF_STEP", "step_db": step_db,
                     "applied_db": self.applied_db, "revertable": True}
            self.events.append(event)
            return event
        if self.droop_seconds >= HF_DRIFT_HOLD_S and not self.opt_in:
            return {"name": "HF_DROOP_ALERT", "droop_db": round(droop, 1),
                    "suggestion": "включить HF-полку (opt-in) или ручной shelf"}
        return None

    def rollback(self) -> float:
        delta = -self.applied_db
        self.applied_db = 0.0
        self.events.append({"name": "HF_SHELF_ROLLED_BACK"})
        return delta


@dataclass
class SPLSupervisor:
    """Надзор LAeq по юрисдикции: предиктивный алерт при лимит−1.5 дБ с
    восходящим трендом; мастер-трим −1…−2 дБ ТОЛЬКО при armed."""
    jurisdiction: str = "germany_din_15905_5"
    calibrated: bool = False
    _history: List[Tuple[float, float]] = field(default_factory=list)

    def step(
        self,
        t_s: float,
        laeq_db: float,
        *,
        operator_armed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        limit = SPL_LIMITS[self.jurisdiction]["laeq_db"]
        self._history.append((t_s, float(laeq_db)))
        self._history = self._history[-60:]
        if not self.calibrated:
            note = "UNCALIBRATED_RELATIVE_ONLY"
        else:
            note = None
        trend_up = (len(self._history) >= 3
                    and self._history[-1][1] > self._history[-3][1])
        if laeq_db >= limit:
            return {"name": "SPL_LIMIT_EXCEEDED", "laeq_db": laeq_db,
                    "limit_db": limit, "note": note,
                    "action": {"master_trim_db": -2.0} if operator_armed else None}
        if laeq_db >= limit - SPL_PREDICTIVE_MARGIN_DB and trend_up:
            return {"name": "SPL_PREDICTIVE_ALERT", "laeq_db": laeq_db,
                    "limit_db": limit, "note": note,
                    "suggestion": "мастер-трим −1…−2 дБ",
                    "action": {"master_trim_db": -1.0} if operator_armed else None}
        return None


def limiter_gr_alert(gr_db: float, sustained_s: float) -> Optional[Dict[str, Any]]:
    """GR лимитера устойчиво > 3 дБ → «система в лимите, снизьте мастер»."""
    if gr_db > 3.0 and sustained_s >= 5.0:
        return {"name": "SYSTEM_IN_LIMIT",
                "gr_db": round(gr_db, 1),
                "suggestion": "снизить мастер — система в лимите"}
    return None


def bass_excess_indicator(lceq_db: float, laeq_db: float) -> Dict[str, Any]:
    """LCeq−LAeq: индикатор избытка баса (типично 8–15 дБ)."""
    diff = float(lceq_db) - float(laeq_db)
    return {"lceq_minus_laeq_db": round(diff, 1),
            "excess": diff > 15.0}
