"""Pan v3 — venue-зависимое панорамирование (синтез FOH-документа).

Панорама в живом звуке ≠ студийная: стерео работает только для «малого
треугольника» аудитории (McCarthy: «stereo effects don't scale…»), вне
оси precedence притягивает образ к ближнему стеку. Надёжно полезны
только четыре функции: центрирование критичных для ВСЕЙ аудитории,
L/R-симметрия спектра и нагрузки сторон PA, умеренная ширина для
свит-спота, разведение спектрально-похожих для центра зала.

SOUNDCHECK: relevance-gate (mono PA → всё в центр, модуль bypass);
якоря — критичные роли и НЧ-доминантные (>60% энергии < 150 Гц) в
центр; бюджет ширины по классу зала (клуб 50%, театр 70%, арена 30%,
фестиваль 25%); симметричные слоты ±15/±30/±45… с разведением
спектрально-похожих на противоположные стороны; стереопары — только
параметр ширины пары, НЕ индивидуальные паны; верификация: L/R
третьоктавная симметрия ≤1.5 дБ (идеал ±0.8 — Pestana, 928 хитов),
mono-fold корреляция ≥0.5, падение моно-суммы ≤1 дБ, headroom сторон
≤2 дБ.

LIVE: паны ЗАМОРОЖЕНЫ в снапшоте. Opt-in полосы: solo-spotlight
(к центру ≤20%, ≤5%/с), guard L/R-дрейфа (>2.5 дБ 60 с → alert,
armed-apply ≤10%); суммарный дрейф ≤15%, rollback. Прыжки образа
слышнее статически-неоптимальной панорамы.

Сознательное проектное решение (противоречие в литературе): спектр-
центроид НЕ используется как драйвер ширины (Pestana: Assumption 10
«выше частота → шире» = False), только для РАЗВЕДЕНИЯ похожих.
Pan law: constant-power −3 дБ. Proposal-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12

WIDTH_CAP: Dict[str, float] = {
    "small_club": 0.40, "club": 0.50, "theater": 0.70,
    "arena": 0.30, "festival": 0.25,
}
LF_CENTER_THRESHOLD_HZ = 150.0
LF_CENTER_ENERGY_FRACTION = 0.60
SLOT_STEP = 0.15
LR_SYMMETRY_TOL_DB = 1.5
MONO_FOLD_CORRELATION_MIN = 0.5
MONO_SUM_DROP_MAX_DB = 1.0
BUS_HEADROOM_TOL_DB = 2.0

LIVE_RATE_PER_S = 0.05            # ≤5%/с
LIVE_TOTAL_DRIFT_MAX = 0.15       # ≤15% от снапшота
SPOTLIGHT_MAX_TOWARD_CENTER = 0.20
SYMMETRY_ALERT_DB = 2.5
SYMMETRY_ALERT_HOLD_S = 60.0
SYMMETRY_TRIM_MAX = 0.10

CRITICAL_ROLES = frozenset({
    "lead_vocal", "kick", "snare", "bass", "speech", "announce",
})
PAIR_WIDTH_FACTOR = 0.7           # ширина пар/томов = cap·0.7 (аудиторная)


@dataclass
class PanConfig:
    pa_type: str = "mono"          # mono | stereo | lcr; дефолт КОНСЕРВАТИВНЫЙ:
                                   # venue-профиль неизвестен → считаем моно
    venue_class: str = "club"
    width_cap_override: Optional[float] = None
    audience_perspective: bool = True
    pan_follow_to_sends: bool = False   # ПОКАНАЛЬНАЯ верификация на консоли!
    snare_offset: float = 0.0           # 0..0.10
    criticals_override: List[str] = field(default_factory=list)
    lcr_csr_default: float = 0.5
    subs_mono: bool = True
    sub_xover_hz: float = 90.0


def width_budget(config: PanConfig) -> float:
    if config.pa_type == "mono":
        return 0.0
    if config.width_cap_override is not None:
        return float(np.clip(config.width_cap_override, 0.0, 0.85))
    return WIDTH_CAP.get(config.venue_class, 0.40)


def pan_gains(pan_norm: float) -> Tuple[float, float]:
    """Constant-power −3 дБ: L=cos(θ), R=sin(θ); pan_norm ∈ [−1, +1]."""
    theta = (float(np.clip(pan_norm, -1.0, 1.0)) + 1.0) * (math.pi / 4.0)
    return math.cos(theta), math.sin(theta)


def lcr_gains(pan_norm: float, csr: float) -> Dict[str, float]:
    """LCR-раскладка через CSR (Center Side Ratio, Yamaha 0..1)."""
    g_l, g_r = pan_gains(pan_norm)
    csr = float(np.clip(csr, 0.0, 1.0))
    return {"L": g_l * (1.0 - csr), "C": csr * min(g_l, g_r) * 2.0,
            "R": g_r * (1.0 - csr)}


def lf_dominant(x: np.ndarray, sample_rate: int,
                *, threshold_hz: float = LF_CENTER_THRESHOLD_HZ,
                fraction: float = LF_CENTER_ENERGY_FRACTION) -> bool:
    """НЧ-доминантный источник (>60% энергии ниже 150 Гц) → центр.

    Валидировано: Pestana (<200 Гц центрируется универсально),
    Mansbridge/Perez-Gonzalez (порог 200 Гц); НЧ-локализация через
    PA+зал отсутствует, сабы моно.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 512:
        return False
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    total = float(spectrum.sum()) + EPS
    return float(spectrum[freqs < threshold_hz].sum() / total) > fraction


def spectral_centroid_hz(x: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 512:
        return 1000.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + EPS))


@dataclass
class PanSolution:
    pans: Dict[Any, float]                      # −1..+1, 0 = центр
    anchored: Dict[Any, str] = field(default_factory=dict)
    pair_widths: Dict[str, float] = field(default_factory=dict)
    pair_channels: "set" = field(default_factory=set)
    bypass: bool = False
    bypass_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pans": {str(k): round(v, 3) for k, v in self.pans.items()},
            "anchored": {str(k): v for k, v in self.anchored.items()},
            "pair_widths": dict(self.pair_widths),
            "bypass": self.bypass,
            "bypass_reason": self.bypass_reason,
            "notes": list(self.notes),
            "method": "pan_v3_soundcheck",
        }


def assign_pans(
    channels: Mapping[Any, Mapping[str, Any]],
    config: PanConfig,
    sample_rate: int = 48000,
) -> PanSolution:
    """Саундчек-назначение панорамы.

    channels: id -> {"role": str, "audio": np.ndarray (опц.),
                     "stereo_pair": str|None, "level_lufs": float (опц.)}
    """
    # (1) Гейт релевантности: mono PA — панорама полностью bypass
    if config.pa_type == "mono":
        return PanSolution(
            pans={ch: 0.0 for ch in channels},
            anchored={ch: "mono_pa_bypass" for ch in channels},
            bypass=True, bypass_reason="mono PA — panning bypassed",
        )

    cap = width_budget(config)
    solution = PanSolution(pans={})
    criticals = CRITICAL_ROLES | set(config.criticals_override)

    pairs: Dict[str, List[Any]] = {}
    movable: List[Tuple[Any, float, float]] = []  # (ch, centroid, level)
    for ch, payload in channels.items():
        role = str(payload.get("role", "line"))
        audio = payload.get("audio")
        pair = payload.get("stereo_pair")

        # (2) якоря: критичные роли и НЧ-доминантные — в центр
        if role in criticals:
            pan = 0.0
            if role == "snare" and config.snare_offset > 0.0:
                pan = float(np.clip(config.snare_offset, 0.0, 0.10))
            solution.pans[ch] = pan
            solution.anchored[ch] = "critical_role_center"
            continue
        if audio is not None and lf_dominant(np.asarray(audio), sample_rate):
            solution.pans[ch] = 0.0
            solution.anchored[ch] = "lf_dominant_center"
            continue
        if pair:
            pairs.setdefault(str(pair), []).append(ch)
            continue
        centroid = (spectral_centroid_hz(np.asarray(audio), sample_rate)
                    if audio is not None else 1000.0)
        movable.append((ch, centroid, float(payload.get("level_lufs", -25.0))))

    # (3) стереопары: только ширина пары, симметрично; индивидуальные
    # паны пары не назначаются никогда
    for pair_name, members in pairs.items():
        width = cap * PAIR_WIDTH_FACTOR
        solution.pair_widths[pair_name] = round(width, 3)
        members_sorted = sorted(members, key=str)
        for idx, ch in enumerate(members_sorted[:2]):
            sign = -1.0 if idx == 0 else 1.0
            if not config.audience_perspective:
                sign = -sign
            solution.pans[ch] = sign * width
            solution.pair_channels.add(ch)
        for extra in members_sorted[2:]:
            solution.pans[extra] = 0.0
            solution.anchored[extra] = "pair_overflow_center"

    # (4) симметричные слоты для подвижных; спектрально-похожие —
    # на противоположные стороны (сосед по центроиду = другая сторона)
    slots: List[float] = []
    k = 1
    while SLOT_STEP * k <= cap + 1e-9:
        slots += [SLOT_STEP * k, -SLOT_STEP * k]
        k += 1
    movable.sort(key=lambda item: item[1])   # по центроиду
    for idx, (ch, _centroid, _lvl) in enumerate(movable):
        solution.pans[ch] = slots[idx % len(slots)] if slots else 0.0

    # баланс энергии сторон: если суммарный уровень L и R разъехался,
    # зеркалим самый слабый слот (простая итеративная перестановка)
    solution.notes.append(f"width_cap={cap} ({config.venue_class})")
    _enforce_side_balance(solution, channels)
    return solution


def _enforce_side_balance(
    solution: PanSolution,
    channels: Mapping[Any, Mapping[str, Any]],
    tol_db: float = LR_SYMMETRY_TOL_DB,
    max_iterations: int = 8,
) -> None:
    for _ in range(max_iterations):
        left_p, right_p = 0.0, 0.0
        for ch, pan in solution.pans.items():
            if abs(pan) < 1e-9 or ch in solution.anchored:
                continue
            level = float(channels.get(ch, {}).get("level_lufs", -25.0))
            power = 10.0 ** (level / 10.0)
            g_l, g_r = pan_gains(pan)
            left_p += power * g_l ** 2
            right_p += power * g_r ** 2
        if left_p <= EPS or right_p <= EPS:
            return
        diff_db = 10.0 * np.log10(left_p / right_p)
        if abs(diff_db) <= tol_db:
            return
        heavy_sign = -1.0 if diff_db > 0 else 1.0   # какая сторона тяжелее
        candidates = [
            (ch, pan) for ch, pan in solution.pans.items()
            if np.sign(pan) == heavy_sign and ch not in solution.anchored
            and ch not in solution.pair_channels   # симметрия пар неприкосновенна
        ]
        if not candidates:
            return
        ch, pan = min(candidates, key=lambda item: abs(item[1]))
        solution.pans[ch] = -pan
        solution.notes.append(f"side balance: ch {ch} mirrored to {-pan:+.2f}")


# ── Верификация ──────────────────────────────────────────────────

def synthesize_lr(
    channels_audio: Mapping[Any, np.ndarray],
    pans: Mapping[Any, float],
) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(np.asarray(a)) for a in channels_audio.values())
    left = np.zeros(n)
    right = np.zeros(n)
    for ch, audio in channels_audio.items():
        g_l, g_r = pan_gains(float(pans.get(ch, 0.0)))
        a = np.asarray(audio, dtype=np.float64)[:n]
        left += g_l * a
        right += g_r * a
    return left, right


def lr_spectral_symmetry_db(
    left: np.ndarray, right: np.ndarray, sample_rate: int,
    band_hz: Tuple[float, float] = (250.0, 8000.0),
) -> float:
    """Максимальный |Δ| энергии L/R по третьоктавным полосам, дБ."""
    def bands(x: np.ndarray) -> List[float]:
        spec = np.abs(np.fft.rfft(np.asarray(x) * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
        out = []
        f = band_hz[0]
        while f < band_hz[1]:
            hi = f * 2 ** (1 / 3)
            m = (freqs >= f) & (freqs < hi)
            out.append(10.0 * np.log10(float(spec[m].sum()) + EPS))
            f = hi
        return out
    return float(max(abs(a - b) for a, b in zip(bands(left), bands(right))))


def mono_fold_metrics(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = (float(np.std(left)) * float(np.std(right))) or EPS
    corr = float(np.mean((left - left.mean()) * (right - right.mean())) / denom)
    mono = left + right
    stereo_db = 10.0 * np.log10(float(np.mean(left ** 2 + right ** 2)) + EPS)
    mono_db = 10.0 * np.log10(float(np.mean(mono ** 2)) / 2.0 + EPS)
    peak_l = 20.0 * np.log10(float(np.max(np.abs(left))) + EPS)
    peak_r = 20.0 * np.log10(float(np.max(np.abs(right))) + EPS)
    return {
        "correlation": corr,
        "mono_drop_db": stereo_db - mono_db,
        "headroom_diff_db": abs(peak_l - peak_r),
    }


def verify_pan_solution(
    channels_audio: Mapping[Any, np.ndarray],
    pans: Mapping[Any, float],
    sample_rate: int,
) -> Dict[str, Any]:
    left, right = synthesize_lr(channels_audio, pans)
    symmetry = lr_spectral_symmetry_db(left, right, sample_rate)
    fold = mono_fold_metrics(left, right)
    return {
        "lr_symmetry_db": round(symmetry, 2),
        "mono_fold_correlation": round(fold["correlation"], 3),
        "mono_drop_db": round(fold["mono_drop_db"], 2),
        "headroom_diff_db": round(fold["headroom_diff_db"], 2),
        "passed": bool(
            symmetry <= LR_SYMMETRY_TOL_DB
            and fold["correlation"] >= MONO_FOLD_CORRELATION_MIN
            and fold["mono_drop_db"] <= MONO_SUM_DROP_MAX_DB
            and fold["headroom_diff_db"] <= BUS_HEADROOM_TOL_DB
        ),
    }


# ── LIVE: freeze + охраняемые полосы ─────────────────────────────

@dataclass
class LivePanState:
    """LIVE: паны заморожены; разрешены только opt-in полосы с
    rate-limit ≤5%/с и суммарным дрейфом ≤15% от снапшота."""
    snapshot_pans: Dict[Any, float]
    trims: Dict[Any, float] = field(default_factory=dict)
    symmetry_seconds: float = 0.0
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def _apply_trim(self, ch: Any, desired: float, dt_s: float) -> float:
        current = self.trims.get(ch, 0.0)
        step = float(np.clip(desired - current,
                             -LIVE_RATE_PER_S * dt_s, LIVE_RATE_PER_S * dt_s))
        new = float(np.clip(current + step,
                            -LIVE_TOTAL_DRIFT_MAX, LIVE_TOTAL_DRIFT_MAX))
        self.trims[ch] = new
        return new

    def solo_spotlight_step(
        self, ch: Any, dt_s: float,
        *, solo_active: bool, confidence: float = 1.0,
        operator_armed: bool = False,
    ) -> float:
        """Solo-канал подтягивается к центру (≤20%), только armed + conf>0.9."""
        if not (solo_active and operator_armed and confidence > 0.9):
            return self._apply_trim(ch, 0.0, dt_s)   # плавный возврат
        snap = self.snapshot_pans.get(ch, 0.0)
        desired = float(np.clip(-snap, -SPOTLIGHT_MAX_TOWARD_CENTER,
                                SPOTLIGHT_MAX_TOWARD_CENTER))
        return self._apply_trim(ch, desired, dt_s)

    def symmetry_guard_step(
        self, lr_diff_db: float, dt_s: float,
        *, operator_armed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Дрейф L/R-симметрии: >2.5 дБ дольше 60 с → alert; armed →
        pan-trim ≤10% (rate-limited)."""
        if abs(lr_diff_db) > SYMMETRY_ALERT_DB:
            self.symmetry_seconds += dt_s
            if self.symmetry_seconds >= SYMMETRY_ALERT_HOLD_S:
                alert = {"name": "LR_SYMMETRY_DRIFT",
                         "diff_db": round(lr_diff_db, 2),
                         "suggestion": "pan-trim ≤10% или fader-side",
                         "auto_applied": bool(operator_armed)}
                self.alerts.append(alert)
                if operator_armed:
                    sign = -1.0 if lr_diff_db > 0 else 1.0
                    alert["trim"] = float(np.clip(
                        sign * SYMMETRY_TRIM_MAX,
                        -SYMMETRY_TRIM_MAX, SYMMETRY_TRIM_MAX))
                return alert
        else:
            self.symmetry_seconds = 0.0
        return None

    def rollback(self) -> Dict[Any, float]:
        """Полный откат live-тримов к снапшоту."""
        self.trims = {}
        self.alerts.append({"name": "PAN_TRIMS_ROLLED_BACK"})
        return dict(self.snapshot_pans)
