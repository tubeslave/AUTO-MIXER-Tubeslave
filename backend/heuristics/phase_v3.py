"""Phase v3 — выравнивание полярности и задержек (синтез FOH-документа).

Триада приоритетов: (1) ПОЛЯРНОСТЬ — всегда, дёшево (одна кнопка, ноль
латентности), выигрыш единицы дБ НЧ на сумме кик/снейр; (2) грубые
задержки 2–10 мс внутри кластера; (3) субмиллиметровая подстройка — в
live убывающая отдача.

КЛАСТЕРЫ ВЫРАВНИВАНИЯ: выравниваются только механически фиксированные
группы (кит+OH, кабинетные пары, DI+микрофон). Референс — политика
farthest (школа оверхедов, дефолт) / closest (рок-панч) / manual.
Задержки только положительные: delay_i = t_ref − t_i ≥ 0, кэп per
cluster, квантование к шагу консоли.

ЖЁСТКИЕ ЗАПРЕТЫ (белый список «не выравнивать»):
  - стереопары НИКОГДА не выравниваются друг к другу (межканальная
    задержка AB = стереоинформация; «коррекция» схлопывает образ);
  - room-микрофоны (намеренная глубина, 9–30 мс — вне кэпа);
  - вокал и подвижные источники: 0 добавленной задержки (комфорт IEM
    ≤ 5 мс по Shure; собственный голос через кость + задержанный IEM
    = гребёнка).

РЕЖИМЫ: задержка/полярность — параметры ТОЛЬКО САУНДЧЕКА (изменение
задержки в эфире = флэнжинг/чирп/щелчки). LIVE = детекция дрейфа
(> 1.5 мс, 2 окна подряд) и полярностных флипов + оповещение;
применение — только в паузе-тишине с armed-подтверждением оператора.

Правила проекта: gcc_phat(target, reference) — порядок аргументов;
onset-gated окна; полярность/задержки применяются только при
MAD ≤ 0.5 мс и достаточном числе ударов. Proposal-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from auto_phase_gcc_phat import GCCPHATAnalyzer  # noqa: E402

SPEED_OF_SOUND_MS_PER_M = 2.915        # при 20 °C (c ≈ 343 м/с)
CONSOLE_DELAY_STEP_MS = 0.1            # подтверждено на WING 2026-07-05:
                                       # 1.23 -> /ch/N/in/set/dly -> 1.2
POLARITY_R_MIN = 0.4
COHERENCE_MIN = 0.6
MIN_HITS = 8
IQR_SANITY_MS = 1.0
MAD_APPLY_LIMIT_MS = 0.5               # правило CLAUDE.md
DRIFT_THRESHOLD_MS = 1.5
DRIFT_CONSECUTIVE_WINDOWS = 2
VOCAL_MAX_ADDED_DELAY_MS = 0.0         # вокал: 0 добавленной задержки
ISOLATION_GUARD_MS = 30.0
HIT_WINDOW_PRE_MS = 10.0
HIT_WINDOW_POST_MS = 80.0   # окно >= 4096 сэмплов для GCC-PHAT
SUM_VERIFY_BANDS_HZ = ((50.0, 120.0), (150.0, 300.0))
GEOMETRY_PLAUSIBLE_M = (0.2, 4.0)

# Роли, которые НИКОГДА не выравниваются (подвижные источники / глубина)
NEVER_ALIGN_ROLES = frozenset({
    "lead_vocal", "backing_vocal", "room", "playback",
})
# Роли-стереопары: выравнивание ВНУТРИ пары запрещено
STEREO_PAIR_INTERIOR_ROLES = frozenset({"overheads", "room"})
UNKNOWN_ROLES = frozenset({"unknown", "line", "custom", ""})

# Канонизация вариантов ролей классификатора в семейства (live 2026-07-06:
# oh_l/oh_r — РАЗНЫЕ строки, проверка role_a == role_b пропускала стереопару
# оверхедов). Snare top/bottom и kick in/out — многомикрофонные кластеры
# ОДНОГО источника: канонизируются и остаются выравниваемыми.
PHASE_ROLE_FAMILIES: Dict[str, str] = {
    "oh_l": "overheads", "oh_r": "overheads", "overhead": "overheads",
    "cymbals": "overheads", "drum_room": "room",
    "snare_top": "snare", "snare_bottom": "snare",
    "kick_in": "kick", "kick_out": "kick",
    "rack_tom": "tom", "floor_tom": "tom",
    "leadVocal": "lead_vocal", "backVocal": "backing_vocal",
    "vocal": "lead_vocal", "tracks": "playback",
}


def phase_role_family(role: str) -> str:
    role = str(role or "unknown")
    return PHASE_ROLE_FAMILIES.get(role, role)


@dataclass(frozen=True)
class AlignmentCluster:
    name: str
    member_roles: Tuple[str, ...]
    reference_policy: str = "farthest"   # farthest | closest | manual
    cap_ms: float = 10.0
    align: bool = True
    polarity_prior_flip_roles: Tuple[str, ...] = ()
    manual_reference_role: Optional[str] = None


DEFAULT_CLUSTERS: Tuple[AlignmentCluster, ...] = (
    AlignmentCluster(
        name="drums_close_oh",
        member_roles=("kick", "snare", "tom", "floor_tom", "overheads"),
        reference_policy="farthest", cap_ms=10.0,
        # снейр-bottom: приорный флип (диафрагма навстречу верхней)
        polarity_prior_flip_roles=("snare_bottom",),
    ),
    AlignmentCluster(
        name="guitar_cab",
        member_roles=("guitar",),
        reference_policy="farthest", cap_ms=1.0,
    ),
    AlignmentCluster(
        name="bass_di_mic",
        member_roles=("bass",),
        reference_policy="manual", manual_reference_role="bass_mic",
        cap_ms=3.0,
    ),
    AlignmentCluster(
        name="room_mics",
        member_roles=("room",),
        align=False, cap_ms=0.0,     # намеренно НЕ выравниваются
    ),
)


def pair_alignment_allowed(role_a: str, role_b: str) -> Tuple[bool, str]:
    """Жёсткий белый список: можно ли вообще выравнивать пару ролей."""
    fam_a, fam_b = phase_role_family(role_a), phase_role_family(role_b)
    if fam_a in NEVER_ALIGN_ROLES or fam_b in NEVER_ALIGN_ROLES:
        return False, "never_align_role_moving_source_or_room"
    if (fam_a == fam_b) and fam_a in STEREO_PAIR_INTERIOR_ROLES:
        return False, "stereo_pair_interior_delay_is_stereo_information"
    # Неизвестная роль = консервативный отказ: пара может оказаться
    # стереопарой (белый список, а не чёрный). Live-прогон 2026-07-06.
    if fam_a in UNKNOWN_ROLES or fam_b in UNKNOWN_ROLES:
        return False, "unknown_role_conservative_deny"
    return True, "ok"


# ── Онсеты и изолированные удары ─────────────────────────────────

def detect_onsets(
    x: np.ndarray, sample_rate: int,
    *, min_gap_s: float = 0.15, jump_db: float = 8.0, max_onsets: int = 32,
) -> List[int]:
    """Индексы сэмплов ударов: скачок огибающей над локальной медианой."""
    hop = max(int(0.01 * sample_rate), 64)
    x = np.asarray(x, dtype=np.float64)
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2))
                    for i in range(0, len(x) - hop, hop)])
    if env.size < 4:
        return []
    env_db = 20.0 * np.log10(env + 1e-10)
    med = float(np.median(env_db))
    onsets, last = [], -10 ** 9
    for i in range(2, len(env_db)):
        if env_db[i] > med + 12.0 and env_db[i] - env_db[i - 2] > jump_db:
            idx = i * hop
            if idx - last >= int(min_gap_s * sample_rate):
                onsets.append(idx)
                last = idx
                if len(onsets) >= max_onsets:
                    break
    return onsets


def isolated_hits(
    primary: np.ndarray,
    other: np.ndarray,
    sample_rate: int,
    *, guard_ms: float = ISOLATION_GUARD_MS,
) -> List[int]:
    """Удары primary БЕЗ одновременного удара other в ±guard_ms.

    Одновременный кик+снейр смещает GCC-оценку (Clifford & Reiss) —
    берём только изолированные удары.
    """
    on_p = detect_onsets(primary, sample_rate)
    on_o = set()
    guard = int(guard_ms / 1000.0 * sample_rate)
    for t in detect_onsets(other, sample_rate):
        on_o.add(t)
    return [t for t in on_p
            if not any(abs(t - to) < guard for to in on_o)]


# ── Оценка задержки и полярности пер-пара ────────────────────────

@dataclass
class PairDelayEstimate:
    delay_ms: float
    mad_ms: float
    iqr_ms: float
    n_hits: int
    accepted: bool
    reject_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delay_ms": round(self.delay_ms, 3),
            "mad_ms": round(self.mad_ms, 3),
            "iqr_ms": round(self.iqr_ms, 3),
            "n_hits": self.n_hits,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
        }


def estimate_pair_delay(
    target: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    hit_indices: Sequence[int],
    *,
    min_hits: int = MIN_HITS,
) -> PairDelayEstimate:
    """Медианная задержка target относительно reference по onset-окнам.

    Порядок аргументов — правило CLAUDE.md: gcc_phat(target, reference).
    Принимается только при MAD ≤ 0.5 мс, IQR ≤ 1 мс и ≥ min_hits ударов.
    """
    pre = int(HIT_WINDOW_PRE_MS / 1000.0 * sample_rate)
    post = int(HIT_WINDOW_POST_MS / 1000.0 * sample_rate)
    delays: List[float] = []
    for t in hit_indices:
        lo, hi = max(0, t - pre), min(len(reference), t + post)
        if hi - lo < 256:
            continue
        analyzer = GCCPHATAnalyzer(sample_rate=sample_rate)
        m = analyzer.compute_delay(
            ref_signal=np.asarray(reference[lo:hi], dtype=np.float64),
            tgt_signal=np.asarray(target[lo:hi], dtype=np.float64),
        )
        # Валидность окна для PHAT — PSR (после фазового отбеливания
        # амплитуда пика мала по построению, correlation_peak ~0.05 даже
        # на идеальном сигнале); устойчивость обеспечивают MAD/IQR-гейты.
        if m is not None and m.psr >= 3.0:
            delays.append(float(m.delay_ms))
    n = len(delays)
    if n == 0:
        return PairDelayEstimate(0.0, 0.0, 0.0, 0, False, "no_valid_hits")
    arr = np.asarray(delays)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    if n < min_hits:
        return PairDelayEstimate(med, mad, iqr, n, False, "insufficient_hits")
    if mad > MAD_APPLY_LIMIT_MS:
        return PairDelayEstimate(med, mad, iqr, n, False,
                                 "mad_above_0.5ms_needs_solo_pass")
    if iqr > IQR_SANITY_MS:
        return PairDelayEstimate(med, mad, iqr, n, False, "iqr_unstable")
    return PairDelayEstimate(med, mad, iqr, n, True)


def polarity_decision(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: int,
    lag_ms: float = 0.0,
    *,
    r_min: float = POLARITY_R_MIN,
) -> Dict[str, Any]:
    """Полярность по знаку нормированной корреляции на лаге.

    flip=None + flag="manual" — уверенности нет, автофлип запрещён.
    """
    lag = int(round(lag_ms / 1000.0 * sample_rate))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y)) - abs(lag)
    if n < sample_rate // 10:
        return {"flip": None, "flag": "manual", "r": 0.0}
    if lag >= 0:
        a, b = x[lag:lag + n], y[:n]
    else:
        a, b = x[:n], y[-lag:-lag + n]
    denom = (float(np.std(a)) * float(np.std(b))) or 1e-12
    r = float(np.mean((a - a.mean()) * (b - b.mean())) / denom)
    if abs(r) < r_min:
        return {"flip": None, "flag": "manual", "r": round(r, 3)}
    return {"flip": bool(r < 0), "confidence": round(abs(r), 3),
            "r": round(r, 3)}


# ── Решатель кластера ────────────────────────────────────────────

def quantize_delay_ms(delay_ms: float, step_ms: float = CONSOLE_DELAY_STEP_MS) -> float:
    return float(np.round(delay_ms / step_ms) * step_ms)


def align_cluster(
    arrival_offsets_ms: Mapping[Any, float],
    roles: Mapping[Any, str],
    cluster: AlignmentCluster,
    *,
    console_step_ms: float = CONSOLE_DELAY_STEP_MS,
) -> Dict[str, Any]:
    """Решение задержек кластера: delay_i = t_ref − t_i ≥ 0, кэп, квант.

    arrival_offsets_ms — время прихода относительно любого общего нуля
    (позднее прибытие = больше = дальше). Возвращает {delays_ms, reference,
    excluded: [(ch, reason)], geometry}.
    """
    if not cluster.align:
        return {"delays_ms": {}, "reference": None,
                "excluded": [(ch, "cluster_align_disabled")
                             for ch in arrival_offsets_ms],
                "geometry_m": {}}

    eligible: Dict[Any, float] = {}
    excluded: List[Tuple[Any, str]] = []
    for ch, t in arrival_offsets_ms.items():
        role = roles.get(ch, "line")
        # индивидуальные запреты: подвижные источники и room
        if role in NEVER_ALIGN_ROLES:
            excluded.append((ch, "never_align_role"))
            continue
        eligible[ch] = float(t)
    if len(eligible) < 2:
        return {"delays_ms": {}, "reference": None,
                "excluded": excluded + [(ch, "cluster_too_small")
                                        for ch in eligible],
                "geometry_m": {}}

    if cluster.reference_policy == "manual" and cluster.manual_reference_role:
        ref = next((ch for ch, r in roles.items()
                    if r == cluster.manual_reference_role and ch in eligible),
                   max(eligible, key=eligible.get))
    elif cluster.reference_policy == "closest":
        ref = min(eligible, key=eligible.get)
    else:  # farthest — школа оверхедов, дефолт
        ref = max(eligible, key=eligible.get)

    delays: Dict[Any, float] = {}
    geometry: Dict[Any, float] = {}
    for ch, t in eligible.items():
        d = eligible[ref] - t
        if d < 0.0:
            # при closest-политике дальние члены пришлось бы «ускорить» —
            # физически невозможно, оставляем 0 (они и так позже)
            d = 0.0
        if d > cluster.cap_ms:
            excluded.append((ch, f"exceeds_cap_{cluster.cap_ms}ms"))
            continue
        role = roles.get(ch, "line")
        if role in ("lead_vocal", "backing_vocal") and d > VOCAL_MAX_ADDED_DELAY_MS:
            excluded.append((ch, "vocal_zero_added_delay_iem"))
            continue
        delays[ch] = quantize_delay_ms(d, console_step_ms)
        geometry[ch] = round(d / SPEED_OF_SOUND_MS_PER_M, 2)  # метры
    return {"delays_ms": delays, "reference": ref,
            "excluded": excluded, "geometry_m": geometry}


def geometry_plausible(delay_ms: float) -> bool:
    """Дистанция = задержка × c: правдоподобие 0.2–4 м для кластера кита."""
    dist = delay_ms / SPEED_OF_SOUND_MS_PER_M
    return GEOMETRY_PLAUSIBLE_M[0] <= dist <= GEOMETRY_PLAUSIBLE_M[1]


def sum_energy_gain_db(
    a: np.ndarray,
    b: np.ndarray,
    delay_ms: float,
    sample_rate: int,
    bands_hz: Sequence[Tuple[float, float]] = SUM_VERIFY_BANDS_HZ,
) -> float:
    """Верификация: прирост энергии суммы в полосах перекрытия (дБ).

    Ожидаем +1…+3 дБ НЧ после выравнивания; ухудшение → откат."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    shift = int(round(delay_ms / 1000.0 * sample_rate))
    b_aligned = np.roll(b, shift)

    def band_energy(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
        total = 0.0
        for lo, hi in bands_hz:
            m = (freqs >= lo) & (freqs < hi)
            total += float(spec[m].sum())
        return 10.0 * np.log10(total + 1e-12)

    n = min(len(a), len(b_aligned))
    return band_energy(a[:n] + b_aligned[:n]) - band_energy(a[:n] + b[:n])


# ── Снапшот и LIVE-мониторинг (детекция, не запись) ──────────────

@dataclass
class PhaseSnapshot:
    delays_ms: Dict[Any, float]
    polarity_flips: Dict[Any, bool]
    polarity_r_signs: Dict[Any, float] = field(default_factory=dict)
    estimates: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delays_ms": {str(k): v for k, v in self.delays_ms.items()},
            "polarity_flips": {str(k): v for k, v in self.polarity_flips.items()},
            "estimates": {str(k): v for k, v in self.estimates.items()},
        }


@dataclass
class LiveDriftMonitor:
    """LIVE: только детекция + оповещение. НИКАКОЙ автозаписи задержек
    в эфире (изменение задержки в эфире = флэнжинг/щелчки); применение
    предложений — только в паузе-тишине с armed-подтверждением."""
    snapshot: PhaseSnapshot
    threshold_ms: float = DRIFT_THRESHOLD_MS
    consecutive_required: int = DRIFT_CONSECUTIVE_WINDOWS
    _consecutive: int = 0
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def check_drift(self, current_delays_ms: Mapping[Any, float]) -> Optional[Dict[str, Any]]:
        drift = {
            ch: abs(float(current_delays_ms.get(ch, self.snapshot.delays_ms[ch]))
                    - self.snapshot.delays_ms[ch])
            for ch in self.snapshot.delays_ms
        }
        if drift and max(drift.values()) > self.threshold_ms:
            self._consecutive += 1
            if self._consecutive >= self.consecutive_required:
                alert = {"name": "DELAY_DRIFT",
                         "drift_ms": {str(k): round(v, 2) for k, v in drift.items()},
                         "suggestion": "применить в паузе-тишине (armed)"}
                self.alerts.append(alert)
                return alert
        else:
            self._consecutive = 0
        return None

    def check_polarity_flip(self, current_r_by_channel: Mapping[Any, float]) -> Optional[Dict[str, Any]]:
        for ch, r in current_r_by_channel.items():
            snap_r = self.snapshot.polarity_r_signs.get(ch)
            if snap_r is None or abs(r) < POLARITY_R_MIN:
                continue
            if np.sign(r) != np.sign(snap_r):
                alert = {"name": "POLARITY_FLIP", "channel": ch,
                         "suggestion": "перепатч/замена кабеля? приоритетная проверка"}
                self.alerts.append(alert)
                return alert
        return None


def can_apply_live(silence_detected: bool, operator_armed: bool) -> bool:
    """Применение задержек в LIVE разрешено ТОЛЬКО в тишине + armed."""
    return bool(silence_detected and operator_armed)
