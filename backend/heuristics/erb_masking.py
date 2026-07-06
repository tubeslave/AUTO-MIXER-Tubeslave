"""EQ v3, слой 2: ERB-матрица маскирования и комплементарное распределение.

Маскирование канала считается ОТНОСИТЕЛЬНО СУММЫ всех остальных каналов
(Hafezi & Reiss 2015; Ronan et al. 2018), а не попарно: попарная
оптимизация может улучшать одну пару, ухудшая другую. Полосы разнесены
по ERB-шкале (Glasberg & Moore: ERBn = 21.4·log10(4.37·f/1000 + 1)) —
на низах корректнее третьоктавного разбиения.

Распределение: жадное, cut-only (валидировано: подъём в live снижает
запас до самовозбуждения), в конфликтной полосе режется канал с МЕНЬШИМ
приоритетом; lead vocal не режется никогда (Pestana & Reiss #04: «no
element should mask the vocals», вокал в среднем на 3-8 LU громче).
Кэп −6 дБ («harsh filtering» глубже — Hafezi & Reiss constrained-версия).

Известное ограничение (по прямому утверждению Hafezi & Reiss): объективные
метрики маскирования плохо коррелируют с воспринимаемым — поэтому метрика
здесь НЕ единственный критерий: результат проходит предсказательную A/B
(eq_ab_check) и operator-proposal путь. Proposal-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

N_ERB_BANDS = 36
FREQ_MIN_HZ, FREQ_MAX_HZ = 50.0, 16000.0
MASK_THRESHOLD_DB = 6.0
MAX_MASKING_CUT_DB = 6.0
CUT_STEP_DB = 1.0
# Защита тембрового ядра маскера (вопрос оператора 2026-07-06 «EQ просто
# давит громкое»): полосы в пределах IDENTITY_CORE_WINDOW_DB от максимума
# канала — его голос; срезать их глубже IDENTITY_CORE_CUT_CAP_DB нельзя,
# даже если они конфликтуют с суммой. Конфликт решается в НЕ-ядерных
# полосах маскера или балансом/аранжировкой, не выпиливанием характера.
IDENTITY_CORE_WINDOW_DB = 6.0
IDENTITY_CORE_CUT_CAP_DB = 3.0
MASKING_CUT_Q = 4.0            # constrained-версия Hafezi & Reiss (Q=5, у нас кэп WING)
MAX_CUTS_PER_CHANNEL = 3

# Приоритеты ролей P (вокал = 1.0); в конфликте режется меньший P
ROLE_PRIORITY: Dict[str, float] = {
    "lead_vocal": 1.0,
    "solo": 0.95,
    "snare": 0.8,
    "kick": 0.8,
    "bass": 0.75,
    "keys": 0.6,
    "guitar": 0.55,
    "acoustic_guitar": 0.55,
    "playback": 0.55,
    "backing_vocal": 0.5,
    "tom": 0.5,
    "floor_tom": 0.5,
    "hihat": 0.4,
    "overheads": 0.35,
    "line": 0.5,
}
PROTECTED_ROLES = frozenset({"lead_vocal"})


def erb_number(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 21.4 * np.log10(4.37 * np.asarray(freq_hz, dtype=np.float64) / 1000.0 + 1.0)


def erb_to_freq(erb: np.ndarray | float) -> np.ndarray | float:
    return (10.0 ** (np.asarray(erb, dtype=np.float64) / 21.4) - 1.0) * 1000.0 / 4.37


def erb_band_edges(
    n_bands: int = N_ERB_BANDS,
    fmin: float = FREQ_MIN_HZ,
    fmax: float = FREQ_MAX_HZ,
) -> np.ndarray:
    """n_bands+1 границ, равномерных по ERB-шкале."""
    return np.asarray(erb_to_freq(np.linspace(erb_number(fmin), erb_number(fmax), n_bands + 1)))


def band_energies_db(
    x: np.ndarray,
    sample_rate: int,
    edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Энергия сигнала по ERB-полосам, дБ."""
    edges = erb_band_edges() if edges is None else edges
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 512:
        return np.full(len(edges) - 1, -120.0)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 / len(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    out = np.empty(len(edges) - 1)
    for b in range(len(edges) - 1):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        e = float(spectrum[m].sum()) if m.any() else 0.0
        out[b] = 10.0 * np.log10(e + 1e-12)
    return out


@dataclass
class MaskingCut:
    """Предложенный маскинг-рез (proposal-only)."""
    channel: Any
    freq_hz: float
    gain_db: float           # < 0
    q: float
    band_index: int
    masked_channel: Any
    mask_before_db: float
    mask_after_db: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "freq_hz": round(self.freq_hz, 1),
            "gain_db": round(self.gain_db, 2),
            "q": round(self.q, 2),
            "band_index": self.band_index,
            "masked_channel": self.masked_channel,
            "mask_before_db": round(self.mask_before_db, 2),
            "mask_after_db": round(self.mask_after_db, 2),
            "rule": "erb_masking_cut_v3",
        }


def masking_vector_db(
    bands_by_channel: Mapping[Any, np.ndarray],
    maskee: Any,
) -> np.ndarray:
    """M[b] = (сумма мощностей остальных) − maskee, дБ по полосам."""
    maskee_db = bands_by_channel[maskee]
    others_power = None
    for ch, bands in bands_by_channel.items():
        if ch == maskee:
            continue
        p = 10.0 ** (np.asarray(bands) / 10.0)
        others_power = p if others_power is None else others_power + p
    if others_power is None:
        return np.full_like(maskee_db, -120.0)
    return 10.0 * np.log10(others_power + 1e-12) - np.asarray(maskee_db)


def allocate_complementary_cuts(
    bands_by_channel: Mapping[Any, np.ndarray],
    roles: Mapping[Any, str],
    *,
    edges: Optional[np.ndarray] = None,
    mask_threshold_db: float = MASK_THRESHOLD_DB,
    max_cut_db: float = MAX_MASKING_CUT_DB,
    max_cuts_per_channel: int = MAX_CUTS_PER_CHANNEL,
    active_channels: Optional[frozenset] = None,
) -> List[MaskingCut]:
    """Жадное комплементарное распределение маскинг-резов.

    Для каждой конфликтной ERB-полосы каждого маскируемого канала (по
    убыванию приоритета) режется доминирующий маскер с меньшим приоритетом,
    шагами CUT_STEP_DB, пока маскирование не упадёт ниже порога или не
    исчерпан кэп. Защищённые роли (lead_vocal) не режутся.
    """
    edges = erb_band_edges() if edges is None else edges
    centers = np.sqrt(edges[:-1] * edges[1:])
    working = {ch: np.asarray(b, dtype=np.float64).copy()
               for ch, b in bands_by_channel.items()}
    if active_channels is not None:
        working = {ch: b for ch, b in working.items() if ch in active_channels}

    priority = {ch: ROLE_PRIORITY.get(roles.get(ch, "line"), 0.5) for ch in working}
    cut_budget = {ch: {} for ch in working}   # ch -> band -> cut_db (>0)
    # тембровое ядро каждого канала — по ИСХОДНОМУ спектру (до срезов)
    identity_core = {
        ch: {int(b) for b in np.flatnonzero(
            np.asarray(bands) >= float(np.max(bands)) - IDENTITY_CORE_WINDOW_DB)}
        for ch, bands in working.items()
    }
    cuts: List[MaskingCut] = []

    maskees = sorted(working, key=lambda ch: -priority[ch])
    for maskee in maskees:
        mask = masking_vector_db(working, maskee)
        # обрабатываем полосы по убыванию перекрытия
        for b in np.argsort(-mask):
            b = int(b)
            if mask[b] < mask_threshold_db:
                break
            # маскируемая полоса должна содержать сигнал самого канала
            if working[maskee][b] <= -80.0:
                continue
            # доминирующий маскер с меньшим приоритетом и запасом бюджета
            def _band_cap(ch: Any) -> float:
                # в тембровом ядре маскера срез ограничен гигиеническим кэпом
                return (IDENTITY_CORE_CUT_CAP_DB if b in identity_core.get(ch, set())
                        else max_cut_db)

            candidates = [
                ch for ch in working
                if ch != maskee
                and priority[ch] < priority[maskee]
                and roles.get(ch, "") not in PROTECTED_ROLES
                and cut_budget[ch].get(b, 0.0) < _band_cap(ch)
            ]
            if not candidates:
                continue
            masker = max(candidates, key=lambda ch: working[ch][b])
            masker_cap = _band_cap(masker)
            mask_before = float(mask[b])
            applied = 0.0
            while (cut_budget[masker].get(b, 0.0) + applied < masker_cap):
                working[masker][b] -= CUT_STEP_DB
                applied += CUT_STEP_DB
                mask = masking_vector_db(working, maskee)
                if mask[b] < mask_threshold_db:
                    break
            if applied <= 0.0:
                continue
            total = cut_budget[masker].get(b, 0.0) + applied
            cut_budget[masker][b] = total
            cuts.append(MaskingCut(
                channel=masker,
                freq_hz=float(centers[b]),
                gain_db=-float(total),
                q=MASKING_CUT_Q,
                band_index=b,
                masked_channel=maskee,
                mask_before_db=mask_before,
                mask_after_db=float(mask[b]),
            ))

    # схлопнуть повторные записи одной полосы (осталась последняя total)
    final: Dict[Tuple[Any, int], MaskingCut] = {}
    for c in cuts:
        final[(c.channel, c.band_index)] = c
    # лимит резов на канал: сильнейшие по глубине
    by_channel: Dict[Any, List[MaskingCut]] = {}
    for c in final.values():
        by_channel.setdefault(c.channel, []).append(c)
    out: List[MaskingCut] = []
    for ch, items in by_channel.items():
        items.sort(key=lambda c: c.gain_db)  # глубже — раньше
        out.extend(items[:max_cuts_per_channel])
    return out
