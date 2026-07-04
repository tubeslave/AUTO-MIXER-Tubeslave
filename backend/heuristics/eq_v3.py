"""EQ v3 — двухконтурный метод эквализации (синтез документа FOH и проекта).

САУНДЧЕК-контур (агрессивный, однократный → baseline-снимок):
  1. HPF по фундаментальной (0.75·f0, границы роли) — resonance_eq
  2. Резонанс-резы (prominence, Q 2..8, гармонический фильтр) — resonance_eq
  3. ERB-маскирование против суммы, комплементарные cut-only резы по
     приоритету, вокал защищён — erb_masking
  4. Предсказательная A/B по BS.1770 (loudness-bias гейт) — eq_ab_check

LIVE-контур (дифференциальный, поверх baseline):
  ±3 дБ от снапшота, запрет подъёма возле feedback-частот, freeze в паузах,
  bleed-гейт. Скорость/гистерезис/бюджет — SafetyGate v2 live-конфигом
  (max_rate_db_per_sec, min_hold_sec, correction_budget_db).

Ролевые slope-профили (spectral_ceiling_eq) в v3 понижены до приоров и
мастер-шины: целевой спад −5 дБ/окт (Pestana et al. 2013, 928 хитов) —
статистика МИКСА, к каналу не применяется. Парный mirror EQ
(cross_adaptive_eq) для канального маскирования замещён ERB-матрицей.

Proposal-only: все решения идут в operator-proposal путь.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from heuristics.resonance_eq import (  # noqa: E402
    ResonanceCut,
    detect_resonances,
    estimate_fundamental_hz,
    hpf_from_fundamental,
)
from heuristics.erb_masking import (  # noqa: E402
    MaskingCut,
    allocate_complementary_cuts,
    band_energies_db,
    erb_band_edges,
    masking_vector_db,
)
from heuristics.eq_ab_check import (  # noqa: E402
    ABCheckResult,
    predictive_ab_check,
)

LIVE_DEVIATION_DB = 3.0             # ±3 дБ от снапшота (JND узкой полосы)
FEEDBACK_GUARD_OCTAVES = 1.0 / 6.0  # запрет boost в ±1/6 окт от feedback-частот
FREEZE_LEVEL_DB = -50.0             # тише — hold
BLEED_RATIO_MAX = 0.6               # канал/наведённое: выше — низкая уверенность
ACTIVE_ST_LUFS = -45.0
FFT_SIZE = 8192


@dataclass
class EQv3ChannelProposal:
    channel: Any
    role: str
    hpf_hz: Optional[float]
    fundamental_hz: Optional[float]
    resonance_cuts: List[ResonanceCut] = field(default_factory=list)
    masking_cuts: List[MaskingCut] = field(default_factory=list)
    ab_check: Optional[ABCheckResult] = None
    applied_bands: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "role": self.role,
            "hpf_hz": self.hpf_hz,
            "fundamental_hz": (
                None if self.fundamental_hz is None else round(self.fundamental_hz, 1)
            ),
            "resonance_cuts": [c.to_dict() for c in self.resonance_cuts],
            "masking_cuts": [c.to_dict() for c in self.masking_cuts],
            "ab_check": None if self.ab_check is None else self.ab_check.to_dict(),
            "bands": list(self.applied_bands),
            "warnings": list(self.warnings),
            "method": "eq_v3_soundcheck",
        }


def _merge_overlapping_bands(
    bands: List[Dict[str, Any]],
    min_separation_oct: float = 1.0 / 6.0,
) -> List[Dict[str, Any]]:
    """Анти-стакинг: перекрывающиеся полосы не суммируются.

    Резонанс-рез и маскинг-рез одной зоны иначе складываются в −15 дБ —
    A/B-гейт это ловит, но правильнее не предлагать: остаётся более
    глубокая полоса, дубли в пределах 1/6 октавы отбрасываются.
    """
    ordered = sorted(bands, key=lambda b: -abs(float(b["gain_db"])))
    kept: List[Dict[str, Any]] = []
    for band in ordered:
        f = float(band["freq_hz"])
        if all(abs(np.log2(f / float(k["freq_hz"]))) > min_separation_oct
               for k in kept):
            kept.append(band)
    return kept


def _spectrum_db(x: np.ndarray, sample_rate: int):
    x = np.asarray(x, dtype=np.float64)
    n = min(len(x), FFT_SIZE)
    if n < 1024:
        return np.array([]), np.array([])
    seg = x[-n:] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    return np.fft.rfftfreq(n, 1.0 / sample_rate), 20.0 * np.log10(mag + 1e-10)


def soundcheck_eq_v3(
    channels: Mapping[Any, Mapping[str, Any]],
    sample_rate: int,
    *,
    max_bands_per_channel: int = 4,
) -> Dict[Any, EQv3ChannelProposal]:
    """Саундчек-контур EQ v3.

    channels: id -> {"audio": np.ndarray, "role": str}. Возвращает
    proposal-only результат на канал; консольных полос — не больше
    max_bands_per_channel (4 у WING), сильнейшие по |gain|.
    """
    proposals: Dict[Any, EQv3ChannelProposal] = {}
    bands_by_channel: Dict[Any, np.ndarray] = {}
    roles: Dict[Any, str] = {}
    edges = erb_band_edges()

    # Слой 1-2: HPF по f0 + резонансы, ERB-энергии для маскирования
    for ch, payload in channels.items():
        audio = np.asarray(payload["audio"], dtype=np.float64)
        role = str(payload.get("role", "line"))
        roles[ch] = role
        f0 = estimate_fundamental_hz(audio, sample_rate)
        proposal = EQv3ChannelProposal(
            channel=ch,
            role=role,
            hpf_hz=hpf_from_fundamental(role, f0),
            fundamental_hz=f0,
            resonance_cuts=detect_resonances(audio, sample_rate),
        )
        proposals[ch] = proposal
        bands_by_channel[ch] = band_energies_db(audio, sample_rate, edges)

    # Слой 3: комплементарное распределение маскинг-резов
    for cut in allocate_complementary_cuts(bands_by_channel, roles, edges=edges):
        proposals[cut.channel].masking_cuts.append(cut)

    # Слой 4: сборка консольных полос + предсказательная A/B
    for ch, proposal in proposals.items():
        audio = np.asarray(channels[ch]["audio"], dtype=np.float64)
        bands = [
            {"freq_hz": c.freq_hz, "gain_db": c.gain_db, "q": c.q,
             "rule": "resonance_cut_v3"}
            for c in proposal.resonance_cuts
        ] + [
            {"freq_hz": c.freq_hz, "gain_db": c.gain_db, "q": c.q,
             "rule": "erb_masking_cut_v3"}
            for c in proposal.masking_cuts
        ]
        bands = _merge_overlapping_bands(bands)[:max_bands_per_channel]

        freqs, spec = _spectrum_db(audio, sample_rate)
        if freqs.size:
            # A/B с мягкой деградацией: если суммарный сдвиг громкости вне
            # безопасной фейдерной компенсации — резы масштабируются вниз
            # (×0.8 итеративно), а не отклоняются целиком.
            for attempt in range(4):
                proposal.ab_check = predictive_ab_check(
                    freqs, spec, bands, hpf_hz=proposal.hpf_hz,
                    sample_rate=sample_rate,
                )
                if proposal.ab_check.allowed or not bands:
                    break
                bands = [{**b, "gain_db": round(b["gain_db"] * 0.8, 2)}
                         for b in bands]
                proposal.warnings.append(
                    f"A/B: резы уменьшены ×0.8 (попытка {attempt + 1}) — "
                    "сдвиг громкости был вне безопасной компенсации"
                )
            if not proposal.ab_check.allowed:
                proposal.warnings.append(
                    "A/B: loudness bias вне безопасной компенсации — полосы отклонены"
                )
                bands = []
            elif proposal.ab_check.compensation_db:
                proposal.warnings.append(
                    f"A/B: требуется фейдерная компенсация "
                    f"{proposal.ab_check.compensation_db:+.1f} dB"
                )
        else:
            proposal.warnings.append("сигнал слишком короткий для A/B")
            bands = []
        proposal.applied_bands = bands
    return proposals


# ── LIVE-контур: дифференциальный EQ поверх baseline ─────────────

def near_feedback(freq_hz: float, feedback_freqs: Sequence[float]) -> bool:
    for fb in feedback_freqs:
        if fb > 0 and abs(np.log2(max(freq_hz, 1e-6) / fb)) <= FEEDBACK_GUARD_OCTAVES:
            return True
    return False


def live_eq_differential(
    snapshot_bands: Sequence[Mapping[str, Any]],
    desired_bands: Sequence[Mapping[str, Any]],
    *,
    feedback_freqs: Sequence[float] = (),
    channel_level_db: float = 0.0,
    bleed_ratio: float = 0.0,
    max_deviation_db: float = LIVE_DEVIATION_DB,
    freeze_level_db: float = FREEZE_LEVEL_DB,
    bleed_ratio_max: float = BLEED_RATIO_MAX,
) -> Dict[str, Any]:
    """Live-огибающая: желаемые полосы прижимаются к снапшоту ±3 дБ.

    Возвращает {"hold": bool, "reason": str, "bands": [...]}, где bands —
    полосы с клампом отклонения; возле feedback-частот подъём запрещён
    (только вниз). Скорость/гистерезис/min-hold/бюджет добавляет
    SafetyGate v2 (live-конфиг) — здесь не дублируются.
    """
    if channel_level_db < freeze_level_db:
        return {"hold": True, "reason": "freeze_below_gate", "bands": []}
    if bleed_ratio > bleed_ratio_max:
        return {"hold": True, "reason": "bleed_confidence_low", "bands": []}

    snap_by_freq = {round(float(b["freq_hz"]), 1): float(b["gain_db"])
                    for b in snapshot_bands}
    out: List[Dict[str, Any]] = []
    for band in desired_bands:
        f = round(float(band["freq_hz"]), 1)
        desired = float(band["gain_db"])
        base = snap_by_freq.get(f, 0.0)
        g = float(np.clip(desired, base - max_deviation_db, base + max_deviation_db))
        if near_feedback(f, feedback_freqs):
            g = min(g, base)  # подъём относительно снапшота запрещён
        out.append({"freq_hz": f, "gain_db": round(g, 2),
                    "q": float(band.get("q", 1.0)),
                    "snapshot_gain_db": base})
    return {"hold": False, "reason": "ok", "bands": out}
