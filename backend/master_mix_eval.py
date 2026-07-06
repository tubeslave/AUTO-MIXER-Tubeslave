"""Оценка результата сведения по ЗВУЧАНИЮ мастер-суммы (closed loop).

Директива оператора 2026-07-06: Main LR пульта заведён на USB-отправки
47/48 (`/io/out/USB/47|48`: grp=MAIN, in=L/R) — проект принимает общий
микс и оценивает его как результат сведения. Это ЕДИНСТВЕННЫЙ замкнутый
контур в системе: pre-trim tap каналов не слышит ни одной применённой
коррекции, мастер-петля слышит их все (trim, gate, EQ, comp, fader,
pan, FX, mirror-EQ).

Метрики (informational, proposal-only — модуль ничего не пишет на пульт):
  - integrated loudness по BS.1770 (K-weighting, gating проекта);
  - true peak (4x oversampling) и crest;
  - стерео: L/R RMS-баланс, межканальная корреляция, mono-fold потеря;
  - спектральный наклон против целевой кривой pa_v3 (live_standard);
  - вокальная зона 1-4 кГц относительно суммы (разборчивость);
  - вердикт со списком конкретных проблем.

Честность: тишина в отводе (например, Main LR замьючен) — отдельный
вердикт NO_SIGNAL, а не «отличный тихий микс».
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parents[0])
if _BACKEND_DIR not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, _BACKEND_DIR)

from heuristics.pa_v3 import target_curve_db  # noqa: E402
from lufs_gain_staging import KWeightingFilter, TruePeakMeter  # noqa: E402

SILENCE_RMS_DBFS = -80.0
TARGET_MIX_LUFS = -20.0            # соундчек-плоскость мастера (калибруется)
MIX_LUFS_TOLERANCE_LU = 6.0
BALANCE_LIMIT_DB = 1.5             # L/R (правило pan v3)
MONO_FOLD_LIMIT = 0.5              # относительная потеря энергии в mono
CORRELATION_MIN = 0.0              # ниже нуля = фазовые проблемы суммы
TILT_DEVIATION_LIMIT_DB = 6.0      # медианное |отклонение| от целевой кривой
VOCAL_BAND_HZ = (1000.0, 4000.0)
VOCAL_PRESENCE_MIN_DB = -18.0      # зона 1-4к не глубже этого от суммы
TRUE_PEAK_CEILING_DBTP = -1.0      # правило проекта


@dataclass
class MasterMixVerdict:
    status: str                    # OK | ISSUES | NO_SIGNAL
    lufs_integrated: Optional[float] = None
    true_peak_dbtp: Optional[float] = None
    crest_db: Optional[float] = None
    lr_balance_db: Optional[float] = None
    correlation: Optional[float] = None
    mono_fold_ratio: Optional[float] = None
    tilt_deviation_db: Optional[float] = None
    vocal_presence_db: Optional[float] = None
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "lufs_integrated": _r(self.lufs_integrated),
            "true_peak_dbtp": _r(self.true_peak_dbtp),
            "crest_db": _r(self.crest_db),
            "lr_balance_db": _r(self.lr_balance_db),
            "correlation": _r(self.correlation, 3),
            "mono_fold_ratio": _r(self.mono_fold_ratio, 3),
            "tilt_deviation_db": _r(self.tilt_deviation_db),
            "vocal_presence_db": _r(self.vocal_presence_db),
            "issues": list(self.issues),
            "method": "master_mix_eval",
        }


def _r(value: Optional[float], nd: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), nd)


def _integrated_lufs(mono: np.ndarray, sample_rate: int) -> float:
    """BS.1770: K-weighting + абсолютный гейт −70 LUFS + относительный −10 LU."""
    kf = KWeightingFilter(sample_rate)
    weighted = kf.process(mono.astype(np.float64))
    block = int(0.4 * sample_rate)
    hop = block // 4
    loudness: List[float] = []
    for i in range(0, len(weighted) - block, hop):
        ms = float(np.mean(weighted[i:i + block] ** 2))
        loudness.append(-0.691 + 10.0 * np.log10(ms + 1e-12))
    if not loudness:
        return -120.0
    blocks = np.asarray(loudness)
    gated = blocks[blocks > -70.0]
    if gated.size == 0:
        return -120.0
    ungated_mean = 10.0 * np.log10(np.mean(10.0 ** (gated / 10.0)))
    gated = gated[gated > ungated_mean - 10.0]
    if gated.size == 0:
        return -120.0
    return float(10.0 * np.log10(np.mean(10.0 ** (gated / 10.0))))


def evaluate_master_mix(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    *,
    target_curve: str = "live_standard",
    target_lufs: float = TARGET_MIX_LUFS,
) -> MasterMixVerdict:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    n = min(len(left), len(right))
    if n < sample_rate:
        return MasterMixVerdict(status="NO_SIGNAL",
                                issues=["слишком короткий захват"])
    left, right = left[:n], right[:n]

    rms_l = float(np.sqrt(np.mean(left ** 2)))
    rms_r = float(np.sqrt(np.mean(right ** 2)))
    rms_db = 20.0 * np.log10(max(rms_l, rms_r) + 1e-12)
    if rms_db < SILENCE_RMS_DBFS:
        return MasterMixVerdict(
            status="NO_SIGNAL",
            issues=["в мастер-отводе тишина — проверить mute Main LR и "
                    "роутинг /io/out/USB/47|48"],
        )

    mono = 0.5 * (left + right)
    issues: List[str] = []

    lufs = _integrated_lufs(mono, sample_rate)
    tp_meter = TruePeakMeter(sample_rate)
    tp = float(max(tp_meter.process(left), tp_meter.process(right)))
    peak_db = 20.0 * np.log10(float(np.max(np.abs(mono))) + 1e-12)
    crest = peak_db - 20.0 * np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-12)

    balance = 20.0 * np.log10((rms_l + 1e-12) / (rms_r + 1e-12))
    denom = float(np.std(left) * np.std(right))
    correlation = float(np.mean(
        (left - np.mean(left)) * (right - np.mean(right))) / (denom + 1e-12))
    e_lr = float(np.mean(left ** 2) + np.mean(right ** 2))
    mono_fold = float(np.mean((left + right) ** 2) / (2.0 * e_lr + 1e-12))

    spec = np.abs(np.fft.rfft(mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band = (freqs >= 40.0) & (freqs <= 16000.0)
    mag_db = 20.0 * np.log10(spec[band] + 1e-12)
    f_band = freqs[band]
    target = target_curve_db(f_band, target_curve)
    deviation = (mag_db - float(np.median(mag_db))) - (target - float(np.median(target)))
    # медианное |отклонение| по октавно-сглаженной сетке
    edges = 40.0 * (2.0 ** np.arange(0, 9, 1.0 / 3.0))
    dev_bands: List[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f_band >= lo) & (f_band < hi)
        if m.any():
            dev_bands.append(float(np.median(deviation[m])))
    tilt_dev = float(np.median(np.abs(dev_bands))) if dev_bands else 0.0

    vocal_m = (f_band >= VOCAL_BAND_HZ[0]) & (f_band <= VOCAL_BAND_HZ[1])
    vocal_presence = (float(np.median(mag_db[vocal_m])) - float(np.median(mag_db))
                      if vocal_m.any() else None)

    if tp > TRUE_PEAK_CEILING_DBTP:
        issues.append(f"true peak {tp:.1f} dBTP выше потолка {TRUE_PEAK_CEILING_DBTP}")
    if abs(lufs - target_lufs) > MIX_LUFS_TOLERANCE_LU:
        issues.append(f"громкость микса {lufs:.1f} LUFS вне {target_lufs}±{MIX_LUFS_TOLERANCE_LU}")
    if abs(balance) > BALANCE_LIMIT_DB:
        issues.append(f"L/R дисбаланс {balance:+.1f} дБ (> {BALANCE_LIMIT_DB})")
    if correlation < CORRELATION_MIN:
        issues.append(f"межканальная корреляция {correlation:+.2f} < 0 — фазовые проблемы")
    if mono_fold < MONO_FOLD_LIMIT:
        issues.append(f"mono-fold {mono_fold:.2f} < {MONO_FOLD_LIMIT} — сумма теряет энергию")
    if tilt_dev > TILT_DEVIATION_LIMIT_DB:
        issues.append(f"спектр отклоняется от '{target_curve}' на {tilt_dev:.1f} дБ (медиана)")
    if vocal_presence is not None and vocal_presence < VOCAL_PRESENCE_MIN_DB:
        issues.append(f"зона вокала 1-4 кГц {vocal_presence:.1f} дБ от суммы — маскируется")

    return MasterMixVerdict(
        status="ISSUES" if issues else "OK",
        lufs_integrated=lufs,
        true_peak_dbtp=tp,
        crest_db=crest,
        lr_balance_db=balance,
        correlation=correlation,
        mono_fold_ratio=mono_fold,
        tilt_deviation_db=tilt_dev,
        vocal_presence_db=vocal_presence,
        issues=issues,
    )
