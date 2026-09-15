"""Stereo, phase and mono-compatibility features."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .dsp_utils import EPS, amp_to_db, channels, iter_frames
from .spectral import BANDS


def _rms(samples: np.ndarray) -> float:
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data)) + EPS))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    ls = float(np.std(left))
    rs = float(np.std(right))
    if ls < 1e-9 and rs < 1e-9:
        return 1.0
    if ls < 1e-9 or rs < 1e-9:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _frequency_width(left: np.ndarray, right: np.ndarray, sample_rate: int, n_fft: int = 4096) -> Dict[str, float]:
    """Average M/S power over every FFT frame instead of only the first frame."""
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float64)
    power_m = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    power_s = np.zeros_like(power_m)
    count = 0

    left_frames = list(iter_frames(left, n_fft, hop, pad_end=True))
    right_frames = list(iter_frames(right, n_fft, hop, pad_end=True))
    for lf, rf in zip(left_frames, right_frames):
        L = np.fft.rfft(lf.astype(np.float64, copy=False) * window)
        R = np.fft.rfft(rf.astype(np.float64, copy=False) * window)
        M = (L + R) * 0.5
        S = (L - R) * 0.5
        power_m += np.square(np.abs(M))
        power_s += np.square(np.abs(S))
        count += 1

    if count:
        power_m /= count
        power_s /= count
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
    widths: Dict[str, float] = {}
    nyquist = sample_rate / 2.0
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < min(hi, nyquist + 1.0))
        m = float(np.sum(power_m[mask])) if np.any(mask) else 0.0
        s = float(np.sum(power_s[mask])) if np.any(mask) else 0.0
        widths[name] = round(s / (m + s + EPS), 6) if (m + s) > EPS else 0.0
    return widths


def compute_stereo_metrics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute finite stereo width, correlation and mono fold-down risk."""
    data = channels(audio)
    if data.shape[1] < 2:
        return {
            "is_stereo": False,
            "stereo_width": 0.0,
            "inter_channel_correlation": 1.0,
            "mid_side_energy_ratio": 0.0,
            "mono_fold_down_loss_db": 0.0,
            "phase_cancellation_risk": False,
            "low_frequency_stereo_width": 0.0,
            "low_frequency_stereo_width_warning": False,
            "frequency_dependent_width": {},
            "limitations": ["Mono source: stereo width and mono compatibility are not meaningful."],
        }

    left = data[:, 0].astype(np.float64, copy=False)
    right = data[:, 1].astype(np.float64, copy=False)
    corr = _correlation(left, right)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_energy = _rms(mid) ** 2
    side_energy = _rms(side) ** 2
    stereo_width = float(side_energy / (mid_energy + side_energy + EPS)) if (mid_energy + side_energy) > EPS else 0.0
    mono = mid
    stereo_rms = _rms(np.column_stack([left, right]))
    mono_rms = _rms(mono)
    if stereo_rms <= EPS:
        mono_loss_db = 0.0
    else:
        mono_loss_db = max(0.0, amp_to_db(stereo_rms) - amp_to_db(mono_rms))

    widths = _frequency_width(left, right, sample_rate)
    low_width = max(widths.get("sub", 0.0), widths.get("bass", 0.0))
    side_mid_ratio = side_energy / (mid_energy + EPS) if side_energy > EPS else 0.0
    risk = bool(corr < 0.1 or mono_loss_db > 2.0 or low_width > 0.5)

    return {
        "is_stereo": True,
        "stereo_width": round(stereo_width, 6),
        "inter_channel_correlation": round(corr, 6),
        "mid_side_energy_ratio": round(float(side_mid_ratio), 6),
        "mono_fold_down_loss_db": round(float(mono_loss_db), 3),
        "phase_cancellation_risk": risk,
        "low_frequency_stereo_width": round(float(low_width), 6),
        "low_frequency_stereo_width_warning": bool(low_width > 0.25),
        "frequency_dependent_width": widths,
        "limitations": [],
    }
