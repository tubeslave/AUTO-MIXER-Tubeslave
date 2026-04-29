"""Stereo, phase and mono-compatibility features."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .loudness import amp_to_db
from .spectral import BANDS


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)) + 1e-12))


def compute_stereo_metrics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute stereo width, correlation and mono fold-down risk."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1 or data.shape[1] < 2:
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

    left = data[:, 0]
    right = data[:, 1]
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        corr = 1.0
    else:
        corr = float(np.corrcoef(left, right)[0, 1])
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_energy = _rms(mid) ** 2
    side_energy = _rms(side) ** 2
    stereo_width = float(side_energy / (mid_energy + side_energy + 1e-12))
    mono = (left + right) * 0.5
    stereo_rms = _rms(data)
    mono_loss_db = amp_to_db(stereo_rms) - amp_to_db(_rms(mono))

    n_fft = 4096
    if len(left) < n_fft:
        pad = n_fft - len(left)
        left = np.pad(left, (0, pad))
        right = np.pad(right, (0, pad))
    window = np.hanning(n_fft).astype(np.float32)
    L = np.fft.rfft(left[:n_fft] * window)
    R = np.fft.rfft(right[:n_fft] * window)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    widths: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        m = float(np.sum(np.abs(M[mask]) ** 2))
        s = float(np.sum(np.abs(S[mask]) ** 2))
        widths[name] = round(s / (m + s + 1e-12), 6)

    low_width = max(widths.get("sub", 0.0), widths.get("bass", 0.0))
    return {
        "is_stereo": True,
        "stereo_width": round(stereo_width, 6),
        "inter_channel_correlation": round(corr, 6),
        "mid_side_energy_ratio": round(side_energy / (mid_energy + 1e-12), 6),
        "mono_fold_down_loss_db": round(float(mono_loss_db), 3),
        "phase_cancellation_risk": bool(corr < 0.1 or mono_loss_db > 2.0),
        "low_frequency_stereo_width": round(float(low_width), 6),
        "low_frequency_stereo_width_warning": bool(low_width > 0.25),
        "frequency_dependent_width": widths,
        "limitations": [],
    }
