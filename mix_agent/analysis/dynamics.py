"""Dynamic and transient feature extraction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .loudness import amp_to_db, to_mono


def compute_dynamics_metrics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute coarse dynamics metrics used by conservative rules."""
    mono = to_mono(audio)
    frame = max(1, int(sample_rate * 0.05))
    hop = max(1, frame // 2)
    rms_db = []
    for start in range(0, max(1, len(mono) - frame + 1), hop):
        chunk = mono[start:start + frame]
        rms_db.append(amp_to_db(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))))
    if not rms_db:
        rms_db = [-120.0]
    values = np.asarray(rms_db, dtype=np.float32)
    active = values[values > -70.0]
    active = active if len(active) else values
    diff = np.diff(active) if len(active) > 1 else np.asarray([], dtype=np.float32)
    transient_rises = diff[diff > 3.0]
    duration = max(len(mono) / float(sample_rate), 1e-6)
    pumping = 0.0
    if len(diff) > 3:
        pumping = float(np.mean(np.abs(diff)) / (np.std(active) + 1e-6))
    return {
        "transient_density": round(float(len(transient_rises) / duration), 6),
        "transient_strength_db": round(float(np.mean(transient_rises)), 3)
        if len(transient_rises)
        else 0.0,
        "attack_statistics": {
            "fast_rise_count": int(len(transient_rises)),
            "p95_rise_db": round(float(np.percentile(transient_rises, 95)), 3)
            if len(transient_rises)
            else 0.0,
        },
        "gain_envelope_variance": round(float(np.var(active)), 6),
        "short_term_loudness_variance": round(float(np.std(active)), 6),
        "compression_pumping_proxy": round(pumping, 6),
        "macro_dynamics": {
            "p10_db": round(float(np.percentile(active, 10)), 3),
            "p50_db": round(float(np.percentile(active, 50)), 3),
            "p90_db": round(float(np.percentile(active, 90)), 3),
        },
    }
