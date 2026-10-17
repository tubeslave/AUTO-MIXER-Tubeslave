"""Explicit post-insert level compensation; no new compression or mix acceptance.

The fixed activity mask comes only from the previous processed source. RMS is
channel power, not a mono fold-down. This is source-level evidence only: all
shared buses, source-dependent returns and the actual exports still need checks.
"""
from __future__ import annotations

import numpy as np

from .compression import as_audio


def match_processed_source(reference: np.ndarray, candidate: np.ndarray, sr: int, *,
                           active_percentile: float = 65.0,
                           max_gain_db: float = 1.0,
                           tolerance_db: float = .05) -> tuple[np.ndarray, dict]:
    """Match the candidate to the no-change source using one constant gain.

    No peaks are normalized, no control trace is rescaled, and no per-section
    riding is added. Excessive required compensation raises instead of silently
    capping gain and reporting a successful balance match. Headroom must be checked
    in the actual downstream graph, not by silently attenuating one A/B side.
    """
    a, b = as_audio(reference), as_audio(candidate)
    if a.shape != b.shape:
        raise ValueError('source and candidate shapes must match')
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 1000:
        raise ValueError('sample rate must be an integer >= 1000 Hz')
    if (not np.isfinite(active_percentile) or not 0 <= active_percentile < 100
            or not np.isfinite(max_gain_db) or not 0 <= max_gain_db <= 6
            or not np.isfinite(tolerance_db) or not 0 < tolerance_db <= .1):
        raise ValueError('invalid level preservation policy')
    hop = max(1, round(.02 * sr))
    if len(a) < hop:
        raise ValueError('at least one 20 ms frame is required')
    starts = np.arange(0, len(a), hop)
    counts = np.minimum(hop, len(a) - starts)

    def power(x):
        p = np.asarray(x, dtype=np.float64) ** 2
        if x.ndim == 2:
            p = np.mean(p, axis=1)
        return np.add.reduceat(p, starts) / counts

    ap, bp = power(a), power(b)
    if float(ap.max()) <= 1e-20 or float(bp.max()) <= 1e-20:
        raise ValueError('silent source/candidate cannot define a level match')
    # Stable recordings with equal frame power must still select activity.
    mask = ap >= max(float(np.percentile(ap, active_percentile)),
                     float(ap.max()) * 1e-4, 1e-20)

    def level(p):
        e = float(np.average(p[mask], weights=counts[mask]))
        if e <= 1e-20:
            raise ValueError('candidate has no energy in reference-active windows')
        return float(10 * np.log10(e))

    ref_db, before_db = level(ap), level(bp)
    required = ref_db - before_db
    if abs(required) > max_gain_db + 1e-9:
        raise ValueError('required compensation exceeds declared gain bound')
    gain = np.float32(10 ** (required / 20))
    y = (b * gain).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite matched output')
    residual = level(power(y)) - ref_db
    if abs(residual) > tolerance_db:
        raise ValueError('level preservation measurement failed')
    return y, {
        'schema': 'processed-source-level-preservation-v1',
        'active_percentile': float(active_percentile),
        'reference_active_frames': int(mask.sum()),
        'reference_active_rms_dbfs': ref_db,
        'candidate_before_active_rms_dbfs': before_db,
        'uncompensated_delta_db': before_db - ref_db,
        'compensation_gain_db': required,
        'matched_active_rms_delta_db': residual,
        'balance_match_passed': True,
        'mask_source': 'fixed_previous_processed_source',
        'processing': 'one_constant_gain_after_compressor',
        'compressor_settings_changed': False,
        'headroom_verified': False,
        'requires_full_session_rerender': True,
        'requires_human_listening': True,
        'requires_human_review': True,
        'baseline_eligible': False,
    }
