"""Before/after evaluation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np

from mix_agent.analysis.loudness import compute_level_metrics
from mix_agent.analysis.spectral import compute_spectral_metrics
from mix_agent.analysis.stereo import compute_stereo_metrics


def compare_before_after(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
) -> Dict[str, object]:
    """Compute a small before/after metric delta report."""
    before_level, _ = compute_level_metrics(before, sample_rate)
    after_level, _ = compute_level_metrics(after, sample_rate)
    before_spec = compute_spectral_metrics(before, sample_rate)
    after_spec = compute_spectral_metrics(after, sample_rate)
    before_stereo = compute_stereo_metrics(before, sample_rate)
    after_stereo = compute_stereo_metrics(after, sample_rate)
    return {
        "before": {
            "level": before_level,
            "spectral": before_spec,
            "stereo": before_stereo,
        },
        "after": {
            "level": after_level,
            "spectral": after_spec,
            "stereo": after_stereo,
        },
        "delta": {
            "true_peak_dbtp": round(
                float(after_level.get("true_peak_dbtp", 0.0))
                - float(before_level.get("true_peak_dbtp", 0.0)),
                3,
            ),
            "integrated_lufs": round(
                float(after_level.get("integrated_lufs", 0.0))
                - float(before_level.get("integrated_lufs", 0.0)),
                3,
            ),
            "mono_fold_down_loss_db": round(
                float(after_stereo.get("mono_fold_down_loss_db", 0.0))
                - float(before_stereo.get("mono_fold_down_loss_db", 0.0)),
                3,
            ),
            "muddiness_proxy": round(
                float(after_spec.get("muddiness_proxy", 0.0))
                - float(before_spec.get("muddiness_proxy", 0.0)),
                6,
            ),
        },
    }
