"""Technical artifact helpers."""

from __future__ import annotations

from typing import Any, Dict


def artifact_summary(level_metrics: Dict[str, Any], stereo_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Collect artifact-related flags from level and stereo metrics."""
    return {
        "clip_count": int(level_metrics.get("clip_count", 0)),
        "true_peak_dbtp": level_metrics.get("true_peak_dbtp"),
        "dc_offset": level_metrics.get("dc_offset"),
        "click_pop_estimate": int(level_metrics.get("click_pop_estimate", 0)),
        "noise_floor_dbfs": level_metrics.get("noise_floor_dbfs"),
        "phase_correlation": stereo_metrics.get("inter_channel_correlation"),
        "phase_cancellation_risk": bool(stereo_metrics.get("phase_cancellation_risk", False)),
        "tail_truncation_proxy": "not_estimated",
    }
