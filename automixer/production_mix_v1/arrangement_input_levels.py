"""Offline arrangement-aware input level staging.

This adapter reuses the backend arrangement automation modules, but applies a
single static input-level offset per rendered stem. It intentionally does not
run realtime fader automation and does not send OSC.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import amp_to_db, ensure_stereo

try:
    from arrangement_automation.activity_detector import ActivityDetector
    from arrangement_automation.arrangement_density import ArrangementDensityAnalyzer
    from arrangement_automation.channel_role_classifier import ChannelRoleClassifier
    from arrangement_automation.level_automation_planner import LevelAutomationPlanner
    from arrangement_automation.masking_analyzer import MaskingAnalyzer
    from arrangement_automation.mix_priority_engine import MixPriorityEngine
    from arrangement_automation.safety_limiter import SafetyLimiter
    from arrangement_automation.section_detector import SectionDetector
except ImportError:  # pragma: no cover - used when repo root, not backend/, is on PYTHONPATH.
    from backend.arrangement_automation.activity_detector import ActivityDetector
    from backend.arrangement_automation.arrangement_density import ArrangementDensityAnalyzer
    from backend.arrangement_automation.channel_role_classifier import ChannelRoleClassifier
    from backend.arrangement_automation.level_automation_planner import LevelAutomationPlanner
    from backend.arrangement_automation.masking_analyzer import MaskingAnalyzer
    from backend.arrangement_automation.mix_priority_engine import MixPriorityEngine
    from backend.arrangement_automation.safety_limiter import SafetyLimiter
    from backend.arrangement_automation.section_detector import SectionDetector


ARRANGEMENT_INPUT_LEVEL_STAGE = "arrangement_input_levels_pre_phase"
REPLACED_STAGE = "ayaic_channel_levels_pre_phase"
DEFAULT_TRUE_PEAK_CEILING_DBTP = -1.0
DEFAULT_OFFLINE_MAX_STATIC_STEP_DB = 4.0
OVERHEAD_MAX_INPUT_CUT_DB = -0.75


def apply_arrangement_input_levels(
    stems: list[Any],
    *,
    stage: str = ARRANGEMENT_INPUT_LEVEL_STAGE,
    lufs_fn: Callable[[np.ndarray, int], float],
    peak_fn: Callable[[np.ndarray], float],
    primary_audio_fn: Callable[[Any], np.ndarray],
    band_energy_fn: Callable[[np.ndarray, int], dict[str, float]],
    db_to_amp_fn: Callable[[float], float],
    true_peak_ceiling_dbtp: float = DEFAULT_TRUE_PEAK_CEILING_DBTP,
    max_static_step_db: float = DEFAULT_OFFLINE_MAX_STATIC_STEP_DB,
) -> list[dict[str, Any]]:
    """Apply one arrangement-aware input gain offset per offline stem.

    The planner computes relative musical offsets instead of matching every
    stem to one LUFS target. Positive moves are additionally constrained by a
    4x-oversampled true-peak headroom check.
    """

    if not stems:
        return []

    classifier = ChannelRoleClassifier()
    activity_detector = ActivityDetector(
        min_active_db=-55.0,
        noise_margin_db=8.0,
        attack_hold_sec=0.0,
        release_hold_sec=0.0,
    )
    density_analyzer = ArrangementDensityAnalyzer(max_channels=len(stems))
    section_detector = SectionDetector(hold_time_sec=0.0)
    priority_engine = MixPriorityEngine()
    masking_analyzer = MaskingAnalyzer()
    planner = LevelAutomationPlanner()
    safety_limiter = SafetyLimiter(
        {
            "max_step_db_per_tick": float(max_static_step_db),
            "deadband_db": 0.0,
            "min_confidence_for_live_apply": 0.65,
            "fader_ceiling_db": 999.0,
        }
    )

    roles = {}
    activities = {}
    band_energy = {}
    measurements: dict[int, dict[str, float | str | bool | None]] = {}

    for index, stem in enumerate(stems, start=1):
        channel_id = int(getattr(stem, "channel_id", index))
        sample_rate = int(getattr(stem, "sample_rate", 48_000))
        channel_name = str(getattr(stem, "name", f"Ch {channel_id}"))
        measurement_audio = _measurement_audio(stem, primary_audio_fn)
        full_audio = ensure_stereo(getattr(stem, "audio"))

        measured_lufs = _safe_level(lufs_fn(measurement_audio, sample_rate))
        full_track_lufs = _safe_level(lufs_fn(full_audio, sample_rate))
        rms_db = _rms_db(measurement_audio)
        peak_db = _safe_level(peak_fn(measurement_audio))
        full_peak_dbfs = _safe_level(peak_fn(full_audio))
        true_peak_dbtp = _true_peak_dbtp(full_audio)
        bands = _with_band_aliases(band_energy_fn(measurement_audio, sample_rate))

        roles[channel_id] = classifier.classify(
            channel_id,
            channel_name,
            audio_features=_audio_features_from_bands(bands),
        )
        activities[channel_id] = activity_detector.update(
            channel_id=channel_id,
            rms_db=rms_db,
            peak_db=peak_db,
            short_term_loudness_db=measured_lufs,
            spectral_energy_db=max(bands.values()) if bands else rms_db,
            noise_floor_db=min(-80.0, rms_db - 30.0),
            timestamp=0.0,
        )
        band_energy[channel_id] = bands
        measurements[channel_id] = {
            "measurement_scope": getattr(stem, "bleed_analysis", {}).get("analysis_mode", "full_track"),
            "primary_active_ratio": getattr(stem, "bleed_analysis", {}).get("analysis_active_ratio"),
            "measured_lufs": measured_lufs,
            "full_track_lufs": full_track_lufs,
            "rms_db": rms_db,
            "peak_db": peak_db,
            "full_peak_dbfs": full_peak_dbfs,
            "true_peak_dbtp": true_peak_dbtp,
        }

    density = density_analyzer.analyze(activities, roles, band_energy)
    section = section_detector.update(activities, roles, density, timestamp=0.0)
    priority = priority_engine.analyze(roles, activities, section)
    masking = masking_analyzer.analyze(
        roles,
        activities,
        band_energy,
        primary_channel_id=priority.primary_channel_id,
    )
    if masking:
        density = density_analyzer.analyze(
            activities,
            roles,
            band_energy,
            masking_pressure_on_lead=max(item.masking_score for item in masking),
        )
        section = section_detector.update(activities, roles, density, timestamp=1.0)
        priority = priority_engine.analyze(roles, activities, section)
        masking = masking_analyzer.analyze(
            roles,
            activities,
            band_energy,
            primary_channel_id=priority.primary_channel_id,
        )

    planned = planner.plan(roles, activities, density, section, masking, priority)
    safety = safety_limiter.limit(
        planned,
        roles,
        base_fader_positions=None,
        automation_enabled=True,
        live_apply_enabled=False,
    )
    masking_by_channel = {item.channel_id: item for item in masking}

    report: list[dict[str, Any]] = []
    for index, stem in enumerate(stems, start=1):
        channel_id = int(getattr(stem, "channel_id", index))
        role = roles[channel_id]
        activity = activities[channel_id]
        measurement = measurements[channel_id]
        planned_offset = planned[channel_id]
        planner_safe_offset = float(safety.safe_offsets.get(channel_id, 0.0))
        safe_offset, role_guard_reason = _offline_input_role_guard(
            role.role,
            planner_safe_offset,
        )
        final_gain, peak_reason = _apply_peak_headroom_limit(
            safe_offset,
            float(measurement["true_peak_dbtp"]),
            true_peak_ceiling_dbtp,
        )

        if abs(final_gain) > 1e-6:
            stem.audio = ensure_stereo(stem.audio) * db_to_amp_fn(final_gain)
            stem.track_gain_db += final_gain

        reasons = list(safety.reasons.get(channel_id, []))
        if role_guard_reason:
            reasons.append(role_guard_reason)
        if peak_reason:
            reasons.append(peak_reason)
        masking_source = masking_by_channel.get(channel_id)
        post_true_peak = _sum_level(float(measurement["true_peak_dbtp"]), final_gain)
        report.append(
            {
                "stage": stage,
                "type": "arrangement_aware_input_level",
                "replacement_for": REPLACED_STAGE,
                "channel": channel_id,
                "file": Path(getattr(stem, "path", role.channel_name)).name,
                "role": role.role,
                "role_confidence": round(role.confidence, 4),
                "group": role.group,
                "priority": role.priority,
                "allowed_gain_range_db": [
                    round(role.allowed_gain_range_db[0], 3),
                    round(role.allowed_gain_range_db[1], 3),
                ],
                "section": section.section,
                "section_confidence": round(section.confidence, 4),
                "section_reason": section.reason,
                "arrangement_density_index": round(density.density_index, 4),
                "density_label": density.density_label,
                "density_components": density.components,
                "primary_source_role": priority.primary_source_role,
                "primary_channel": priority.primary_channel_id,
                "active": bool(activity.active),
                "activity_confidence": round(activity.activity_confidence, 4),
                "activity_duration_sec": round(activity.activity_duration_sec, 3),
                "measurement_scope": measurement["measurement_scope"],
                "primary_active_ratio": measurement["primary_active_ratio"],
                "measured_lufs": round(float(measurement["measured_lufs"]), 3),
                "full_track_lufs": round(float(measurement["full_track_lufs"]), 3),
                "rms_db": round(float(measurement["rms_db"]), 3),
                "peak_db": round(float(measurement["peak_db"]), 3),
                "full_peak_dbfs": round(float(measurement["full_peak_dbfs"]), 3),
                "pre_true_peak_dbtp": round(float(measurement["true_peak_dbtp"]), 3),
                "true_peak_ceiling_dbtp": round(float(true_peak_ceiling_dbtp), 3),
                "requested_offset_db": round(planned_offset.requested_offset_db, 3),
                "planner_safe_offset_db": round(planner_safe_offset, 3),
                "safe_offset_db": round(safe_offset, 3),
                "gain_db": round(final_gain, 3),
                "post_lufs": round(_sum_level(float(measurement["measured_lufs"]), final_gain), 3),
                "post_full_track_lufs": round(_sum_level(float(measurement["full_track_lufs"]), final_gain), 3),
                "post_true_peak_dbtp": round(post_true_peak, 3),
                "masking_score": round(masking_source.masking_score, 4) if masking_source else 0.0,
                "masking_action": masking_source.suggested_action if masking_source else "none",
                "masking_gain_reduction_db": (
                    round(masking_source.suggested_gain_reduction_db, 3)
                    if masking_source
                    else 0.0
                ),
                "planner_components": planned_offset.components,
                "safety_reasons": reasons,
                "blocked": channel_id in safety.blocked_offsets,
                "reason": planned_offset.reason,
            }
        )

    return report


def _offline_input_role_guard(role: str, safe_offset_db: float) -> tuple[float, str | None]:
    """Keep ambience mics from being mixed away during the input staging pass.

    Arrangement input staging replaces the old Ayaic pre-phase leveler, so it
    should create headroom and rough musical order, not solve the final drum
    image balance. Overhead audibility is decided later by the output-balance
    stage after EQ, compression, panning, and phase correction are known.
    """

    if role == "overhead" and safe_offset_db < OVERHEAD_MAX_INPUT_CUT_DB:
        return OVERHEAD_MAX_INPUT_CUT_DB, "offline_overhead_air_preservation_floor"
    return safe_offset_db, None


def _measurement_audio(stem: Any, primary_audio_fn: Callable[[Any], np.ndarray]) -> np.ndarray:
    audio = ensure_stereo(primary_audio_fn(stem))
    if audio.size == 0:
        audio = ensure_stereo(getattr(stem, "audio"))
    return audio


def _safe_level(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else -120.0


def _sum_level(level_db: float, gain_db: float) -> float:
    if level_db <= -119.0:
        return level_db
    return level_db + gain_db


def _rms_db(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio).astype(np.float64, copy=False)
    if arr.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
    return amp_to_db(rms)


def _with_band_aliases(bands: dict[str, float]) -> dict[str, float]:
    out = {str(key): float(value) for key, value in (bands or {}).items()}
    if not out:
        return out
    out.setdefault("lf", max(out.get("sub", -100.0), out.get("bass", -100.0)))
    out.setdefault("lmf", max(out.get("low_mid", -100.0), out.get("mid", -100.0)))
    out.setdefault("umf", max(out.get("mid", -100.0), out.get("high_mid", -100.0)))
    out.setdefault("hf", max(out.get("high", -100.0), out.get("air", -100.0)))
    return out


def _audio_features_from_bands(bands: dict[str, float]) -> dict[str, float]:
    if not bands:
        return {}
    centers = {
        "sub": 40.0,
        "bass": 120.0,
        "low_mid": 375.0,
        "mid": 1200.0,
        "high_mid": 3000.0,
        "high": 6000.0,
        "air": 11000.0,
    }
    weights = []
    weighted = []
    for name, center in centers.items():
        if name not in bands:
            continue
        weight = 10.0 ** (float(bands[name]) / 20.0)
        weights.append(weight)
        weighted.append(weight * center)
    if not weights or sum(weights) <= 0.0:
        return {}
    return {"spectral_centroid": sum(weighted) / sum(weights)}


def _apply_peak_headroom_limit(
    safe_offset_db: float,
    true_peak_dbtp: float,
    ceiling_dbtp: float,
) -> tuple[float, str | None]:
    if safe_offset_db <= 0.0:
        return safe_offset_db, None
    headroom_db = ceiling_dbtp - true_peak_dbtp
    if headroom_db >= safe_offset_db:
        return safe_offset_db, None
    limited = max(0.0, headroom_db)
    return limited, "true_peak_headroom_limited"


def _true_peak_dbtp(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    try:
        from scipy import signal

        max_peak = 0.0
        chunk_size = 1_000_000
        overlap = 64
        for channel_index in range(arr.shape[1]):
            channel = np.asarray(arr[:, channel_index], dtype=np.float64)
            for start in range(0, len(channel), chunk_size):
                lo = max(0, start - overlap)
                hi = min(len(channel), start + chunk_size + overlap)
                upsampled = signal.resample_poly(channel[lo:hi], 4, 1)
                if upsampled.size:
                    max_peak = max(max_peak, float(np.max(np.abs(upsampled))))
        return amp_to_db(max_peak)
    except Exception:
        return amp_to_db(float(np.max(np.abs(arr))))
