"""Deterministic STUDIO compression candidate proposer.

This module does not choose a musical winner. It measures macro event timing and
returns a small, bounded set of CompressorConfig candidates for audition. Every
candidate is subjective and therefore requires human listening before promotion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal

from ..mastering.analyzer import true_peak_dbtp
from .compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class RolePolicy:
    ratio: float
    max_gr_db: float
    knee_db: float
    rms_ms: float
    fallback_attack_ms: float
    fallback_release_ms: float
    attack_bounds_ms: tuple[float, float]
    release_bounds_ms: tuple[float, float]
    frame_ms: float
    hop_ms: float
    smooth_ms: float
    min_event_gap_ms: float
    prominence_db: float


POLICIES = {
    "vocal": RolePolicy(3.2, 5.0, 6.0, 3.0, 10, 120, (3, 80), (45, 600), 20, 10, 35, 160, 4),
    "bass": RolePolicy(3.0, 4.0, 6.0, 3.0, 15, 180, (4, 80), (45, 800), 12, 6, 18, 70, 5),
    "kick": RolePolicy(2.0, 2.5, 4.0, 1.0, 20, 100, (3, 60), (30, 450), 6, 3, 6, 80, 7),
    "snare": RolePolicy(2.2, 3.0, 4.0, 1.0, 12, 100, (2, 50), (30, 450), 6, 3, 6, 80, 7),
    "toms": RolePolicy(2.0, 2.5, 4.0, 1.2, 20, 160, (3, 80), (40, 650), 8, 4, 8, 90, 7),
    "guitar": RolePolicy(1.8, 2.5, 6.0, 3.0, 25, 180, (5, 100), (50, 800), 16, 8, 24, 120, 4),
    "keys": RolePolicy(1.6, 2.0, 6.0, 3.0, 30, 220, (5, 120), (60, 900), 20, 10, 30, 140, 4),
    "playback": RolePolicy(1.5, 1.5, 6.0, 3.0, 30, 230, (5, 120), (60, 900), 20, 10, 30, 140, 4),
    "cymbals": RolePolicy(1.3, 1.0, 6.0, 3.0, 35, 240, (8, 140), (70, 1000), 12, 6, 18, 100, 5),
}


def _linked_power(x: np.ndarray) -> np.ndarray:
    power = x.astype(np.float64) ** 2
    return power if x.ndim == 1 else np.mean(power, axis=1)


def _frame_db(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(x):
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    frame = max(1, int(round(sr * frame_ms / 1000)))
    hop = max(1, int(round(sr * hop_ms / 1000)))
    starts = np.arange(0, len(x), hop, dtype=int)
    ends = np.minimum(starts + frame, len(x))
    counts = ends - starts
    power = _linked_power(x)
    integral = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    frame_power = (integral[ends] - integral[starts]) / counts
    db = 10 * np.log10(np.maximum(frame_power, 1e-30))
    times = (starts + counts * .5) / sr
    return db, times, counts


def analyze_events(x: np.ndarray, sr: int, role: str) -> dict:
    """Measure macro envelope timing, not note/onset transcription.

    Peak spacing and prominence are intentionally role-bounded so bleed and
    waveform-scale ripples do not masquerade as thousands of musical events.
    """
    x = as_audio(x)
    if role not in POLICIES:
        raise ValueError(f"unknown compression role: {role}")
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 1000:
        raise ValueError("sample rate must be an integer >= 1000 Hz")
    p = POLICIES[role]
    db, times, _ = _frame_db(x, sr, p.frame_ms, p.hop_ms)
    if not len(db) or float(np.max(db)) < -100:
        return {
            "status": "no_active_audio", "event_count": 0, "active_duration_s": 0.0,
            "event_density_hz": 0.0, "median_attack_ms": None,
            "median_recovery_ms": None, "median_inter_event_ms": None,
            "active_p90_dbfs": None, "crest_db": None,
        }
    sigma = max(.5, p.smooth_ms / (2.355 * p.hop_ms))
    envelope_db = ndimage.gaussian_filter1d(db, sigma)
    active_threshold = max(-100.0, float(np.max(envelope_db)) - 45.0,
                           float(np.percentile(envelope_db, 40)))
    active = envelope_db >= active_threshold
    active_times = times[active]
    active_duration = (float(active_times[-1] - active_times[0] + p.frame_ms / 1000)
                       if len(active_times) else 0.0)
    hop_s = float(np.median(np.diff(times))) if len(times) > 1 else p.hop_ms / 1000
    peaks, _ = signal.find_peaks(
        envelope_db,
        height=active_threshold,
        prominence=p.prominence_db,
        distance=max(1, int(round((p.min_event_gap_ms / 1000) / hop_s))),
    )
    attacks: list[float] = []
    recoveries: list[float] = []
    max_back = max(1, int(round(.20 / hop_s)))
    max_forward = max(1, int(round(1.20 / hop_s)))
    for index, peak in enumerate(peaks):
        six_down = envelope_db[peak] - 6.0
        left_limit = max(0, peak - max_back)
        left = int(peak)
        while left > left_limit and envelope_db[left] > six_down:
            left -= 1
        attacks.append((peak - left) * hop_s * 1000)
        next_peak_limit = int(peaks[index + 1] - 1) if index + 1 < len(peaks) else len(envelope_db) - 1
        right_limit = min(len(envelope_db) - 1, peak + max_forward, next_peak_limit)
        right = int(peak)
        while right < right_limit and envelope_db[right] > six_down:
            right += 1
        recoveries.append((right - peak) * hop_s * 1000)
    gaps_ms = np.diff(times[peaks]) * 1000 if len(peaks) > 1 else np.zeros(0)
    rms = float(np.sqrt(np.mean(_linked_power(x)) + 1e-30))
    sample_peak = float(np.max(np.abs(x)) + 1e-30)
    return {
        "status": "event_evidence" if len(peaks) >= 3 else "sparse_event_fallback",
        "event_count": int(len(peaks)),
        "active_duration_s": active_duration,
        "event_density_hz": float(len(peaks) / active_duration) if active_duration else 0.0,
        "median_attack_ms": float(np.median(attacks)) if attacks else None,
        "median_recovery_ms": float(np.median(recoveries)) if recoveries else None,
        "median_inter_event_ms": float(np.median(gaps_ms)) if len(gaps_ms) else None,
        "active_p90_dbfs": float(np.percentile(envelope_db[active], 90)) if np.any(active) else None,
        "crest_db": float(20 * np.log10(sample_peak / rms)),
        "active_threshold_dbfs": float(active_threshold),
        "measurement_scope": "role-smoothed macro RMS envelope; not transcription or listening judgement",
    }


def _bounded(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


def _threshold_for_target(anchor_db: float, ratio: float, target_gr_db: float) -> float:
    slope = 1 - 1 / ratio
    return float(np.clip(anchor_db - target_gr_db / slope, -80, 6))


def propose_candidates(x: np.ndarray, sr: int, role: str) -> dict:
    """Return three auditable audition candidates; deliberately no winner/ranking."""
    analysis = analyze_events(x, sr, role)
    policy = POLICIES[role]
    measured_attack = (analysis["median_attack_ms"] if analysis["median_attack_ms"] is not None
                       else policy.fallback_attack_ms)
    recovery = (analysis["median_recovery_ms"] if analysis["median_recovery_ms"] is not None
                else policy.fallback_release_ms)
    gap = (analysis["median_inter_event_ms"] if analysis["median_inter_event_ms"] is not None
           else policy.fallback_release_ms * 2)
    release_anchor = _bounded(.55 * recovery + .30 * gap, policy.release_bounds_ms)
    anchor_db = analysis["active_p90_dbfs"] if analysis["active_p90_dbfs"] is not None else -18.0
    variants = [
        ("preserve_transient", 1.50, 1.30, .45, 1.00),
        ("balanced", .85, 1.00, .65, 1.00),
        ("control", .40, .65, .85, 1.15),
    ]
    candidates = []
    for identifier, attack_factor, release_factor, gr_fraction, ratio_factor in variants:
        ratio = float(np.clip(policy.ratio * ratio_factor, 1.05, 8.0))
        target_gr = float(min(3.5, policy.max_gr_db * gr_fraction))
        config = CompressorConfig(
            threshold_dbfs=_threshold_for_target(anchor_db, ratio, target_gr),
            ratio=ratio,
            attack_ms=_bounded(measured_attack * attack_factor, policy.attack_bounds_ms),
            release_ms=_bounded(release_anchor * release_factor, policy.release_bounds_ms),
            knee_db=policy.knee_db,
            max_gr_db=policy.max_gr_db,
            detector="rms",
            rms_ms=policy.rms_ms,
        )
        config.validate(sr)
        candidates.append({
            "id": identifier,
            "compressor": asdict(config),
            "requested_static_gr_db": target_gr,
            "requires_human_review": True,
            "requires_human_listening": True,
            "baseline_eligible": False,
            "evidence_basis": "measured macro attack/recovery/spacing plus role safety bounds",
        })
    return {
        "schema": "compression-director-v1",
        "role": role,
        "analysis": analysis,
        "candidates": candidates,
        "selection_policy": "candidate set only; no machine winner or baseline promotion",
        "requires_human_review": True,
        "baseline_eligible": False,
    }


def config_from_candidate(candidate: dict) -> CompressorConfig:
    config = CompressorConfig(**dict(candidate["compressor"]))
    return config


def render_core_candidate(x: np.ndarray, sr: int, candidate: dict) -> tuple[np.ndarray, dict]:
    """Render compressor-only candidate; no rider, makeup, clipper or hidden match."""
    source = as_audio(x)
    config = config_from_candidate(candidate)
    y, gr = LinkedCompressor(sr, config).process(source)
    return y, {
        "schema": "compression-director-v1-render",
        "candidate_id": str(candidate["id"]),
        "max_gr_db": float(np.max(gr)) if len(gr) else 0.0,
        "p95_gr_db": float(np.percentile(gr, 95)) if len(gr) else 0.0,
        "compression_duty_fraction": float(np.mean(gr > .1)) if len(gr) else 0.0,
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def render_candidate(x: np.ndarray, sr: int, role: str, candidate: dict, *,
                     enable_ride: bool = False) -> tuple[np.ndarray, dict]:
    """Render through the accepted dynamics adapter without changing its defaults."""
    from . import dynamics  # lazy import: director remains optional, no circular default path
    config = config_from_candidate(candidate)
    y, report = dynamics.apply(
        x, sr, role, compressor_config=config, enable_ride=enable_ride,
    )
    report = dict(report)
    report.update({
        "compression_director_candidate_id": str(candidate["id"]),
        "compression_director_schema": "compression-director-v1",
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
    })
    return y, report


def _match_frames(x: np.ndarray, sr: int, hop_s: float = .02
                  ) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(sr * hop_s)))
    starts = np.arange(0, len(x), hop, dtype=int)
    if not len(starts):
        return np.zeros(0), np.zeros(0, dtype=int)
    ends = np.minimum(starts + hop, len(x))
    counts = ends - starts
    power = _linked_power(x)
    integral = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    return (integral[ends] - integral[starts]) / counts, counts


def level_match_evidence(reference: np.ndarray, candidate: np.ndarray, sr: int, *,
                         ceiling_dbtp: float | None = None,
                         tolerance_db: float = .05) -> dict:
    """Report exact gain needed to match RAW-active RMS and any ceiling conflict.

    With ceiling_dbtp=None this is internal-float evidence only: no arbitrary
    source-level true-peak rule is invented. Passing a ceiling models a specific
    audition/export constraint and explicitly reports residual mismatch.
    """
    reference = as_audio(reference)
    candidate = as_audio(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    if not np.isfinite(tolerance_db) or tolerance_db < 0:
        raise ValueError("tolerance_db must be finite and non-negative")
    if ceiling_dbtp is not None and not np.isfinite(ceiling_dbtp):
        raise ValueError("ceiling_dbtp must be finite or None")
    ref_power, counts = _match_frames(reference, sr)
    cand_power, _ = _match_frames(candidate, sr)
    if not len(ref_power) or float(np.max(ref_power)) <= 1e-20:
        return {
            "status": "no_active_reference", "required_gain_db": None,
            "applied_gain_db": 0.0, "residual_match_error_db": None,
            "headroom_limited": False, "match_passed": None,
            "ceiling_dbtp": ceiling_dbtp,
        }
    ref_db = 10 * np.log10(np.maximum(ref_power, 1e-30))
    active_threshold = max(-100.0, float(np.max(ref_db)) - 40.0, float(np.percentile(ref_db, 55)))
    active = ref_db >= active_threshold
    ref_rms_db = float(10 * np.log10(np.average(ref_power[active], weights=counts[active])))
    cand_rms_db = float(10 * np.log10(np.average(cand_power[active], weights=counts[active])))
    required = ref_rms_db - cand_rms_db
    candidate_tp = true_peak_dbtp(candidate)
    allowed_gain = None if ceiling_dbtp is None else float(ceiling_dbtp - candidate_tp)
    applied = required if allowed_gain is None else min(required, allowed_gain)
    residual = cand_rms_db + applied - ref_rms_db
    return {
        "status": "measured",
        "reference_active_rms_dbfs": ref_rms_db,
        "candidate_active_rms_dbfs": cand_rms_db,
        "required_gain_db": float(required),
        "applied_gain_db": float(applied),
        "residual_match_error_db": float(residual),
        "candidate_true_peak_dbtp": float(candidate_tp),
        "ceiling_dbtp": ceiling_dbtp,
        "maximum_gain_allowed_by_ceiling_db": allowed_gain,
        "headroom_limited": bool(allowed_gain is not None and applied < required - 1e-9),
        "match_passed": bool(abs(residual) <= tolerance_db),
        "measurement_scope": "fixed reference-active 20 ms RMS mask",
    }


def audition_pair_plan(reference: np.ndarray, candidate: np.ndarray, sr: int, *,
                       ceiling_dbtp: float = -3.0) -> dict:
    """Plan a fair A/B: exact active-RMS match, then one common headroom trim.

    The common trim prevents the export ceiling from changing the A/B loudness
    relationship. No limiting/clipping is introduced by this planner.
    """
    if not np.isfinite(ceiling_dbtp):
        raise ValueError("ceiling_dbtp must be finite")
    match = level_match_evidence(reference, candidate, sr, ceiling_dbtp=None)
    if match["status"] != "measured":
        return {
            "status": match["status"], "candidate_match_gain_db": None,
            "common_trim_db": 0.0, "reference_gain_db": 0.0,
            "candidate_total_gain_db": None, "ceiling_dbtp": ceiling_dbtp,
            "match_passed": None,
        }
    match_gain = float(match["required_gain_db"])
    reference_tp = true_peak_dbtp(reference)
    candidate_tp_matched = true_peak_dbtp(candidate) + match_gain
    common_trim = float(min(0.0, ceiling_dbtp - max(reference_tp, candidate_tp_matched)))
    return {
        "status": "measured",
        "candidate_match_gain_db": match_gain,
        "common_trim_db": common_trim,
        "reference_gain_db": common_trim,
        "candidate_total_gain_db": match_gain + common_trim,
        "reference_true_peak_after_dbtp": reference_tp + common_trim,
        "candidate_true_peak_after_dbtp": candidate_tp_matched + common_trim,
        "ceiling_dbtp": float(ceiling_dbtp),
        "residual_match_error_db": float(match["residual_match_error_db"]),
        "match_passed": bool(match["match_passed"]),
        "method": "exact fixed-mask active-RMS match followed by common pair trim; no limiter",
    }
