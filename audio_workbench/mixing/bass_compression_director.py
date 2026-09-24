"""Bass-specific STUDIO compression proposal and technical screening.

The generic Compression Director measures macro timing for many source roles. Bass
needs an additional objective: note/body level stability without flattening the
attack-to-body relationship or holding gain reduction across following notes.

This module never selects a musical winner. It proposes a small bounded family,
calibrates threshold against the actual compressor GR, and can compare a rendered
candidate with an explicit reference using fixed event windows. Passing that screen
only means technically safe enough for full-mix A/B and human listening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, as_audio
from .compression_director import analyze_events
from .compression_director_v11 import calibrate_threshold, timing_censor_evidence


@dataclass(frozen=True)
class BassCompressionPolicy:
    min_attack_ms: float = 5.0
    max_attack_ms: float = 25.0
    min_release_ms: float = 75.0
    max_release_ms: float = 180.0
    knee_db: float = 5.0
    max_gr_db: float = 5.0
    rms_ms: float = 3.0
    event_frame_ms: float = 12.0
    event_hop_ms: float = 6.0
    event_smooth_ms: float = 18.0
    event_gap_ms: float = 70.0
    event_prominence_db: float = 5.0
    body_start_ms: float = 20.0
    body_end_ms: float = 100.0
    attack_lookback_ms: float = 12.0
    attack_lookahead_ms: float = 6.0
    max_body_spread_regression_db: float = 0.05
    max_attack_contrast_loss_db: float = 0.35


POLICY = BassCompressionPolicy()


def _event_windows(x: np.ndarray, sr: int, policy: BassCompressionPolicy = POLICY
                   ) -> list[tuple[int, int, int, int]]:
    """Fixed bass macro-event windows from the input, reused for every candidate."""
    x = as_audio(x)
    if x.ndim != 1:
        raise ValueError("bass event analysis expects mono source")
    frame = max(3, int(round(policy.event_frame_ms * sr / 1000)))
    hop = max(1, int(round(policy.event_hop_ms * sr / 1000)))
    power = ndimage.uniform_filter1d(x.astype(np.float64) ** 2, size=frame)
    db = 10 * np.log10(np.maximum(power[::hop], 1e-30))
    if not len(db) or float(np.max(db)) < -100:
        return []
    sigma = max(.5, policy.event_smooth_ms / (2.355 * policy.event_hop_ms))
    envelope = ndimage.gaussian_filter1d(db, sigma)
    threshold = max(float(np.percentile(envelope, 40)), float(np.max(envelope)) - 45)
    peaks, _ = signal.find_peaks(
        envelope,
        prominence=policy.event_prominence_db,
        height=threshold,
        distance=max(1, int(round(policy.event_gap_ms / policy.event_hop_ms))),
    )
    windows: list[tuple[int, int, int, int]] = []
    for index, peak in enumerate(peaks):
        center = int(peak * hop)
        following = int(peaks[index + 1] * hop) if index + 1 < len(peaks) else len(x)
        attack_start = max(0, center - int(round(policy.attack_lookback_ms * sr / 1000)))
        attack_end = min(len(x), center + int(round(policy.attack_lookahead_ms * sr / 1000)))
        body_start = center + int(round(policy.body_start_ms * sr / 1000))
        body_end = min(
            len(x),
            center + int(round(policy.body_end_ms * sr / 1000)),
            following - int(round(policy.body_start_ms * sr / 1000)),
        )
        if attack_end > attack_start and body_end > body_start:
            windows.append((attack_start, attack_end, body_start, body_end))
    return windows


def measure_event_body(x: np.ndarray, windows: list[tuple[int, int, int, int]]) -> dict:
    x = as_audio(x)
    if x.ndim != 1:
        raise ValueError("bass body measurement expects mono source")
    bodies: list[float] = []
    contrasts: list[float] = []
    for attack_start, attack_end, body_start, body_end in windows:
        attack_peak = float(np.max(np.abs(x[attack_start:attack_end])))
        body_rms = float(np.sqrt(np.mean(x[body_start:body_end].astype(np.float64) ** 2)))
        if body_rms <= 1e-12:
            continue
        bodies.append(float(20 * np.log10(body_rms + 1e-15)))
        contrasts.append(float(20 * np.log10((attack_peak + 1e-15) / (body_rms + 1e-15))))
    if not bodies:
        return {
            "valid_windows": 0,
            "body_level_p90_minus_p10_db": None,
            "median_attack_peak_to_body_rms_db": None,
            "measurement_scope": "fixed input-derived bass macro-event windows",
        }
    return {
        "valid_windows": len(bodies),
        "body_level_p90_minus_p10_db": float(np.percentile(bodies, 90) - np.percentile(bodies, 10)),
        "median_attack_peak_to_body_rms_db": float(np.median(contrasts)),
        "measurement_scope": "fixed input-derived bass macro-event windows; not note transcription or listening judgement",
    }


def _bounded(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def propose_bass_candidates(x: np.ndarray, sr: int, *, tolerance_db: float = .08,
                            policy: BassCompressionPolicy = POLICY) -> dict:
    """Return three calibrated bass candidates, deliberately without ranking."""
    x = as_audio(x)
    if x.ndim != 1:
        raise ValueError("bass compression director expects mono source")
    analysis = analyze_events(x, sr, "bass")
    attack = analysis.get("median_attack_ms") or 35.0
    gap = analysis.get("median_inter_event_ms") or 650.0
    variants = (
        ("preserve_transient", 3.5, .28, .18, 3.8),
        ("balanced", 3.5, .22, .17, 4.1),
        ("control", 4.0, .18, .15, 4.3),
    )
    candidates = []
    for identifier, ratio, attack_factor, release_factor, target_gr in variants:
        config = CompressorConfig(
            threshold_dbfs=-20.0,
            ratio=ratio,
            attack_ms=_bounded(attack * attack_factor, policy.min_attack_ms, policy.max_attack_ms),
            release_ms=_bounded(gap * release_factor, policy.min_release_ms, policy.max_release_ms),
            knee_db=policy.knee_db,
            max_gr_db=policy.max_gr_db,
            detector="rms",
            rms_ms=policy.rms_ms,
        )
        calibrated, calibration = calibrate_threshold(
            x, sr, config, target_gr, tolerance_db=tolerance_db)
        candidates.append({
            "id": identifier,
            "compressor": asdict(calibrated),
            "target_active_p95_gr_db": target_gr,
            "calibration": calibration,
            "requires_human_review": True,
            "requires_human_listening": True,
            "baseline_eligible": False,
            "evidence_basis": "bass macro attack/spacing + short recovery bounds + actual active-p95 GR calibration",
        })
    return {
        "schema": "bass-compression-director-v1",
        "analysis": analysis,
        "timing_censoring": timing_censor_evidence(x, sr, "bass"),
        "event_window_count": len(_event_windows(x, sr, policy)),
        "candidates": candidates,
        "selection_policy": "technical candidate family only; no machine musical winner or baseline promotion",
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def technical_screen(reference: np.ndarray, candidate: np.ndarray,
                     event_source: np.ndarray, sr: int, *,
                     policy: BassCompressionPolicy = POLICY) -> dict:
    """Objective gate for full-mix audition eligibility, not musical acceptance."""
    reference = as_audio(reference); candidate = as_audio(candidate); event_source = as_audio(event_source)
    if reference.ndim != 1 or candidate.ndim != 1 or event_source.ndim != 1:
        raise ValueError("bass technical screen expects mono sources")
    if len(reference) != len(candidate) or len(reference) != len(event_source):
        raise ValueError("bass technical screen sources must have equal length")
    windows = _event_windows(event_source, sr, policy)
    before = measure_event_body(reference, windows)
    after = measure_event_body(candidate, windows)
    failures: list[str] = []
    if before["valid_windows"] < 8 or after["valid_windows"] < 8:
        failures.append("insufficient_event_evidence")
    elif before["body_level_p90_minus_p10_db"] is not None and after["body_level_p90_minus_p10_db"] is not None:
        spread_delta = after["body_level_p90_minus_p10_db"] - before["body_level_p90_minus_p10_db"]
        contrast_delta = (after["median_attack_peak_to_body_rms_db"]
                          - before["median_attack_peak_to_body_rms_db"])
        if spread_delta > policy.max_body_spread_regression_db:
            failures.append("body_stability_regressed")
        if contrast_delta < -policy.max_attack_contrast_loss_db:
            failures.append("attack_contrast_overflattened")
    else:
        failures.append("invalid_event_metrics")
        spread_delta = contrast_delta = None
    if before["valid_windows"] >= 8 and after["valid_windows"] >= 8:
        spread_delta = after["body_level_p90_minus_p10_db"] - before["body_level_p90_minus_p10_db"]
        contrast_delta = (after["median_attack_peak_to_body_rms_db"]
                          - before["median_attack_peak_to_body_rms_db"])
    return {
        "schema": "bass-compression-technical-screen-v1",
        "passed": not failures,
        "failures": failures,
        "reference": before,
        "candidate": after,
        "body_spread_delta_db": None if before["valid_windows"] < 8 else float(spread_delta),
        "attack_contrast_delta_db": None if before["valid_windows"] < 8 else float(contrast_delta),
        "requires_full_mix_rerender": not failures,
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
        "note": "pass means eligible for contextual A/B only, never automatic musical acceptance",
    }
