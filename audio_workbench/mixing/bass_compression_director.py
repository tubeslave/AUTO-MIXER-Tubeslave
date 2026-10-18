"""Baseline-aware STUDIO bass compression candidate proposer.

This director exists because the generic role candidates can be weaker than an
already-good bass compressor. It treats the current first-stage compressor as a
technical reference and proposes only small, source-aware perturbations around
that reference. No candidate is a musical winner; full routed A/B and human
listening remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, LinkedCompressor, as_audio
from .compression_director import POLICIES, _frame_db
from .compression_director_v11 import _active_sample_mask


@dataclass(frozen=True)
class BassBaselinePolicy:
    max_extra_active_p95_gr_db: float = 0.65
    max_attack_change_fraction: float = 0.30
    max_release_change_fraction: float = 0.45
    max_ratio_change_fraction: float = 0.12
    min_body_spread_improvement_db: float = 0.10
    max_attack_body_loss_db: float = 0.40
    min_events_for_comparison: int = 8


def _validate_baseline(config: CompressorConfig, sr: int) -> None:
    config.validate(sr)
    if config.detector != "rms":
        raise ValueError("bass baseline director currently requires an RMS detector")
    if config.ratio <= 1.0 or config.max_gr_db <= 0:
        raise ValueError("bass baseline compressor must perform compression")


def _macro_event_centers(x: np.ndarray, sr: int) -> np.ndarray:
    x = as_audio(x)
    p = POLICIES["bass"]
    db, times, _ = _frame_db(x, sr, p.frame_ms, p.hop_ms)
    if not len(db) or float(np.max(db)) < -100:
        return np.zeros(0, dtype=np.int64)
    sigma = max(.5, p.smooth_ms / (2.355 * p.hop_ms))
    env = ndimage.gaussian_filter1d(db, sigma)
    threshold = max(-100.0, float(np.max(env)) - 45.0, float(np.percentile(env, 40)))
    hop_s = float(np.median(np.diff(times))) if len(times) > 1 else p.hop_ms / 1000
    peaks, _ = signal.find_peaks(
        env,
        height=threshold,
        prominence=p.prominence_db,
        distance=max(1, int(round((p.min_event_gap_ms / 1000) / hop_s))),
    )
    return np.asarray(np.round(times[peaks] * sr), dtype=np.int64)


def _window_rms_db(x: np.ndarray, start: int, end: int) -> float | None:
    start = max(0, int(start)); end = min(len(x), int(end))
    if end - start < 4:
        return None
    seg = np.asarray(x[start:end], dtype=np.float64)
    if seg.ndim == 2:
        power = np.mean(seg * seg, axis=1)
    else:
        power = seg * seg
    value = float(np.sqrt(np.mean(power) + 1e-30))
    return 20 * np.log10(max(value, 1e-15))


def bass_stability_evidence(detection_source: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    """Measure fixed macro-event body spread and attack/body contrast.

    Event centers are found only from ``detection_source`` and then reused for
    every candidate. These are macro-envelope windows, not note transcription.
    """
    detection_source = as_audio(detection_source)
    audio = as_audio(audio)
    if len(detection_source) != len(audio):
        raise ValueError("detection source and measured audio must have equal frames")
    centers = _macro_event_centers(detection_source, sr)
    body: list[float] = []
    attack_body: list[float] = []
    attack_pre = int(round(.018 * sr)); attack_post = int(round(.018 * sr))
    body_start = int(round(.030 * sr)); body_end = int(round(.115 * sr))
    for center in centers:
        attack = _window_rms_db(audio, center - attack_pre, center + attack_post)
        sustain = _window_rms_db(audio, center + body_start, center + body_end)
        if attack is None or sustain is None or not np.isfinite(attack + sustain):
            continue
        body.append(sustain)
        attack_body.append(attack - sustain)
    if not body:
        return {
            "event_count": 0,
            "body_level_spread_db": None,
            "median_attack_body_db": None,
            "body_level_median_dbfs": None,
            "measurement_scope": "fixed bass macro-event windows; not note transcription or listening judgement",
        }
    body_arr = np.asarray(body, dtype=np.float64)
    contrast = np.asarray(attack_body, dtype=np.float64)
    return {
        "event_count": int(len(body_arr)),
        "body_level_spread_db": float(np.percentile(body_arr, 90) - np.percentile(body_arr, 10)),
        "median_attack_body_db": float(np.median(contrast)),
        "body_level_median_dbfs": float(np.median(body_arr)),
        "measurement_scope": "fixed bass macro-event windows; not note transcription or listening judgement",
    }


def assess_against_baseline(detection_source: np.ndarray, baseline_audio: np.ndarray,
                            candidate_audio: np.ndarray, sr: int,
                            *, policy: BassBaselinePolicy | None = None) -> dict:
    policy = policy or BassBaselinePolicy()
    baseline = bass_stability_evidence(detection_source, baseline_audio, sr)
    candidate = bass_stability_evidence(detection_source, candidate_audio, sr)
    failures: list[str] = []
    if baseline["event_count"] < policy.min_events_for_comparison or candidate["event_count"] < policy.min_events_for_comparison:
        failures.append("insufficient_macro_events")
        body_delta = attack_delta = None
    else:
        body_delta = float(candidate["body_level_spread_db"] - baseline["body_level_spread_db"])
        attack_delta = float(candidate["median_attack_body_db"] - baseline["median_attack_body_db"])
        if body_delta > -policy.min_body_spread_improvement_db:
            failures.append("body_stability_not_improved")
        if attack_delta < -policy.max_attack_body_loss_db:
            failures.append("attack_body_contrast_reduced")
    return {
        "schema": "bass-compression-baseline-assessment-v1",
        "baseline": baseline,
        "candidate": candidate,
        "body_spread_delta_db": body_delta,
        "attack_body_delta_db": attack_delta,
        "technically_survives": not failures,
        "failures": failures,
        "selection_policy": "technical proxy gate only; no musical winner or automatic baseline promotion",
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def propose_baseline_aware_candidates(x: np.ndarray, sr: int,
                                      baseline_config: CompressorConfig,
                                      baseline_active_p95_gr_db: float,
                                      *, policy: BassBaselinePolicy | None = None) -> dict:
    """Propose causal timing probes around an existing bass compressor.

    Real Belye Stai evidence showed that generic lower-GR and stronger-GR families
    both worsened the fixed-window body-spread proxy. This v1 therefore freezes
    threshold, ratio, knee, detector, RMS integration and max-GR, and varies only
    attack/release in small bounded moves. ``no_change`` is an explicit reference.
    """
    x = as_audio(x)
    policy = policy or BassBaselinePolicy()
    _validate_baseline(baseline_config, sr)
    if not np.isfinite(baseline_active_p95_gr_db) or baseline_active_p95_gr_db < 0:
        raise ValueError("baseline_active_p95_gr_db must be finite and non-negative")
    if baseline_active_p95_gr_db > baseline_config.max_gr_db + .25:
        raise ValueError("baseline_active_p95_gr_db is inconsistent with max_gr_db")

    variants = (
        ("faster_recovery", 1.00, .80),
        ("longer_recovery", 1.00, 1.30),
        ("quicker_attack_longer_release", .80, 1.30),
    )
    candidates = []
    for identifier, attack_factor, release_factor in variants:
        attack_factor = float(np.clip(
            attack_factor,
            1 - policy.max_attack_change_fraction,
            1 + policy.max_attack_change_fraction,
        ))
        release_factor = float(np.clip(
            release_factor,
            1 - policy.max_release_change_fraction,
            1 + policy.max_release_change_fraction,
        ))
        config = replace(
            baseline_config,
            attack_ms=max(.01, baseline_config.attack_ms * attack_factor),
            release_ms=max(.01, baseline_config.release_ms * release_factor),
        )
        config.validate(sr)
        candidates.append({
            "id": identifier,
            "compressor": asdict(config),
            "change_scope": "attack_release_only",
            "baseline_relative": {
                "attack_factor": attack_factor,
                "release_factor": release_factor,
                "threshold_delta_db": 0.0,
                "ratio_delta": 0.0,
            },
            "requires_full_session_rerender": True,
            "requires_human_review": True,
            "requires_human_listening": True,
            "baseline_eligible": False,
        })
    return {
        "schema": "bass-compression-director-baseline-aware-v1",
        "role": "bass",
        "reference": {"id": "no_change", "compressor": asdict(baseline_config)},
        "baseline_active_p95_gr_db": float(baseline_active_p95_gr_db),
        "policy": asdict(policy),
        "candidates": candidates,
        "selection_policy": "technical probe family only; no ranking, musical winner or automatic baseline promotion",
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def render_candidate(x: np.ndarray, sr: int, candidate: dict) -> tuple[np.ndarray, dict]:
    source = as_audio(x)
    config = CompressorConfig(**dict(candidate["compressor"]))
    y, gr = LinkedCompressor(sr, config).process(source)
    active = gr[_active_sample_mask(source, sr)]
    return y, {
        "schema": "bass-compression-director-baseline-aware-v1-render",
        "candidate_id": str(candidate["id"]),
        "active_p95_gr_db": float(np.percentile(active, 95)) if len(active) else 0.0,
        "whole_track_p95_gr_db": float(np.percentile(gr, 95)) if len(gr) else 0.0,
        "max_gr_db": float(np.max(gr)) if len(gr) else 0.0,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }
