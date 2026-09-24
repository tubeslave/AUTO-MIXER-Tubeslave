"""Baseline-aware tom transient/body/decay gate for STUDIO/offline use.

Events are detected once from each frozen pre-compression close mic and reused for
baseline and candidates.  The gate asks for a small body-level consistency gain
while protecting attack/body contrast, quiet hits, decay shape and between-hit
spill.  These are technical proxies only; a survivor requires a full routed mix
rerender and human listening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class TomGatePolicy:
    min_body_spread_improvement_db: float = 0.08
    max_attack_body_loss_db: float = 0.35
    max_quiet_hit_loss_db: float = 0.35
    max_decay_shape_change_db: float = 0.45
    max_between_hit_floor_increase_db: float = 0.40
    min_events_for_comparison: int = 16
    max_attack_change_fraction: float = 0.30
    max_release_change_fraction: float = 0.35


def _mono(x: np.ndarray) -> np.ndarray:
    x = as_audio(x)
    return np.mean(x, axis=1).astype(np.float32) if x.ndim == 2 else x


def _frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 12.0,
                  hop_ms: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    x = _mono(x)
    frame = max(4, int(round(frame_ms * sr / 1000)))
    hop = max(1, int(round(hop_ms * sr / 1000)))
    if len(x) < frame:
        return np.zeros(0, np.float64), np.zeros(0, np.float64)
    count = 1 + (len(x) - frame) // hop
    starts = np.arange(count, dtype=np.int64) * hop
    power = x.astype(np.float64) ** 2
    integral = np.concatenate([[0.0], np.cumsum(power)])
    rms = np.sqrt(np.maximum((integral[starts + frame] - integral[starts]) / frame, 1e-30))
    return 20 * np.log10(rms), (starts + frame / 2) / sr


def tom_event_centers(detection_source: np.ndarray, sr: int) -> np.ndarray:
    x = _mono(detection_source)
    if not len(x) or float(np.max(np.abs(x))) < 1e-10:
        return np.zeros(0, dtype=np.int64)
    detector = signal.sosfiltfilt(
        signal.butter(2, [70, 500], btype="bandpass", fs=sr, output="sos"), x
    ).astype(np.float32)
    db, times = _frame_rms_db(detector, sr)
    if not len(db) or float(np.max(db)) < -100:
        return np.zeros(0, dtype=np.int64)
    env = ndimage.gaussian_filter1d(db, 1.5)
    threshold = max(float(np.percentile(env, 98)), float(np.max(env)) - 24.0)
    hop_s = float(np.median(np.diff(times))) if len(times) > 1 else .004
    peaks, _ = signal.find_peaks(
        env,
        height=threshold,
        prominence=6.0,
        distance=max(1, int(round(.180 / hop_s))),
    )
    return np.asarray(np.round(times[peaks] * sr), dtype=np.int64)


def _rms_db(x: np.ndarray, start: int, end: int) -> float | None:
    start = max(0, int(start)); end = min(len(x), int(end))
    if end - start < 8:
        return None
    seg = np.asarray(x[start:end], np.float64)
    power = np.mean(seg * seg, axis=1) if seg.ndim == 2 else seg * seg
    return 10 * np.log10(max(float(np.mean(power)), 1e-30))


def tom_transient_evidence(detection_source: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    detection_source = as_audio(detection_source); audio = as_audio(audio)
    if len(detection_source) != len(audio):
        raise ValueError("tom detection source and measured audio must have equal frames")
    centers = tom_event_centers(detection_source, sr)
    body: list[float] = []
    attack_body: list[float] = []
    body_tail: list[float] = []
    hit_levels: list[float] = []
    floor_segments: list[float] = []
    for idx, center in enumerate(centers):
        attack = _rms_db(audio, center - int(.008 * sr), center + int(.022 * sr))
        sustain = _rms_db(audio, center + int(.035 * sr), center + int(.120 * sr))
        tail = _rms_db(audio, center + int(.140 * sr), center + int(.260 * sr))
        if attack is not None and sustain is not None and tail is not None and np.isfinite(attack + sustain + tail):
            body.append(sustain)
            attack_body.append(attack - sustain)
            body_tail.append(sustain - tail)
            hit_levels.append(attack)
        next_center = centers[idx + 1] if idx + 1 < len(centers) else len(audio)
        lo = center + int(.300 * sr)
        hi = min(next_center - int(.030 * sr), center + int(.500 * sr))
        floor = _rms_db(audio, lo, hi) if hi > lo else None
        if floor is not None and np.isfinite(floor):
            floor_segments.append(floor)
    if not body:
        return {
            "event_count": 0,
            "body_level_spread_db": None,
            "median_attack_body_db": None,
            "median_body_tail_db": None,
            "quiet_hit_level_dbfs": None,
            "between_hit_floor_dbfs": None,
            "measurement_scope": "fixed tom event windows; not listening judgement",
        }
    b = np.asarray(body); ab = np.asarray(attack_body); bt = np.asarray(body_tail); hits = np.asarray(hit_levels)
    return {
        "event_count": int(len(b)),
        "body_level_spread_db": float(np.percentile(b, 90) - np.percentile(b, 10)),
        "median_attack_body_db": float(np.median(ab)),
        "median_body_tail_db": float(np.median(bt)),
        "quiet_hit_level_dbfs": float(np.percentile(hits, 25)),
        "between_hit_floor_dbfs": float(np.median(floor_segments)) if floor_segments else None,
        "measurement_scope": "fixed tom event windows; not listening judgement",
    }


def assess_against_baseline(detection_source: np.ndarray, baseline_audio: np.ndarray,
                            candidate_audio: np.ndarray, sr: int, *,
                            policy: TomGatePolicy | None = None) -> dict:
    policy = policy or TomGatePolicy()
    base = tom_transient_evidence(detection_source, baseline_audio, sr)
    cand = tom_transient_evidence(detection_source, candidate_audio, sr)
    failures: list[str] = []
    keys = ("body_spread_delta_db", "attack_body_delta_db", "body_tail_delta_db",
            "quiet_hit_delta_db", "between_hit_floor_delta_db")
    if base["event_count"] < policy.min_events_for_comparison or cand["event_count"] < policy.min_events_for_comparison:
        failures.append("insufficient_tom_events")
        deltas = {key: None for key in keys}
    else:
        deltas = {
            "body_spread_delta_db": float(cand["body_level_spread_db"] - base["body_level_spread_db"]),
            "attack_body_delta_db": float(cand["median_attack_body_db"] - base["median_attack_body_db"]),
            "body_tail_delta_db": float(cand["median_body_tail_db"] - base["median_body_tail_db"]),
            "quiet_hit_delta_db": float(cand["quiet_hit_level_dbfs"] - base["quiet_hit_level_dbfs"]),
            "between_hit_floor_delta_db": None if base["between_hit_floor_dbfs"] is None or cand["between_hit_floor_dbfs"] is None else float(cand["between_hit_floor_dbfs"] - base["between_hit_floor_dbfs"]),
        }
        if deltas["body_spread_delta_db"] > -policy.min_body_spread_improvement_db:
            failures.append("tom_body_stability_not_improved")
        if deltas["attack_body_delta_db"] < -policy.max_attack_body_loss_db:
            failures.append("tom_attack_body_contrast_reduced")
        if abs(deltas["body_tail_delta_db"]) > policy.max_decay_shape_change_db:
            failures.append("tom_decay_shape_changed")
        if deltas["quiet_hit_delta_db"] < -policy.max_quiet_hit_loss_db:
            failures.append("quiet_tom_hits_reduced")
        if deltas["between_hit_floor_delta_db"] is not None and deltas["between_hit_floor_delta_db"] > policy.max_between_hit_floor_increase_db:
            failures.append("between_hit_spill_increased")
    return {
        "schema": "tom-compression-baseline-assessment-v1",
        "baseline": base,
        "candidate": cand,
        **deltas,
        "technically_survives": not failures,
        "failures": failures,
        "selection_policy": "technical proxy gate only; no musical winner or automatic baseline promotion",
        "requires_full_session_rerender": not failures,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def propose_baseline_aware_candidates(sr: int, baseline_config: CompressorConfig, *,
                                      policy: TomGatePolicy | None = None) -> dict:
    policy = policy or TomGatePolicy(); baseline_config.validate(sr)
    if baseline_config.detector != "rms" or baseline_config.ratio <= 1:
        raise ValueError("tom baseline requires an active RMS compressor")
    variants = (
        ("preserve_attack", 1.25, .85),
        ("tighter_body", .80, .80),
        ("longer_decay", 1.00, 1.25),
    )
    candidates = []
    for identifier, attack_factor, release_factor in variants:
        attack_factor = float(np.clip(attack_factor, 1 - policy.max_attack_change_fraction, 1 + policy.max_attack_change_fraction))
        release_factor = float(np.clip(release_factor, 1 - policy.max_release_change_fraction, 1 + policy.max_release_change_fraction))
        cfg = replace(
            baseline_config,
            attack_ms=max(.01, baseline_config.attack_ms * attack_factor),
            release_ms=max(.01, baseline_config.release_ms * release_factor),
        )
        cfg.validate(sr)
        candidates.append({
            "id": identifier,
            "compressor": asdict(cfg),
            "change_scope": "attack_release_only",
            "baseline_relative": {
                "attack_factor": attack_factor,
                "release_factor": release_factor,
                "threshold_delta_db": 0.0,
                "ratio_delta": 0.0,
            },
            "requires_full_session_rerender": True,
            "requires_human_listening": True,
            "baseline_eligible": False,
        })
    return {
        "schema": "tom-compression-director-baseline-aware-v1",
        "role": "tom_close_mic",
        "reference": {"id": "no_change", "compressor": asdict(baseline_config)},
        "policy": asdict(policy),
        "candidates": candidates,
        "selection_policy": "bounded timing probes only; no ranking or musical winner",
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def render_candidate(x: np.ndarray, sr: int, candidate: dict) -> tuple[np.ndarray, dict]:
    cfg = CompressorConfig(**dict(candidate["compressor"]))
    y, gr = LinkedCompressor(sr, cfg).process(as_audio(x))
    return y, {
        "candidate_id": str(candidate["id"]),
        "whole_track_p95_gr_db": float(np.percentile(gr, 95)) if len(gr) else 0.0,
        "max_gr_db": float(np.max(gr)) if len(gr) else 0.0,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }
