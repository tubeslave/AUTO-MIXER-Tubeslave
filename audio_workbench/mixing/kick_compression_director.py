"""Baseline-aware grouped-kick compression gate for STUDIO/offline use.

The gate detects kick events once from the frozen pre-compression microphone sum,
then measures identical windows in baseline and candidates. It protects initial
attack, quiet-hit audibility and between-hit spill while asking for a small body-
level consistency improvement. Metrics are technical proxies, never a musical
winner; surviving candidates still require full-session A/B and human listening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class KickGatePolicy:
    min_body_spread_improvement_db: float = 0.08
    max_attack_body_loss_db: float = 0.35
    max_quiet_hit_loss_db: float = 0.35
    max_between_hit_floor_increase_db: float = 0.40
    min_events_for_comparison: int = 24
    max_attack_change_fraction: float = 0.30
    max_release_change_fraction: float = 0.40


def _mono(x: np.ndarray) -> np.ndarray:
    x = as_audio(x)
    return np.mean(x, axis=1).astype(np.float32) if x.ndim == 2 else x


def _frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 5.0,
                  hop_ms: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
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


def kick_event_centers(detection_source: np.ndarray, sr: int) -> np.ndarray:
    db, times = _frame_rms_db(detection_source, sr)
    if not len(db) or float(np.max(db)) < -100:
        return np.zeros(0, dtype=np.int64)
    envelope = ndimage.gaussian_filter1d(db, 1.2)
    threshold = max(float(np.percentile(envelope, 75)), float(np.max(envelope)) - 30.0)
    hop_s = float(np.median(np.diff(times))) if len(times) > 1 else .0025
    peaks, _ = signal.find_peaks(
        envelope,
        height=threshold,
        prominence=3.0,
        distance=max(1, int(round(.070 / hop_s))),
    )
    return np.asarray(np.round(times[peaks] * sr), dtype=np.int64)


def _rms_db(x: np.ndarray, start: int, end: int) -> float | None:
    start = max(0, int(start))
    end = min(len(x), int(end))
    if end - start < 4:
        return None
    segment = np.asarray(x[start:end], np.float64)
    if segment.ndim == 2:
        segment = np.mean(segment * segment, axis=1)
    else:
        segment = segment * segment
    return 10 * np.log10(max(float(np.mean(segment)), 1e-30))


def kick_transient_evidence(detection_source: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    detection_source = as_audio(detection_source)
    audio = as_audio(audio)
    if len(detection_source) != len(audio):
        raise ValueError('kick detection source and measured audio must have equal frames')
    centers = kick_event_centers(detection_source, sr)
    body, attack_body, hit_levels, floor_segments = [], [], [], []
    for index, center in enumerate(centers):
        attack = _rms_db(audio, center - int(.008 * sr), center + int(.018 * sr))
        sustain = _rms_db(audio, center + int(.025 * sr), center + int(.095 * sr))
        if attack is not None and sustain is not None and np.isfinite(attack + sustain):
            body.append(sustain)
            attack_body.append(attack - sustain)
            hit_levels.append(attack)
        next_center = centers[index + 1] if index + 1 < len(centers) else len(audio)
        low = center + int(.145 * sr)
        high = min(next_center - int(.025 * sr), center + int(.280 * sr))
        floor = _rms_db(audio, low, high) if high > low else None
        if floor is not None and np.isfinite(floor):
            floor_segments.append(floor)
    if not body:
        return {
            'event_count': 0,
            'body_level_spread_db': None,
            'median_attack_body_db': None,
            'quiet_hit_level_dbfs': None,
            'between_hit_floor_dbfs': None,
            'measurement_scope': 'fixed grouped-kick event windows; not listening judgement',
        }
    body_array = np.asarray(body)
    attack_array = np.asarray(attack_body)
    hit_array = np.asarray(hit_levels)
    return {
        'event_count': int(len(body_array)),
        'body_level_spread_db': float(np.percentile(body_array, 90) - np.percentile(body_array, 10)),
        'median_attack_body_db': float(np.median(attack_array)),
        'quiet_hit_level_dbfs': float(np.percentile(hit_array, 25)),
        'between_hit_floor_dbfs': float(np.median(floor_segments)) if floor_segments else None,
        'measurement_scope': 'fixed grouped-kick event windows; not listening judgement',
    }


def assess_against_baseline(detection_source: np.ndarray, baseline_audio: np.ndarray,
                            candidate_audio: np.ndarray, sr: int, *,
                            policy: KickGatePolicy | None = None) -> dict:
    policy = policy or KickGatePolicy()
    baseline = kick_transient_evidence(detection_source, baseline_audio, sr)
    candidate = kick_transient_evidence(detection_source, candidate_audio, sr)
    failures: list[str] = []
    if baseline['event_count'] < policy.min_events_for_comparison or candidate['event_count'] < policy.min_events_for_comparison:
        failures.append('insufficient_kick_events')
        deltas = {key: None for key in (
            'body_spread_delta_db', 'attack_body_delta_db',
            'quiet_hit_delta_db', 'between_hit_floor_delta_db')}
    else:
        deltas = {
            'body_spread_delta_db': float(candidate['body_level_spread_db'] - baseline['body_level_spread_db']),
            'attack_body_delta_db': float(candidate['median_attack_body_db'] - baseline['median_attack_body_db']),
            'quiet_hit_delta_db': float(candidate['quiet_hit_level_dbfs'] - baseline['quiet_hit_level_dbfs']),
            'between_hit_floor_delta_db': None if baseline['between_hit_floor_dbfs'] is None or candidate['between_hit_floor_dbfs'] is None else float(candidate['between_hit_floor_dbfs'] - baseline['between_hit_floor_dbfs']),
        }
        if deltas['body_spread_delta_db'] > -policy.min_body_spread_improvement_db:
            failures.append('kick_body_stability_not_improved')
        if deltas['attack_body_delta_db'] < -policy.max_attack_body_loss_db:
            failures.append('kick_attack_body_contrast_reduced')
        if deltas['quiet_hit_delta_db'] < -policy.max_quiet_hit_loss_db:
            failures.append('quiet_kick_hits_reduced')
        if deltas['between_hit_floor_delta_db'] is not None and deltas['between_hit_floor_delta_db'] > policy.max_between_hit_floor_increase_db:
            failures.append('between_hit_spill_increased')
    return {
        'schema': 'kick-compression-baseline-assessment-v1',
        'baseline': baseline,
        'candidate': candidate,
        **deltas,
        'technically_survives': not failures,
        'failures': failures,
        'selection_policy': 'technical proxy gate only; no musical winner or automatic baseline promotion',
        'requires_full_session_rerender': not failures,
        'requires_human_listening': True,
        'baseline_eligible': False,
    }


def propose_baseline_aware_candidates(sr: int, baseline_config: CompressorConfig, *,
                                      policy: KickGatePolicy | None = None) -> dict:
    policy = policy or KickGatePolicy()
    baseline_config.validate(sr)
    if baseline_config.detector != 'rms' or baseline_config.ratio <= 1:
        raise ValueError('kick baseline requires an active RMS compressor')
    variants = (
        ('more_punch', 1.25, .85),
        ('tighter_body', .80, .75),
        ('longer_body', 1.00, 1.25),
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
            'id': identifier,
            'compressor': asdict(config),
            'change_scope': 'attack_release_only',
            'baseline_relative': {
                'attack_factor': attack_factor,
                'release_factor': release_factor,
                'threshold_delta_db': 0.0,
                'ratio_delta': 0.0,
            },
            'requires_full_session_rerender': True,
            'requires_human_listening': True,
            'baseline_eligible': False,
        })
    return {
        'schema': 'kick-compression-director-baseline-aware-v1',
        'role': 'grouped_kick',
        'reference': {'id': 'no_change', 'compressor': asdict(baseline_config)},
        'policy': asdict(policy),
        'candidates': candidates,
        'selection_policy': 'bounded timing probes only; no ranking or musical winner',
        'requires_human_listening': True,
        'baseline_eligible': False,
    }


def render_candidate(x: np.ndarray, sr: int, candidate: dict) -> tuple[np.ndarray, dict]:
    config = CompressorConfig(**dict(candidate['compressor']))
    y, gain_reduction = LinkedCompressor(sr, config).process(as_audio(x))
    return y, {
        'candidate_id': str(candidate['id']),
        'whole_track_p95_gr_db': float(np.percentile(gain_reduction, 95)) if len(gain_reduction) else 0.0,
        'max_gr_db': float(np.max(gain_reduction)) if len(gain_reduction) else 0.0,
        'requires_human_listening': True,
        'baseline_eligible': False,
    }
