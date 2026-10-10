"""Baseline-aware grouped-snare compression gate for STUDIO/offline use.

Events are detected once from the frozen SN_T/SN_B pre-compression sum and reused
for baseline and every candidate.  The gate asks for a small body-consistency
improvement while protecting crack/attack contrast, quiet/ghost hit audibility
and between-hit bleed.  Metrics are technical proxies only; a survivor still
requires full-session rerender and human listening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, as_audio


@dataclass(frozen=True)
class SnareGatePolicy:
    min_body_spread_improvement_db: float = 0.08
    max_attack_body_loss_db: float = 0.35
    max_ghost_hit_loss_db: float = 0.40
    max_between_hit_bleed_increase_db: float = 0.40
    min_events_for_comparison: int = 36
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
    p = x.astype(np.float64) ** 2
    integral = np.concatenate([[0.0], np.cumsum(p)])
    rms = np.sqrt(np.maximum((integral[starts + frame] - integral[starts]) / frame, 1e-30))
    return 20 * np.log10(rms), (starts + frame / 2) / sr


def snare_event_centers(detection_source: np.ndarray, sr: int) -> np.ndarray:
    db, times = _frame_rms_db(detection_source, sr)
    if not len(db) or float(np.max(db)) < -100:
        return np.zeros(0, dtype=np.int64)
    env = ndimage.gaussian_filter1d(db, 1.15)
    threshold = max(float(np.percentile(env, 72)), float(np.max(env)) - 34.0)
    hop_s = float(np.median(np.diff(times))) if len(times) > 1 else .0025
    peaks, _ = signal.find_peaks(
        env,
        height=threshold,
        prominence=2.3,
        distance=max(1, int(round(.050 / hop_s))),
    )
    return np.asarray(np.round(times[peaks] * sr), dtype=np.int64)


def _rms_db(x: np.ndarray, start: int, end: int) -> float | None:
    start = max(0, int(start)); end = min(len(x), int(end))
    if end - start < 4:
        return None
    seg = np.asarray(x[start:end], np.float64)
    if seg.ndim == 2:
        power = np.mean(seg * seg, axis=1)
    else:
        power = seg * seg
    return 10 * np.log10(max(float(np.mean(power)), 1e-30))


def snare_transient_evidence(detection_source: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    detection_source = as_audio(detection_source); audio = as_audio(audio)
    if len(detection_source) != len(audio):
        raise ValueError('snare detection source and measured audio must have equal frames')
    centers = snare_event_centers(detection_source, sr)
    bodies: list[float] = []
    attack_body: list[float] = []
    attack_levels: list[float] = []
    bleed: list[float] = []
    for idx, center in enumerate(centers):
        attack = _rms_db(audio, center - int(.006 * sr), center + int(.016 * sr))
        body = _rms_db(audio, center + int(.022 * sr), center + int(.085 * sr))
        if attack is not None and body is not None and np.isfinite(attack + body):
            attack_levels.append(attack); bodies.append(body); attack_body.append(attack - body)
        next_center = centers[idx + 1] if idx + 1 < len(centers) else len(audio)
        lo = center + int(.105 * sr)
        hi = min(next_center - int(.018 * sr), center + int(.210 * sr))
        floor = _rms_db(audio, lo, hi) if hi > lo else None
        if floor is not None and np.isfinite(floor):
            bleed.append(floor)
    if not bodies:
        return {
            'event_count': 0, 'body_level_spread_db': None,
            'median_attack_body_db': None, 'ghost_hit_level_dbfs': None,
            'between_hit_bleed_dbfs': None,
            'measurement_scope': 'fixed grouped-snare event windows; not listening judgement or ghost-note transcription',
        }
    body_arr = np.asarray(bodies, np.float64)
    attack_arr = np.asarray(attack_levels, np.float64)
    contrast = np.asarray(attack_body, np.float64)
    # Lower attack quartile is a deliberately simple proxy for quieter/ghost hits.
    ghost_cut = float(np.percentile(attack_arr, 25))
    ghost = attack_arr[attack_arr <= ghost_cut]
    return {
        'event_count': int(len(body_arr)),
        'body_level_spread_db': float(np.percentile(body_arr, 90) - np.percentile(body_arr, 10)),
        'median_attack_body_db': float(np.median(contrast)),
        'ghost_hit_level_dbfs': float(np.median(ghost)) if len(ghost) else float(np.percentile(attack_arr, 25)),
        'between_hit_bleed_dbfs': float(np.median(bleed)) if bleed else None,
        'measurement_scope': 'fixed grouped-snare event windows; quiet-hit proxy is lower attack quartile, not semantic ghost-note transcription',
    }


def assess_against_baseline(detection_source: np.ndarray, baseline_audio: np.ndarray,
                            candidate_audio: np.ndarray, sr: int, *,
                            policy: SnareGatePolicy | None = None) -> dict:
    policy = policy or SnareGatePolicy()
    base = snare_transient_evidence(detection_source, baseline_audio, sr)
    cand = snare_transient_evidence(detection_source, candidate_audio, sr)
    failures: list[str] = []
    if base['event_count'] < policy.min_events_for_comparison or cand['event_count'] < policy.min_events_for_comparison:
        failures.append('insufficient_snare_events')
        deltas = {k: None for k in (
            'body_spread_delta_db', 'attack_body_delta_db',
            'ghost_hit_delta_db', 'between_hit_bleed_delta_db')}
    else:
        deltas = {
            'body_spread_delta_db': float(cand['body_level_spread_db'] - base['body_level_spread_db']),
            'attack_body_delta_db': float(cand['median_attack_body_db'] - base['median_attack_body_db']),
            'ghost_hit_delta_db': float(cand['ghost_hit_level_dbfs'] - base['ghost_hit_level_dbfs']),
            'between_hit_bleed_delta_db': None if base['between_hit_bleed_dbfs'] is None or cand['between_hit_bleed_dbfs'] is None else float(cand['between_hit_bleed_dbfs'] - base['between_hit_bleed_dbfs']),
        }
        if deltas['body_spread_delta_db'] > -policy.min_body_spread_improvement_db:
            failures.append('snare_body_stability_not_improved')
        if deltas['attack_body_delta_db'] < -policy.max_attack_body_loss_db:
            failures.append('snare_attack_body_contrast_reduced')
        if deltas['ghost_hit_delta_db'] < -policy.max_ghost_hit_loss_db:
            failures.append('quiet_snare_hits_reduced')
        if (deltas['between_hit_bleed_delta_db'] is not None
                and deltas['between_hit_bleed_delta_db'] > policy.max_between_hit_bleed_increase_db):
            failures.append('between_hit_bleed_increased')
    return {
        'schema': 'snare-compression-baseline-assessment-v1',
        'baseline': base, 'candidate': cand, **deltas,
        'technically_survives': not failures, 'failures': failures,
        'selection_policy': 'technical proxy gate only; no musical winner or automatic baseline promotion',
        'requires_full_session_rerender': not failures,
        'requires_human_listening': True,
        'baseline_eligible': False,
    }


def propose_baseline_aware_candidates(sr: int, baseline_config: CompressorConfig, *,
                                      policy: SnareGatePolicy | None = None) -> dict:
    policy = policy or SnareGatePolicy(); baseline_config.validate(sr)
    if baseline_config.detector != 'rms' or baseline_config.ratio <= 1:
        raise ValueError('snare baseline requires an active RMS compressor')
    variants = (
        ('preserve_crack', 1.25, .85),
        ('tighter_body', .80, .80),
        ('longer_body', 1.00, 1.25),
    )
    candidates = []
    for identifier, attack_factor, release_factor in variants:
        attack_factor = float(np.clip(
            attack_factor, 1 - policy.max_attack_change_fraction,
            1 + policy.max_attack_change_fraction))
        release_factor = float(np.clip(
            release_factor, 1 - policy.max_release_change_fraction,
            1 + policy.max_release_change_fraction))
        cfg = replace(
            baseline_config,
            attack_ms=max(.01, baseline_config.attack_ms * attack_factor),
            release_ms=max(.01, baseline_config.release_ms * release_factor),
        )
        cfg.validate(sr)
        candidates.append({
            'id': identifier,
            'compressor': asdict(cfg),
            'change_scope': 'attack_release_only',
            'baseline_relative': {
                'attack_factor': attack_factor, 'release_factor': release_factor,
                'threshold_delta_db': 0.0, 'ratio_delta': 0.0,
            },
            'requires_full_session_rerender': True,
            'requires_human_listening': True,
            'baseline_eligible': False,
        })
    return {
        'schema': 'snare-compression-director-baseline-aware-v1',
        'role': 'grouped_snare',
        'reference': {'id': 'no_change', 'compressor': asdict(baseline_config)},
        'policy': asdict(policy), 'candidates': candidates,
        'selection_policy': 'bounded timing probes only; no ranking or musical winner',
        'requires_human_listening': True, 'baseline_eligible': False,
    }
