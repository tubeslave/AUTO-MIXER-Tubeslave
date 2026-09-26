"""Role-aware KEYS / PLAYBACK dynamics gate for STUDIO/offline use.

This module deliberately separates source roles before proposing compression.
KEYS may need transient-safe body control; PLAYBACK is assumed to contain
intentional programmed dynamics and therefore has a stricter no-change-first
policy. All evidence is technical proxy evidence. Musical acceptance remains
human-listening only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, as_audio

Role = Literal["keys", "playback"]


@dataclass(frozen=True)
class KeysPlaybackPolicy:
    # Existing spectral helpers keep these limits for backwards compatibility.
    keys_lowmid_max_db: float = 1.2
    playback_lowmid_max_db: float = 0.9
    playback_ride_max_db: float = 0.6
    keys_width_max_db: float = 0.7

    # Dynamics actionability. These are intentionally conservative.
    keys_actionable_local_spread_db: float = 3.0
    playback_actionable_local_spread_db: float = 3.5
    playback_max_macro_spread_for_compression_db: float = 2.5
    playback_section_ride_review_db: float = 3.0
    min_active_blocks: int = 10
    keys_min_transient_events: int = 18
    playback_min_transient_events: int = 12

    # Candidate-vs-baseline technical gates.
    keys_min_local_spread_improvement_db: float = 0.10
    playback_min_local_spread_improvement_db: float = 0.12
    keys_max_attack_loss_db: float = 0.25
    playback_max_attack_loss_db: float = 0.18
    keys_max_macro_dynamics_change_db: float = 0.30
    playback_max_macro_dynamics_change_db: float = 0.15
    max_side_mid_change_db: float = 0.15
    max_correlation_change: float = 0.03


def _validate_role(role: str) -> Role:
    if role not in {"keys", "playback"}:
        raise ValueError("role must be 'keys' or 'playback'")
    return role  # type: ignore[return-value]


def _validate_stereo(x: np.ndarray, sr: int) -> np.ndarray:
    x = as_audio(x)
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 1000:
        raise ValueError("sample rate must be an integer >= 1000 Hz")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("KEYS / PLAYBACK evidence requires explicit stereo samples x 2")
    if len(x) < max(256, int(0.25 * sr)):
        raise ValueError("KEYS / PLAYBACK evidence requires at least 250 ms")
    if float(np.max(np.abs(x))) < 1e-10:
        raise ValueError("silent KEYS / PLAYBACK source cannot define dynamics evidence")
    return x


def band_env(x: np.ndarray, sr: int, lo: float, hi: float, ms: float = 45) -> np.ndarray:
    """Smoothed mono band envelope used by legacy low-mid helpers."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(1)
    y = signal.sosfiltfilt(signal.butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos"), x)
    return np.sqrt(ndimage.uniform_filter1d(y * y, max(3, int(ms * sr / 1000))) + 1e-12)


def dynamic_lowmid_db(x: np.ndarray, sr: int, lo: float, hi: float, max_db: float) -> np.ndarray:
    """Legacy bounded dynamic low-mid attenuation helper."""
    lm = band_env(x, sr, lo, hi)
    body = band_env(x, sr, 600, 2200)
    ratio = 20 * np.log10((lm + 1e-9) / (body + 1e-9))
    th = float(np.percentile(ratio, 86))
    gr = np.clip((ratio - th) * .36, 0, max_db)
    return -ndimage.gaussian_filter1d(gr.astype("float32"), sigma=max(1, int(.03 * sr)))


def playback_section_ride(frame_db: np.ndarray, active: np.ndarray, max_db: float = .6) -> np.ndarray:
    """Legacy bounded section ride. It is not compressor gain reduction."""
    frame_db = np.asarray(frame_db, dtype=float)
    active = np.asarray(active, dtype=bool)
    if frame_db.ndim != 1 or active.shape != frame_db.shape:
        raise ValueError("frame_db and active must be equal 1-D arrays")
    if not active.any():
        return np.zeros_like(frame_db, dtype=np.float32)
    target = float(np.median(frame_db[active]))
    move = np.where(active, np.clip((target - frame_db) * .22, -max_db, max_db), 0)
    return ndimage.gaussian_filter1d(move.astype("float32"), sigma=2)


def _frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 50., hop_ms: float = 25.) -> tuple[np.ndarray, np.ndarray]:
    mono = np.mean(x.astype(np.float64), axis=1)
    frame = max(8, round(frame_ms * sr / 1000))
    hop = max(1, round(hop_ms * sr / 1000))
    if len(mono) < frame:
        return np.zeros(0), np.zeros(0)
    starts = np.arange(1 + (len(mono) - frame) // hop, dtype=np.int64) * hop
    power = mono * mono
    integ = np.concatenate([[0.], np.cumsum(power)])
    rms = np.sqrt(np.maximum((integ[starts + frame] - integ[starts]) / frame, 1e-30))
    return 20 * np.log10(rms), (starts + frame / 2) / sr


def _rms_db(x: np.ndarray, start: int, end: int) -> float | None:
    start = max(0, int(start)); end = min(len(x), int(end))
    if end - start < 8:
        return None
    s = np.asarray(x[start:end], dtype=np.float64)
    return 10 * np.log10(max(float(np.mean(s * s)), 1e-30))


def _transient_centers(x: np.ndarray, sr: int) -> np.ndarray:
    """Fixed high-band onset proxy; source detection is reused for A/B."""
    mono = np.mean(x.astype(np.float64), axis=1)
    nyq = 0.5 * sr
    lo, hi = 900.0, min(8000.0, 0.90 * nyq)
    if hi <= lo * 1.1:
        return np.zeros(0, dtype=np.int64)
    high = signal.sosfiltfilt(signal.butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos"), mono)
    frame = max(16, round(.012 * sr)); hop = max(4, round(.004 * sr))
    if len(high) < frame:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(1 + (len(high) - frame) // hop, dtype=np.int64) * hop
    p = high * high; integ = np.concatenate([[0.], np.cumsum(p)])
    env = 10 * np.log10(np.maximum((integ[starts + frame] - integ[starts]) / frame, 1e-30))
    novelty = ndimage.gaussian_filter1d(env, 1.0) - ndimage.gaussian_filter1d(env, 7.0)
    threshold = max(float(np.percentile(novelty, 93)), 1.3)
    peaks, _ = signal.find_peaks(novelty, height=threshold, prominence=.7,
                                 distance=max(1, round(.06 * sr / hop)))
    return starts[peaks] + frame // 2


def _stereo_metrics(x: np.ndarray) -> tuple[float, float]:
    l = x[:, 0].astype(np.float64); r = x[:, 1].astype(np.float64)
    mid = (l + r) * .5; side = (l - r) * .5
    side_mid = 10 * np.log10(max(float(np.mean(side * side)), 1e-30) /
                             max(float(np.mean(mid * mid)), 1e-30))
    denom = np.sqrt(max(float(np.mean(l * l) * np.mean(r * r)), 1e-30))
    corr = float(np.mean(l * r) / denom)
    return float(side_mid), float(np.clip(corr, -1., 1.))


def dynamics_evidence(detection_source: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    """Measure role-neutral dynamics using fixed detection events/windows."""
    detection_source = _validate_stereo(detection_source, sr)
    audio = _validate_stereo(audio, sr)
    if len(detection_source) != len(audio):
        raise ValueError("detection source and measured audio must have equal frames")
    db, t = _frame_rms_db(audio, sr)
    global_th = max(float(np.percentile(db, 30)), float(np.max(db)) - 42.)
    block_ids = np.floor(t / 2.0).astype(int)
    local_spread, block_means = [], []
    for bid in np.unique(block_ids):
        vals = db[(block_ids == bid) & (db >= global_th)]
        if len(vals) >= 24:
            local_spread.append(float(np.percentile(vals, 90) - np.percentile(vals, 10)))
            block_means.append(float(np.mean(vals)))
    centers = _transient_centers(detection_source, sr)
    attack_body = []
    mono = np.mean(audio, axis=1)
    for c in centers:
        attack = _rms_db(mono, c - int(.004 * sr), c + int(.016 * sr))
        body = _rms_db(mono, c + int(.025 * sr), c + int(.085 * sr))
        if attack is not None and body is not None and np.isfinite(attack + body):
            attack_body.append(attack - body)
    side_mid, corr = _stereo_metrics(audio)
    return {
        "active_block_count": len(local_spread),
        "transient_event_count": len(attack_body),
        "median_local_spread_db": float(np.median(local_spread)) if local_spread else None,
        "macro_spread_db": (float(np.percentile(block_means, 90) - np.percentile(block_means, 10))
                            if block_means else None),
        "median_attack_body_db": float(np.median(attack_body)) if attack_body else None,
        "side_mid_db": side_mid,
        "stereo_correlation": corr,
        "measurement_scope": "2 s active-frame P90-P10 + fixed high-band transient windows + integrated stereo metrics; technical proxy, not listening judgement",
    }


def baseline_actionability(role: str, detection_source: np.ndarray, baseline_audio: np.ndarray,
                           sr: int, *, policy: KeysPlaybackPolicy | None = None) -> dict:
    """Fail-closed role-specific decision before any compression candidate exists."""
    role = _validate_role(role); policy = policy or KeysPlaybackPolicy()
    e = dynamics_evidence(detection_source, baseline_audio, sr); failures: list[str] = []
    if e["active_block_count"] < policy.min_active_blocks:
        failures.append("insufficient_active_blocks")
    min_events = policy.keys_min_transient_events if role == "keys" else policy.playback_min_transient_events
    if e["transient_event_count"] < min_events:
        failures.append("insufficient_transient_events")
    if not failures:
        local_limit = (policy.keys_actionable_local_spread_db if role == "keys"
                       else policy.playback_actionable_local_spread_db)
        if e["median_local_spread_db"] < local_limit:
            failures.append("no_actionable_local_dynamics_problem")
        if (role == "playback" and e["macro_spread_db"] >
                policy.playback_max_macro_spread_for_compression_db):
            failures.append("playback_macro_dynamics_may_be_programmed")
    if role == "playback" and e["macro_spread_db"] is not None and \
            e["macro_spread_db"] >= policy.playback_section_ride_review_db:
        alternate = "review_section_level_ride_not_compression"
    else:
        alternate = None
    return {
        "schema": "keys-playback-compression-actionability-v1",
        "role": role, "evidence": e, "actionable": not failures, "failures": failures,
        "alternate_action": alternate,
        "selection_policy": "role-aware no-change first; programmed PLAYBACK dynamics get stricter protection",
        "requires_human_listening": True, "baseline_eligible": False,
    }


def assess_against_baseline(role: str, detection_source: np.ndarray, baseline_audio: np.ndarray,
                            candidate_audio: np.ndarray, sr: int,
                            *, policy: KeysPlaybackPolicy | None = None) -> dict:
    role = _validate_role(role); policy = policy or KeysPlaybackPolicy()
    base = dynamics_evidence(detection_source, baseline_audio, sr)
    cand = dynamics_evidence(detection_source, candidate_audio, sr)
    failures: list[str] = []
    min_events = policy.keys_min_transient_events if role == "keys" else policy.playback_min_transient_events
    if min(base["active_block_count"], cand["active_block_count"]) < policy.min_active_blocks or \
            min(base["transient_event_count"], cand["transient_event_count"]) < min_events:
        failures.append("insufficient_role_evidence")
        deltas = {"local_spread_delta_db": None, "attack_body_delta_db": None,
                  "macro_spread_delta_db": None, "side_mid_delta_db": None,
                  "correlation_delta": None}
    else:
        deltas = {
            "local_spread_delta_db": float(cand["median_local_spread_db"] - base["median_local_spread_db"]),
            "attack_body_delta_db": float(cand["median_attack_body_db"] - base["median_attack_body_db"]),
            "macro_spread_delta_db": float(cand["macro_spread_db"] - base["macro_spread_db"]),
            "side_mid_delta_db": float(cand["side_mid_db"] - base["side_mid_db"]),
            "correlation_delta": float(cand["stereo_correlation"] - base["stereo_correlation"]),
        }
        min_improve = (policy.keys_min_local_spread_improvement_db if role == "keys"
                       else policy.playback_min_local_spread_improvement_db)
        max_attack_loss = policy.keys_max_attack_loss_db if role == "keys" else policy.playback_max_attack_loss_db
        max_macro = (policy.keys_max_macro_dynamics_change_db if role == "keys"
                     else policy.playback_max_macro_dynamics_change_db)
        if deltas["local_spread_delta_db"] > -min_improve:
            failures.append(f"{role}_local_stability_not_improved")
        if deltas["attack_body_delta_db"] < -max_attack_loss:
            failures.append(f"{role}_transient_attack_reduced")
        if abs(deltas["macro_spread_delta_db"]) > max_macro:
            failures.append(f"{role}_macro_dynamics_changed")
        if abs(deltas["side_mid_delta_db"]) > policy.max_side_mid_change_db:
            failures.append(f"{role}_stereo_side_mid_changed")
        if abs(deltas["correlation_delta"]) > policy.max_correlation_change:
            failures.append(f"{role}_stereo_correlation_changed")
    return {
        "schema": "keys-playback-compression-baseline-assessment-v1",
        "role": role, "baseline": base, "candidate": cand, **deltas,
        "technically_survives": not failures, "failures": failures,
        "requires_full_session_rerender": not failures,
        "requires_human_listening": True, "baseline_eligible": False,
    }


def propose_compression_candidates(role: str, sr: int, baseline_config: CompressorConfig,
                                   actionability: dict,
                                   *, policy: KeysPlaybackPolicy | None = None) -> dict:
    """Propose bounded timing-only candidates after actionability has passed."""
    role = _validate_role(role); policy = policy or KeysPlaybackPolicy(); baseline_config.validate(sr)
    if actionability.get("role") != role:
        raise ValueError("actionability role does not match requested role")
    reference = {"id": "no_change", "compressor": asdict(baseline_config)}
    if not actionability.get("actionable", False):
        return {
            "schema": "keys-playback-compression-director-v1", "role": role,
            "reference": reference, "policy": asdict(policy), "candidates": [],
            "decision": "no_change", "reason": actionability.get("failures", []),
            "alternate_action": actionability.get("alternate_action"),
            "requires_human_listening": True, "baseline_eligible": False,
        }
    if role == "keys":
        variants = (("preserve_attack", 1.30, .92), ("balanced", 1.0, .82), ("tighten_body", .82, .72))
    else:
        # PLAYBACK changes are intentionally narrower: it is already arranged/programmed audio.
        variants = (("preserve_program", 1.18, .95), ("gentle_control", 1.0, .88))
    candidates = []
    for ident, attack_factor, release_factor in variants:
        cfg = replace(baseline_config,
                      attack_ms=baseline_config.attack_ms * attack_factor,
                      release_ms=baseline_config.release_ms * release_factor)
        cfg.validate(sr)
        candidates.append({
            "id": ident, "compressor": asdict(cfg), "change_scope": "attack_release_only",
            "stereo_link_required": True, "requires_full_session_rerender": True,
            "requires_human_listening": True, "baseline_eligible": False,
        })
    return {
        "schema": "keys-playback-compression-director-v1", "role": role,
        "reference": reference, "policy": asdict(policy), "candidates": candidates,
        "decision": "evaluate_bounded_candidates", "alternate_action": None,
        "requires_human_listening": True, "baseline_eligible": False,
    }


def accept(metrics: dict) -> dict:
    """Legacy whole-mix guard retained for callers using low-mid/ride operations."""
    fail = []
    if abs(metrics.get("mix_loudness_change_lu", 0)) > .35:
        fail.append("loudness_cheat")
    if abs(metrics.get("mix_lowmid_shift_db", 0)) > .45:
        fail.append("excess_lowmid_change")
    if metrics.get("side_energy_gain_db", 0) > 1.0:
        fail.append("excess_width")
    if metrics.get("playback_section_spread_reduction_db", 0) > 3.0:
        fail.append("playback_overflattened")
    return {"accept": not fail, "failures": fail}
