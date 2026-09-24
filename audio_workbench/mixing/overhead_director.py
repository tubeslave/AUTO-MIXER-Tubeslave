"""No-change-first overhead/cymbal compression gate for STUDIO/offline use.

Compression is proposed only when fixed high-band event evidence demonstrates
inconsistent bright peaks. Stereo image, cymbal decay and inter-event ambience
are protected. A technical survivor is permission for a full routed render and
human A/B only; it is never a musical winner or an audio-baseline promotion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class OverheadPolicy:
    actionable_peak_excess_p95_db: float = 6.0
    actionable_peak_excess_spread_db: float = 3.0
    min_events: int = 18
    min_active_blocks: int = 8
    min_peak_excess_spread_improvement_db: float = 0.15
    max_peak_excess_p95_growth_db: float = 0.50
    max_decay_change_db: float = 0.25
    max_floor_ratio_change_db: float = 0.25
    max_side_mid_change_db: float = 0.12
    max_correlation_change: float = 0.025
    max_macro_change_db: float = 0.25


def _validate_stereo(x: np.ndarray, sr: int) -> np.ndarray:
    x = as_audio(x)
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 2000:
        raise ValueError("sample rate must be an integer >= 2000 Hz")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("overhead evidence requires explicit stereo samples x 2")
    if len(x) < max(512, int(0.5 * sr)):
        raise ValueError("overhead evidence requires at least 500 ms")
    if float(np.max(np.abs(x))) < 1e-10:
        raise ValueError("silent overhead source cannot define dynamics evidence")
    return x


def _band_stereo(x: np.ndarray, sr: int, low_hz: float = 3200.0,
                 high_hz: float = 12000.0) -> np.ndarray:
    nyq = 0.5 * sr
    high = min(high_hz, 0.90 * nyq)
    low = min(low_hz, high * 0.55)
    if high <= max(100.0, low * 1.1):
        return np.zeros_like(x, dtype=np.float32)
    sos = signal.butter(2, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


def _frame_db(mono: np.ndarray, sr: int, frame_ms: float = 35.0,
              hop_ms: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    frame = max(16, round(frame_ms * sr / 1000))
    hop = max(4, round(hop_ms * sr / 1000))
    if len(mono) < frame:
        return np.zeros(0), np.zeros(0)
    starts = np.arange(1 + (len(mono) - frame) // hop, dtype=np.int64) * hop
    p = mono.astype(np.float64) ** 2
    integ = np.concatenate([[0.0], np.cumsum(p)])
    db = 10 * np.log10(np.maximum((integ[starts + frame] - integ[starts]) / frame, 1e-30))
    return db, (starts + frame / 2) / sr


def _rms_db(x: np.ndarray, start: int, end: int) -> float | None:
    start = max(0, int(start)); end = min(len(x), int(end))
    if end - start < 8:
        return None
    y = np.asarray(x[start:end], dtype=np.float64)
    return 10 * np.log10(max(float(np.mean(y * y)), 1e-30))


def _event_centers(detection_source: np.ndarray, sr: int) -> np.ndarray:
    bright = _band_stereo(detection_source, sr)
    mono = np.mean(bright.astype(np.float64), axis=1)
    frame = max(16, round(.012 * sr)); hop = max(4, round(.004 * sr))
    if len(mono) < frame:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(1 + (len(mono) - frame) // hop, dtype=np.int64) * hop
    p = mono * mono; integ = np.concatenate([[0.0], np.cumsum(p)])
    env = 10 * np.log10(np.maximum((integ[starts + frame] - integ[starts]) / frame, 1e-30))
    novelty = ndimage.gaussian_filter1d(env, 1.0) - ndimage.gaussian_filter1d(env, 9.0)
    finite = novelty[np.isfinite(novelty)]
    if not len(finite):
        return np.zeros(0, dtype=np.int64)
    threshold = max(float(np.percentile(finite, 90)), 1.0)
    peaks, _ = signal.find_peaks(
        novelty, height=threshold, prominence=.55,
        distance=max(1, round(.075 * sr / hop)),
    )
    return starts[peaks] + frame // 2


def _stereo_metrics(x: np.ndarray) -> tuple[float, float]:
    l = x[:, 0].astype(np.float64); r = x[:, 1].astype(np.float64)
    mid = .5 * (l + r); side = .5 * (l - r)
    side_mid = 10 * np.log10(max(float(np.mean(side * side)), 1e-30) /
                             max(float(np.mean(mid * mid)), 1e-30))
    denom = np.sqrt(max(float(np.mean(l * l) * np.mean(r * r)), 1e-30))
    corr = float(np.mean(l * r) / denom)
    return float(side_mid), float(np.clip(corr, -1.0, 1.0))


def dynamics_evidence(detection_source: np.ndarray, audio: np.ndarray, sr: int) -> dict:
    """Measure fixed-event cymbal dynamics; all metrics are technical proxies."""
    detection_source = _validate_stereo(detection_source, sr)
    audio = _validate_stereo(audio, sr)
    if len(detection_source) != len(audio):
        raise ValueError("detection source and measured audio must have equal frames")

    bright = _band_stereo(audio, sr)
    bright_mono = np.mean(bright, axis=1)
    full_mono = np.mean(audio, axis=1)
    centers = _event_centers(detection_source, sr)

    frame_db, frame_t = _frame_db(bright_mono, sr)
    active_th = max(float(np.percentile(frame_db, 25)), float(np.max(frame_db)) - 45.0)
    block_ids = np.floor(frame_t / 2.0).astype(int)
    block_means = []
    for bid in np.unique(block_ids):
        vals = frame_db[(block_ids == bid) & (frame_db >= active_th)]
        if len(vals) >= 30:
            block_means.append(float(np.mean(vals)))

    peak_excess, decay_drop, floor_ratio = [], [], []
    for c in centers:
        attack = _rms_db(bright_mono, c - int(.004 * sr), c + int(.020 * sr))
        decay = _rms_db(bright_mono, c + int(.090 * sr), c + int(.260 * sr))
        floor = _rms_db(full_mono, c - int(.120 * sr), c - int(.030 * sr))
        local_a = max(0, c - int(1.0 * sr)); local_b = min(len(bright_mono), c + int(1.0 * sr))
        local = _rms_db(bright_mono, local_a, local_b)
        event_body = _rms_db(full_mono, c + int(.030 * sr), c + int(.180 * sr))
        if attack is None or decay is None or local is None or floor is None or event_body is None:
            continue
        if not np.isfinite(attack + decay + local + floor + event_body):
            continue
        peak_excess.append(attack - local)
        decay_drop.append(attack - decay)
        floor_ratio.append(floor - event_body)

    side_mid, corr = _stereo_metrics(audio)
    if peak_excess:
        p95 = float(np.percentile(peak_excess, 95))
        spread = float(np.percentile(peak_excess, 90) - np.percentile(peak_excess, 10))
    else:
        p95 = spread = None
    return {
        "active_block_count": len(block_means),
        "event_count": len(peak_excess),
        "bright_peak_excess_p95_db": p95,
        "bright_peak_excess_spread_db": spread,
        "median_decay_drop_db": float(np.median(decay_drop)) if decay_drop else None,
        "median_pre_floor_to_body_db": float(np.median(floor_ratio)) if floor_ratio else None,
        "macro_spread_db": (float(np.percentile(block_means, 90) - np.percentile(block_means, 10))
                            if block_means else None),
        "side_mid_db": side_mid,
        "stereo_correlation": corr,
        "measurement_scope": (
            "fixed 3.2-12 kHz onset events + local bright-peak excess + 90-260 ms decay + "
            "pre-event floor/body + 2 s macro blocks + integrated stereo; technical proxy, not listening judgement"
        ),
    }


def baseline_actionability(detection_source: np.ndarray, baseline_audio: np.ndarray, sr: int,
                           *, policy: OverheadPolicy | None = None) -> dict:
    policy = policy or OverheadPolicy()
    e = dynamics_evidence(detection_source, baseline_audio, sr)
    failures: list[str] = []
    if e["event_count"] < policy.min_events:
        failures.append("insufficient_cymbal_events")
    if e["active_block_count"] < policy.min_active_blocks:
        failures.append("insufficient_active_blocks")
    if not failures:
        if e["bright_peak_excess_p95_db"] < policy.actionable_peak_excess_p95_db:
            failures.append("no_actionable_bright_peak_excess")
        if e["bright_peak_excess_spread_db"] < policy.actionable_peak_excess_spread_db:
            failures.append("bright_peak_excess_already_consistent")
    return {
        "schema": "overhead-compression-actionability-v1",
        "evidence": e,
        "actionable": not failures,
        "failures": failures,
        "selection_policy": "no-change first; compress OH only for demonstrated inconsistent bright peaks",
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def assess_against_baseline(detection_source: np.ndarray, baseline_audio: np.ndarray,
                            candidate_audio: np.ndarray, sr: int,
                            *, policy: OverheadPolicy | None = None) -> dict:
    policy = policy or OverheadPolicy()
    base = dynamics_evidence(detection_source, baseline_audio, sr)
    cand = dynamics_evidence(detection_source, candidate_audio, sr)
    failures: list[str] = []
    if min(base["event_count"], cand["event_count"]) < policy.min_events or \
            min(base["active_block_count"], cand["active_block_count"]) < policy.min_active_blocks:
        failures.append("insufficient_overhead_evidence")
        deltas = {k: None for k in [
            "peak_excess_p95_delta_db", "peak_excess_spread_delta_db", "decay_drop_delta_db",
            "floor_ratio_delta_db", "macro_spread_delta_db", "side_mid_delta_db", "correlation_delta"]}
    else:
        deltas = {
            "peak_excess_p95_delta_db": float(cand["bright_peak_excess_p95_db"] - base["bright_peak_excess_p95_db"]),
            "peak_excess_spread_delta_db": float(cand["bright_peak_excess_spread_db"] - base["bright_peak_excess_spread_db"]),
            "decay_drop_delta_db": float(cand["median_decay_drop_db"] - base["median_decay_drop_db"]),
            "floor_ratio_delta_db": float(cand["median_pre_floor_to_body_db"] - base["median_pre_floor_to_body_db"]),
            "macro_spread_delta_db": float(cand["macro_spread_db"] - base["macro_spread_db"]),
            "side_mid_delta_db": float(cand["side_mid_db"] - base["side_mid_db"]),
            "correlation_delta": float(cand["stereo_correlation"] - base["stereo_correlation"]),
        }
        if deltas["peak_excess_spread_delta_db"] > -policy.min_peak_excess_spread_improvement_db:
            failures.append("bright_peak_consistency_not_improved")
        if deltas["peak_excess_p95_delta_db"] > policy.max_peak_excess_p95_growth_db:
            failures.append("bright_peak_excess_upper_tail_worsened")
        if abs(deltas["decay_drop_delta_db"]) > policy.max_decay_change_db:
            failures.append("cymbal_decay_shape_changed")
        if abs(deltas["floor_ratio_delta_db"]) > policy.max_floor_ratio_change_db:
            failures.append("inter_event_floor_relationship_changed")
        if abs(deltas["macro_spread_delta_db"]) > policy.max_macro_change_db:
            failures.append("overhead_macro_dynamics_changed")
        if abs(deltas["side_mid_delta_db"]) > policy.max_side_mid_change_db:
            failures.append("overhead_stereo_side_mid_changed")
        if abs(deltas["correlation_delta"]) > policy.max_correlation_change:
            failures.append("overhead_stereo_correlation_changed")
    return {
        "schema": "overhead-compression-baseline-assessment-v1",
        "baseline": base,
        "candidate": cand,
        **deltas,
        "technically_survives": not failures,
        "failures": failures,
        "requires_full_session_rerender": not failures,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def propose_compression_candidates(sr: int, baseline_config: CompressorConfig,
                                   actionability: dict) -> dict:
    baseline_config.validate(sr)
    if actionability.get("schema") != "overhead-compression-actionability-v1":
        raise ValueError("wrong actionability schema")
    if not actionability.get("actionable", False):
        return {
            "schema": "overhead-compression-proposals-v1",
            "decision": "no_change",
            "candidates": [],
            "reason": list(actionability.get("failures", [])),
            "requires_human_listening": True,
            "baseline_eligible": False,
        }
    timing = {
        "preserve_decay": (baseline_config.attack_ms * 1.25, baseline_config.release_ms * 0.80),
        "balanced": (baseline_config.attack_ms, baseline_config.release_ms * 0.70),
        "catch_bright_peaks": (baseline_config.attack_ms * 0.75, baseline_config.release_ms * 0.60),
    }
    candidates = []
    for name, (attack, release) in timing.items():
        cfg = replace(
            baseline_config,
            attack_ms=float(np.clip(attack, 1.0, 80.0)),
            release_ms=float(np.clip(release, 50.0, 900.0)),
        )
        cfg.validate(sr)
        candidates.append({"id": name, "config": asdict(cfg), "change_scope": "attack_release_only"})
    return {
        "schema": "overhead-compression-proposals-v1",
        "decision": "review_candidates",
        "candidates": candidates,
        "frozen": ["threshold_dbfs", "ratio", "knee_db", "max_gr_db", "detector", "rms_ms", "sidechain_hpf_hz"],
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def render_candidate(audio: np.ndarray, sr: int, config: CompressorConfig) -> tuple[np.ndarray, dict]:
    """Linked stereo render helper; no makeup, clipping, width change or level matching."""
    x = _validate_stereo(audio, sr)
    config.validate(sr)
    y, gr = LinkedCompressor(sr, config).process(x)
    return y, {
        "schema": "overhead-compression-render-v1",
        "config": asdict(config),
        "max_gr_db": float(np.max(gr)) if len(gr) else 0.0,
        "p95_gr_db": float(np.percentile(gr, 95)) if len(gr) else 0.0,
        "linked_stereo": True,
        "makeup_gain_db": 0.0,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }
