"""Style- and tempo-aware offline compression for the Ayaic render pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import amp_to_db, db_to_amp, ensure_stereo


CONTEXTUAL_COMPRESSION_STAGE = "contextual_compression"
DEFAULT_CONTEXTUAL_COMPRESSION_STYLE = "live_pop_rock"
DEFAULT_CONTEXTUAL_COMPRESSION_BPM = 120.0


RoleFn = Callable[[str], str]
AudioFn = Callable[[Any], np.ndarray]
LufsFn = Callable[[np.ndarray, int], float]
PeakFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CompressionMetrics:
    lufs: float
    rms_db: float
    peak_db: float
    crest_factor_db: float
    dynamic_range_db: float
    activity_ratio: float
    transient_density_per_sec: float
    transient_strength_db: float
    spectral_flux: float


ROLE_INTENTS: dict[str, dict[str, float | str]] = {
    "lead_vocal": {
        "target_gr": 3.2,
        "max_gr": 7.0,
        "ratio": 3.0,
        "attack_ms": 14.0,
        "release_ms": 160.0,
        "release_divisor": 3.0,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 4.0,
    },
    "backing_vocal": {
        "target_gr": 3.8,
        "max_gr": 8.0,
        "ratio": 3.4,
        "attack_ms": 12.0,
        "release_ms": 190.0,
        "release_divisor": 3.0,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 5.0,
    },
    "kick": {
        "target_gr": 2.4,
        "max_gr": 6.0,
        "ratio": 4.0,
        "attack_ms": 22.0,
        "release_ms": 95.0,
        "release_divisor": 8.0,
        "knee_db": 2.0,
        "detector": "peak",
        "transient_loss_limit": 2.2,
    },
    "snare": {
        "target_gr": 2.8,
        "max_gr": 6.5,
        "ratio": 4.0,
        "attack_ms": 10.0,
        "release_ms": 120.0,
        "release_divisor": 6.0,
        "knee_db": 2.5,
        "detector": "peak",
        "transient_loss_limit": 2.5,
    },
    "tom": {
        "target_gr": 2.2,
        "max_gr": 6.0,
        "ratio": 3.6,
        "attack_ms": 14.0,
        "release_ms": 150.0,
        "release_divisor": 5.0,
        "knee_db": 2.5,
        "detector": "peak",
        "transient_loss_limit": 2.8,
    },
    "bass": {
        "target_gr": 3.8,
        "max_gr": 8.0,
        "ratio": 4.0,
        "attack_ms": 32.0,
        "release_ms": 240.0,
        "release_divisor": 2.5,
        "knee_db": 4.0,
        "detector": "rms",
        "transient_loss_limit": 5.5,
    },
    "guitar": {
        "target_gr": 1.6,
        "max_gr": 5.0,
        "ratio": 2.4,
        "attack_ms": 22.0,
        "release_ms": 190.0,
        "release_divisor": 3.5,
        "knee_db": 4.0,
        "detector": "rms",
        "transient_loss_limit": 5.0,
    },
    "keys": {
        "target_gr": 1.5,
        "max_gr": 5.0,
        "ratio": 2.2,
        "attack_ms": 28.0,
        "release_ms": 230.0,
        "release_divisor": 3.0,
        "knee_db": 4.0,
        "detector": "rms",
        "transient_loss_limit": 5.0,
    },
    "pad": {
        "target_gr": 1.2,
        "max_gr": 4.0,
        "ratio": 2.0,
        "attack_ms": 38.0,
        "release_ms": 320.0,
        "release_divisor": 2.0,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 6.0,
    },
    "playback": {
        "target_gr": 0.9,
        "max_gr": 3.0,
        "ratio": 1.8,
        "attack_ms": 35.0,
        "release_ms": 300.0,
        "release_divisor": 2.5,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 6.0,
    },
    "overhead": {
        "target_gr": 0.45,
        "max_gr": 1.8,
        "ratio": 1.35,
        "attack_ms": 48.0,
        "release_ms": 430.0,
        "release_divisor": 2.0,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 1.2,
    },
    "room": {
        "target_gr": 0.8,
        "max_gr": 2.8,
        "ratio": 1.7,
        "attack_ms": 42.0,
        "release_ms": 420.0,
        "release_divisor": 1.5,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 2.0,
    },
    "unknown": {
        "target_gr": 0.6,
        "max_gr": 2.0,
        "ratio": 1.5,
        "attack_ms": 35.0,
        "release_ms": 260.0,
        "release_divisor": 3.0,
        "knee_db": 5.0,
        "detector": "rms",
        "transient_loss_limit": 4.0,
    },
}


STYLE_MODIFIERS: dict[str, dict[str, float]] = {
    "live_pop_rock": {
        "target_scale": 1.0,
        "ratio_scale": 1.0,
        "attack_scale": 1.0,
        "release_scale": 1.0,
    },
    "rock": {
        "target_scale": 1.18,
        "ratio_scale": 1.12,
        "attack_scale": 0.9,
        "release_scale": 0.88,
    },
    "dense": {
        "target_scale": 1.25,
        "ratio_scale": 1.08,
        "attack_scale": 0.95,
        "release_scale": 0.9,
    },
    "ballad": {
        "target_scale": 0.78,
        "ratio_scale": 0.9,
        "attack_scale": 1.18,
        "release_scale": 1.28,
    },
    "transparent": {
        "target_scale": 0.62,
        "ratio_scale": 0.82,
        "attack_scale": 1.25,
        "release_scale": 1.18,
    },
}


def apply_contextual_compression(
    stems: list[Any],
    *,
    style: str = DEFAULT_CONTEXTUAL_COMPRESSION_STYLE,
    bpm: float | None = None,
    report_only: bool = False,
    role_fn: RoleFn | None = None,
    primary_audio_fn: AudioFn | None = None,
    lufs_fn: LufsFn | None = None,
    peak_fn: PeakFn | None = None,
) -> dict[str, Any]:
    """Apply role-aware target-GR compression and return an explainable report."""
    style_key = _style_key(style)
    if not stems:
        return {
            "enabled": False,
            "applied": False,
            "stage": CONTEXTUAL_COMPRESSION_STAGE,
            "reason": "no_stems",
            "channels": [],
        }

    sample_rate = int(stems[0].sample_rate)
    tempo_bpm, tempo_confidence, tempo_reason = _resolve_tempo(stems, bpm, sample_rate)
    channels: list[dict[str, Any]] = []
    applied_count = 0
    applied_band_count = 0

    for stem in stems:
        role = _normalize_role(role_fn(stem.name) if role_fn else _fallback_role(stem.name))
        primary_audio = primary_audio_fn(stem) if primary_audio_fn else stem.audio
        metrics = _compression_metrics(
            primary_audio,
            sample_rate=int(stem.sample_rate),
            lufs_fn=lufs_fn,
            peak_fn=peak_fn,
        )
        before_full_lufs = _call_lufs(lufs_fn, stem.audio, int(stem.sample_rate))
        before_full_peak = _call_peak(peak_fn, stem.audio)
        plan = _plan_channel_compression(role, metrics, style_key, tempo_bpm)
        if plan["bypass"]:
            channels.append(
                {
                    "stage": CONTEXTUAL_COMPRESSION_STAGE,
                    "channel": int(stem.channel_id),
                    "file": Path(stem.path).name,
                    "role": role,
                    "enabled": False,
                    "applied": False,
                    "reason": plan["reason"],
                    "metrics": _metrics_report(metrics),
                    "style": style_key,
                    "tempo_bpm": round(float(tempo_bpm), 2),
                    "tempo_confidence": round(float(tempo_confidence), 3),
                }
            )
            continue

        original_plan = dict(plan)
        revision_history: list[dict[str, Any]] = []
        for attempt in range(4):
            params = _solve_threshold_for_target(primary_audio, int(stem.sample_rate), plan)
            compressed, compression_stats = _compress_audio(stem.audio, int(stem.sample_rate), params)
            critic = _critic_report(stem.audio, compressed, metrics, compression_stats, plan)
            if not _critic_requires_revision(critic) or attempt == 3:
                break
            revision_history.append(
                {
                    "attempt": attempt + 1,
                    "decision": critic["decision"],
                    "target_gr_db": round(float(plan["target_gr_db"]), 3),
                    "ratio": round(float(plan["ratio"]), 3),
                }
            )
            plan = _soften_plan(plan, critic)
        if revision_history:
            critic = {
                **critic,
                "revision_applied": True,
                "revision_count": len(revision_history),
                "revision_history": revision_history,
                "original_target_gr_db": round(float(original_plan["target_gr_db"]), 3),
            }

        if not report_only:
            stem.audio = compressed
        applied = (not report_only) and compression_stats["avg_gr_db"] > 0.05
        applied_count += 1 if applied else 0
        applied_band_count += 1 if applied else 0
        after_primary_audio = primary_audio_fn(stem) if primary_audio_fn else stem.audio
        after_full_lufs = _call_lufs(lufs_fn, stem.audio, int(stem.sample_rate))
        after_full_peak = _call_peak(peak_fn, stem.audio)
        channel_report = {
            "stage": CONTEXTUAL_COMPRESSION_STAGE,
            "channel": int(stem.channel_id),
            "file": Path(stem.path).name,
            "role": role,
            "enabled": True,
            "applied": bool(applied),
            "report_only": bool(report_only),
            "style": style_key,
            "tempo_bpm": round(float(tempo_bpm), 2),
            "tempo_confidence": round(float(tempo_confidence), 3),
            "metrics": _metrics_report(metrics),
            "intent": {
                "target_gr_db": round(float(plan["target_gr_db"]), 3),
                "max_gr_db": round(float(plan["max_gr_db"]), 3),
                "detector": str(plan["detector"]),
                "reason": str(plan["reason"]),
            },
            "parameters": {
                "threshold_db": round(float(params["threshold_db"]), 3),
                "ratio": round(float(params["ratio"]), 3),
                "attack_ms": round(float(params["attack_ms"]), 3),
                "release_ms": round(float(params["release_ms"]), 3),
                "knee_db": round(float(params["knee_db"]), 3),
                "makeup_gain_db": round(float(params["makeup_gain_db"]), 3),
                "detector": str(params["detector"]),
            },
            "gain_reduction": {
                "avg_gr_db": round(float(compression_stats["avg_gr_db"]), 3),
                "median_gr_db": round(float(compression_stats["median_gr_db"]), 3),
                "p95_gr_db": round(float(compression_stats["p95_gr_db"]), 3),
                "max_gr_db": round(float(compression_stats["max_gr_db"]), 3),
                "active_frame_ratio": round(float(compression_stats["active_frame_ratio"]), 4),
            },
            "critic": critic,
            "pre_primary_lufs": round(metrics.lufs, 3),
            "post_primary_lufs": round(_call_lufs(lufs_fn, after_primary_audio, int(stem.sample_rate)), 3),
            "pre_full_track_lufs": round(before_full_lufs, 3),
            "post_full_track_lufs": round(after_full_lufs, 3),
            "pre_full_peak_dbfs": round(before_full_peak, 3),
            "post_full_peak_dbfs": round(after_full_peak, 3),
        }
        _append_compression_note(stem, channel_report)
        channels.append(channel_report)

    return {
        "enabled": True,
        "applied": bool(applied_count > 0),
        "stage": CONTEXTUAL_COMPRESSION_STAGE,
        "method": "style_tempo_aware_target_gr_solver",
        "style": style_key,
        "tempo_bpm": round(float(tempo_bpm), 2),
        "tempo_confidence": round(float(tempo_confidence), 3),
        "tempo_reason": tempo_reason,
        "report_only": bool(report_only),
        "applied_channel_count": applied_count,
        "processed_channel_count": len(channels),
        "applied_band_count": applied_band_count,
        "channels": channels,
        "source": {
            "name": "Contextual Compression",
            "source": "local style/tempo-aware target gain-reduction solver",
            "references": [
                {
                    "name": "Digital Dynamic Range Compressor Design - A Tutorial and Analysis",
                    "url": "https://aes2.org/publications/elibrary-page/?id=16354",
                    "use": "feed-forward compressor model, gain computer, envelope smoothing",
                },
                {
                    "name": "Dynamic range compression automation",
                    "url": "https://joshreiss.github.io/documents/2013/Giannoulis%20Massberg%20Reiss%20-%20dynamic%20range%20compression%20automation%20-%20JAES%202013.pdf",
                    "use": "feature-driven compressor parameter automation",
                },
                {
                    "name": "An autonomous method for multi-track dynamic range compression",
                    "url": "http://dafx12.york.ac.uk/papers/dafx12_submission_63.pdf",
                    "use": "multitrack dynamic range compression target selection",
                },
            ],
            "notes": [
                "Tempo/style adaptation is heuristic; no strong evidence supports one universal setting table.",
                "Critic loop softens compression when pumping, peak growth, or transient damage is excessive.",
            ],
        },
    }


def _style_key(style: str) -> str:
    key = str(style or DEFAULT_CONTEXTUAL_COMPRESSION_STYLE).strip().lower().replace("-", "_")
    return key if key in STYLE_MODIFIERS else DEFAULT_CONTEXTUAL_COMPRESSION_STYLE


def _resolve_tempo(stems: list[Any], bpm: float | None, sample_rate: int) -> tuple[float, float, str]:
    if bpm and np.isfinite(float(bpm)) and float(bpm) > 20.0:
        return float(max(40.0, min(240.0, bpm))), 1.0, "explicit_bpm"
    estimated, confidence = estimate_tempo_bpm(stems, sample_rate=sample_rate)
    if confidence < 0.18:
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, confidence, "low_confidence_default_120_bpm"
    return estimated, confidence, "estimated_from_mix_transient_envelope"


def estimate_tempo_bpm(stems: list[Any], *, sample_rate: int) -> tuple[float, float]:
    """Estimate tempo from the summed transient envelope; returns BPM and confidence."""
    if not stems:
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, 0.0
    max_len = max(int(stem.audio.shape[0]) for stem in stems)
    limit = min(max_len, int(sample_rate * 100.0))
    mix = np.zeros(limit, dtype=np.float32)
    for stem in stems:
        audio = ensure_stereo(stem.audio[:limit])
        mix[: audio.shape[0]] += np.mean(audio, axis=1).astype(np.float32)
    if mix.size < int(sample_rate * 4.0) or _rms_linear(mix) <= db_to_amp(-70.0):
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, 0.0
    frame = max(512, int(0.040 * sample_rate))
    hop = max(128, int(0.010 * sample_rate))
    levels = _frame_rms(mix, frame, hop)
    if levels.size < 16:
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, 0.0
    envelope = np.maximum(0.0, np.diff(np.log(levels + 1e-8), prepend=np.log(levels[0] + 1e-8)))
    envelope = envelope - float(np.mean(envelope))
    if float(np.std(envelope)) <= 1e-8:
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, 0.0
    corr = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1:]
    frame_rate = sample_rate / float(hop)
    min_lag = max(1, int(frame_rate * 60.0 / 190.0))
    max_lag = min(len(corr) - 1, int(frame_rate * 60.0 / 55.0))
    if max_lag <= min_lag:
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, 0.0
    search = corr[min_lag:max_lag + 1]
    if search.size == 0:
        return DEFAULT_CONTEXTUAL_COMPRESSION_BPM, 0.0
    lag = int(np.argmax(search)) + min_lag
    bpm = 60.0 * frame_rate / float(lag)
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm *= 0.5
    peak = float(corr[lag])
    baseline = float(np.mean(np.abs(search)) + 1e-12)
    confidence = float(max(0.0, min(1.0, (peak / baseline - 1.0) / 6.0)))
    return float(max(55.0, min(190.0, bpm))), confidence


def _compression_metrics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    lufs_fn: LufsFn | None,
    peak_fn: PeakFn | None,
) -> CompressionMetrics:
    arr = ensure_stereo(audio)
    mono = np.mean(arr, axis=1).astype(np.float32)
    rms = _rms_db(mono)
    peak = _call_peak(peak_fn, arr)
    lufs = _call_lufs(lufs_fn, arr, sample_rate)
    levels, active = _active_frame_levels(mono, sample_rate, detector="rms")
    if active.size:
        dynamic_range = float(np.percentile(active, 95) - np.percentile(active, 10))
        activity_ratio = float(active.size / max(1, levels.size))
    else:
        dynamic_range = 0.0
        activity_ratio = 0.0
    transient_density, transient_strength = _transient_metrics(mono, sample_rate)
    return CompressionMetrics(
        lufs=lufs,
        rms_db=rms,
        peak_db=peak,
        crest_factor_db=max(0.0, peak - rms),
        dynamic_range_db=max(0.0, dynamic_range),
        activity_ratio=max(0.0, min(1.0, activity_ratio)),
        transient_density_per_sec=transient_density,
        transient_strength_db=transient_strength,
        spectral_flux=_spectral_flux(mono, sample_rate),
    )


def _plan_channel_compression(
    role: str,
    metrics: CompressionMetrics,
    style: str,
    bpm: float,
) -> dict[str, Any]:
    intent = ROLE_INTENTS.get(role, ROLE_INTENTS["unknown"])
    style_mod = STYLE_MODIFIERS[style]
    if metrics.lufs < -62.0 or metrics.activity_ratio < 0.005:
        return {"bypass": True, "reason": "signal_too_low_or_inactive"}

    target = float(intent["target_gr"]) * float(style_mod["target_scale"])
    if metrics.dynamic_range_db > 16.0:
        target += min(1.4, (metrics.dynamic_range_db - 16.0) * 0.08)
    if role in {"kick", "snare", "tom"} and metrics.crest_factor_db > 18.0:
        target += 0.4
    if role in {"overhead", "room"} and metrics.transient_density_per_sec > 5.0:
        target *= 0.75
    if role in {"guitar", "keys", "pad", "playback"} and metrics.crest_factor_db < 7.0:
        target *= 0.75
    target = max(0.25, min(float(intent["max_gr"]) * 0.75, target))

    beat_ms = 60000.0 / max(40.0, float(bpm))
    tempo_release = beat_ms / max(1.0, float(intent["release_divisor"]))
    release = 0.55 * float(intent["release_ms"]) + 0.45 * tempo_release
    release *= float(style_mod["release_scale"])
    attack = float(intent["attack_ms"]) * float(style_mod["attack_scale"])
    if metrics.transient_density_per_sec > 6.0:
        release *= 0.88
    if metrics.crest_factor_db > 16.0 and role not in {"kick", "snare", "tom"}:
        attack *= 1.15
    ratio = float(intent["ratio"]) * float(style_mod["ratio_scale"])
    if metrics.dynamic_range_db > 20.0:
        ratio *= 1.08
    if metrics.dynamic_range_db < 7.0:
        ratio *= 0.85

    return {
        "bypass": False,
        "reason": (
            f"{role} target GR from style={style}, bpm={bpm:.1f}, "
            f"DR={metrics.dynamic_range_db:.1f}dB, crest={metrics.crest_factor_db:.1f}dB"
        ),
        "role": role,
        "target_gr_db": float(target),
        "max_gr_db": float(intent["max_gr"]),
        "ratio": float(max(1.1, min(12.0, ratio))),
        "attack_ms": float(max(1.0, min(120.0, attack))),
        "release_ms": float(max(30.0, min(1200.0, release))),
        "knee_db": float(intent["knee_db"]),
        "detector": str(intent["detector"]),
        "transient_loss_limit": float(intent["transient_loss_limit"]),
    }


def _solve_threshold_for_target(audio: np.ndarray, sample_rate: int, plan: dict[str, Any]) -> dict[str, Any]:
    mono = np.mean(ensure_stereo(audio), axis=1).astype(np.float32)
    levels, active = _active_frame_levels(mono, sample_rate, detector=str(plan["detector"]))
    if active.size == 0:
        active = levels
    ratio = float(plan["ratio"])
    knee = float(plan["knee_db"])
    target = float(plan["target_gr_db"])
    max_gr = float(plan["max_gr_db"])
    low = -70.0
    high = min(-1.0, float(np.percentile(active, 98)) + 3.0)
    for _ in range(36):
        mid = (low + high) * 0.5
        gr = _static_gain_reduction_db(active, threshold_db=mid, ratio=ratio, knee_db=knee)
        avg = float(np.mean(gr)) if gr.size else 0.0
        if avg > target:
            low = mid
        else:
            high = mid
    threshold = high
    gr = _static_gain_reduction_db(active, threshold_db=threshold, ratio=ratio, knee_db=knee)
    p95 = float(np.percentile(gr, 95)) if gr.size else 0.0
    if p95 > max_gr:
        for _ in range(24):
            threshold += 0.5
            gr = _static_gain_reduction_db(active, threshold_db=threshold, ratio=ratio, knee_db=knee)
            p95 = float(np.percentile(gr, 95)) if gr.size else 0.0
            if p95 <= max_gr:
                break
    return {
        "threshold_db": float(max(-70.0, min(-1.0, threshold))),
        "ratio": ratio,
        "attack_ms": float(plan["attack_ms"]),
        "release_ms": float(plan["release_ms"]),
        "knee_db": knee,
        "makeup_gain_db": 0.0,
        "detector": str(plan["detector"]),
    }


def _compress_audio(audio: np.ndarray, sample_rate: int, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    arr = ensure_stereo(audio).astype(np.float32, copy=False)
    mono = np.mean(arr, axis=1).astype(np.float32)
    frame = max(256, int(0.010 * sample_rate))
    hop = max(128, int(0.005 * sample_rate))
    levels = _frame_levels_db(mono, frame, hop, detector=str(params["detector"]))
    raw_gr = _static_gain_reduction_db(
        levels,
        threshold_db=float(params["threshold_db"]),
        ratio=float(params["ratio"]),
        knee_db=float(params["knee_db"]),
    )
    smoothed_gr = _smooth_gain_reduction(
        raw_gr,
        attack_ms=float(params["attack_ms"]),
        release_ms=float(params["release_ms"]),
        frame_interval_sec=hop / float(sample_rate),
    )
    gain_frames_db = -smoothed_gr + float(params["makeup_gain_db"])
    if gain_frames_db.size == 0:
        return arr.copy(), {
            "avg_gr_db": 0.0,
            "median_gr_db": 0.0,
            "p95_gr_db": 0.0,
            "max_gr_db": 0.0,
            "active_frame_ratio": 0.0,
        }
    centers = np.arange(gain_frames_db.size, dtype=np.float64) * hop + frame * 0.5
    samples = np.arange(arr.shape[0], dtype=np.float64)
    gain_db = np.interp(samples, centers, gain_frames_db, left=gain_frames_db[0], right=gain_frames_db[-1])
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)
    out = arr * gain_linear[:, None]
    active = raw_gr > 0.05
    active_gr = smoothed_gr[active] if np.any(active) else smoothed_gr
    return out.astype(np.float32), {
        "avg_gr_db": float(np.mean(active_gr)) if active_gr.size else 0.0,
        "median_gr_db": float(np.median(active_gr)) if active_gr.size else 0.0,
        "p95_gr_db": float(np.percentile(active_gr, 95)) if active_gr.size else 0.0,
        "max_gr_db": float(np.max(active_gr)) if active_gr.size else 0.0,
        "active_frame_ratio": float(np.mean(active)) if active.size else 0.0,
        "gr_frame_std_db": float(np.std(smoothed_gr)) if smoothed_gr.size else 0.0,
        "gr_frame_delta_p95_db": float(np.percentile(np.abs(np.diff(smoothed_gr)), 95)) if smoothed_gr.size > 1 else 0.0,
    }


def _critic_report(
    before: np.ndarray,
    after: np.ndarray,
    metrics: CompressionMetrics,
    stats: dict[str, float],
    plan: dict[str, Any],
) -> dict[str, Any]:
    before_peak = _peak_db(before)
    after_peak = _peak_db(after)
    before_crest = before_peak - _rms_db(before)
    after_crest = after_peak - _rms_db(after)
    transient_loss = max(0.0, before_crest - after_crest)
    pumping_score = float(stats.get("gr_frame_delta_p95_db", 0.0) + 0.12 * stats.get("gr_frame_std_db", 0.0))
    notes = []
    decision = "ok"
    if transient_loss > float(plan["transient_loss_limit"]):
        notes.append({"severity": "high", "reason": "transient_loss", "value_db": round(transient_loss, 3)})
        decision = "soften_transient_damage"
    if float(stats.get("p95_gr_db", 0.0)) > float(plan["max_gr_db"]) + 0.5:
        notes.append({"severity": "high", "reason": "excessive_gain_reduction", "value_db": round(float(stats["p95_gr_db"]), 3)})
        decision = "soften_excessive_gr"
    if pumping_score > 0.9 and metrics.activity_ratio > 0.1:
        notes.append({"severity": "medium", "reason": "pumping_risk", "score": round(pumping_score, 3)})
        if decision == "ok":
            decision = "soften_pumping_risk"
    if after_peak > before_peak + 0.5:
        notes.append({"severity": "medium", "reason": "peak_growth", "value_db": round(after_peak - before_peak, 3)})
        if decision == "ok":
            decision = "soften_peak_growth"
    return {
        "decision": decision,
        "notes": notes,
        "transient_loss_db": round(transient_loss, 3),
        "pumping_score": round(pumping_score, 3),
        "before_crest_db": round(before_crest, 3),
        "after_crest_db": round(after_crest, 3),
        "peak_delta_db": round(after_peak - before_peak, 3),
        "revision_applied": False,
    }


def _critic_requires_revision(critic: dict[str, Any]) -> bool:
    return str(critic.get("decision", "ok")) != "ok"


def _soften_plan(plan: dict[str, Any], critic: dict[str, Any]) -> dict[str, Any]:
    revised = dict(plan)
    revised["target_gr_db"] = max(0.2, float(plan["target_gr_db"]) * 0.72)
    revised["ratio"] = max(1.15, 1.0 + (float(plan["ratio"]) - 1.0) * 0.78)
    revised["attack_ms"] = min(160.0, float(plan["attack_ms"]) * 1.35)
    revised["release_ms"] = min(1400.0, float(plan["release_ms"]) * 1.2)
    revised["reason"] = f"{plan['reason']}; critic revision: {critic.get('decision')}"
    return revised


def _static_gain_reduction_db(level_db: np.ndarray, *, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    levels = np.asarray(level_db, dtype=np.float64)
    over = levels - float(threshold_db)
    ratio = max(1.01, float(ratio))
    slope = 1.0 - 1.0 / ratio
    knee = max(0.0, float(knee_db))
    if knee <= 1e-6:
        return np.maximum(0.0, over * slope)
    gr = np.zeros_like(levels)
    lower = -knee * 0.5
    upper = knee * 0.5
    below = over <= lower
    above = over >= upper
    middle = ~(below | above)
    gr[above] = over[above] * slope
    gr[middle] = ((over[middle] - lower) ** 2 / (2.0 * knee)) * slope
    return np.maximum(0.0, gr)


def _smooth_gain_reduction(
    gr: np.ndarray,
    *,
    attack_ms: float,
    release_ms: float,
    frame_interval_sec: float,
) -> np.ndarray:
    raw = np.asarray(gr, dtype=np.float64)
    if raw.size == 0:
        return raw
    attack_coeff = math.exp(-frame_interval_sec / max(0.001, attack_ms / 1000.0))
    release_coeff = math.exp(-frame_interval_sec / max(0.001, release_ms / 1000.0))
    out = np.zeros_like(raw)
    state = 0.0
    for idx, value in enumerate(raw):
        coeff = attack_coeff if value > state else release_coeff
        state = coeff * state + (1.0 - coeff) * value
        out[idx] = state
    return out


def _active_frame_levels(audio: np.ndarray, sample_rate: int, *, detector: str) -> tuple[np.ndarray, np.ndarray]:
    frame = max(256, int(0.025 * sample_rate))
    hop = max(128, int(0.010 * sample_rate))
    levels = _frame_levels_db(audio, frame, hop, detector=detector)
    if levels.size == 0:
        return levels, levels
    high = float(np.percentile(levels, 90))
    threshold = max(-70.0, high - 38.0)
    active = levels[levels > threshold]
    return levels, active


def _frame_levels_db(audio: np.ndarray, frame: int, hop: int, *, detector: str) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    if arr.size < frame:
        value = np.max(np.abs(arr)) if detector == "peak" else _rms_linear(arr)
        return np.array([amp_to_db(float(value))], dtype=np.float64)
    starts = np.arange(0, arr.size - frame + 1, hop, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    if detector == "peak":
        values = np.array([float(np.max(np.abs(arr[start:start + frame]))) for start in starts], dtype=np.float64)
    else:
        values = _frame_rms(arr, frame, hop)
    return np.array([amp_to_db(float(value)) for value in values], dtype=np.float64)


def _frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size < frame:
        return np.array([_rms_linear(arr)], dtype=np.float64)
    power = arr * arr
    cumsum = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, arr.size - frame + 1, hop, dtype=np.int64)
    sums = cumsum[starts + frame] - cumsum[starts]
    return np.sqrt(np.maximum(sums / float(frame), 1e-12))


def _transient_metrics(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    frame = max(256, int(0.020 * sample_rate))
    hop = max(128, int(0.010 * sample_rate))
    rms = _frame_rms(audio, frame, hop)
    if rms.size < 3:
        return 0.0, 0.0
    db = np.array([amp_to_db(float(value)) for value in rms], dtype=np.float64)
    onset = np.maximum(0.0, np.diff(db, prepend=db[0]))
    threshold = float(np.median(onset) + np.std(onset) * 1.25)
    min_gap = max(1, int(0.050 / (hop / float(sample_rate))))
    peaks = []
    last = -min_gap
    for idx, value in enumerate(onset):
        if value > threshold and idx - last >= min_gap:
            peaks.append(float(value))
            last = idx
    duration = max(1e-6, len(audio) / float(sample_rate))
    density = len(peaks) / duration
    strength = float(np.percentile(peaks, 75)) if peaks else 0.0
    return float(density), strength


def _spectral_flux(audio: np.ndarray, sample_rate: int) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    limit = min(arr.size, int(sample_rate * 30.0))
    arr = arr[:limit]
    frame = 2048
    hop = 1024
    if arr.size < frame * 3:
        return 0.0
    flux_values = []
    prev = None
    window = np.hanning(frame)
    for start in range(0, arr.size - frame + 1, hop):
        mag = np.abs(np.fft.rfft(arr[start:start + frame] * window))
        mag = mag / (float(np.sum(mag)) + 1e-12)
        if prev is not None:
            flux_values.append(float(np.sqrt(np.mean(np.maximum(0.0, mag - prev) ** 2))))
        prev = mag
    if not flux_values:
        return 0.0
    return float(max(0.0, min(1.0, np.percentile(flux_values, 90) * 100.0)))


def _normalize_role(role: str) -> str:
    key = str(role or "unknown").strip().lower()
    if key in {"lead_vocal", "vocal", "vox"}:
        return "lead_vocal"
    if key in {"back_vocal", "backing_vocal", "backs"}:
        return "backing_vocal"
    if key in {"snare_top", "snare_bottom", "snare"}:
        return "snare"
    if key in {"tom", "rack_tom", "floor_tom"}:
        return "tom"
    if key in {"bass", "bass_guitar"}:
        return "bass"
    if key in {"guitar", "rhythm_guitar", "lead_guitar", "electric_guitar", "acoustic_guitar"}:
        return "guitar"
    if key in {"accordion", "keys", "piano", "synth"}:
        return "keys"
    if key in {"pad"}:
        return "pad"
    if key in {"overhead", "room", "kick", "playback"}:
        return key
    return "unknown"


def _fallback_role(name: str) -> str:
    key = Path(str(name)).stem.lower()
    if "kick" in key:
        return "kick"
    if "snare" in key:
        return "snare"
    if "tom" in key:
        return "tom"
    if "bass" in key:
        return "bass"
    if "oh" in key or "overhead" in key:
        return "overhead"
    if "room" in key:
        return "room"
    if "back" in key or "backs" in key:
        return "backing_vocal"
    if "vox" in key or "vocal" in key:
        return "lead_vocal"
    if "guitar" in key or "gtr" in key:
        return "guitar"
    if "playback" in key:
        return "playback"
    if "accordion" in key or "keys" in key or "piano" in key:
        return "keys"
    return "unknown"


def _metrics_report(metrics: CompressionMetrics) -> dict[str, float]:
    return {
        "lufs": round(metrics.lufs, 3),
        "rms_db": round(metrics.rms_db, 3),
        "peak_db": round(metrics.peak_db, 3),
        "crest_factor_db": round(metrics.crest_factor_db, 3),
        "dynamic_range_db": round(metrics.dynamic_range_db, 3),
        "activity_ratio": round(metrics.activity_ratio, 4),
        "transient_density_per_sec": round(metrics.transient_density_per_sec, 3),
        "transient_strength_db": round(metrics.transient_strength_db, 3),
        "spectral_flux": round(metrics.spectral_flux, 4),
    }


def _append_compression_note(stem: Any, report: dict[str, Any]) -> None:
    note = {
        "type": CONTEXTUAL_COMPRESSION_STAGE,
        "threshold_db": report["parameters"]["threshold_db"],
        "ratio": report["parameters"]["ratio"],
        "attack_ms": report["parameters"]["attack_ms"],
        "release_ms": report["parameters"]["release_ms"],
        "avg_gr_db": report["gain_reduction"]["avg_gr_db"],
        "p95_gr_db": report["gain_reduction"]["p95_gr_db"],
        "critic_decision": report["critic"]["decision"],
    }
    if hasattr(stem, "compression_notes"):
        stem.compression_notes.append(note)
    elif hasattr(stem, "notes"):
        stem.notes.append(note)


def _call_lufs(lufs_fn: LufsFn | None, audio: np.ndarray, sample_rate: int) -> float:
    if lufs_fn:
        return float(lufs_fn(audio, sample_rate))
    return _rms_db(audio) - 0.691


def _call_peak(peak_fn: PeakFn | None, audio: np.ndarray) -> float:
    if peak_fn:
        return float(peak_fn(audio))
    return _peak_db(audio)


def _peak_db(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    return amp_to_db(float(np.max(np.abs(arr))))


def _rms_db(audio: np.ndarray) -> float:
    return amp_to_db(_rms_linear(audio))


def _rms_linear(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr) + 1e-12))
