"""Offline source dynamics: independent slow ride, causal compression, bounded match.

Role profiles are engineering starting points, not listening judgements. Diagnostic
spread means P90-P10 of RAW-active 20 ms RMS frames, not hit dynamics, crest or LRA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from ..mastering.analyzer import true_peak_dbtp
from .compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class DynamicsProfile:
    ratio: float
    max_gr_db: float
    ride_max_db: float
    ride_window_s: float
    active_percentile: float
    attack_ms: float = 15.0
    release_ms: float = 150.0
    knee_db: float = 6.0
    rms_ms: float = 3.0


PROFILES = {
    "vocal": DynamicsProfile(3.2, 5.0, 2.0, 1.6, 55, 10, 120),
    "bass": DynamicsProfile(3.0, 4.0, 1.5, 1.8, 50, 15, 180),
    "kick": DynamicsProfile(2.0, 2.5, .5, 1.0, 78, 20, 100, 4, 1),
    "snare": DynamicsProfile(2.2, 3.0, .5, 1.0, 78, 12, 100, 4, 1),
    "toms": DynamicsProfile(2.0, 2.5, .5, 1.1, 75, 20, 160, 4, 1),
    "guitar": DynamicsProfile(1.8, 2.5, 1.2, 2.0, 50, 25, 180),
    "keys": DynamicsProfile(1.6, 2.0, 1.0, 2.2, 50, 30, 220),
    "playback": DynamicsProfile(1.5, 1.5, .8, 2.5, 50, 30, 230),
    "cymbals": DynamicsProfile(1.3, 1.0, .35, 2.5, 70, 35, 240),
}


def _frame_power(x: np.ndarray, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if not len(x):
        return np.zeros(0), np.zeros(0, dtype=int)
    starts = np.arange(0, len(x), hop)
    power = x.astype(np.float64) ** 2
    if x.ndim == 2:
        power = np.mean(power, axis=1)
    counts = np.minimum(hop, len(x) - starts)
    return np.add.reduceat(power, starts) / counts, counts


def _db(power: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, 1e-30))


def _spread(db: np.ndarray, active: np.ndarray) -> float:
    values = db[active]
    return float(np.percentile(values, 90) - np.percentile(values, 10)) if len(values) else 0.0


def _active_rms_db(power, counts, active) -> float | None:
    if not np.any(active):
        return None
    return float(_db(np.average(power[active], weights=counts[active])))


def _render(x: np.ndarray, sr: int, role: str, hop_s: float = .02, *,
            compressor_config: CompressorConfig | None = None,
            enable_ride: bool = True, makeup_limit_db: float = 2.0
            ) -> tuple[np.ndarray, dict]:
    x = as_audio(x)
    if role not in PROFILES:
        raise ValueError(f"unknown dynamics role: {role}")
    CompressorConfig().validate(sr)
    if not np.isfinite(hop_s) or hop_s <= 0 or not 0 <= makeup_limit_db <= 6:
        raise ValueError("invalid hop or makeup bound")
    if compressor_config is not None:
        compressor_config.validate(sr)
    hop = max(1, int(round(sr * hop_s)))
    frame_power, counts = _frame_power(x, hop)
    db = _db(frame_power)
    profile = PROFILES[role]
    active = np.zeros(len(db), dtype=bool)
    if len(db) and np.max(db) > -100:
        threshold = max(-100.0, float(np.max(db)) - 40,
                        float(np.percentile(db, profile.active_percentile)))
        active = db >= threshold
    ride = np.zeros(len(db), dtype=np.float32)
    no_op = not np.any(active) or (compressor_config is not None and (
        compressor_config.bypass or compressor_config.ratio == 1 or compressor_config.max_gr_db == 0))
    if enable_ride and not no_op:
        target = float(np.median(db[active]))
        ride = np.where(active, np.clip((target - db) * .42,
                        -profile.ride_max_db, profile.ride_max_db), 0)
        ride = ndimage.gaussian_filter1d(
            ride, max(1.0, profile.ride_window_s / (2.355 * hop_s))).astype(np.float32)
    centers = (np.arange(len(db)) * hop + counts / 2) / sr
    sample_ride = np.interp(np.arange(len(x)) / sr, centers, ride) if len(db) else ride
    ridden = (x * np.power(10, (sample_ride[:, None] if x.ndim == 2
                               else sample_ride) / 20)).astype(np.float32)
    cfg = compressor_config
    if cfg is None:
        # This is a threshold proposal, NOT a promise of actual GR. Ballistics
        # and RMS integration affect the measured result and are reported below.
        requested_static_gr = min(3.0, profile.max_gr_db * .6)
        threshold = (float(np.percentile((db + ride)[active], 90))
                     - requested_static_gr / (1 - 1 / profile.ratio)) if np.any(active) else 0.0
        cfg = CompressorConfig(threshold_dbfs=float(np.clip(threshold, -160, 24)),
                               ratio=profile.ratio, max_gr_db=profile.max_gr_db,
                               attack_ms=profile.attack_ms, release_ms=profile.release_ms,
                               knee_db=profile.knee_db, rms_ms=profile.rms_ms)
    y, gr = (ridden.copy(), np.zeros(len(x), np.float32)) if no_op else LinkedCompressor(sr, cfg).process(ridden)
    before_level = _active_rms_db(frame_power, counts, active)
    compressed_power, _ = _frame_power(y, hop)
    compressed_level = _active_rms_db(compressed_power, counts, active)
    desired_makeup = 0.0 if no_op else float(before_level - compressed_level)
    makeup = float(np.clip(desired_makeup, -makeup_limit_db, makeup_limit_db))
    peak_before_makeup = true_peak_dbtp(y) if len(y) and not no_op else None
    if makeup > 0:
        # No compensation can bypass the -1 dBTP boost headroom rule.
        makeup = min(makeup, max(0.0, -1.0 - peak_before_makeup))
    if makeup:
        y = (y * np.float32(10 ** (makeup / 20))).astype(np.float32)
    after_power, _ = _frame_power(y, hop)
    after_level = _active_rms_db(after_power, counts, active)
    sample_active = np.repeat(active, counts)
    active_gr = gr[sample_active]
    frame_gr = np.add.reduceat(gr.astype(np.float64), np.arange(0, len(gr), hop)) / counts if len(gr) else gr
    peak_after = true_peak_dbtp(y) if len(y) and not no_op else None
    report = {
        "schema": "studio-dynamics-v2", "db": db, "active": active,
        "ride_db": ride, "gr_db": frame_gr, "net_db": ride - frame_gr + makeup,
        "makeup_db": makeup, "requested_makeup_db": desired_makeup,
        "makeup_limited": abs(makeup - desired_makeup) > .01,
        "before_spread_db": _spread(db, active),
        "after_spread_db": _spread(_db(after_power), active),
        "p95_gr_db": float(np.percentile(active_gr, 95)) if len(active_gr) else 0.0,
        "max_gr_db": float(np.max(gr)) if len(gr) else 0.0,
        "whole_track_p95_gr_db": float(np.percentile(gr, 95)) if len(gr) else 0.0,
        "compression_duty_fraction": float(np.mean(gr > .1)) if len(gr) else 0.0,
        "hop_s": hop / sr, "active_frames": int(np.sum(active)),
        "active_rms_before_dbfs": before_level, "active_rms_after_dbfs": after_level,
        "active_rms_delta_db": None if before_level is None else after_level - before_level,
        "true_peak_dbtp": peak_after,
        "true_peak_method": "scipy.resample_poly_4x" if peak_after is not None else "not_measured_no_op",
        "headroom_ok": None if peak_after is None else peak_after <= -.999,
        "compressor": asdict(cfg), "threshold_selection": "explicit" if compressor_config else "active_frame_p90_proposal",
        "rider": {"enabled": bool(enable_ride and not no_op), "mode": "offline_slow_gaussian",
                  "window_s": profile.ride_window_s, "separate_from_compressor_gr": True},
        "measurement_scope": "actual PCM; fixed RAW-active RMS frames; not event dynamics or LRA",
        "status": "no_op" if no_op else "pending_human_review",
        "baseline_eligible": False, "input_unchanged": True,
    }
    return y, report


def analyze_frames(x: np.ndarray, sr: int, role: str, hop_s: float = .02) -> dict:
    """Retained analysis entry point; compression summaries describe actual output."""
    return _render(x, sr, role, hop_s)[1]


def apply(x: np.ndarray, sr: int, role: str, *,
          compressor_config: CompressorConfig | None = None,
          enable_ride: bool = True, makeup_limit_db: float = 2.0
          ) -> tuple[np.ndarray, dict]:
    """Render one full source, never independent per-chunk loudness matching.

    For block processing use LinkedCompressor directly; the rider and fixed-mask
    level match intentionally operate on the full offline source.
    """
    y, report = _render(x, sr, role, compressor_config=compressor_config,
                        enable_ride=enable_ride, makeup_limit_db=makeup_limit_db)
    return y, {k: v for k, v in report.items()
               if k not in {"db", "active", "ride_db", "gr_db", "net_db"}}
