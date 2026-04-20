#!/usr/bin/env python3
"""Offline multitrack mix using the same rule/agent path as live control.

The script treats each WAV file as a console channel, runs a virtual mixer
through MixingAgent + RuleEngine, applies channel strip DSP, then exports
a mastered MP3 for listening checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ai.agent import AgentAction, AgentMode, MixingAgent  # noqa: E402
from ai.knowledge_base import KnowledgeBase  # noqa: E402
from ai.llm_client import LLMClient  # noqa: E402
from ai.rule_engine import RuleEngine  # noqa: E402
from auto_phase_gcc_phat import GCCPHATAnalyzer  # noqa: E402
from cross_adaptive_eq import CrossAdaptiveEQ  # noqa: E402
from auto_fx import AutoFXPlanner, FXBusDecision, FXPlan  # noqa: E402


DRUM_INSTRUMENTS = {
    "kick",
    "snare",
    "rack_tom",
    "floor_tom",
    "hi_hat",
    "ride",
    "percussion",
}


@dataclass
class ChannelPlan:
    path: Path
    name: str
    instrument: str
    pan: float
    hpf: float
    target_rms_db: float
    lpf: float = 0.0
    trim_db: float = 0.0
    fader_db: float = 0.0
    muted: bool = False
    phase_invert: bool = False
    delay_ms: float = 0.0
    input_fade_ms: float = 0.0
    eq_bands: list[tuple[float, float, float]] = field(default_factory=list)
    comp_threshold_db: float = -20.0
    comp_ratio: float = 2.5
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 120.0
    expander_enabled: bool = False
    expander_range_db: float = 0.0
    expander_open_ms: float = 12.0
    expander_close_ms: float = 140.0
    expander_hold_ms: float = 0.0
    expander_threshold_db: float | None = None
    expander_report: dict[str, Any] = field(default_factory=dict)
    event_activity: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    phase_notes: list[dict[str, Any]] = field(default_factory=list)
    pan_notes: list[dict[str, Any]] = field(default_factory=list)
    cross_adaptive_eq: list[dict[str, Any]] = field(default_factory=list)


def _agent_actions_to_dict(actions: list[AgentAction]) -> list[dict[str, Any]]:
    """Serialize action objects for report output."""
    return [action.__dict__ for action in actions]


class VirtualConsole:
    """MixerClient-like adapter for MixingAgent offline actions."""

    def __init__(self, plans: dict[int, ChannelPlan]):
        self.plans = plans
        self.calls: list[dict[str, Any]] = []

    def get_fader(self, channel: int) -> float:
        return self.plans[channel].fader_db

    def set_fader(self, channel: int, value_db: float):
        self.plans[channel].fader_db = float(np.clip(value_db, -100.0, 10.0))
        self.calls.append({"cmd": "set_fader", "channel": channel, "value_db": self.plans[channel].fader_db})

    def get_mute(self, channel: int) -> bool:
        return self.plans[channel].muted

    def set_mute(self, channel: int, muted: bool):
        self.plans[channel].muted = bool(muted)
        self.calls.append({"cmd": "set_mute", "channel": channel, "muted": bool(muted)})

    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        self.plans[channel].hpf = float(freq) if enabled else 0.0
        self.calls.append({"cmd": "set_hpf", "channel": channel, "freq": self.plans[channel].hpf})

    def set_polarity(self, channel: int, inverted: bool):
        self.plans[channel].phase_invert = bool(inverted)
        self.calls.append({"cmd": "set_polarity", "channel": channel, "inverted": bool(inverted)})

    def set_channel_phase_invert(self, channel: int, value: int):
        self.set_polarity(channel, bool(value))

    def set_delay(self, channel: int, delay_ms: float, enabled: bool = True):
        self.plans[channel].delay_ms = float(delay_ms) if enabled else 0.0
        self.calls.append({"cmd": "set_delay", "channel": channel, "delay_ms": self.plans[channel].delay_ms})

    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        self.set_delay(channel, float(value), enabled=True)

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        plan = self.plans[channel]
        band_idx = max(1, int(band)) - 1
        while len(plan.eq_bands) <= band_idx:
            plan.eq_bands.append((1000.0, 0.0, 1.0))
        plan.eq_bands[band_idx] = (float(freq), float(gain), float(q))
        self.calls.append({"cmd": "set_eq_band", "channel": channel, "band": band, "freq": freq, "gain": gain, "q": q})

    def set_compressor(self, channel: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, enabled: bool = True):
        if not enabled:
            return
        plan = self.plans[channel]
        plan.comp_threshold_db = float(threshold_db)
        plan.comp_ratio = float(ratio)
        plan.comp_attack_ms = float(attack_ms)
        plan.comp_release_ms = float(release_ms)
        self.calls.append({
            "cmd": "set_compressor",
            "channel": channel,
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
        })


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def mono_sum(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return mono_sum(audio), int(sr)


def delay_signal(x: np.ndarray, sr: int, delay_ms: float) -> np.ndarray:
    delay_samples = int(round(max(0.0, delay_ms) * sr / 1000.0))
    if delay_samples <= 0:
        return x
    return np.pad(x, (delay_samples, 0))[:len(x)].astype(np.float32)


def declick_start(x: np.ndarray, sr: int, fade_ms: float, threshold_db: float = -70.0) -> np.ndarray:
    if fade_ms <= 0.0 or len(x) == 0:
        return x
    threshold = db_to_amp(threshold_db)
    active = np.flatnonzero(np.abs(x) > threshold)
    if len(active) == 0:
        return x
    fade_len = max(16, int(fade_ms * sr / 1000.0))
    start = max(0, int(active[0]) - fade_len // 4)
    end = min(len(x), start + fade_len)
    if end <= start:
        return x
    out = x.copy()
    out[start:end] *= np.linspace(0.0, 1.0, end - start, dtype=np.float32)
    return out.astype(np.float32)


def _active_segment_start(x: np.ndarray, sr: int, window_sec: float = 3.0) -> int:
    window = max(1024, int(window_sec * sr))
    if len(x) <= window:
        return 0
    hop = max(512, window // 2)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return best_start


def _active_segment_starts(x: np.ndarray, sr: int, window_sec: float = 3.0, count: int = 8) -> list[int]:
    window = max(1024, int(window_sec * sr))
    if len(x) <= window:
        return [0]
    hop = max(512, window // 2)
    candidates = []
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        candidates.append((energy, start))
    candidates.sort(reverse=True)
    return [start for _, start in candidates[:count]]


def _merge_ranges(ranges: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _event_metric_config(instrument: str) -> dict[str, float] | None:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return {
            "frame_ms": 240.0,
            "hop_ms": 60.0,
            "pad_ms": 140.0,
            "detect_hpf_hz": 120.0,
            "detect_lpf_hz": 4500.0,
            "percentile": 82.0,
            "peak_offset_db": 18.0,
            "floor_margin_db": 8.0,
            "min_threshold_db": -42.0,
        }
    if instrument in {"rack_tom", "floor_tom"}:
        return {
            "frame_ms": 150.0,
            "hop_ms": 35.0,
            "pad_ms": 90.0,
            "detect_hpf_hz": 60.0,
            "detect_lpf_hz": 1200.0,
            "percentile": 97.5,
            "peak_offset_db": 14.0,
            "floor_margin_db": 10.0,
            "min_threshold_db": -36.0,
        }
    if instrument == "kick":
        return {
            "frame_ms": 140.0,
            "hop_ms": 30.0,
            "pad_ms": 80.0,
            "detect_hpf_hz": 28.0,
            "detect_lpf_hz": 220.0,
            "percentile": 98.0,
            "peak_offset_db": 12.0,
            "floor_margin_db": 9.0,
            "min_threshold_db": -38.0,
        }
    if instrument == "snare":
        return {
            "frame_ms": 135.0,
            "hop_ms": 30.0,
            "pad_ms": 85.0,
            "detect_hpf_hz": 140.0,
            "detect_lpf_hz": 2600.0,
            "percentile": 97.0,
            "peak_offset_db": 13.0,
            "floor_margin_db": 9.0,
            "min_threshold_db": -38.0,
        }
    if instrument == "hi_hat":
        return {
            "frame_ms": 160.0,
            "hop_ms": 40.0,
            "pad_ms": 110.0,
            "detect_hpf_hz": 1800.0,
            "detect_lpf_hz": 14000.0,
            "percentile": 92.0,
            "peak_offset_db": 16.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    if instrument == "ride":
        return {
            "frame_ms": 170.0,
            "hop_ms": 40.0,
            "pad_ms": 120.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 94.0,
            "peak_offset_db": 18.0,
            "peak_percentile": 99.9,
            "floor_margin_db": 5.0,
            "min_threshold_db": -50.0,
        }
    if instrument == "percussion":
        return {
            "frame_ms": 160.0,
            "hop_ms": 40.0,
            "pad_ms": 110.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 92.0,
            "peak_offset_db": 16.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    return None


def _next_power_of_two(value: int) -> int:
    return 1 << max(1, int(value - 1).bit_length())


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = a[:n] - float(np.mean(a[:n]))
    bb = b[:n] - float(np.mean(b[:n]))
    denom = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb))) + 1e-12
    return float(np.dot(aa, bb) / denom)


def _drum_channel_names(instrument: str) -> bool:
    return instrument in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}


def apply_drum_phase_alignment(plans: dict[int, ChannelPlan], sr: int, max_delay_ms: float = 12.0) -> list[dict[str, Any]]:
    """Apply the project's GCC-PHAT close-mic-to-overhead drum alignment rule."""
    overheads = [plan for plan in plans.values() if plan.instrument == "overhead"]
    if len(overheads) < 2:
        return []

    overhead_signals = []
    for plan in overheads:
        signal, file_sr = read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        overhead_signals.append(signal)
    reference = np.mean(np.vstack(overhead_signals), axis=0).astype(np.float32)
    reference = highpass(reference, sr, 40.0)

    reports: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        if not _drum_channel_names(plan.instrument):
            continue

        target, file_sr = read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        target = highpass(target, sr, 40.0)
        window_len = int(3.0 * sr)
        starts = _active_segment_starts(target, sr)
        measurements = []
        for start in starts:
            window = min(len(reference) - start, len(target) - start, window_len)
            if window < int(0.5 * sr):
                continue
            ref_seg = reference[start:start + window]
            tgt_seg = target[start:start + window]
            fft_size = _next_power_of_two(len(ref_seg))
            if fft_size != len(ref_seg):
                ref_seg = np.pad(ref_seg, (0, fft_size - len(ref_seg)))
                tgt_seg = np.pad(tgt_seg, (0, fft_size - len(tgt_seg)))

            analyzer = GCCPHATAnalyzer(sample_rate=sr, fft_size=fft_size, max_delay_ms=max_delay_ms)
            measurement = analyzer.compute_delay(ref_seg, tgt_seg)
            measured_delay_ms = float(measurement.delay_ms)
            boundary_hit = abs(abs(measured_delay_ms) - max_delay_ms) < 0.05
            if boundary_hit or not np.isfinite(measured_delay_ms):
                continue
            measurements.append((start, measurement))

        if not measurements:
            continue

        negative_measurements = [(start, m) for start, m in measurements if float(m.delay_ms) < -0.3]
        if negative_measurements:
            median_delay = float(np.median([float(m.delay_ms) for _, m in negative_measurements]))
            start, measurement = min(
                negative_measurements,
                key=lambda item: abs(float(item[1].delay_ms) - median_delay),
            )
        else:
            start, measurement = measurements[0]
        measured_delay_ms = float(measurement.delay_ms)

        delay_ms = 0.0
        if measured_delay_ms < -0.3:
            delay_ms = min(max_delay_ms, -measured_delay_ms)
            plan.delay_ms = max(plan.delay_ms, delay_ms)

        aligned = delay_signal(target, sr, plan.delay_ms)
        corr_start = start
        corr_end = min(corr_start + window_len, len(reference), len(aligned))
        corr_ref = reference[corr_start:corr_end]
        corr_target = aligned[corr_start:corr_end]
        current_corr = _norm_corr(corr_ref, -corr_target if plan.phase_invert else corr_target)
        flipped_corr = _norm_corr(corr_ref, corr_target if plan.phase_invert else -corr_target)
        if abs(flipped_corr) > 0.12 and flipped_corr > current_corr + 0.04:
            plan.phase_invert = not plan.phase_invert

        note = {
            "reference": "overhead_pair",
            "measured_delay_ms": round(measured_delay_ms, 3),
            "applied_delay_ms": round(plan.delay_ms, 3),
            "psr_db": round(float(measurement.psr), 2) if np.isfinite(measurement.psr) else None,
            "confidence": round(float(measurement.confidence), 3) if np.isfinite(measurement.confidence) else None,
            "coherence": round(float(measurement.coherence), 3) if np.isfinite(measurement.coherence) else None,
            "phase_invert": plan.phase_invert,
            "corr_current": round(current_corr, 3),
            "corr_flipped": round(flipped_corr, 3),
        }
        plan.phase_notes.append(note)
        reports.append({"channel": channel, "file": plan.path.name, **note})

    return reports


def _equal_power_gains(pan: float) -> tuple[float, float]:
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * math.pi / 4.0
    return math.cos(theta), math.sin(theta)


def _pan_from_lr_diff_db(diff_db: float) -> float:
    ratio = db_to_amp(float(np.clip(diff_db, -48.0, 48.0)))
    theta = math.atan(ratio)
    return float(np.clip((4.0 * theta / math.pi) - 1.0, -1.0, 1.0))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = float(np.sum(sorted_weights)) * 0.5
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def _transient_segment_starts(
    x: np.ndarray,
    sr: int,
    window_ms: float = 220.0,
    count: int = 16,
    min_gap_ms: float = 260.0,
) -> list[int]:
    """Find short drum events so overhead image placement is not averaged out."""
    window = max(256, int(window_ms * 0.001 * sr))
    if len(x) <= window:
        return [0]

    frame = max(128, int(0.018 * sr))
    hop = max(64, int(0.009 * sr))
    if len(x) <= frame:
        return [0]

    starts = np.arange(0, len(x) - frame, hop, dtype=np.int64)
    envelope = np.asarray([
        float(np.sqrt(np.mean(np.square(x[start:start + frame]))) + 1e-12)
        for start in starts
    ], dtype=np.float32)
    if len(envelope) < 3:
        return [0]

    floor = float(np.percentile(envelope, 60))
    threshold = max(floor * 1.4, float(np.percentile(envelope, 82)))
    peaks: list[tuple[float, int]] = []
    for idx in range(1, len(envelope) - 1):
        value = float(envelope[idx])
        if value < threshold:
            continue
        if value >= float(envelope[idx - 1]) and value >= float(envelope[idx + 1]):
            peaks.append((value, int(starts[idx] + frame // 2)))

    if not peaks:
        return _active_segment_starts(x, sr, window_sec=window_ms * 0.001, count=count)

    peaks.sort(reverse=True)
    min_gap = int(min_gap_ms * 0.001 * sr)
    selected: list[int] = []
    for _, center in peaks:
        if all(abs(center - existing) >= min_gap for existing in selected):
            selected.append(center)
        if len(selected) >= count:
            break

    selected.sort()
    return [int(np.clip(center - window // 2, 0, len(x) - window)) for center in selected]


def _overhead_lr(plans: dict[int, ChannelPlan]) -> tuple[tuple[int, ChannelPlan], tuple[int, ChannelPlan]] | None:
    overheads = [(channel, plan) for channel, plan in plans.items() if plan.instrument == "overhead"]
    if len(overheads) < 2:
        return None

    left = next(
        ((channel, plan) for channel, plan in overheads if "overhead l" in plan.name.lower() or plan.name.lower().endswith(" l")),
        None,
    )
    right = next(
        ((channel, plan) for channel, plan in overheads if "overhead r" in plan.name.lower() or plan.name.lower().endswith(" r")),
        None,
    )
    if left and right:
        return left, right

    ordered = sorted(overheads, key=lambda item: item[1].pan)
    return ordered[0], ordered[-1]


def _source_overhead_measurements(
    source: np.ndarray,
    overhead_l: np.ndarray,
    overhead_r: np.ndarray,
    sr: int,
    source_delay_ms: float,
    window_ms: float,
    count: int,
) -> list[dict[str, Any]]:
    window = max(512, int(window_ms * 0.001 * sr))
    delay_samples = int(round(max(0.0, source_delay_ms) * sr / 1000.0))
    starts = _transient_segment_starts(source, sr, window_ms=window_ms, count=count)
    measurements: list[dict[str, Any]] = []
    for start in starts:
        oh_start = start + delay_samples
        oh_end = min(oh_start + window, len(overhead_l), len(overhead_r))
        src_end = min(start + window, len(source))
        if oh_start < 0 or oh_end - oh_start < window // 3 or src_end - start < window // 3:
            continue

        src_seg = source[start:src_end]
        l_seg = overhead_l[oh_start:oh_end]
        r_seg = overhead_r[oh_start:oh_end]
        source_rms = float(np.sqrt(np.mean(np.square(src_seg))) + 1e-12)
        left_rms = float(np.sqrt(np.mean(np.square(l_seg))) + 1e-12)
        right_rms = float(np.sqrt(np.mean(np.square(r_seg))) + 1e-12)
        if source_rms < db_to_amp(-60.0) or (left_rms + right_rms) < db_to_amp(-64.0):
            continue
        measurements.append({
            "start_sec": round(start / sr, 3),
            "left_rms": left_rms,
            "right_rms": right_rms,
            "left_db": round(amp_to_db(left_rms), 2),
            "right_db": round(amp_to_db(right_rms), 2),
            "source_db": round(amp_to_db(source_rms), 2),
            "weight": float(np.clip(db_to_amp(amp_to_db(source_rms) + 34.0), 0.15, 4.0)),
        })
    return measurements


def _overhead_output_diff_db(left_rms: float, right_rms: float, left_pan: float, right_pan: float) -> float:
    left_to_l, left_to_r = _equal_power_gains(left_pan)
    right_to_l, right_to_r = _equal_power_gains(right_pan)
    out_l = max(1e-12, left_rms * left_to_l + right_rms * right_to_l)
    out_r = max(1e-12, left_rms * left_to_r + right_rms * right_to_r)
    return amp_to_db(out_r) - amp_to_db(out_l)


def _weighted_center_error(measurements: list[dict[str, Any]], left_pan: float, right_pan: float) -> tuple[float, float]:
    if not measurements:
        return 0.0, 0.0
    errors = []
    signed = []
    weights = []
    for measurement in measurements:
        diff = _overhead_output_diff_db(
            float(measurement["left_rms"]),
            float(measurement["right_rms"]),
            left_pan,
            right_pan,
        )
        weight = float(measurement["weight"])
        errors.append(abs(diff))
        signed.append(diff)
        weights.append(weight)
    return (
        float(np.average(np.asarray(errors), weights=np.asarray(weights))),
        float(np.average(np.asarray(signed), weights=np.asarray(weights))),
    )


def apply_overhead_anchored_drum_panning(plans: dict[int, ChannelPlan], sr: int) -> dict[str, Any]:
    """Pan drums from the overhead picture: center kick/snare first, then place close mics."""
    pair = _overhead_lr(plans)
    if not pair:
        return {"enabled": False, "reason": "overhead_pair_not_found"}

    (left_channel, left_plan), (right_channel, right_plan) = pair
    overhead_l, file_sr = read_mono(left_plan.path)
    if file_sr != sr:
        raise ValueError(f"{left_plan.path.name}: sample rate mismatch {file_sr} != {sr}")
    overhead_r, file_sr = read_mono(right_plan.path)
    if file_sr != sr:
        raise ValueError(f"{right_plan.path.name}: sample rate mismatch {file_sr} != {sr}")
    overhead_l = highpass(overhead_l, sr, 45.0)
    overhead_r = highpass(overhead_r, sr, 45.0)

    center_measurements: list[dict[str, Any]] = []
    source_measurements: dict[int, list[dict[str, Any]]] = {}
    for channel, plan in plans.items():
        if plan.instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue
        source, file_sr = read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        source = highpass(source, sr, 40.0)
        window_ms = 170.0 if plan.instrument in {"kick", "snare"} else 260.0
        count = 18 if plan.instrument in {"kick", "snare"} else 12
        measurements = _source_overhead_measurements(
            source,
            overhead_l,
            overhead_r,
            sr,
            plan.delay_ms,
            window_ms=window_ms,
            count=count,
        )
        source_measurements[channel] = measurements
        if plan.instrument in {"kick", "snare"}:
            center_measurements.extend([
                {**measurement, "channel": channel, "file": plan.path.name, "instrument": plan.instrument}
                for measurement in measurements
            ])

    before_left = float(left_plan.pan)
    before_right = float(right_plan.pan)
    if center_measurements:
        best: tuple[float, float, float, float, float] | None = None
        for width in np.linspace(0.58, 0.88, 31):
            for shift in np.linspace(-0.18, 0.18, 37):
                left_pan = float(np.clip(-width + shift, -0.95, -0.18))
                right_pan = float(np.clip(width + shift, 0.18, 0.95))
                error, signed = _weighted_center_error(center_measurements, left_pan, right_pan)
                width_penalty = max(0.0, 0.74 - width) * 0.7 + max(0.0, width - 0.86) * 0.25
                score = error + width_penalty + abs(float(shift)) * 0.2
                if best is None or score < best[0]:
                    best = (score, left_pan, right_pan, error, signed)
        assert best is not None
        _, new_left, new_right, after_error, after_signed = best
        left_plan.pan = new_left
        right_plan.pan = new_right
    else:
        after_error, after_signed = _weighted_center_error(center_measurements, before_left, before_right)

    before_error, before_signed = _weighted_center_error(center_measurements, before_left, before_right)
    left_note = {
        "rule": "overhead_anchor_first",
        "before_pan": round(before_left, 3),
        "after_pan": round(left_plan.pan, 3),
        "center_sources": ["kick", "snare"],
        "center_error_before_db": round(before_error, 2),
        "center_error_after_db": round(after_error, 2),
        "center_signed_after_db": round(after_signed, 2),
    }
    right_note = dict(left_note)
    right_note["before_pan"] = round(before_right, 3)
    right_note["after_pan"] = round(right_plan.pan, 3)
    left_plan.pan_notes.append(left_note)
    right_plan.pan_notes.append(right_note)

    placed: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        if plan.instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue

        before_pan = float(plan.pan)
        if plan.instrument in {"kick", "snare"}:
            plan.pan = 0.0
            note = {
                "rule": "kick_snare_center_after_overheads",
                "before_pan": round(before_pan, 3),
                "after_pan": round(plan.pan, 3),
                "reason": "kick/snare image is the center reference for the overhead pair",
            }
            plan.pan_notes.append(note)
            placed.append({"channel": channel, "file": plan.path.name, "instrument": plan.instrument, **note})
            continue

        measurements = source_measurements.get(channel, [])
        pan_values = []
        weights = []
        for measurement in measurements:
            diff = _overhead_output_diff_db(
                float(measurement["left_rms"]),
                float(measurement["right_rms"]),
                left_plan.pan,
                right_plan.pan,
            )
            pan_values.append(_pan_from_lr_diff_db(diff))
            weights.append(float(measurement["weight"]))

        if pan_values:
            estimated_pan = _weighted_median(pan_values, weights)
            if plan.instrument in {"rack_tom", "floor_tom"}:
                estimated_pan = float(np.clip(estimated_pan, -0.68, 0.68))
            else:
                estimated_pan = float(np.clip(estimated_pan, -0.82, 0.82))
            plan.pan = estimated_pan
            source_note = {
                "rule": "close_mic_follows_overhead_image",
                "before_pan": round(before_pan, 3),
                "after_pan": round(plan.pan, 3),
                "measurement_count": len(measurements),
                "median_overhead_image_pan": round(estimated_pan, 3),
                "left_right_db_examples": [
                    {
                        "start_sec": measurement["start_sec"],
                        "left_db": measurement["left_db"],
                        "right_db": measurement["right_db"],
                    }
                    for measurement in measurements[:4]
                ],
            }
        else:
            source_note = {
                "rule": "close_mic_follows_overhead_image",
                "before_pan": round(before_pan, 3),
                "after_pan": round(plan.pan, 3),
                "measurement_count": 0,
                "reason": "no confident overhead image measurements, kept existing pan",
            }
        plan.pan_notes.append(source_note)
        placed.append({"channel": channel, "file": plan.path.name, "instrument": plan.instrument, **source_note})

    return {
        "enabled": True,
        "overhead_left_channel": left_channel,
        "overhead_right_channel": right_channel,
        "overhead_left_file": left_plan.path.name,
        "overhead_right_file": right_plan.path.name,
        "overhead_left_pan": round(left_plan.pan, 3),
        "overhead_right_pan": round(right_plan.pan, 3),
        "center_measurements": len(center_measurements),
        "center_error_before_db": round(before_error, 2),
        "center_error_after_db": round(after_error, 2),
        "center_signed_before_db": round(before_signed, 2),
        "center_signed_after_db": round(after_signed, 2),
        "placed_close_mics": placed,
    }


def _post_fader_lufs(plan: ChannelPlan) -> float:
    return float(plan.metrics["lufs_momentary"] + plan.trim_db + plan.fader_db)


def _music_bed_lufs(plans: dict[int, ChannelPlan]) -> float:
    music_reference = [
        _post_fader_lufs(plan)
        for plan in plans.values()
        if plan.instrument != "lead_vocal" and not plan.muted
    ]
    top_music = sorted(music_reference, reverse=True)[:5]
    return float(np.mean(top_music)) if top_music else -20.0


def apply_vocal_bed_balance(
    plans: dict[int, ChannelPlan],
    desired_vocal_delta_db: float = 6.5,
    max_bed_attenuation_db: float = 3.0,
) -> dict[str, Any]:
    """Lower the whole music bed when vocal cannot safely move far enough up."""
    vocal_plans = [
        (channel, plan)
        for channel, plan in plans.items()
        if plan.instrument == "lead_vocal" and not plan.muted
    ]
    if not vocal_plans:
        return {}

    vocal_channel, vocal_plan = max(vocal_plans, key=lambda item: _post_fader_lufs(item[1]))
    music_lufs = _music_bed_lufs(plans)
    vocal_lufs = _post_fader_lufs(vocal_plan)
    current_delta = vocal_lufs - music_lufs
    shortfall = desired_vocal_delta_db - current_delta
    attenuation = max(0.0, min(max_bed_attenuation_db, shortfall))

    adjusted_channels = []
    if attenuation >= 0.25:
        for channel, plan in plans.items():
            if plan.instrument == "lead_vocal" or plan.muted:
                continue
            plan.fader_db = float(np.clip(plan.fader_db - attenuation, -100.0, 10.0))
            adjusted_channels.append(channel)

    return {
        "vocal_channel": vocal_channel,
        "desired_vocal_delta_db": round(desired_vocal_delta_db, 2),
        "before_vocal_lufs": round(vocal_lufs, 2),
        "before_music_bed_lufs": round(music_lufs, 2),
        "before_delta_db": round(current_delta, 2),
        "bed_attenuation_db": round(attenuation, 2),
        "after_vocal_lufs": round(_post_fader_lufs(vocal_plan), 2),
        "after_music_bed_lufs": round(_music_bed_lufs(plans), 2),
        "after_delta_db": round(_post_fader_lufs(vocal_plan) - _music_bed_lufs(plans), 2),
        "adjusted_channels": adjusted_channels,
    }


def codex_correction_actions(plans: dict[int, ChannelPlan]) -> list[AgentAction]:
    """Human-in-the-loop correction pass used when Codex is the mix agent."""
    actions: list[AgentAction] = []

    vocal_plans = [
        (channel, plan)
        for channel, plan in plans.items()
        if plan.instrument == "lead_vocal" and not plan.muted
    ]
    if len(vocal_plans) > 1:
        anchor_channel, anchor_plan = max(vocal_plans, key=lambda item: _post_fader_lufs(item[1]))
        anchor_lufs = _post_fader_lufs(anchor_plan)
        for channel, plan in vocal_plans:
            if channel == anchor_channel:
                continue
            target_lufs = anchor_lufs - 1.2
            shortfall = target_lufs - _post_fader_lufs(plan)
            if shortfall >= 0.5:
                adjustment = float(np.clip(shortfall, 0.0, 2.0))
                actions.append(AgentAction(
                    action_type="adjust_gain",
                    channel=channel,
                    parameters={"channel": channel, "adjustment_db": adjustment},
                    priority=1,
                    confidence=0.72,
                    reason=(
                        "Codex correction: keep additional lead vocals within about "
                        "1-1.5 dB of the loudest lead so verse/feature lines do not drop behind the band."
                    ),
                    source="codex",
                    risk="low",
                    expected_effect="More even lead-vocal handoff without making all vocal mics equally dominant.",
                    rollback_hint=f"Lower channel {channel} fader by {adjustment:.1f} dB.",
                ))

    for channel, plan in plans.items():
        if plan.instrument != "backing_vocal" or plan.muted:
            continue
        actions.append(AgentAction(
            action_type="reduce_gain",
            channel=channel,
            parameters={"channel": channel, "amount_db": -2.0},
            priority=2,
            confidence=0.74,
            reason=(
                "Codex correction: backing vocals are already compressed and read too forward, "
                "so lower the pair instead of adding more compression or presence."
            ),
            source="codex",
            risk="low",
            expected_effect="Back vocals sit behind the lead vocals with less apparent loudness.",
            rollback_hint=f"Raise channel {channel} fader by 2.0 dB.",
        ))

    return actions


def _set_compressor_plan(plan: ChannelPlan, threshold_db: float, ratio: float, attack_ms: float, release_ms: float):
    plan.comp_threshold_db = float(threshold_db)
    plan.comp_ratio = float(ratio)
    plan.comp_attack_ms = float(attack_ms)
    plan.comp_release_ms = float(release_ms)


def apply_codex_bleed_control(plans: dict[int, ChannelPlan]) -> dict[str, Any]:
    """Reduce cymbal bleed buildup and avoid re-compressing pre-compressed backing vocals."""
    changes: list[dict[str, Any]] = []

    def record(channel: int, plan: ChannelPlan, change: str, before: Any, after: Any, reason: str):
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "change": change,
            "before": before,
            "after": after,
            "reason": reason,
            "source": "codex",
        })

    def add_eq_cut(channel: int, plan: ChannelPlan, freq: float, gain: float, q: float, reason: str):
        before = list(plan.eq_bands)
        plan.eq_bands.append((freq, gain, q))
        record(channel, plan, "append_eq_band", before, list(plan.eq_bands), reason)

    for channel, plan in plans.items():
        if plan.muted:
            continue

        if plan.instrument == "backing_vocal":
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            _set_compressor_plan(plan, 0.0, 1.0, 10.0, 100.0)
            record(
                channel,
                plan,
                "disable_channel_compression",
                before_comp,
                {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio},
                "Backing vocals are already compressed in the source; extra compression makes them appear louder than the leads.",
            )
            before_lpf = plan.lpf
            plan.lpf = 9500.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Keep backing vocal air controlled after removing extra compression.")

        if plan.instrument in {"kick", "snare", "rack_tom", "floor_tom"}:
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            if plan.instrument == "kick":
                _set_compressor_plan(plan, -10.0, 1.4, 12.0, 120.0)
                lpf = 6800.0
            elif plan.instrument == "snare":
                _set_compressor_plan(plan, -11.0, 1.5, 12.0, 130.0)
                lpf = 7600.0
            else:
                _set_compressor_plan(plan, 0.0, 1.0, 12.0, 160.0)
                lpf = 5800.0
            record(
                channel,
                plan,
                "soften_or_disable_drum_compression",
                before_comp,
                {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio, "attack_ms": plan.comp_attack_ms, "release_ms": plan.comp_release_ms},
                "Close drum mics contain strong cymbal bleed; compression was pulling that bleed forward.",
            )
            before_lpf = plan.lpf
            plan.lpf = lpf
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Low-pass close drum mics so cymbal spill does not stack across the kit.")
            add_eq_cut(channel, plan, 6500.0, -2.0, 1.1, "Presence cut targets cymbal bleed in close drum mics.")
            add_eq_cut(channel, plan, 9500.0, -3.0, 0.9, "Air-band cut reduces repeated cymbal spill across close drum mics.")

        if plan.instrument in {"hi_hat", "ride"}:
            before_fader = plan.fader_db
            plan.fader_db = float(np.clip(plan.fader_db - 2.5, -100.0, 10.0))
            record(channel, plan, "adjust_fader_db", round(before_fader, 2), round(plan.fader_db, 2), "Direct cymbal spot mics were adding to already heavy cymbal bleed.")
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            _set_compressor_plan(plan, 0.0, 1.0, 20.0, 180.0)
            record(channel, plan, "disable_channel_compression", before_comp, {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio}, "Do not compress cymbal spot mics when the kit already has cymbal spill.")
            before_lpf = plan.lpf
            plan.lpf = 10500.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Keep cymbal spot tone present but less splashy.")

        if plan.instrument == "overhead":
            before_fader = plan.fader_db
            plan.fader_db = float(np.clip(plan.fader_db - 1.2, -100.0, 10.0))
            record(channel, plan, "adjust_fader_db", round(before_fader, 2), round(plan.fader_db, 2), "Overheads are the main cymbal picture, but the rest of the kit also contains cymbal bleed.")
            before_lpf = plan.lpf
            plan.lpf = 9000.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Softer overhead top end for a less cymbal-heavy drum image.")
            add_eq_cut(channel, plan, 6500.0, -1.5, 1.0, "Presence cut reduces harsh cymbal build-up.")

        if plan.instrument == "room":
            before_fader = plan.fader_db
            plan.fader_db = float(np.clip(plan.fader_db - 2.0, -100.0, 10.0))
            record(channel, plan, "adjust_fader_db", round(before_fader, 2), round(plan.fader_db, 2), "Room mics are useful for space but add broad cymbal wash in this multitrack.")
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            _set_compressor_plan(plan, 0.0, 1.0, 20.0, 220.0)
            record(channel, plan, "disable_channel_compression", before_comp, {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio}, "Avoid lifting cymbal decay from room microphones.")
            before_lpf = plan.lpf
            plan.lpf = 6500.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Keep room size while reducing cymbal splash.")

    return {
        "enabled": True,
        "changes": changes,
        "notes": [
            "Cymbal bleed is handled by reducing high-frequency buildup at every drum capture point, not only overheads.",
            "Backing vocals are treated as pre-compressed sources: lower level, no extra compressor.",
        ],
    }


def _frame_rms_db(x: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    mono = mono_sum(x) if x.ndim > 1 else x
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    starts = np.arange(0, max(1, len(mono) - frame + 1), hop, dtype=np.int64)
    values = []
    for start in starts:
        block = mono[start:start + frame]
        values.append(amp_to_db(float(np.sqrt(np.mean(np.square(block))) + 1e-12)))
    return starts.astype(np.float32), np.asarray(values, dtype=np.float32)


def _smooth_gain_db(target_db: np.ndarray, sr: int, hop: int, attack_ms: float = 180.0, release_ms: float = 900.0) -> np.ndarray:
    attack = math.exp(-hop / max(1.0, attack_ms * 0.001 * sr))
    release = math.exp(-hop / max(1.0, release_ms * 0.001 * sr))
    smoothed = np.zeros_like(target_db, dtype=np.float32)
    last = 0.0
    for i, value in enumerate(target_db):
        coeff = attack if value < last else release
        last = coeff * last + (1.0 - coeff) * float(value)
        smoothed[i] = last
    return smoothed


def apply_dynamic_vocal_priority(
    rendered: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    max_duck_db: float = 3.5,
    duck_drum_instruments: bool = True,
) -> dict[str, Any]:
    """Apply chorus-aware DCA ducking of non-vocal channels while vocal is active.

    If ducking is disabled for drums, kick/snare/toms/cymbal channels stay untouched.
    """
    vocal_channels = [
        channel for channel, plan in plans.items()
        if plan.instrument == "lead_vocal" and channel in rendered and not plan.muted
    ]
    if not vocal_channels:
        return {}

    vocal = sum((rendered[channel] for channel in vocal_channels), np.zeros_like(next(iter(rendered.values()))))
    bed = sum(
        (audio for channel, audio in rendered.items() if channel not in vocal_channels and not plans[channel].muted),
        np.zeros_like(vocal),
    )

    frame = int(0.35 * sr)
    hop = int(0.08 * sr)
    starts, vocal_db = _frame_rms_db(vocal, frame, hop)
    _, bed_db = _frame_rms_db(bed, frame, hop)
    active = vocal_db > -36.0
    if not np.any(active):
        return {}

    active_bed = bed_db[active]
    bed_mid = float(np.percentile(active_bed, 55))
    bed_loud = float(np.percentile(active_bed, 88))
    density_span = max(1.0, bed_loud - bed_mid)
    density = np.clip((bed_db - bed_mid) / density_span, 0.0, 1.0)

    # Stronger ducking is only requested when the arrangement gets dense and
    # the vocal-to-bed ratio is not generous enough for a live-style lead.
    vocal_margin = vocal_db - bed_db
    margin_shortfall = np.clip((4.0 - vocal_margin) / 8.0, 0.0, 1.0)
    duck_db = -max_duck_db * density * margin_shortfall
    duck_db = np.where(active, duck_db, 0.0).astype(np.float32)
    duck_db = _smooth_gain_db(duck_db, sr, hop)

    sample_points = np.clip(starts + frame // 2, 0, len(vocal) - 1)
    full_points = np.concatenate(([0.0], sample_points, [float(len(vocal) - 1)]))
    full_duck = np.concatenate(([duck_db[0]], duck_db, [duck_db[-1]]))
    envelope_db = np.interp(np.arange(len(vocal)), full_points, full_duck).astype(np.float32)
    gain = (10.0 ** (envelope_db / 20.0)).astype(np.float32)

    adjusted_channels = []
    for channel, audio in rendered.items():
        if channel in vocal_channels or plans[channel].muted:
            continue
        if not duck_drum_instruments and plans[channel].instrument in DRUM_INSTRUMENTS:
            continue
        rendered[channel] = (audio * gain[:, None]).astype(np.float32)
        adjusted_channels.append(channel)

    ducked = duck_db < -0.25
    return {
        "vocal_channels": vocal_channels,
        "adjusted_channels": adjusted_channels,
        "max_duck_db": round(abs(float(np.min(duck_db))), 2),
        "mean_duck_when_active_db": round(abs(float(np.mean(duck_db[active]))), 2),
        "ducked_sec": round(float(np.sum(ducked) * hop / sr), 2),
        "active_vocal_sec": round(float(np.sum(active) * hop / sr), 2),
        "bed_mid_db": round(bed_mid, 2),
        "bed_loud_db": round(bed_loud, 2),
    }


def _priority_for_instrument(instrument: str) -> int:
    """Lower number means the channel is more important in mirror EQ."""
    if instrument == "lead_vocal":
        return 1
    if instrument in {"kick", "snare", "bass_guitar"}:
        return 2
    if instrument in {"electric_guitar", "accordion", "playback", "backing_vocal"}:
        return 3
    if instrument in {"overhead", "room", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
        return 4
    return 3


def _analysis_block(x: np.ndarray, sr: int, window_sec: float = 18.0) -> np.ndarray:
    window = min(len(x), max(1024, int(window_sec * sr)))
    if len(x) <= window:
        return x
    hop = max(512, window // 3)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return x[best_start:best_start + window]


def _event_activity_ranges(x: np.ndarray, sr: int, instrument: str | None) -> dict[str, Any] | None:
    config = _event_metric_config(instrument or "")
    if not config or len(x) == 0:
        return None

    detect = highpass(x, sr, config["detect_hpf_hz"])
    detect = lowpass(detect, sr, config["detect_lpf_hz"])
    frame = max(256, int(config["frame_ms"] * sr / 1000.0))
    hop = max(64, int(config["hop_ms"] * sr / 1000.0))
    starts, rms_db = _frame_rms_db(detect, frame, hop)
    if len(rms_db) == 0:
        return {
            "config": config,
            "frame": frame,
            "hop": hop,
            "ranges": [],
            "threshold_db": None,
            "active_samples": 0,
        }

    peak_percentile = float(config.get("peak_percentile", 100.0))
    if len(detect):
        if peak_percentile >= 100.0:
            detect_peak = float(np.max(np.abs(detect)))
        else:
            detect_peak = float(np.percentile(np.abs(detect), peak_percentile))
    else:
        detect_peak = 0.0
    detect_peak_db = amp_to_db(detect_peak)
    noise_floor_db = float(np.percentile(rms_db, 50))
    threshold_db = max(
        float(np.percentile(rms_db, config["percentile"])),
        detect_peak_db - config["peak_offset_db"],
        noise_floor_db + config["floor_margin_db"],
        config["min_threshold_db"],
    )
    active_idx = np.flatnonzero(rms_db >= threshold_db)
    if len(active_idx) == 0:
        return {
            "config": config,
            "frame": frame,
            "hop": hop,
            "ranges": [],
            "threshold_db": round(threshold_db, 2),
            "active_samples": 0,
        }

    pad = int(config["pad_ms"] * sr / 1000.0)
    ranges = []
    for idx in active_idx:
        start = max(0, int(starts[idx]) - pad)
        end = min(len(x), int(starts[idx]) + frame + pad)
        if end > start:
            ranges.append((start, end))
    merged = _merge_ranges(ranges, gap=pad // 2)
    active_samples = sum(end - start for start, end in merged)
    return {
        "config": config,
        "frame": frame,
        "hop": hop,
        "ranges": merged,
        "threshold_db": round(threshold_db, 2),
        "active_samples": active_samples,
    }


def _analysis_signal_for_metrics(x: np.ndarray, sr: int, instrument: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    activity = _event_activity_ranges(x, sr, instrument)
    if not activity:
        block = _analysis_block(x, sr)
        return block, {
            "analysis_mode": "windowed_full_track",
            "analysis_active_sec": round(len(block) / sr, 3) if sr else 0.0,
            "analysis_active_ratio": round(len(block) / max(1, len(x)), 4),
            "analysis_threshold_db": None,
        }

    merged = activity["ranges"]
    threshold_db = activity["threshold_db"]
    frame = activity["frame"]
    if not merged:
        block = _analysis_block(x, sr)
        return block, {
            "analysis_mode": "windowed_full_track_fallback",
            "analysis_active_sec": round(len(block) / sr, 3) if sr else 0.0,
            "analysis_active_ratio": round(len(block) / max(1, len(x)), 4),
            "analysis_threshold_db": round(threshold_db, 2),
        }
    block = np.concatenate([x[start:end] for start, end in merged], axis=0) if merged else _analysis_block(x, sr)
    if len(block) < max(512, frame // 2):
        block = _analysis_block(x, sr)
        mode = "windowed_full_track_fallback"
    else:
        mode = "event_based"
    active_samples = activity["active_samples"]
    return block, {
        "analysis_mode": mode,
        "analysis_active_sec": round(active_samples / sr, 3) if sr else 0.0,
        "analysis_active_ratio": round(active_samples / max(1, len(x)), 4),
        "analysis_threshold_db": round(threshold_db, 2),
    }


def _event_expander_profile(instrument: str, metrics: dict[str, Any]) -> dict[str, float] | None:
    if metrics.get("analysis_mode") != "event_based":
        return None

    active_ratio = float(metrics.get("analysis_active_ratio") or 0.0)
    threshold_db = metrics.get("analysis_threshold_db")
    if threshold_db is None:
        return None

    if instrument == "lead_vocal":
        return {
            "range_db": 5.5 if active_ratio < 0.3 else 4.5,
            "open_ms": 18.0,
            "close_ms": 140.0,
            "hold_ms": 180.0,
            "threshold_db": float(threshold_db),
        }
    if instrument == "backing_vocal":
        return {
            "range_db": 6.0 if active_ratio < 0.35 else 5.0,
            "open_ms": 20.0,
            "close_ms": 150.0,
            "hold_ms": 150.0,
            "threshold_db": float(threshold_db),
        }
    if instrument in {"rack_tom", "floor_tom"}:
        return {
            "range_db": 12.0 if active_ratio < 0.08 else 10.0,
            "open_ms": 10.0,
            "close_ms": 110.0,
            "hold_ms": 130.0,
            "threshold_db": float(threshold_db),
        }
    if instrument == "kick":
        return {
            "range_db": 7.5 if active_ratio < 0.2 else 6.0,
            "open_ms": 8.0,
            "close_ms": 115.0,
            "hold_ms": 130.0,
            "threshold_db": float(threshold_db),
        }
    if instrument == "snare":
        return {
            "range_db": 8.5 if active_ratio < 0.22 else 7.0,
            "open_ms": 8.0,
            "close_ms": 120.0,
            "hold_ms": 145.0,
            "threshold_db": float(threshold_db),
        }
    if instrument in {"hi_hat", "ride", "percussion"}:
        return {
            "range_db": 5.0 if active_ratio < 0.45 else 4.0,
            "open_ms": 14.0,
            "close_ms": 160.0,
            "hold_ms": 180.0,
            "threshold_db": float(threshold_db),
        }
    return None


def apply_event_based_dynamics(plans: dict[int, ChannelPlan]) -> dict[str, Any]:
    """Configure gentle event-based expanders so bleed does not drive dynamics."""
    changes: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        if plan.muted:
            continue
        profile = _event_expander_profile(plan.instrument, plan.metrics)
        if not profile:
            continue
        plan.expander_enabled = True
        plan.expander_range_db = float(profile["range_db"])
        plan.expander_open_ms = float(profile["open_ms"])
        plan.expander_close_ms = float(profile["close_ms"])
        plan.expander_hold_ms = float(profile["hold_ms"])
        plan.expander_threshold_db = float(profile["threshold_db"])
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "analysis_active_ratio": round(float(plan.metrics.get("analysis_active_ratio") or 0.0), 4),
            "analysis_threshold_db": plan.expander_threshold_db,
            "range_db": round(plan.expander_range_db, 2),
            "open_ms": round(plan.expander_open_ms, 2),
            "close_ms": round(plan.expander_close_ms, 2),
            "hold_ms": round(plan.expander_hold_ms, 2),
            "reason": "Event-based activity windows drive a soft expander so bleed between hits or phrases does not keep the channel artificially open.",
        })

    return {
        "enabled": bool(changes),
        "changes": changes,
        "notes": [
            "Lead and backing vocals get a gentle downward expander between phrases.",
            "Rack and floor toms get a stronger event-based expander between hits.",
            "Kick, snare, and cymbal spot mics also use event-based expansion when bleed would otherwise keep them open.",
            "Ambient microphones such as overheads and rooms stay ungated.",
        ],
    }


def _band_energy_for_rendered(audio: np.ndarray, sr: int) -> dict[str, float]:
    mono = mono_sum(audio)
    block = _analysis_block(mono, sr)
    if len(block) < 1024:
        return {}
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    bands = {
        "sub": (20.0, 60.0),
        "bass": (60.0, 250.0),
        "low_mid": (250.0, 500.0),
        "mid": (500.0, 2000.0),
        "high_mid": (2000.0, 4000.0),
        "high": (4000.0, 8000.0),
        "air": (8000.0, 14000.0),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            out[name] = -100.0
            continue
        band_rms = float(np.sqrt(np.sum(np.square(spec[idx]))) / len(windowed))
        out[name] = amp_to_db(band_rms)
    return out


def _cross_band_for_frequency(freq: float) -> str:
    centers = CrossAdaptiveEQ.BAND_CENTERS
    return min(centers.keys(), key=lambda name: abs(centers[name] - freq))


def apply_cross_adaptive_eq(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
) -> dict[str, Any]:
    """Run priority-based CrossAdaptiveEQ and add conservative channel EQ moves."""
    preview = {
        channel: render_channel(plan.path, plan, target_len, sr)
        for channel, plan in plans.items()
        if not plan.muted
    }
    channel_band_energy = {
        channel: _band_energy_for_rendered(audio, sr)
        for channel, audio in preview.items()
    }
    channel_priorities = {
        channel: _priority_for_instrument(plan.instrument)
        for channel, plan in plans.items()
        if channel in preview
    }

    processor = CrossAdaptiveEQ(
        min_band_level_db=-82.0,
        overlap_tolerance_db=8.0,
        max_cut_db=-2.5,
        max_boost_db=1.2,
    )
    raw_adjustments = processor.calculate_corrections(channel_band_energy, channel_priorities)

    aggregated: dict[tuple[int, float], dict[str, Any]] = {}
    skipped = 0
    for adj in raw_adjustments:
        channel = int(adj.channel_id)
        plan = plans.get(channel)
        if not plan:
            skipped += 1
            continue
        band = _cross_band_for_frequency(float(adj.frequency_hz))
        priority = channel_priorities.get(channel, 3)
        gain = float(adj.gain_db)

        if band == "air":
            skipped += 1
            continue
        if gain < 0.0 and priority == 1:
            skipped += 1
            continue
        if gain > 0.0 and priority != 1:
            skipped += 1
            continue
        if gain > 0.0 and band in {"sub", "bass", "low_mid"}:
            skipped += 1
            continue

        if band in {"mid", "high_mid", "high"}:
            scale = 0.36 if gain < 0.0 else 0.32
        elif band in {"low_mid", "bass"}:
            scale = 0.24 if gain < 0.0 else 0.20
        else:
            scale = 0.20

        key = (channel, float(adj.frequency_hz))
        existing = aggregated.setdefault(key, {
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "priority": priority,
            "frequency_hz": float(adj.frequency_hz),
            "band": band,
            "gain_db": 0.0,
            "q": float(adj.q_factor),
            "source_count": 0,
        })
        existing["gain_db"] += gain * scale
        existing["source_count"] += 1

    applied = []
    by_channel: dict[int, list[dict[str, Any]]] = {}
    for item in aggregated.values():
        channel = int(item["channel"])
        priority = int(item["priority"])
        gain = float(item["gain_db"])
        if gain < 0.0:
            lower = -2.5 if priority >= 4 else (-2.0 if priority == 3 else -1.0)
            gain = max(lower, gain)
        else:
            gain = min(0.9, gain)
        if abs(gain) < 0.35:
            skipped += 1
            continue
        item["gain_db"] = round(gain, 2)
        item["q"] = round(float(item["q"]), 2)
        by_channel.setdefault(channel, []).append(item)

    for channel, items in by_channel.items():
        # Keep this implementable on a real console: at most three anti-mask
        # moves per channel, with vocal-presence bands taking precedence.
        items.sort(key=lambda item: (
            0 if item["band"] in {"mid", "high_mid", "high"} else 1,
            -abs(float(item["gain_db"])),
        ))
        for item in items[:3]:
            plan = plans[channel]
            plan.eq_bands.append((
                float(item["frequency_hz"]),
                float(item["gain_db"]),
                float(item["q"]),
            ))
            plan.cross_adaptive_eq.append(dict(item))
            applied.append(dict(item))

    applied.sort(key=lambda item: (item["priority"], item["channel"], item["frequency_hz"]))
    return {
        "enabled": True,
        "raw_adjustments": len(raw_adjustments),
        "applied_adjustments": len(applied),
        "skipped_adjustments": skipped,
        "channel_priorities": {
            str(channel): priority
            for channel, priority in sorted(channel_priorities.items())
        },
        "applied": applied,
    }


def classify_track(path: Path) -> tuple[str, float, float, float, list[tuple[float, float, float]], tuple[float, float, float, float], bool]:
    name = path.stem.lower()
    if "kick" in name:
        return "kick", 0.0, 35.0, -20.0, [(60, 3.0, 0.9), (320, -3.0, 1.3), (4200, 2.0, 1.2)], (-18, 4.0, 8, 90), False
    if "snare bottom" in name or "snare b" in name:
        return "snare", 0.02, 120.0, -27.0, [(220, 1.0, 1.0), (5200, 2.0, 1.2), (850, -2.0, 2.0)], (-22, 3.5, 8, 110), True
    if "snare top" in name or name == "snare t" or name.startswith("snare t ") or name.endswith(" snare t") or name == "snaret":
        return "snare", -0.02, 90.0, -22.0, [(200, 2.0, 1.0), (850, -2.5, 2.0), (5200, 3.0, 1.0)], (-20, 4.0, 6, 120), False
    if "snare" in name or "snate" in name:
        return "snare", 0.0, 110.0, -23.0, [(220, -2.0, 1.1), (750, -2.5, 1.7), (4800, 2.0, 1.0)], (-19, 3.0, 8, 130), False
    if "floor" in name or name.startswith("f tom") or name.startswith("ftom"):
        return "floor_tom", 0.25, 55.0, -24.5, [(95, 2.0, 1.0), (360, -2.5, 1.5), (4200, 1.5, 1.2)], (-22, 3.0, 12, 150), False
    if "tom" in name:
        return "rack_tom", -0.20, 65.0, -25.0, [(120, 2.0, 1.0), (380, -2.0, 1.5), (4300, 1.5, 1.2)], (-22, 3.0, 12, 150), False
    if "hi hat" in name or "hi-hat" in name or "hihat" in name or name in {"hh", "hat"}:
        return "hi_hat", 0.0, 180.0, -28.0, [(450, -1.5, 1.2), (6500, 1.5, 1.0), (9500, 1.0, 0.8)], (-18, 1.8, 18, 180), False
    if "ride" in name:
        return "ride", 0.0, 170.0, -28.5, [(420, -1.2, 1.2), (4800, 1.2, 1.0), (9000, 1.0, 0.8)], (-18, 1.8, 20, 200), False
    if "overhead l" in name or name in {"oh l", "ohl"} or name.endswith(" oh l"):
        return "overhead", -0.72, 150.0, -27.0, [(350, -1.5, 1.2), (3500, -1.0, 1.5), (10500, 1.5, 0.8)], (-18, 1.6, 25, 300), False
    if "overhead r" in name or name in {"oh r", "ohr"} or name.endswith(" oh r"):
        return "overhead", 0.72, 150.0, -27.0, [(350, -1.5, 1.2), (3500, -1.0, 1.5), (10500, 1.5, 0.8)], (-18, 1.6, 25, 300), False
    if name.startswith("room dr l") or name.startswith("room l") or name in {"roomdr l", "room l"}:
        return "room", -0.78, 80.0, -31.0, [(250, -1.5, 1.2), (2500, -1.0, 1.4), (8500, 0.8, 1.0)], (-20, 2.0, 20, 220), False
    if name.startswith("room dr r") or name.startswith("room r") or name in {"roomdr r", "room r"}:
        return "room", 0.78, 80.0, -31.0, [(250, -1.5, 1.2), (2500, -1.0, 1.4), (8500, 0.8, 1.0)], (-20, 2.0, 20, 220), False
    if "room" in name:
        return "room", 0.0, 80.0, -31.0, [(250, -1.5, 1.2), (2500, -1.0, 1.4), (8500, 0.8, 1.0)], (-20, 2.0, 20, 220), False
    if "bass" in name:
        return "bass_guitar", 0.0, 35.0, -21.0, [(80, 2.0, 0.9), (250, -2.5, 1.2), (750, 1.2, 1.0)], (-24, 4.0, 18, 180), False
    if "guitar l" in name:
        return "electric_guitar", -0.65, 90.0, -25.5, [(250, -2.0, 1.2), (2500, 1.5, 1.0), (6200, -1.0, 1.0)], (-20, 2.0, 12, 140), False
    if "guitar r" in name:
        return "electric_guitar", 0.65, 90.0, -25.5, [(250, -2.0, 1.2), (2500, 1.5, 1.0), (6200, -1.0, 1.0)], (-20, 2.0, 12, 140), False
    if "accordion" in name:
        return "accordion", -0.10, 100.0, -23.5, [(350, -1.5, 1.3), (2300, 1.3, 1.0), (7000, 0.8, 1.0)], (-21, 2.2, 15, 160), False
    if "playback l" in name:
        return "playback", -0.78, 30.0, -23.0, [(180, -0.8, 1.0), (3500, 0.8, 1.0)], (-16, 1.5, 25, 250), False
    if "playbacks r" in name or "playback r" in name:
        return "playback", 0.78, 30.0, -23.0, [(180, -0.8, 1.0), (3500, 0.8, 1.0)], (-16, 1.5, 25, 250), False
    if "back vox l" in name or "back vocal l" in name or "bvox l" in name or name in {"backs l", "back l", "bgv l"}:
        return "backing_vocal", -0.32, 110.0, -25.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-23, 2.5, 8, 140), False
    if "back vox r" in name or "back vocal r" in name or "bvox r" in name or name in {"backs r", "back r", "bgv r"}:
        return "backing_vocal", 0.32, 110.0, -25.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-23, 2.5, 8, 140), False
    if "back vox" in name or "back vocal" in name or "bvox" in name or "backs" in name or "bgv" in name:
        return "backing_vocal", 0.0, 110.0, -25.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-23, 2.5, 8, 140), False
    if "vox" in name or "vocal" in name:
        return "lead_vocal", 0.0, 90.0, -20.0, [(250, -2.5, 1.4), (3100, 2.5, 1.0), (10500, 1.5, 0.8)], (-22, 3.2, 5, 120), False
    return "custom", 0.0, 80.0, -24.0, [(300, -1.0, 1.2), (3000, 0.8, 1.0)], (-20, 2.0, 10, 150), False


def highpass(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    if freq <= 0:
        return x
    b, a = butter(2, freq / (sr * 0.5), btype="highpass")
    return lfilter(b, a, x).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    if freq <= 0 or freq >= sr * 0.49:
        return x
    b, a = butter(2, freq / (sr * 0.5), btype="lowpass")
    return lfilter(b, a, x).astype(np.float32)


def peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    if abs(gain_db) < 1e-4:
        return x
    a = db_to_amp(gain_db)
    w0 = 2.0 * math.pi * float(freq) / sr
    alpha = math.sin(w0) / (2.0 * max(float(q), 0.05))
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    aa = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, aa, x).astype(np.float32)


def compressor(x: np.ndarray, sr: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_db: float = 0.0) -> np.ndarray:
    abs_x = np.maximum(np.abs(x), 1e-8)
    level_db = 20.0 * np.log10(abs_x)
    over = level_db - threshold_db
    gain_reduction_db = np.where(over > 0, over * (1.0 - 1.0 / max(ratio, 1.0)), 0.0)
    attack = math.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr))
    release = math.exp(-1.0 / max(1.0, release_ms * 0.001 * sr))
    smoothed = np.zeros_like(gain_reduction_db, dtype=np.float32)
    last = 0.0
    for i, gr in enumerate(gain_reduction_db):
        coeff = attack if gr > last else release
        last = coeff * last + (1.0 - coeff) * float(gr)
        smoothed[i] = last
    gain = 10.0 ** ((makeup_db - smoothed) / 20.0)
    return (x * gain).astype(np.float32)


def pan_mono(x: np.ndarray, pan: float) -> np.ndarray:
    left, right = _equal_power_gains(pan)
    return np.column_stack((x * left, x * right)).astype(np.float32)


def metrics_for(x: np.ndarray, sr: int, instrument: str | None = None) -> dict[str, Any]:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    block, analysis_meta = _analysis_signal_for_metrics(x, sr, instrument)
    rms = float(np.sqrt(np.mean(np.square(block))) + 1e-12) if len(block) else 0.0
    freqs = np.fft.rfftfreq(min(len(block), sr * 8), 1.0 / sr)
    block = block[: len(freqs) * 2 - 2] if len(freqs) > 1 else block[:0]
    if len(block) > 0:
        spec = np.abs(np.fft.rfft(block * np.hanning(len(block)))) + 1e-12
        total = float(np.sum(spec))
        def band(lo: float, hi: float) -> float:
            idx = (freqs >= lo) & (freqs < hi)
            if not np.any(idx):
                return -100.0
            return amp_to_db(float(np.sum(spec[idx]) / total))
        band_energy = {
            "sub": band(20, 60),
            "bass": band(60, 250),
            "low_mid": band(250, 500),
            "mid": band(500, 2000),
            "presence": band(4000, 8000),
        }
    else:
        band_energy = {}
    return {
        "peak_db": amp_to_db(peak),
        "true_peak_db": amp_to_db(peak),
        "rms_db": amp_to_db(rms),
        "lufs_momentary": amp_to_db(rms),
        "dynamic_range_db": max(0.0, amp_to_db(peak) - amp_to_db(rms)),
        "band_energy": band_energy,
        "channel_armed": rms > db_to_amp(-55.0) or peak > db_to_amp(-45.0),
        "needs_attention": peak > db_to_amp(-6.0) or peak < db_to_amp(-30.0),
        "is_muted": False,
        **analysis_meta,
    }


def apply_event_based_expander(x: np.ndarray, sr: int, plan: ChannelPlan) -> tuple[np.ndarray, dict[str, Any]]:
    if not plan.expander_enabled or len(x) == 0:
        return x, {"enabled": False}

    activity = plan.event_activity or _event_activity_ranges(x, sr, plan.instrument)
    if not activity or not activity["ranges"]:
        return x, {
            "enabled": False,
            "reason": "no_event_ranges_detected",
        }

    frame = int(activity["frame"])
    hop = int(activity["hop"])
    starts = np.arange(0, max(1, len(x) - frame + 1), hop, dtype=np.int64)
    if len(starts) == 0:
        return x, {"enabled": False, "reason": "no_frames"}

    hold = int(plan.expander_hold_ms * sr / 1000.0)
    active = np.zeros(len(starts), dtype=bool)
    for start, end in activity["ranges"]:
        extended_end = min(len(x), end + hold)
        active |= (starts < extended_end) & ((starts + frame) > start)
    if not np.any(active):
        return x, {"enabled": False, "reason": "no_active_frames"}

    target_db = np.where(active, 0.0, -plan.expander_range_db).astype(np.float32)
    smoothed_db = _smooth_gain_db(
        target_db,
        sr,
        hop,
        attack_ms=plan.expander_close_ms,
        release_ms=plan.expander_open_ms,
    )

    sample_points = np.clip(starts + frame // 2, 0, len(x) - 1)
    full_points = np.concatenate(([0.0], sample_points.astype(np.float32), [float(len(x) - 1)]))
    full_gain = np.concatenate(([smoothed_db[0]], smoothed_db, [smoothed_db[-1]])).astype(np.float32)
    envelope_db = np.interp(np.arange(len(x), dtype=np.float32), full_points, full_gain).astype(np.float32)
    gain = np.power(10.0, envelope_db / 20.0, dtype=np.float32)
    out = (x * gain).astype(np.float32)

    inactive = ~active
    return out, {
        "enabled": True,
        "mode": "event_based_expander",
        "range_db": round(plan.expander_range_db, 2),
        "threshold_db": activity["threshold_db"],
        "open_ms": round(plan.expander_open_ms, 2),
        "close_ms": round(plan.expander_close_ms, 2),
        "hold_ms": round(plan.expander_hold_ms, 2),
        "active_ratio": round(float(np.mean(active)), 4),
        "active_sec": round(float(np.sum(active) * hop / sr), 3),
        "max_reduction_db": round(float(np.max(-smoothed_db)), 2),
        "mean_reduction_inactive_db": round(float(np.mean(-smoothed_db[inactive])), 2) if np.any(inactive) else 0.0,
        "mean_reduction_active_db": round(float(np.mean(-smoothed_db[active])), 2),
    }


def render_channel(path: Path, plan: ChannelPlan, target_len: int, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{path.name}: expected {sr} Hz, got {file_sr} Hz")
    mono = mono_sum(audio)
    if len(mono) < target_len:
        mono = np.pad(mono, (0, target_len - len(mono)))
    mono = mono[:target_len]
    mono = declick_start(mono, sr, plan.input_fade_ms)
    if plan.phase_invert:
        mono = -mono
    mono = delay_signal(mono, sr, plan.delay_ms)
    x = mono * db_to_amp(plan.trim_db + plan.fader_db)
    x = highpass(x, sr, plan.hpf)
    if plan.lpf > 0.0:
        x = lowpass(x, sr, plan.lpf)
    x, plan.expander_report = apply_event_based_expander(x, sr, plan)
    for freq, gain, q in plan.eq_bands:
        x = peaking_eq(x, sr, freq, gain, q)
    x = compressor(
        x,
        sr,
        threshold_db=plan.comp_threshold_db,
        ratio=plan.comp_ratio,
        attack_ms=plan.comp_attack_ms,
        release_ms=plan.comp_release_ms,
        makeup_db=0.0,
    )
    return pan_mono(x, plan.pan)


def apply_live_channel_peak_headroom(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    channel_peak_ceiling_db: float = -3.0,
) -> dict[str, Any]:
    """Use static channel fader trims for no-master-limiter live headroom."""
    adjusted = []
    for channel, plan in plans.items():
        if plan.muted:
            continue
        rendered = render_channel(plan.path, plan, target_len, sr)
        peak_db = amp_to_db(float(np.max(np.abs(rendered))) if len(rendered) else 0.0)
        reduction_db = min(0.0, channel_peak_ceiling_db - peak_db)
        if reduction_db < -0.05:
            plan.fader_db = float(np.clip(plan.fader_db + reduction_db, -100.0, 10.0))
            adjusted.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "rendered_peak_before_dbfs": round(peak_db, 2),
                "fader_reduction_db": round(reduction_db, 2),
                "fader_after_db": round(plan.fader_db, 2),
            })
    return {
        "enabled": True,
        "channel_peak_ceiling_dbfs": round(channel_peak_ceiling_db, 2),
        "adjusted_channels": adjusted,
    }


def apply_bass_drum_push(plans: dict[int, ChannelPlan], boost_db: float) -> dict[str, Any]:
    """Push bass and drum stems for a fuller low-end and punchy kit picture."""
    if boost_db <= 0.0:
        return {"enabled": False}

    target_instruments = {
        "bass_guitar",
        "kick",
        "snare",
        "rack_tom",
        "floor_tom",
        "hi_hat",
        "ride",
    }
    changes = []
    for channel, plan in plans.items():
        if plan.muted or plan.instrument not in target_instruments:
            continue
        before = plan.fader_db
        after = float(np.clip(before + boost_db, -100.0, 10.0))
        if after == before:
            continue
        plan.fader_db = after
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "fader_before_db": round(before, 2),
            "fader_after_db": round(after, 2),
            "boost_db": round(boost_db, 2),
        })

    return {
        "enabled": bool(changes),
        "boost_db": round(boost_db, 2),
        "affected_channels": len(changes),
        "changes": changes,
    }


def apply_kick_presence_boost(plans: dict[int, ChannelPlan], boost_db: float) -> dict[str, Any]:
    """Raise only the kick stem and add subtle low-end shaping for stronger punch."""
    if boost_db <= 0.0:
        return {"enabled": False}

    changes = []
    for channel, plan in plans.items():
        if plan.muted or plan.instrument != "kick":
            continue

        fader_before = plan.fader_db
        fader_after = float(np.clip(fader_before + boost_db, -100.0, 10.0))
        if fader_after != fader_before:
            plan.fader_db = fader_after
            changes.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "fader_before_db": round(fader_before, 2),
                "fader_after_db": round(fader_after, 2),
                "boost_db": round(boost_db, 2),
            })

        eq_before = list(plan.eq_bands)
        if boost_db >= 2.0:
            # Emphasize body + beater click for clearer kick translation on big systems.
            plan.eq_bands.append((65.0, 1.2, 1.0))
            plan.eq_bands.append((3000.0, 0.9, 1.8))
            changes.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "change_type": "eq_shape_added",
                "eq_before_count": len(eq_before),
                "eq_after_count": len(plan.eq_bands),
            })

    return {
        "enabled": bool(changes),
        "boost_db": round(boost_db, 2),
        "changes": changes,
    }


def apply_cymbal_cleanup_for_kick_focus(plans: dict[int, ChannelPlan], cymbal_atten_db: float) -> dict[str, Any]:
    """Reduce cymbal stems slightly so kick body is easier to hear."""
    if cymbal_atten_db <= 0.0:
        return {"enabled": False}

    target_instruments = {
        "hi_hat",
        "ride",
    }
    changes = []
    for channel, plan in plans.items():
        if plan.muted or plan.instrument not in target_instruments:
            continue
        before = plan.fader_db
        after = float(np.clip(before - cymbal_atten_db, -100.0, 10.0))
        if after == before:
            continue
        plan.fader_db = after
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "fader_before_db": round(before, 2),
            "fader_after_db": round(after, 2),
            "atten_db": round(cymbal_atten_db, 2),
        })

    return {
        "enabled": bool(changes),
        "atten_db": round(cymbal_atten_db, 2),
        "changes": changes,
    }


def _filter_stereo_return(audio: np.ndarray, sr: int, hpf_hz: float, lpf_hz: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    for ch in range(2):
        out[:, ch] = highpass(out[:, ch], sr, hpf_hz)
        out[:, ch] = lowpass(out[:, ch], sr, lpf_hz)
    return out.astype(np.float32)


def _simple_reverb(bus_input: np.ndarray, sr: int, bus: FXBusDecision) -> np.ndarray:
    predelay = int(float(bus.params.get("predelay_ms", 30.0)) * sr / 1000.0)
    decay_s = max(0.2, float(bus.params.get("decay_s", 1.0)))
    density = float(bus.params.get("density", 0.7))
    brightness = float(bus.params.get("brightness", 0.5))
    wet = np.zeros_like(bus_input, dtype=np.float32)
    source = np.pad(bus_input, ((predelay, 0), (0, 0)))[:len(bus_input)]

    # Sparse multi-tap reverb: intentionally lightweight for full-song offline
    # renders while still producing early reflections and a short musical tail.
    early_ms = [19, 29, 37, 43, 53, 67, 79, 97]
    tail_ms = np.linspace(115.0, max(170.0, decay_s * 820.0), int(10 + density * 12))
    all_taps = list(early_ms) + [float(v) for v in tail_ms]
    for idx, delay_ms_value in enumerate(all_taps):
        delay = max(1, int(delay_ms_value * sr / 1000.0))
        if delay >= len(source):
            continue
        time_s = delay / sr
        gain = math.exp(-time_s / max(0.15, decay_s * 0.52)) * (0.30 if idx < len(early_ms) else 0.18)
        gain *= 1.0 + (density - 0.5) * 0.18
        if idx % 2 == 0:
            wet[delay:, 0] += source[:-delay, 0] * gain
            wet[delay:, 1] += source[:-delay, 1] * gain * 0.82
        else:
            wet[delay:, 0] += source[:-delay, 1] * gain * 0.82
            wet[delay:, 1] += source[:-delay, 0] * gain

    wet *= 1.35
    wet = _filter_stereo_return(wet, sr, bus.hpf_hz, min(bus.lpf_hz, 4200.0 + brightness * 5200.0))
    return wet.astype(np.float32)


def _tempo_delay(bus_input: np.ndarray, sr: int, bus: FXBusDecision) -> np.ndarray:
    left_delay = int(float(bus.params.get("left_delay_ms", 375.0)) * sr / 1000.0)
    right_delay = int(float(bus.params.get("right_delay_ms", 500.0)) * sr / 1000.0)
    feedback = float(np.clip(bus.params.get("feedback", 0.2), 0.0, 0.55))
    wet = np.zeros_like(bus_input, dtype=np.float32)
    mono = mono_sum(bus_input)
    for repeat in range(1, 4):
        gain = feedback ** (repeat - 1)
        l_start = left_delay * repeat
        r_start = right_delay * repeat
        if l_start < len(wet):
            wet[l_start:, 0] += mono[:len(wet) - l_start] * gain
        if r_start < len(wet):
            wet[r_start:, 1] += mono[:len(wet) - r_start] * gain
    wet *= 1.0
    return _filter_stereo_return(wet, sr, bus.hpf_hz, bus.lpf_hz)


def _chorus_doubler(bus_input: np.ndarray, sr: int, bus: FXBusDecision) -> np.ndarray:
    left_delay = int(float(bus.params.get("left_delay_ms", 11.0)) * sr / 1000.0)
    right_delay = int(float(bus.params.get("right_delay_ms", 17.0)) * sr / 1000.0)
    depth = float(np.clip(bus.params.get("depth", 0.16), 0.0, 0.5))
    mono = mono_sum(bus_input)
    wet = np.zeros_like(bus_input, dtype=np.float32)
    if left_delay < len(wet):
        wet[left_delay:, 0] += mono[:len(wet) - left_delay]
    if right_delay < len(wet):
        wet[right_delay:, 1] += mono[:len(wet) - right_delay]
    # A second quiet tap mimics chorus spread without a phase-heavy modulated insert.
    second_l = left_delay + int(7.0 * sr / 1000.0)
    second_r = right_delay + int(9.0 * sr / 1000.0)
    if second_l < len(wet):
        wet[second_l:, 0] -= mono[:len(wet) - second_l] * 0.35
    if second_r < len(wet):
        wet[second_r:, 1] -= mono[:len(wet) - second_r] * 0.35
    wet *= depth * 2.4
    return _filter_stereo_return(wet, sr, bus.hpf_hz, bus.lpf_hz)


def _duck_fx_return(
    fx_return: np.ndarray,
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    source_instrument: str,
    depth_db: float,
) -> np.ndarray:
    if depth_db <= 0.0:
        return fx_return
    source_channels = [
        channel for channel, plan in plans.items()
        if plan.instrument == source_instrument and channel in rendered_channels and not plan.muted
    ]
    if not source_channels:
        return fx_return
    source = sum((rendered_channels[channel] for channel in source_channels), np.zeros_like(fx_return))
    frame = int(0.22 * sr)
    hop = int(0.05 * sr)
    starts, source_db = _frame_rms_db(source, frame, hop)
    active = np.clip((source_db + 36.0) / 12.0, 0.0, 1.0)
    duck_db = _smooth_gain_db((-depth_db * active).astype(np.float32), sr, hop, attack_ms=90.0, release_ms=650.0)
    sample_points = np.clip(starts + frame // 2, 0, len(fx_return) - 1)
    full_points = np.concatenate(([0.0], sample_points, [float(len(fx_return) - 1)]))
    full_duck = np.concatenate(([duck_db[0]], duck_db, [duck_db[-1]]))
    envelope_db = np.interp(np.arange(len(fx_return)), full_points, full_duck).astype(np.float32)
    return (fx_return * (10.0 ** (envelope_db / 20.0))[:, None]).astype(np.float32)


def apply_offline_fx_plan(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    tempo_bpm: float = 120.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render shared stereo FX returns from the rule-based FX plan."""
    planner = AutoFXPlanner(tempo_bpm=tempo_bpm, vocal_priority=True)
    fx_plan: FXPlan = planner.create_plan({
        channel: plan.instrument
        for channel, plan in plans.items()
        if channel in rendered_channels and not plan.muted
    })

    target_len = len(next(iter(rendered_channels.values()))) if rendered_channels else 0
    returns: dict[str, np.ndarray] = {}
    return_reports: list[dict[str, Any]] = []
    sends_by_bus: dict[int, list[dict[str, Any]]] = {}

    for bus in fx_plan.buses:
        bus_input = np.zeros((target_len, 2), dtype=np.float32)
        active_sends = []
        for send in fx_plan.sends:
            if send.bus_id != bus.bus_id or send.channel_id not in rendered_channels:
                continue
            bus_input += rendered_channels[send.channel_id] * db_to_amp(send.send_db)
            item = send.__dict__.copy()
            item["file"] = plans[send.channel_id].path.name
            active_sends.append(item)
        sends_by_bus[bus.bus_id] = active_sends

        if not active_sends or target_len == 0:
            continue
        if bus.fx_type == "reverb":
            fx_return = _simple_reverb(bus_input, sr, bus)
        elif bus.fx_type == "delay":
            fx_return = _tempo_delay(bus_input, sr, bus)
        elif bus.fx_type == "chorus":
            fx_return = _chorus_doubler(bus_input, sr, bus)
        else:
            continue

        if bus.duck_source:
            fx_return = _duck_fx_return(fx_return, rendered_channels, plans, sr, bus.duck_source, bus.duck_depth_db)
        fx_return = (fx_return * db_to_amp(bus.return_level_db)).astype(np.float32)
        key = f"{bus.bus_id}_{bus.name.lower().replace(' ', '_')}"
        returns[key] = fx_return
        return_reports.append({
            **bus.__dict__,
            "active_send_count": len(active_sends),
            "return_peak_dbfs": round(amp_to_db(float(np.max(np.abs(fx_return))) if len(fx_return) else 0.0), 2),
        })

    return returns, {
        "enabled": True,
        "tempo_bpm": tempo_bpm,
        "plan": fx_plan.to_dict(),
        "returns": return_reports,
        "sends_by_bus": sends_by_bus,
    }


def _soft_limiter(mix: np.ndarray, drive_db: float, ceiling_db: float = -1.0) -> np.ndarray:
    drive = db_to_amp(drive_db)
    ceiling = db_to_amp(ceiling_db)
    shaped = np.tanh(mix * drive) / np.tanh(drive)
    peak = np.max(np.abs(shaped))
    if peak > 0:
        shaped = shaped / peak * min(ceiling, peak)
    return shaped.astype(np.float32)


def master_process(
    mix: np.ndarray,
    sr: int,
    target_lufs: float = -16.0,
    final_limiter: bool = True,
    live_peak_ceiling_db: float = -3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Console-like 2-bus cleanup and glue.
    for ch in range(2):
        mix[:, ch] = highpass(mix[:, ch], sr, 28.0)
        mix[:, ch] = compressor(mix[:, ch], sr, threshold_db=-11.0, ratio=1.6, attack_ms=25, release_ms=250)

    meter = pyln.Meter(sr)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    pre_lufs = None
    try:
        pre_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass

    if not final_limiter:
        peak_db = amp_to_db(peak)
        target_gain_db = 0.0
        if pre_lufs is not None and np.isfinite(pre_lufs):
            target_gain_db = float(np.clip(target_lufs - pre_lufs, -6.0, 6.0))
        peak_safe_gain_db = live_peak_ceiling_db - peak_db
        static_master_gain_db = min(target_gain_db, peak_safe_gain_db)
        mix = (mix * db_to_amp(static_master_gain_db)).astype(np.float32)
        post_lufs = None
        try:
            post_lufs = float(meter.integrated_loudness(mix))
        except Exception:
            pass
        return np.asarray(mix, dtype=np.float32), {
            "final_limiter": False,
            "soft_limiter": False,
            "static_master_gain_db": round(static_master_gain_db, 2),
            "live_peak_ceiling_dbfs": round(live_peak_ceiling_db, 2),
            "pre_master_peak_dbfs": round(peak_db, 2),
            "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
            "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
            "note": "No final limiting or clipping stage; only static master trim is used for live-style headroom.",
        }

    ceiling = db_to_amp(-1.0)
    if peak > 0.95:
        mix = mix / peak * 0.95

    # Iterative loudness lift with soft limiting, similar to pushing a console
    # into a gentle final limiter. This keeps the MP3 listenable without letting
    # short drum transients dictate the whole mix level.
    for _ in range(4):
        try:
            loudness = meter.integrated_loudness(mix)
        except Exception:
            break
        if not np.isfinite(loudness):
            break
        needed = target_lufs - loudness
        if abs(needed) < 0.4:
            break
        mix = mix * db_to_amp(min(max(needed, -6.0), 6.0))
        peak = np.max(np.abs(mix))
        if peak > ceiling:
            over_db = amp_to_db(peak / ceiling)
            mix = _soft_limiter(mix / max(peak, 1e-9), drive_db=min(12.0, max(3.0, over_db + 3.0)), ceiling_db=-1.0)

    peak = np.max(np.abs(mix))
    if peak > ceiling:
        mix = mix / peak * ceiling
    post_lufs = None
    try:
        post_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass
    return np.asarray(mix, dtype=np.float32), {
        "final_limiter": True,
        "soft_limiter": True,
        "ceiling_dbfs": -1.0,
        "pre_master_peak_dbfs": round(amp_to_db(peak), 2),
        "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
        "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(Path.home() / "Desktop" / "MIX"))
    parser.add_argument("--output", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT.mp3"))
    parser.add_argument("--report", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT_report.json"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-final-limiter", action="store_true")
    parser.add_argument("--no-drum-vocal-duck", action="store_true")
    parser.add_argument("--live-peak-ceiling-db", type=float, default=-3.0)
    parser.add_argument("--live-channel-peak-ceiling-db", type=float, default=-3.0)
    parser.add_argument("--live-input-fade-ms", type=float, default=25.0)
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--disable-fx", action="store_true")
    parser.add_argument("--bass-drum-boost-db", type=float, default=0.0)
    parser.add_argument("--kick-presence-boost-db", type=float, default=0.0)
    parser.add_argument("--kick-focus-cymbal-cut-db", type=float, default=0.0)
    parser.add_argument("--codex-correction-pass", action="store_true")
    parser.add_argument("--codex-orchestrator", action="store_true")
    parser.add_argument("--codex-orchestrator-dry-run", action="store_true")
    parser.add_argument("--codex-orchestrator-allow-llm", action="store_true")
    parser.add_argument("--codex-orchestrator-max-actions", type=int, default=5)
    parser.add_argument("--soft-master", action="store_true")
    parser.add_argument("--master-target-lufs", type=float, default=-16.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    wavs = sorted(p for p in input_dir.glob("*.wav") if p.is_file())
    if not wavs:
        raise SystemExit(f"No WAV files found in {input_dir}")

    info = sf.info(str(wavs[0]))
    sr = int(info.samplerate)
    target_len = int(info.frames)
    plans: dict[int, ChannelPlan] = {}

    for idx, path in enumerate(wavs, start=1):
        instrument, pan, hpf, target_rms, eq, comp, phase = classify_track(path)
        mono, file_sr = read_mono(path)
        if file_sr != sr:
            raise ValueError(f"{path.name}: sample rate mismatch {file_sr} != {sr}")
        m = metrics_for(mono, sr, instrument=instrument)
        trim = float(np.clip(target_rms - m["rms_db"], -18.0, 12.0))
        threshold, ratio, attack, release = comp
        plan = ChannelPlan(
            path=path,
            name=path.stem,
            instrument=instrument,
            pan=pan,
            hpf=hpf,
            target_rms_db=target_rms,
            trim_db=trim,
            input_fade_ms=args.live_input_fade_ms if args.no_final_limiter else 0.0,
            eq_bands=list(eq),
            comp_threshold_db=threshold,
            comp_ratio=ratio,
            comp_attack_ms=attack,
            comp_release_ms=release,
            phase_invert=phase,
            event_activity=_event_activity_ranges(mono, sr, instrument) or {},
            metrics=m,
        )
        plans[idx] = plan
        del mono

    phase_report = apply_drum_phase_alignment(plans, sr)
    drum_pan_rule = apply_overhead_anchored_drum_panning(plans, sr)

    config = yaml.safe_load((REPO_ROOT / "config" / "automixer.yaml").read_text(encoding="utf-8"))
    ai_config = config.get("ai", {})
    use_llm_in_orchestrator = bool(args.codex_orchestrator_allow_llm and args.codex_orchestrator)
    llm = None
    if not args.no_llm and (not args.codex_orchestrator or use_llm_in_orchestrator):
        llm = LLMClient(
            backend=ai_config.get("llm_backend", "auto"),
            model=ai_config.get("llm_model", "gpt-5.4"),
            ollama_url=ai_config.get("ollama_url", "http://localhost:11434"),
            model_fallbacks=ai_config.get("model_fallbacks") or None,
            kimi_timeout_sec=float(ai_config.get("kimi_timeout_sec", 120)),
            kimi_cli_path=ai_config.get("kimi_cli_path") or None,
            kimi_work_dir=ai_config.get("kimi_work_dir") or None,
        )

    console = VirtualConsole(plans)
    agent_mode = AgentMode.SUGGEST if args.codex_orchestrator else AgentMode.AUTO
    agent = MixingAgent(
        knowledge_base=KnowledgeBase(
            knowledge_dir=ai_config.get("knowledge_dir") or None,
            use_vector_db=False,
            allowed_categories=KnowledgeBase.AGENT_RUNTIME_CATEGORIES,
        ),
        rule_engine=RuleEngine(),
        llm_client=llm,
        mixer_client=console,
        mode=agent_mode,
        cycle_interval=0.5,
    )
    agent.configure_safety_limits(max_fader_step_db=3.0, max_fader_db=6.0, max_eq_step_db=2.0, max_comp_threshold_step_db=3.0)
    agent._max_actions_per_cycle = max(1, int(args.codex_orchestrator_max_actions))
    music_bed_lufs = _music_bed_lufs(plans)

    states = {}
    for ch, plan in plans.items():
        state = dict(plan.metrics)
        level_offset = plan.trim_db + plan.fader_db
        for key in ("peak_db", "true_peak_db", "rms_db", "lufs_momentary"):
            if key in state:
                state[key] = float(state[key]) + level_offset
        state.update({
            "channel_id": ch,
            "name": plan.name,
            "instrument": plan.instrument,
            "is_muted": False,
            "mix_lufs": music_bed_lufs,
            "vocal_target_delta_db": 6.0,
        })
        states[ch] = state
    agent.update_channel_states_batch(states)
    codex_dry_run = bool(args.codex_orchestrator_dry_run or args.codex_orchestrator)

    orchestration_report = {
        "enabled": bool(args.codex_orchestrator),
        "dry_run": codex_dry_run,
        "llm_enabled": llm is not None,
        "mode": agent.state.mode.value,
        "max_actions_per_cycle": agent._max_actions_per_cycle,
    }
    if args.codex_orchestrator:
        try:
            proposed_actions = agent._prepare_actions(agent._decide({"channels": states}))
            orchestration_report.update({
                "proposed_actions": _agent_actions_to_dict(proposed_actions),
                "proposed_count": len(proposed_actions),
                "applied_count": 0,
            })
            if not codex_dry_run and proposed_actions:
                agent._queue_pending_actions(proposed_actions)
                orchestration_report["pending_before_approve"] = len(agent.state.pending_actions)
                approved = agent.approve_all_pending()
                orchestration_report["applied_count"] = approved
            else:
                orchestration_report["pending_before_approve"] = 0
            orchestration_report["mode"] = "orchestrated"
        except Exception as e:
            orchestration_report["error"] = str(e)
    else:
        actions = agent._prepare_actions(agent._decide({"channels": states}))
        asyncio.run(agent._act(actions))
        orchestration_report.update({
            "proposed_actions": _agent_actions_to_dict(actions),
            "proposed_count": len(actions),
            "applied_count": len(actions),
            "pending_before_approve": 0,
            "mode": "direct",
        })

    codex_corrections: dict[str, Any] = {"enabled": False, "actions": []}
    if args.codex_correction_pass:
        raw_codex_actions = codex_correction_actions(plans)
        prepared_codex_actions = agent._prepare_actions(raw_codex_actions)
        asyncio.run(agent._act(prepared_codex_actions))
        codex_corrections = {
            "enabled": True,
            "actions": [action.__dict__ for action in prepared_codex_actions],
            "notes": [
                "External LLM recommendations were not used for this correction pass.",
                "Codex applied deterministic balance/clarity decisions after the rule engine.",
            ],
        }

    codex_bleed_control = apply_codex_bleed_control(plans) if args.codex_correction_pass else {"enabled": False}
    event_based_dynamics = apply_event_based_dynamics(plans)
    bass_drum_push = apply_bass_drum_push(plans, args.bass_drum_boost_db)
    kick_presence_boost = apply_kick_presence_boost(plans, args.kick_presence_boost_db)
    cymbal_focus_cleanup = apply_cymbal_cleanup_for_kick_focus(plans, args.kick_focus_cymbal_cut_db)
    vocal_bed_balance = apply_vocal_bed_balance(plans)
    cross_adaptive_eq = apply_cross_adaptive_eq(plans, target_len, sr)
    live_channel_headroom = (
        apply_live_channel_peak_headroom(plans, target_len, sr, args.live_channel_peak_ceiling_db)
        if args.no_final_limiter
        else {"enabled": False}
    )

    rendered_channels: dict[int, np.ndarray] = {}
    channel_reports = []
    for ch, plan in plans.items():
        if plan.muted:
            continue
        rendered = render_channel(plan.path, plan, target_len, sr)
        rendered_channels[ch] = rendered
        channel_reports.append({
            "channel": ch,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "trim_db": round(plan.trim_db, 2),
            "agent_fader_db": round(plan.fader_db, 2),
            "pan": plan.pan,
            "hpf": plan.hpf,
            "lpf": plan.lpf,
            "phase_invert": plan.phase_invert,
            "delay_ms": round(plan.delay_ms, 3),
            "input_fade_ms": round(plan.input_fade_ms, 2),
            "compressor": {
                "threshold_db": round(plan.comp_threshold_db, 2),
                "ratio": round(plan.comp_ratio, 2),
                "attack_ms": round(plan.comp_attack_ms, 2),
                "release_ms": round(plan.comp_release_ms, 2),
            },
            "event_expander": plan.expander_report,
            "phase_notes": plan.phase_notes,
            "pan_notes": plan.pan_notes,
            "cross_adaptive_eq": plan.cross_adaptive_eq,
            "peak_db": round(plan.metrics["peak_db"], 2),
            "rms_db": round(plan.metrics["rms_db"], 2),
            "analysis_mode": plan.metrics.get("analysis_mode"),
            "analysis_active_sec": plan.metrics.get("analysis_active_sec"),
            "analysis_active_ratio": plan.metrics.get("analysis_active_ratio"),
            "analysis_threshold_db": plan.metrics.get("analysis_threshold_db"),
        })

    dynamic_vocal_priority = apply_dynamic_vocal_priority(
        rendered_channels,
        plans,
        sr,
        duck_drum_instruments=not args.no_drum_vocal_duck,
    )
    fx_returns, fx_report = (
        ({}, {"enabled": False})
        if args.disable_fx
        else apply_offline_fx_plan(rendered_channels, plans, sr, tempo_bpm=args.tempo_bpm)
    )
    mix = np.zeros((target_len, 2), dtype=np.float32)
    for rendered in rendered_channels.values():
        mix += rendered
    for rendered in fx_returns.values():
        mix += rendered

    final_limiter = (not args.no_final_limiter) or args.soft_master
    mix, master_report = master_process(
        mix,
        sr,
        target_lufs=args.master_target_lufs,
        final_limiter=final_limiter,
        live_peak_ceiling_db=args.live_peak_ceiling_db,
    )

    tmp_wav = Path(args.output).with_suffix(".wav")
    sf.write(tmp_wav, mix, sr, subtype="PCM_24")

    output = Path(args.output)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(tmp_wav),
        "-codec:a", "libmp3lame",
        "-b:a", "320k",
        str(output),
    ]
    subprocess.run(cmd, check=True)

    meter = pyln.Meter(sr)
    final_lufs = None
    try:
        final_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass
    report = {
        "input_dir": str(input_dir),
        "output": str(output),
        "sample_rate": sr,
        "codex_orchestrator": orchestration_report,
        "duration_sec": round(target_len / sr, 3),
        "final_peak_dbfs": round(amp_to_db(float(np.max(np.abs(mix)))), 2),
        "final_lufs": round(final_lufs, 2) if final_lufs is not None and np.isfinite(final_lufs) else None,
        "music_bed_lufs": round(_music_bed_lufs(plans), 2),
        "vocal_bed_balance": vocal_bed_balance,
        "bass_drum_boost": bass_drum_push,
        "cross_adaptive_eq": cross_adaptive_eq,
        "dynamic_vocal_priority": dynamic_vocal_priority,
        "kick_presence_boost": kick_presence_boost,
        "kick_focus_cymbal_cut": cymbal_focus_cleanup,
        "soft_master": args.soft_master,
        "master_target_lufs": round(args.master_target_lufs, 2),
        "codex_corrections": codex_corrections,
        "codex_bleed_control": codex_bleed_control,
        "event_based_dynamics": event_based_dynamics,
        "fx": fx_report,
        "master_processing": master_report,
        "live_channel_headroom": live_channel_headroom,
        "phase_alignment": phase_report,
        "drum_pan_rule": drum_pan_rule,
        "agent_actions": agent.get_action_history(100),
        "agent_audit": agent.get_action_audit_log(100),
        "virtual_console_calls": console.calls,
        "channels": channel_reports,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
