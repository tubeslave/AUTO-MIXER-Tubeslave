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
import tempfile
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
from auto_mastering import AutoMaster, MasteringResult  # noqa: E402
from auto_phase_gcc_phat import GCCPHATAnalyzer  # noqa: E402
from autofoh_analysis import build_fractional_octave_bands, build_stem_contribution_matrix, extract_analysis_features  # noqa: E402
from autofoh_detectors import (  # noqa: E402
    HarshnessExcessDetector,
    LeadMaskingAnalyzer,
    LowEndAnalyzer,
    MudExcessDetector,
    SibilanceExcessDetector,
    aggregate_stem_features,
)
from autofoh_models import ConfidenceRisk, DetectedProblem, RuntimeState  # noqa: E402
from autofoh_safety import (  # noqa: E402
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    ChannelGainMove,
    ChannelEQMove,
    ChannelFaderMove,
    TypedCorrectionAction,
)
from auto_fx import AutoFXPlanner, FXBusDecision, FXPlan  # noqa: E402
from channel_recognizer import classification_from_legacy_preset  # noqa: E402
from cross_adaptive_eq import CrossAdaptiveEQ  # noqa: E402
from ml.style_transfer import InstrumentStyle, StyleProfile, StyleTransfer  # noqa: E402


DRUM_INSTRUMENTS = {
    "kick",
    "snare",
    "rack_tom",
    "floor_tom",
    "hi_hat",
    "ride",
    "percussion",
}

REFERENCE_STYLE_INSTRUMENTS = {
    "lead_vocal": "vocals",
    "backing_vocal": "vocals",
    "kick": "kick",
    "snare": "snare",
    "rack_tom": "toms",
    "floor_tom": "toms",
    "hi_hat": "hihat",
    "ride": "hihat",
    "overhead": "overheads",
    "oh_l": "overheads",
    "oh_r": "overheads",
    "cymbals": "overheads",
    "bass": "bass",
    "bass_di": "bass",
    "bass_mic": "bass",
    "synth_bass": "bass",
    "guitar": "electric_guitar",
    "electric_guitar": "electric_guitar",
    "lead_guitar": "electric_guitar",
    "rhythm_guitar": "electric_guitar",
    "acoustic_guitar": "acoustic_guitar",
    "keys": "keys",
    "piano": "keys",
    "organ": "keys",
    "synth": "keys",
    "pad": "keys",
    "lead_synth": "keys",
    "percussion": "percussion",
}

REFERENCE_COMPRESSOR_SKIP = {
    "hi_hat",
    "ride",
    "overhead",
    "oh_l",
    "oh_r",
    "cymbals",
    "playback",
    "tracks",
    "music",
    "fx_return",
    "reverb_return",
    "delay_return",
    "room",
}

REFERENCE_AUDIO_SUFFIXES = {
    ".wav",
    ".wave",
    ".aif",
    ".aiff",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
}

REFERENCE_PRESET_SUFFIXES = {
    ".json",
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
    autofoh_actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReferenceMixContext:
    path: Path
    source_type: str
    style_profile: StyleProfile
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    source_paths: list[Path] = field(default_factory=list)


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
        self.plans[channel].fader_db = float(np.clip(value_db, -30.0, 0.0))
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


class OfflineAutoFOHConsole:
    """Minimal mixer adapter so offline analyzer actions use the same safety layer."""

    def __init__(self, plans: dict[int, ChannelPlan]):
        self.plans = plans
        self.calls: list[dict[str, Any]] = []

    def get_fader(self, channel: int) -> float:
        return float(self.plans[channel].fader_db)

    def set_fader(self, channel: int, value_db: float):
        plan = self.plans[channel]
        before = float(plan.fader_db)
        plan.fader_db = float(np.clip(value_db, -30.0, 0.0))
        self.calls.append({
            "cmd": "set_fader",
            "channel": channel,
            "before_db": round(before, 2),
            "after_db": round(plan.fader_db, 2),
        })

    def get_gain(self, channel: int) -> float:
        return float(self.plans[channel].trim_db)

    def set_gain(self, channel: int, value_db: float):
        plan = self.plans[channel]
        before = float(plan.trim_db)
        plan.trim_db = float(np.clip(value_db, -30.0, 12.0))
        self.calls.append({
            "cmd": "set_gain",
            "channel": channel,
            "before_db": round(before, 2),
            "after_db": round(plan.trim_db, 2),
        })

    @staticmethod
    def _band_index(band: int | str) -> int:
        token = str(band).strip().lower().rstrip("g")
        return max(1, int(token)) - 1

    def get_eq_band_gain(self, channel: int, band: int | str) -> float:
        plan = self.plans[channel]
        band_idx = self._band_index(band)
        if 0 <= band_idx < len(plan.eq_bands):
            return float(plan.eq_bands[band_idx][1])
        return 0.0

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        plan = self.plans[channel]
        band_idx = self._band_index(band)
        while len(plan.eq_bands) <= band_idx:
            plan.eq_bands.append((1000.0, 0.0, 1.0))
        before = plan.eq_bands[band_idx]
        plan.eq_bands[band_idx] = (float(freq), float(gain), float(q))
        self.calls.append({
            "cmd": "set_eq_band",
            "channel": channel,
            "band": band_idx + 1,
            "before": tuple(round(float(v), 2) for v in before),
            "after": (
                round(float(freq), 2),
                round(float(gain), 2),
                round(float(q), 2),
            ),
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


def normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim <= 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
        return arr.T.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Read audio with a small ffmpeg fallback so MP3 references work too."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return normalize_audio_shape(audio), int(sr)
    except Exception:
        with tempfile.TemporaryDirectory() as tmpdir:
            decoded = Path(tmpdir) / "decoded_reference.wav"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                str(decoded),
            ]
            subprocess.run(cmd, check=True)
            audio, sr = sf.read(str(decoded), dtype="float32", always_2d=False)
            return normalize_audio_shape(audio), int(sr)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = normalize_audio_shape(audio)
    if src_sr == dst_sr or len(arr) == 0:
        return arr.astype(np.float32, copy=True)

    dst_len = max(1, int(round(len(arr) * float(dst_sr) / float(src_sr))))
    src_pos = np.linspace(0.0, 1.0, num=len(arr), endpoint=False, dtype=np.float64)
    dst_pos = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float64)

    if arr.ndim == 1:
        return np.interp(dst_pos, src_pos, arr.astype(np.float64)).astype(np.float32)

    channels = [
        np.interp(dst_pos, src_pos, arr[:, idx].astype(np.float64)).astype(np.float32)
        for idx in range(arr.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def _reference_channel_count(audio: np.ndarray) -> int:
    arr = normalize_audio_shape(audio)
    return 1 if arr.ndim == 1 else int(arr.shape[1])


def _match_reference_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    arr = normalize_audio_shape(audio)
    if channels <= 1:
        return mono_sum(arr).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return np.column_stack([arr] * channels).astype(np.float32)
    if arr.shape[1] == channels:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] > channels:
        return arr[:, :channels].astype(np.float32, copy=False)
    last = arr[:, -1:]
    pads = [last] * (channels - arr.shape[1])
    return np.concatenate([arr, *pads], axis=1).astype(np.float32)


def _reference_excerpt(audio: np.ndarray, sr: int, max_duration_sec: float = 45.0) -> np.ndarray:
    arr = normalize_audio_shape(audio)
    max_samples = max(1, int(round(float(sr) * max_duration_sec)))
    if len(arr) <= max_samples:
        return arr.astype(np.float32, copy=True)
    start = max(0, (len(arr) - max_samples) // 2)
    return arr[start:start + max_samples].astype(np.float32, copy=True)


def _median_value(values: list[float], default: float = 0.0) -> float:
    usable: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            usable.append(numeric)
    if not usable:
        return float(default)
    return float(np.median(np.asarray(usable, dtype=np.float64)))


def _merge_style_profiles(style_profiles: list[StyleProfile], name: str) -> StyleProfile:
    if not style_profiles:
        raise ValueError("Cannot merge an empty style profile collection")
    if len(style_profiles) == 1:
        profile = style_profiles[0]
        return StyleProfile(
            name=name,
            spectral_balance=dict(profile.spectral_balance),
            dynamic_range=float(profile.dynamic_range),
            stereo_width=float(profile.stereo_width),
            loudness_lufs=float(profile.loudness_lufs),
            crest_factor=float(profile.crest_factor),
            per_instrument_settings=dict(profile.per_instrument_settings),
        )

    merged = StyleProfile(name=name)
    spectral_keys = sorted({
        band
        for profile in style_profiles
        for band in profile.spectral_balance.keys()
    })
    merged.spectral_balance = {
        band: round(
            _median_value(
                [profile.spectral_balance.get(band, -80.0) for profile in style_profiles],
                default=-80.0,
            ),
            4,
        )
        for band in spectral_keys
    }
    merged.dynamic_range = round(_median_value([p.dynamic_range for p in style_profiles], default=0.0), 4)
    merged.stereo_width = round(_median_value([p.stereo_width for p in style_profiles], default=0.0), 4)
    merged.loudness_lufs = round(_median_value([p.loudness_lufs for p in style_profiles], default=-14.0), 4)
    merged.crest_factor = round(_median_value([p.crest_factor for p in style_profiles], default=10.0), 4)

    instrument_names = sorted({
        instrument
        for profile in style_profiles
        for instrument in profile.per_instrument_settings.keys()
    })
    for instrument in instrument_names:
        variants = [
            profile.per_instrument_settings[instrument]
            for profile in style_profiles
            if instrument in profile.per_instrument_settings
        ]
        if not variants:
            continue
        merged.per_instrument_settings[instrument] = InstrumentStyle(
            instrument_type=variants[0].instrument_type,
            gain_db=round(_median_value([item.gain_db for item in variants], default=0.0), 4),
            eq_low_shelf_db=round(_median_value([item.eq_low_shelf_db for item in variants], default=0.0), 4),
            eq_low_mid_db=round(_median_value([item.eq_low_mid_db for item in variants], default=0.0), 4),
            eq_mid_db=round(_median_value([item.eq_mid_db for item in variants], default=0.0), 4),
            eq_high_mid_db=round(_median_value([item.eq_high_mid_db for item in variants], default=0.0), 4),
            eq_high_shelf_db=round(_median_value([item.eq_high_shelf_db for item in variants], default=0.0), 4),
            compression_ratio=round(_median_value([item.compression_ratio for item in variants], default=1.0), 4),
            compression_threshold_db=round(_median_value([item.compression_threshold_db for item in variants], default=-10.0), 4),
            gate_threshold_db=round(_median_value([item.gate_threshold_db for item in variants], default=-60.0), 4),
            pan=round(_median_value([item.pan for item in variants], default=0.0), 4),
            bus_send_level=round(_median_value([item.bus_send_level for item in variants], default=-96.0), 4),
        )
    return merged


def _iter_reference_sources(path: Path) -> list[Path]:
    candidates = [
        candidate.resolve()
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in (REFERENCE_AUDIO_SUFFIXES | REFERENCE_PRESET_SUFFIXES)
    ]
    if not candidates:
        raise FileNotFoundError(f"No supported reference files found in: {path}")
    return candidates


def prepare_reference_mix_context(reference_path: str | Path | None) -> ReferenceMixContext | None:
    if not reference_path:
        return None

    path = Path(reference_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference track not found: {path}")

    transfer = StyleTransfer(fft_size=4096, hop_size=1024)
    if path.is_file() and path.suffix.lower() == ".json":
        style_profile = transfer.load_preset(str(path))
        return ReferenceMixContext(
            path=path,
            source_type="style_preset",
            style_profile=style_profile,
            source_paths=[path],
        )
    if path.is_file():
        audio, ref_sr = read_audio_file(path)
        style_profile = transfer.extract_style(audio, sr=ref_sr, name=path.stem)
        return ReferenceMixContext(
            path=path,
            source_type="audio",
            style_profile=style_profile,
            audio=audio,
            sample_rate=ref_sr,
            source_paths=[path],
        )

    source_paths = _iter_reference_sources(path)
    style_profiles: list[StyleProfile] = []
    audio_sources: list[tuple[np.ndarray, int]] = []
    has_audio = False
    has_preset = False

    for source in source_paths:
        if source.suffix.lower() in REFERENCE_PRESET_SUFFIXES:
            style_profiles.append(transfer.load_preset(str(source)))
            has_preset = True
            continue
        audio, ref_sr = read_audio_file(source)
        style_profiles.append(transfer.extract_style(audio, sr=ref_sr, name=source.stem))
        audio_sources.append((audio, ref_sr))
        has_audio = True

    if not style_profiles:
        raise FileNotFoundError(f"No usable reference profiles could be loaded from: {path}")

    merged_profile = _merge_style_profiles(style_profiles, name=path.name)
    combined_audio = None
    target_sr = None
    if audio_sources:
        target_sr = int(audio_sources[0][1])
        target_channels = max(_reference_channel_count(audio) for audio, _ in audio_sources)
        excerpts: list[np.ndarray] = []
        for audio, ref_sr in audio_sources:
            prepared = audio
            if ref_sr != target_sr:
                prepared = resample_audio(prepared, ref_sr, target_sr)
            prepared = _reference_excerpt(prepared, target_sr)
            prepared = _match_reference_channels(prepared, target_channels)
            peak = float(np.max(np.abs(prepared))) if len(prepared) else 0.0
            if peak > 1e-6:
                prepared = (prepared / peak * 0.92).astype(np.float32)
            excerpts.append(prepared.astype(np.float32, copy=False))
        if excerpts:
            combined_audio = np.concatenate(excerpts, axis=0).astype(np.float32, copy=False)

    if has_audio and has_preset:
        source_type = "mixed_reference_directory"
    elif has_audio:
        source_type = "audio_directory"
    else:
        source_type = "style_preset_directory"

    return ReferenceMixContext(
        path=path,
        source_type=source_type,
        style_profile=merged_profile,
        audio=combined_audio,
        sample_rate=target_sr,
        source_paths=source_paths,
    )


def _reference_style_instrument(instrument: str) -> str:
    return REFERENCE_STYLE_INSTRUMENTS.get(instrument, "other")


def _reference_fader_scale(instrument: str) -> float:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.22
    if instrument in {"kick", "snare", "bass", "bass_guitar", "bass_di", "bass_mic", "synth_bass"}:
        return 0.18
    if instrument in {"hi_hat", "ride", "overhead", "oh_l", "oh_r", "cymbals"}:
        return 0.12
    return 0.15


def _reference_eq_scale(instrument: str) -> float:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.25
    if instrument in {"hi_hat", "ride", "overhead", "oh_l", "oh_r", "cymbals"}:
        return 0.18
    return 0.22


def _reference_comp_scale(instrument: str) -> float:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.3
    if instrument in {"kick", "snare", "bass", "bass_guitar", "bass_di", "bass_mic", "synth_bass"}:
        return 0.25
    return 0.18


def _merge_reference_eq_adjustment(
    plan: ChannelPlan,
    *,
    freq: float,
    gain_db: float,
    q: float,
) -> dict[str, Any] | None:
    if abs(gain_db) < 0.15:
        return None

    target_idx = None
    best_distance = float("inf")
    for idx, (band_freq, _, _) in enumerate(plan.eq_bands):
        distance = abs(math.log2(max(freq, 20.0) / max(float(band_freq), 20.0)))
        if distance < 0.45 and distance < best_distance:
            best_distance = distance
            target_idx = idx

    if target_idx is None:
        band = (
            float(freq),
            float(np.clip(gain_db, -6.0, 6.0)),
            float(np.clip(q, 0.5, 2.5)),
        )
        plan.eq_bands.append(band)
        return {
            "mode": "append",
            "after": {
                "freq_hz": round(band[0], 2),
                "gain_db": round(band[1], 2),
                "q": round(band[2], 2),
            },
        }

    before = plan.eq_bands[target_idx]
    after = (
        float(before[0]),
        float(np.clip(before[1] + gain_db, -6.0, 6.0)),
        float(np.clip((before[2] + q) * 0.5, 0.5, 2.5)),
    )
    plan.eq_bands[target_idx] = after
    return {
        "mode": "merge",
        "before": {
            "freq_hz": round(float(before[0]), 2),
            "gain_db": round(float(before[1]), 2),
            "q": round(float(before[2]), 2),
        },
        "after": {
            "freq_hz": round(float(after[0]), 2),
            "gain_db": round(float(after[1]), 2),
            "q": round(float(after[2]), 2),
        },
    }


def apply_reference_mix_guidance(
    plans: dict[int, ChannelPlan],
    sr: int,
    reference_context: ReferenceMixContext | None,
) -> dict[str, Any]:
    if reference_context is None:
        return {"enabled": False, "reason": "no_reference_supplied"}

    transfer = StyleTransfer(fft_size=4096, hop_size=1024)
    channel_audios: dict[str, np.ndarray] = {}
    channel_types: dict[str, str] = {}
    skipped_channels: list[dict[str, Any]] = []

    for channel, plan in plans.items():
        try:
            mono, file_sr = read_mono(plan.path)
            if file_sr != sr:
                raise ValueError(f"sample rate mismatch {file_sr} != {sr}")
            analysis_audio, _ = _analysis_signal_for_metrics(mono, sr, plan.instrument)
            channel_audios[f"ch{channel}"] = analysis_audio.astype(np.float32, copy=False)
            channel_types[f"ch{channel}"] = _reference_style_instrument(plan.instrument)
        except Exception as exc:
            skipped_channels.append({
                "channel": channel,
                "file": plan.path.name,
                "reason": str(exc),
            })

    if not channel_audios:
        return {
            "enabled": False,
            "reason": "no_channels_available",
            "reference_path": str(reference_context.path),
            "source_type": reference_context.source_type,
            "reference_sources": [str(path) for path in reference_context.source_paths],
            "skipped_channels": skipped_channels,
        }

    mixing_params = transfer.apply_style(
        reference_context.style_profile,
        channel_audios,
        channel_types,
        sr=sr,
    )

    actions: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        params = mixing_params.get(f"ch{channel}")
        if not params:
            continue

        action = {
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "reference_type": channel_types.get(f"ch{channel}", "other"),
            "fader": None,
            "eq": [],
            "compressor": None,
        }

        fader_step = float(
            np.clip(
                float(params.get("fader_db", 0.0)) * _reference_fader_scale(plan.instrument),
                -1.5,
                1.5,
            )
        )
        if abs(fader_step) >= 0.12:
            before = float(plan.fader_db)
            plan.fader_db = float(np.clip(plan.fader_db + fader_step, -30.0, 0.0))
            action["fader"] = {
                "before_db": round(before, 2),
                "delta_db": round(plan.fader_db - before, 2),
                "after_db": round(plan.fader_db, 2),
            }

        eq_candidates = sorted(
            params.get("eq_bands", []),
            key=lambda item: abs(float(item.get("gain_db", 0.0))),
            reverse=True,
        )[:3]
        eq_scale = _reference_eq_scale(plan.instrument)
        for band in eq_candidates:
            scaled_gain = float(np.clip(float(band.get("gain_db", 0.0)) * eq_scale, -1.25, 1.25))
            if abs(scaled_gain) < 0.2:
                continue
            q = 0.75 if band.get("type") in {"low_shelf", "high_shelf"} else float(band.get("q", 1.2))
            merged = _merge_reference_eq_adjustment(
                plan,
                freq=float(band.get("frequency", 1000.0)),
                gain_db=scaled_gain,
                q=q,
            )
            if merged is not None:
                merged["requested_gain_db"] = round(float(band.get("gain_db", 0.0)), 2)
                action["eq"].append(merged)

        if plan.instrument not in REFERENCE_COMPRESSOR_SKIP:
            comp = params.get("compression") or {}
            if comp:
                comp_scale = _reference_comp_scale(plan.instrument)
                ratio_before = float(plan.comp_ratio)
                threshold_before = float(plan.comp_threshold_db)
                ratio_step = float(np.clip((float(comp.get("ratio", ratio_before)) - ratio_before) * comp_scale, -0.6, 0.6))
                threshold_step = float(np.clip((float(comp.get("threshold_db", threshold_before)) - threshold_before) * comp_scale, -2.0, 2.0))
                if abs(ratio_step) >= 0.05 or abs(threshold_step) >= 0.1:
                    plan.comp_ratio = float(np.clip(ratio_before + ratio_step, 1.0, 8.0))
                    plan.comp_threshold_db = float(np.clip(threshold_before + threshold_step, -40.0, 0.0))
                    action["compressor"] = {
                        "ratio_before": round(ratio_before, 2),
                        "ratio_after": round(plan.comp_ratio, 2),
                        "threshold_before_db": round(threshold_before, 2),
                        "threshold_after_db": round(plan.comp_threshold_db, 2),
                    }

        if action["fader"] or action["eq"] or action["compressor"]:
            actions.append(action)

    style = reference_context.style_profile
    return {
        "enabled": True,
        "reference_path": str(reference_context.path),
        "source_type": reference_context.source_type,
        "reference_sources": [str(path) for path in reference_context.source_paths],
        "applied_channel_count": len(actions),
        "style_profile": {
            "name": style.name,
            "loudness_lufs": round(float(style.loudness_lufs), 2),
            "dynamic_range_db": round(float(style.dynamic_range), 2),
            "stereo_width": round(float(style.stereo_width), 3),
            "crest_factor_db": round(float(style.crest_factor), 2),
            "spectral_balance": {
                key: round(float(value), 2)
                for key, value in style.spectral_balance.items()
            },
        },
        "actions": actions,
        "skipped_channels": skipped_channels,
    }


def delay_signal(x: np.ndarray, sr: int, delay_ms: float) -> np.ndarray:
    delay_samples = int(round(max(0.0, delay_ms) * sr / 1000.0))
    if delay_samples <= 0:
        return x
    return np.pad(x, (delay_samples, 0))[:len(x)].astype(np.float32)


def apply_genre_mix_profile(
    plans: dict[int, ChannelPlan],
    genre: str | None,
) -> dict[str, Any]:
    genre_token = str(genre or "").strip().lower()
    if not genre_token:
        return {"enabled": False, "reason": "no_genre_supplied"}
    if genre_token != "rock":
        return {
            "enabled": False,
            "requested_genre": genre_token,
            "reason": "unsupported_genre",
        }

    actions: list[dict[str, Any]] = []
    snare_layers = [
        plan.path.name
        for plan in plans.values()
        if plan.instrument == "snare"
    ]

    for channel, plan in plans.items():
        changes: list[dict[str, Any]] = []

        if plan.instrument == "lead_vocal":
            before = {
                "trim_db": float(plan.trim_db),
                "target_rms_db": float(plan.target_rms_db),
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
                "attack_ms": float(plan.comp_attack_ms),
                "release_ms": float(plan.comp_release_ms),
            }
            plan.target_rms_db = max(plan.target_rms_db, -18.5)
            plan.trim_db = float(np.clip(plan.trim_db + 1.4, -18.0, 12.0))
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -26.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 4.8))
            plan.comp_attack_ms = float(min(plan.comp_attack_ms, 4.0))
            plan.comp_release_ms = float(np.clip(plan.comp_release_ms, 105.0, 125.0))
            changes.append({
                "type": "lead_vocal_glue",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "trim_db": round(float(plan.trim_db), 2),
                    "target_rms_db": round(float(plan.target_rms_db), 2),
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                    "attack_ms": round(float(plan.comp_attack_ms), 2),
                    "release_ms": round(float(plan.comp_release_ms), 2),
                },
            })
            if plan.expander_enabled:
                before_expander = {
                    "range_db": float(plan.expander_range_db),
                    "threshold_db": float(plan.expander_threshold_db or 0.0),
                    "open_ms": float(plan.expander_open_ms),
                    "close_ms": float(plan.expander_close_ms),
                    "hold_ms": float(plan.expander_hold_ms),
                }
                plan.expander_range_db = float(min(plan.expander_range_db, 3.2))
                if plan.expander_threshold_db is not None:
                    plan.expander_threshold_db = float(plan.expander_threshold_db - 1.0)
                plan.expander_open_ms = float(max(plan.expander_open_ms, 22.0))
                plan.expander_close_ms = float(max(plan.expander_close_ms, 240.0))
                plan.expander_hold_ms = float(max(plan.expander_hold_ms, 300.0))
                changes.append({
                    "type": "lead_vocal_phrase_stability",
                    "before": {key: round(value, 2) for key, value in before_expander.items()},
                    "after": {
                        "range_db": round(float(plan.expander_range_db), 2),
                        "threshold_db": round(float(plan.expander_threshold_db or 0.0), 2),
                        "open_ms": round(float(plan.expander_open_ms), 2),
                        "close_ms": round(float(plan.expander_close_ms), 2),
                        "hold_ms": round(float(plan.expander_hold_ms), 2),
                    },
                })

        elif plan.instrument == "backing_vocal":
            before = {
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
                "attack_ms": float(plan.comp_attack_ms),
                "release_ms": float(plan.comp_release_ms),
            }
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -24.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 3.3))
            plan.comp_attack_ms = float(min(plan.comp_attack_ms, 6.0))
            plan.comp_release_ms = float(np.clip(plan.comp_release_ms, 115.0, 145.0))
            changes.append({
                "type": "backing_vocal_tighten",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                    "attack_ms": round(float(plan.comp_attack_ms), 2),
                    "release_ms": round(float(plan.comp_release_ms), 2),
                },
            })

        elif plan.instrument == "snare":
            before = {
                "pan": float(plan.pan),
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
            }
            plan.pan = 0.0
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -23.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 4.0))
            changes.append({
                "type": "rock_snare_center",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "pan": round(float(plan.pan), 2),
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                },
            })

        elif plan.instrument == "kick":
            before = {
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
            }
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -20.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 4.2))
            changes.append({
                "type": "rock_kick_anchor",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                },
            })

        if changes:
            actions.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "changes": changes,
            })

    return {
        "enabled": bool(actions),
        "genre": genre_token,
        "notes": [
            "Rock mode tightens lead vocal glue with stronger compression and steadier between-phrase behavior.",
            "Snare layers stay treated as one center snare voice when both top and bottom mics are present.",
        ],
        "snare_layers": snare_layers,
        "actions": actions,
    }


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
            "percentile": 95.0,
            "peak_offset_db": 14.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
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


LEGACY_PRESET_BY_INSTRUMENT = {
    "kick": "kick",
    "snare": "snare",
    "rack_tom": "tom",
    "floor_tom": "tom",
    "hi_hat": "hihat",
    "ride": "ride",
    "overhead": "overheads",
    "room": "room",
    "bass_guitar": "bass",
    "electric_guitar": "electricGuitar",
    "acoustic_guitar": "acousticGuitar",
    "accordion": "accordion",
    "synth": "synth",
    "playback": "playback",
    "lead_vocal": "leadVocal",
    "backing_vocal": "backVocal",
    "custom": "custom",
}


def _legacy_preset_for_instrument(instrument: str) -> str:
    return LEGACY_PRESET_BY_INSTRUMENT.get(str(instrument or "").strip().lower(), "custom")


def _measurement_window_for_instrument(instrument: str) -> float:
    if instrument in {"kick", "snare", "rack_tom", "floor_tom"}:
        return 0.55
    if instrument in {"hi_hat", "ride", "percussion"}:
        return 0.8
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 1.2
    if instrument in {"bass_guitar", "electric_guitar", "acoustic_guitar", "accordion", "playback", "synth"}:
        return 1.5
    return 1.0


def _focused_fft_measurement_block(
    x: np.ndarray,
    sr: int,
    instrument: str,
    *,
    fft_size: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    analysis_signal, analysis_meta = _analysis_signal_for_metrics(x, sr, instrument)
    search_window = _analysis_block(
        analysis_signal,
        sr,
        window_sec=_measurement_window_for_instrument(instrument),
    )
    if len(search_window) == 0:
        return search_window.astype(np.float32), {
            **analysis_meta,
            "measurement_block_sec": 0.0,
            "measurement_block_samples": 0,
        }

    if len(search_window) <= fft_size:
        block = search_window
    else:
        hop = max(256, fft_size // 4)
        best_start = 0
        best_energy = -1.0
        for start in range(0, len(search_window) - fft_size + 1, hop):
            energy = float(np.mean(np.square(search_window[start:start + fft_size])))
            if energy > best_energy:
                best_energy = energy
                best_start = start
        block = search_window[best_start:best_start + fft_size]

    return block.astype(np.float32), {
        **analysis_meta,
        "measurement_block_sec": round(len(block) / sr, 4) if sr else 0.0,
        "measurement_block_samples": int(len(block)),
    }


def _typed_action_to_dict(action: TypedCorrectionAction) -> dict[str, Any]:
    payload = {
        "action_type": action.action_type,
        "reason": action.reason,
    }
    payload.update(action.__dict__)
    return payload


def _problem_to_dict(problem: Any) -> dict[str, Any]:
    if problem is None:
        return {}
    confidence = getattr(problem, "confidence_risk", None)
    return {
        "problem_type": getattr(problem, "problem_type", ""),
        "description": getattr(problem, "description", ""),
        "channel_id": getattr(problem, "channel_id", None),
        "stem": getattr(problem, "stem", None),
        "band_name": getattr(problem, "band_name", None),
        "persistence_sec": getattr(problem, "persistence_sec", 0.0),
        "expected_effect": getattr(problem, "expected_effect", ""),
        "confidence_risk": {
            "problem_confidence": round(float(getattr(confidence, "problem_confidence", 0.0)), 3),
            "culprit_confidence": round(float(getattr(confidence, "culprit_confidence", 0.0)), 3),
            "action_confidence": round(float(getattr(confidence, "action_confidence", 0.0)), 3),
            "risk_score": round(float(getattr(confidence, "risk_score", 0.0)), 3),
        },
    }


def _build_autofoh_measurement_snapshot(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    autofoh_config: dict[str, Any] | None = None,
):
    autofoh_config = autofoh_config or {}
    analysis_config = autofoh_config.get("analysis", {})
    fft_size = int(analysis_config.get("fft_size", 4096))
    octave_fraction = int(analysis_config.get("octave_fraction", 3))
    slope_db = float(analysis_config.get("slope_compensation_db_per_octave", 4.5))

    channel_features = {}
    channel_stems = {}
    channel_priorities = {}
    channel_measurements = {}

    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        mono = mono_sum(audio)
        block, measurement_meta = _focused_fft_measurement_block(
            mono,
            sr,
            plan.instrument,
            fft_size=fft_size,
        )
        if len(block) == 0:
            continue
        features = extract_analysis_features(
            block,
            sample_rate=sr,
            fft_size=fft_size,
            octave_fraction=octave_fraction,
            slope_compensation_db_per_octave=slope_db,
        )
        if features.confidence <= 0.0:
            continue

        classification = classification_from_legacy_preset(
            _legacy_preset_for_instrument(plan.instrument),
            channel_name=plan.name,
            confidence=1.0,
            match_type="offline_measurement",
        )
        stems = [stem.value for stem in classification.stem_roles if stem.value != "MASTER"]
        if not stems:
            stems = ["LEAD"] if plan.instrument == "lead_vocal" else ["UNKNOWN"]

        channel_features[channel] = features
        channel_stems[channel] = stems
        channel_priorities[channel] = float(classification.priority)
        channel_measurements[channel] = {
            "file": plan.path.name,
            "instrument": plan.instrument,
            "source_role": classification.source_role.value,
            "stem_roles": stems,
            "priority": round(float(classification.priority), 3),
            **measurement_meta,
        }

    if not channel_features:
        return {}, {}, None, {}, {}, {}

    stem_features = aggregate_stem_features(channel_features, channel_stems)
    contribution_matrix = build_stem_contribution_matrix(
        {
            stem_name: features
            for stem_name, features in stem_features.items()
            if stem_name != "MASTER"
        }
    )
    return (
        channel_features,
        stem_features,
        contribution_matrix,
        channel_stems,
        channel_priorities,
        channel_measurements,
    )


def _measured_channel_level_db(plan: ChannelPlan, features: Any) -> float:
    return float(getattr(features, "rms_db", -100.0) + plan.trim_db + plan.fader_db)


def _lead_handoff_balance_recommendations(
    plans: dict[int, ChannelPlan],
    channel_features: dict[int, Any],
    lead_channels: list[int],
) -> list[tuple[str, Any, list[TypedCorrectionAction]]]:
    measured_leads: list[tuple[int, float, float]] = []
    for channel in lead_channels:
        plan = plans.get(channel)
        features = channel_features.get(channel)
        if plan is None or features is None or plan.muted:
            continue
        active_ratio = float(plan.metrics.get("analysis_active_ratio") or 0.0)
        if active_ratio < 0.015:
            continue
        measured_leads.append((channel, _measured_channel_level_db(plan, features), active_ratio))

    if len(measured_leads) < 2:
        return []

    anchor_channel, anchor_level_db, _ = max(measured_leads, key=lambda item: item[1])
    anchor_target_db = anchor_level_db - 0.8
    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []

    for channel, level_db, active_ratio in measured_leads:
        if channel == anchor_channel:
            continue
        shortfall_db = anchor_target_db - level_db
        if shortfall_db < 0.75:
            continue

        plan = plans[channel]
        boost_db = float(min(1.5, shortfall_db))
        if plan.fader_db <= -0.25:
            action: TypedCorrectionAction = ChannelFaderMove(
                channel_id=channel,
                target_db=min(0.0, float(plan.fader_db + boost_db)),
                delta_db=boost_db,
                is_lead=True,
                reason=f"Measured lead handoff balance vs {plans[anchor_channel].path.name}",
            )
        else:
            action = ChannelGainMove(
                channel_id=channel,
                target_db=float(np.clip(plan.trim_db + boost_db, -12.0, 12.0)),
                reason=f"Measured lead handoff balance vs {plans[anchor_channel].path.name}",
            )

        confidence = min(1.0, 0.55 + shortfall_db / 3.0)
        recommendations.append((
            "lead_handoff_balance",
            DetectedProblem(
                problem_type="lead_handoff_balance",
                description="Measured secondary lead sits below the anchor lead level",
                channel_id=channel,
                stem="LEAD",
                band_name="RMS",
                persistence_sec=max(0.5, float(active_ratio * 10.0)),
                features=channel_features[channel],
                confidence_risk=ConfidenceRisk(
                    problem_confidence=confidence,
                    culprit_confidence=1.0,
                    action_confidence=min(1.0, confidence * 0.92),
                    risk_score=0.18,
                ),
                expected_effect="Bring quieter lead handoffs closer to the anchor lead without flattening the vocal hierarchy.",
            ),
            [action],
        ))

    return recommendations


def _cymbal_buildup_recommendations(
    plans: dict[int, ChannelPlan],
    channel_features: dict[int, Any],
    master_features: Any | None,
) -> list[tuple[str, Any, list[TypedCorrectionAction]]]:
    if master_features is None:
        return []

    harshness_excess = max(0.0, float(master_features.mix_indexes.harshness_index))
    sibilance_excess = max(0.0, float(master_features.mix_indexes.sibilance_index))
    if max(harshness_excess, sibilance_excess) < 1.75:
        return []

    direct_candidates: list[tuple[int, float]] = []
    ambient_candidates: list[tuple[int, float]] = []
    for channel, features in channel_features.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        if plan.instrument not in {"hi_hat", "ride", "overhead", "room"}:
            continue
        score = (
            max(0.0, float(features.mix_indexes.harshness_index)) * 0.8
            + max(0.0, float(features.mix_indexes.sibilance_index)) * 1.2
            + max(0.0, float(features.mix_indexes.air_index)) * 0.4
        )
        if plan.instrument in {"hi_hat", "ride"}:
            score *= 1.15
            direct_candidates.append((channel, score))
        else:
            ambient_candidates.append((channel, score))

    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []
    dominant_band = "SIBILANCE" if sibilance_excess >= harshness_excess else "HARSHNESS"
    dominant_freq = 7500.0 if dominant_band == "SIBILANCE" else 4200.0
    dominant_q = 2.0 if dominant_band == "SIBILANCE" else 1.8

    if direct_candidates:
        direct_channel, direct_score = max(direct_candidates, key=lambda item: item[1])
        if direct_score >= 0.7:
            plan = plans[direct_channel]
            cut_db = float(min(1.5, 0.75 + 0.2 * max(harshness_excess, sibilance_excess)))
            target_db = max(-30.0, float(plan.fader_db - cut_db))
            recommendations.append((
                "cymbal_buildup",
                DetectedProblem(
                    problem_type="cymbal_buildup",
                    description="Measured direct cymbal energy is dominating the upper bands",
                    channel_id=direct_channel,
                    stem="DRUMS",
                    band_name=dominant_band,
                    persistence_sec=max(1.0, direct_score),
                    features=channel_features[direct_channel],
                    confidence_risk=ConfidenceRisk(
                        problem_confidence=min(1.0, 0.5 + direct_score / 3.0),
                        culprit_confidence=min(1.0, direct_score / 3.0),
                        action_confidence=0.82,
                        risk_score=0.22,
                    ),
                    expected_effect="Reduce direct cymbal dominance while preserving the drum image.",
                ),
                [ChannelFaderMove(
                    channel_id=direct_channel,
                    target_db=target_db,
                    delta_db=target_db - float(plan.fader_db),
                    reason=f"Measured cymbal buildup on {plan.path.name}",
                )],
            ))

    if ambient_candidates:
        ambient_channel, ambient_score = max(ambient_candidates, key=lambda item: item[1])
        if ambient_score >= 0.55:
            plan = plans[ambient_channel]
            recommendations.append((
                "cymbal_buildup",
                DetectedProblem(
                    problem_type="cymbal_buildup",
                    description="Measured overhead or room cymbal wash is dominating the upper bands",
                    channel_id=ambient_channel,
                    stem="DRUMS",
                    band_name=dominant_band,
                    persistence_sec=max(1.0, ambient_score),
                    features=channel_features[ambient_channel],
                    confidence_risk=ConfidenceRisk(
                        problem_confidence=min(1.0, 0.48 + ambient_score / 3.5),
                        culprit_confidence=min(1.0, ambient_score / 3.5),
                        action_confidence=0.76,
                        risk_score=0.24,
                    ),
                    expected_effect="Reduce cymbal wash in the ambient drum capture without collapsing width.",
                ),
                [ChannelEQMove(
                    channel_id=ambient_channel,
                    band=4 if dominant_band == "SIBILANCE" else 3,
                    freq_hz=dominant_freq,
                    gain_db=-1.0,
                    q=dominant_q,
                    reason=f"Measured cymbal wash cleanup on {plan.path.name}",
                )],
            ))

    return recommendations


def apply_autofoh_measurement_corrections(
    plans: dict[int, ChannelPlan],
    rendered_channels: dict[int, np.ndarray],
    sr: int,
    autofoh_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    autofoh_config = autofoh_config or {}
    detector_config = autofoh_config.get("detectors", {})
    safety_config = AutoFOHSafetyConfig.from_config(
        autofoh_config.get("safety", {}).get("action_limits", {})
    )
    (
        channel_features,
        stem_features,
        contribution_matrix,
        channel_stems,
        channel_priorities,
        channel_measurements,
    ) = _build_autofoh_measurement_snapshot(rendered_channels, plans, sr, autofoh_config)

    if not channel_features or contribution_matrix is None:
        return {
            "enabled": False,
            "reason": "no_measurement_snapshot",
            "applied_actions": [],
            "detected_problems": [],
            "measurement_channels": channel_measurements,
        }

    lead_masking_config = detector_config.get("lead_masking", {})
    mud_config = detector_config.get("mud_excess", {})
    harshness_config = detector_config.get("harshness_excess", {})
    sibilance_config = detector_config.get("sibilance_excess", {})
    low_end_config = detector_config.get("low_end", {})

    lead_channels = [
        channel for channel, stems in channel_stems.items()
        if "LEAD" in stems or plans[channel].instrument == "lead_vocal"
    ]
    current_faders = {
        channel: float(plans[channel].fader_db)
        for channel in channel_features
    }

    analyzers = {
        "lead_masking": (
            bool(lead_masking_config.get("enabled", True)) and bool(lead_channels),
            LeadMaskingAnalyzer(
                masking_threshold_db=float(lead_masking_config.get("masking_threshold_db", 3.0)),
                culprit_share_threshold=float(lead_masking_config.get("min_culprit_contribution", 0.35)),
                persistence_required_cycles=1,
                lead_boost_db=float(lead_masking_config.get("lead_boost_db", 0.5)),
            ),
        ),
        "mud_excess": (
            bool(mud_config.get("enabled", True)),
            MudExcessDetector(
                threshold_db=float(mud_config.get("threshold_db", 2.5)),
                persistence_required_cycles=1,
                hysteresis_db=float(mud_config.get("hysteresis_db", 0.75)),
            ),
        ),
        "harshness_excess": (
            bool(harshness_config.get("enabled", True)),
            HarshnessExcessDetector(
                threshold_db=float(harshness_config.get("threshold_db", 2.5)),
                persistence_required_cycles=1,
                hysteresis_db=float(harshness_config.get("hysteresis_db", 0.75)),
            ),
        ),
        "sibilance_excess": (
            bool(sibilance_config.get("enabled", True)),
            SibilanceExcessDetector(
                threshold_db=float(sibilance_config.get("threshold_db", 2.5)),
                persistence_required_cycles=1,
                hysteresis_db=float(sibilance_config.get("hysteresis_db", 0.75)),
            ),
        ),
        "low_end": (
            bool(low_end_config.get("enabled", True)),
            LowEndAnalyzer(
                sub_threshold_db=float(low_end_config.get("sub_threshold_db", 4.0)),
                bass_threshold_db=float(low_end_config.get("bass_threshold_db", 3.0)),
                body_threshold_db=float(low_end_config.get("body_threshold_db", 2.5)),
                culprit_share_threshold=float(low_end_config.get("min_culprit_contribution", 0.35)),
                persistence_required_cycles=1,
                hysteresis_db=float(low_end_config.get("hysteresis_db", 0.75)),
            ),
        ),
    }

    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []
    if analyzers["lead_masking"][0]:
        lead_result = analyzers["lead_masking"][1].analyze(
            channel_features=channel_features,
            channel_stems=channel_stems,
            stem_features=stem_features,
            contribution_matrix=contribution_matrix,
            lead_channel_ids=lead_channels,
            current_faders_db=current_faders,
            lead_priorities=channel_priorities,
            runtime_state=RuntimeState.PRE_SHOW_CHECK,
        )
        if lead_result.problem:
            recommendations.append(("lead_masking", lead_result.problem, lead_result.candidate_actions))

    recommendations.extend(
        _lead_handoff_balance_recommendations(
            plans,
            channel_features,
            lead_channels,
        )
    )

    master_features = stem_features.get("MASTER")
    if master_features is not None:
        if analyzers["low_end"][0]:
            low_end_result = analyzers["low_end"][1].analyze(
                master_features=master_features,
                contribution_matrix=contribution_matrix,
                channel_features=channel_features,
                channel_stems=channel_stems,
            )
            if low_end_result.problem:
                recommendations.append(("low_end", low_end_result.problem, low_end_result.candidate_actions))

        for label in ("mud_excess", "harshness_excess", "sibilance_excess"):
            enabled, detector = analyzers[label]
            if not enabled:
                continue
            recommendation = detector.observe(
                master_features=master_features,
                contribution_matrix=contribution_matrix,
                channel_features=channel_features,
                channel_stems=channel_stems,
            )
            if recommendation.problem:
                recommendations.append((label, recommendation.problem, recommendation.candidate_actions))

        recommendations.extend(
            _cymbal_buildup_recommendations(
                plans,
                channel_features,
                master_features,
            )
        )

    adapter = OfflineAutoFOHConsole(plans)
    safety_controller = AutoFOHSafetyController(adapter, config=safety_config)
    applied_actions = []
    detected_problems = []
    for label, problem, actions in recommendations:
        detected_problems.append({
            "label": label,
            "problem": _problem_to_dict(problem),
            "candidate_actions": [_typed_action_to_dict(action) for action in actions],
        })
        if not actions:
            continue
        action = actions[0]
        decision = safety_controller.execute(action, RuntimeState.PRE_SHOW_CHECK)
        action_report = {
            "label": label,
            "requested_action": _typed_action_to_dict(action),
            "applied_action": _typed_action_to_dict(decision.action),
            "sent": bool(decision.sent),
            "bounded": bool(decision.bounded),
            "allowed": bool(decision.allowed),
            "supported": bool(decision.supported),
            "message": decision.message,
            "problem": _problem_to_dict(problem),
        }
        applied_actions.append(action_report)
        if decision.sent:
            channel_id = getattr(decision.action, "channel_id", None)
            if channel_id in plans:
                plans[channel_id].autofoh_actions.append(action_report)

    master_indexes = {}
    if master_features is not None:
        master_indexes = {
            key: round(float(value), 3)
            for key, value in master_features.mix_indexes.as_dict().items()
        }

    return {
        "enabled": True,
        "measurement_mode": "autofoh_analyzers",
        "measurement_channels": channel_measurements,
        "master_indexes": master_indexes,
        "detected_problems": detected_problems,
        "applied_actions": applied_actions,
        "virtual_console_calls": adapter.calls,
        "notes": [
            "All additional offline correction moves in this pass come from measured AutoFOH detector outputs.",
            "Lead handoff and cymbal buildup decisions are derived from the same measured analyzer snapshot.",
            "Legacy codex heuristic correction layers are disabled in analyzer-only mode.",
        ],
    }


def apply_autofoh_analyzer_pass(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    autofoh_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rendered_channels = {
        channel: render_channel(plan.path, plan, target_len, sr)
        for channel, plan in plans.items()
        if not plan.muted
    }
    return apply_autofoh_measurement_corrections(
        plans,
        rendered_channels,
        sr,
        autofoh_config=autofoh_config,
    )


def apply_vocal_bed_balance(
    plans: dict[int, ChannelPlan],
    desired_vocal_delta_db: float = 6.5,
    max_bed_attenuation_db: float = 3.0,
    protected_instruments: tuple[str, ...] = ("kick", "snare", "bass_guitar"),
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
            if (
                plan.instrument == "lead_vocal"
                or plan.muted
                or plan.instrument in protected_instruments
            ):
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
        "protected_instruments": list(protected_instruments),
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
    if instrument == "kick":
        return 2
    if instrument in {"snare", "bass_guitar"}:
        return 3
    if instrument in {"electric_guitar", "accordion", "playback", "backing_vocal"}:
        return 4
    if instrument in {"overhead", "room", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
        return 5
    return 4


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
            "range_db": 5.8 if active_ratio < 0.3 else 4.8,
            "open_ms": 22.0,
            "close_ms": 200.0,
            "hold_ms": 240.0,
            "threshold_db": float(threshold_db) - 1.5,
        }
    if instrument == "backing_vocal":
        return {
            "range_db": 5.0 if active_ratio < 0.35 else 4.2,
            "open_ms": 22.0,
            "close_ms": 180.0,
            "hold_ms": 210.0,
            "threshold_db": float(threshold_db) - 0.8,
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
        if plan.instrument == "kick" and gain < 0.0 and band in {"sub", "bass", "low_mid"}:
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
        if plan.instrument == "bass_guitar" and band in {"sub", "bass"} and gain < 0.0:
            scale = 0.36

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
            if plans[channel].instrument == "bass_guitar" and item["band"] in {"sub", "bass"}:
                lower = -1.8
            else:
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
        "notes": [
            "Cross-adaptive EQ is priority-driven: lower numeric priority means the channel is protected first.",
            "Lead vocals keep highest EQ priority, while competing accompaniment receives most anti-mask cuts.",
        ],
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
        return "backing_vocal", -0.32, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "back vox r" in name or "back vocal r" in name or "bvox r" in name or name in {"backs r", "back r", "bgv r"}:
        return "backing_vocal", 0.32, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "back vox" in name or "back vocal" in name or "bvox" in name or "backs" in name or "bgv" in name:
        return "backing_vocal", 0.0, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "vox" in name or "vocal" in name:
        return "lead_vocal", 0.0, 90.0, -18.8, [(250, -2.5, 1.4), (3100, 2.5, 1.0), (10500, 1.5, 0.8)], (-25, 4.4, 4, 115), False
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


def compressor(
    x: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float = 0.0,
    auto_makeup: bool = True,
) -> np.ndarray:
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
    gain = 10.0 ** (-smoothed / 20.0)
    compressed = (x * gain).astype(np.float32)

    total_makeup_db = float(makeup_db)
    if auto_makeup and len(compressed):
        input_rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
        output_rms = float(np.sqrt(np.mean(np.square(compressed))) + 1e-12)
        suppressed_db = max(0.0, amp_to_db(input_rms) - amp_to_db(output_rms))
        total_makeup_db += suppressed_db

    if abs(total_makeup_db) > 1e-4:
        compressed = (compressed * db_to_amp(total_makeup_db)).astype(np.float32)
    return compressed


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
    mono = mono * db_to_amp(plan.trim_db)
    if plan.phase_invert:
        mono = -mono
    mono = delay_signal(mono, sr, plan.delay_ms)
    x = mono
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
    stereo = pan_mono(x, plan.pan)
    return (stereo * db_to_amp(plan.fader_db)).astype(np.float32)


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


def _band_rms_db(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    mono = mono_sum(audio)
    block = _analysis_block(mono, sr)
    if len(block) < 256:
        return -100.0
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    idx = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(idx):
        return -100.0
    band_rms = float(np.sqrt(np.mean(np.square(spec[idx]))) + 1e-12)
    return amp_to_db(band_rms)


def _band_power(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    mono = mono_sum(audio)
    if len(mono) < 256:
        return 1e-12
    windowed = mono * np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    power = np.square(spec)
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    idx = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(idx):
        return 1e-12
    return float(np.sum(power[idx]) + 1e-12)


def _stem_roles_for_plan(plan: ChannelPlan) -> list[str]:
    classification = classification_from_legacy_preset(
        _legacy_preset_for_instrument(plan.instrument),
        channel_name=plan.name,
        confidence=1.0,
        match_type="offline_measurement",
    )
    stems = [stem.value for stem in classification.stem_roles if stem.value != "MASTER"]
    if not stems:
        stems = ["LEAD"] if plan.instrument == "lead_vocal" else ["UNKNOWN"]
    return stems


def _build_rendered_stem_groups(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]], dict[str, list[str]]]:
    stem_audio: dict[str, np.ndarray] = {}
    stem_channels: dict[str, list[int]] = {}
    stem_files: dict[str, list[str]] = {}

    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        for stem in _stem_roles_for_plan(plan):
            if stem not in stem_audio:
                stem_audio[stem] = np.zeros_like(audio)
                stem_channels[stem] = []
                stem_files[stem] = []
            stem_audio[stem] += audio
            stem_channels[stem].append(channel)
            stem_files[stem].append(plan.path.name)

    if rendered_channels:
        first_audio = next(iter(rendered_channels.values()))
        master_audio = np.zeros_like(first_audio)
        for audio in rendered_channels.values():
            master_audio += audio
        stem_audio["MASTER"] = master_audio
        stem_channels["MASTER"] = sorted(rendered_channels)
        stem_files["MASTER"] = [plans[channel].path.name for channel in sorted(rendered_channels) if channel in plans]

    return stem_audio, stem_channels, stem_files


def _activity_window_audio(
    audio: np.ndarray,
    ranges: list[tuple[int, int]] | None,
    sr: int,
    *,
    max_total_sec: float = 4.0,
) -> np.ndarray:
    mono = mono_sum(audio)
    if len(mono) == 0:
        return mono.astype(np.float32)
    if not ranges:
        return _analysis_block(mono, sr, window_sec=min(max_total_sec, 3.0))

    total_limit = max(1024, int(max_total_sec * sr))
    parts: list[np.ndarray] = []
    collected = 0
    for start, end in ranges:
        start_i = max(0, int(start))
        end_i = min(len(mono), int(end))
        if end_i <= start_i:
            continue
        segment = mono[start_i:end_i]
        if len(segment) == 0:
            continue
        remaining = total_limit - collected
        if remaining <= 0:
            break
        if len(segment) > remaining:
            segment = segment[:remaining]
        parts.append(segment.astype(np.float32))
        collected += len(segment)
    if not parts:
        return _analysis_block(mono, sr, window_sec=min(max_total_sec, 3.0))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _ltas_tilt_profile(
    audio: np.ndarray,
    sr: int,
    *,
    compensation_db_per_octave: float = 4.5,
    window_sec: float = 3.0,
    segments: int = 6,
    smoothing_fraction: int = 3,
) -> dict[str, Any]:
    mono = mono_sum(audio)
    if len(mono) == 0:
        return {
            "compensation_db_per_octave": compensation_db_per_octave,
            "window_sec": window_sec,
            "segments": 0,
            "smoothing": f"1/{smoothing_fraction} octave",
            "regions_db": {},
            "weight_60_120_vs_plateau_db": 0.0,
            "sub_35_60_vs_plateau_db": 0.0,
            "high_4k5_12k_vs_plateau_db": 0.0,
            "plateau_spread_db": 0.0,
            "curve": [],
        }

    window = max(2048, int(window_sec * sr))
    starts = _active_segment_starts(mono, sr, window_sec=window_sec, count=segments)
    if not starts:
        starts = [0]

    spectra = []
    for start in starts:
        end = min(len(mono), start + window)
        block = mono[start:end]
        if len(block) < window:
            block = np.pad(block, (0, window - len(block)))
        block = block.astype(np.float32) * np.hanning(len(block)).astype(np.float32)
        spec = np.abs(np.fft.rfft(block)) + 1e-12
        spectra.append(np.square(spec))
    avg_power = np.mean(np.stack(spectra, axis=0), axis=0)
    freqs = np.fft.rfftfreq(window, 1.0 / sr)
    bands = build_fractional_octave_bands(fraction=smoothing_fraction, start_hz=20.0, stop_hz=20000.0)

    curve = []
    levels = {}
    for band in bands:
        mask = (freqs >= band.low_hz) & (freqs < band.high_hz)
        if not np.any(mask):
            continue
        raw_level_db = 10.0 * np.log10(float(np.sum(avg_power[mask])) + 1e-12)
        compensated_level_db = raw_level_db + compensation_db_per_octave * math.log2(max(band.center_hz, 20.0) / 100.0)
        entry = {
            "center_hz": round(float(band.center_hz), 2),
            "raw_db": round(float(raw_level_db), 2),
            "compensated_db": round(float(compensated_level_db), 2),
        }
        curve.append(entry)
        levels[band.center_hz] = float(compensated_level_db)

    def region(lo: float, hi: float) -> float:
        values = [level for hz, level in levels.items() if lo <= hz < hi]
        if not values:
            return -100.0
        return float(np.mean(values))

    plateau_values = [level for hz, level in levels.items() if 90.0 <= hz < 4500.0]
    plateau_db = float(np.mean(plateau_values)) if plateau_values else -100.0
    regions_db = {
        "infra_20_35": round(region(20.0, 35.0), 2),
        "sub_35_60": round(region(35.0, 60.0), 2),
        "weight_60_120": round(region(60.0, 120.0), 2),
        "plateau_90_4500": round(plateau_db, 2),
        "high_4500_12000": round(region(4500.0, 12000.0), 2),
        "air_12000_20000": round(region(12000.0, 20000.0), 2),
    }

    return {
        "compensation_db_per_octave": compensation_db_per_octave,
        "window_sec": round(window_sec, 2),
        "segments": len(starts),
        "smoothing": f"1/{smoothing_fraction} octave",
        "regions_db": regions_db,
        "weight_60_120_vs_plateau_db": round(regions_db["weight_60_120"] - regions_db["plateau_90_4500"], 2),
        "sub_35_60_vs_plateau_db": round(regions_db["sub_35_60"] - regions_db["plateau_90_4500"], 2),
        "high_4500_12000_vs_plateau_db": round(regions_db["high_4500_12000"] - regions_db["plateau_90_4500"], 2),
        "plateau_spread_db": round(
            float(max(plateau_values) - min(plateau_values)) if plateau_values else 0.0,
            2,
        ),
        "curve": curve,
    }


def _stem_mix_snapshot(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
) -> dict[str, Any]:
    rendered_channels = {
        channel: render_channel(plan.path, plan, target_len, sr)
        for channel, plan in plans.items()
        if not plan.muted
    }
    stem_audio, stem_channels, stem_files = _build_rendered_stem_groups(rendered_channels, plans)
    (
        channel_features,
        stem_features,
        contribution_matrix,
        _channel_stems,
        _channel_priorities,
        _channel_measurements,
    ) = _build_autofoh_measurement_snapshot(rendered_channels, plans, sr)

    stems_report = []
    for stem_name in sorted(stem_audio):
        audio = stem_audio[stem_name]
        mono = mono_sum(audio)
        basic_metrics = metrics_for(mono, sr, instrument="custom")
        features = stem_features.get(stem_name)
        mix_indexes = getattr(features, "mix_indexes", None)
        stems_report.append({
            "stem": stem_name,
            "channel_count": len(stem_channels.get(stem_name, [])),
            "channels": stem_channels.get(stem_name, []),
            "files": stem_files.get(stem_name, []),
            "rms_db": round(float(basic_metrics.get("rms_db", -100.0)), 2),
            "dynamic_range_db": round(float(basic_metrics.get("dynamic_range_db", 0.0)), 2),
            "crest_factor_db": round(float(getattr(features, "crest_factor_db", basic_metrics.get("dynamic_range_db", 0.0))), 2),
            "band_energy": {
                band_name: round(float(level_db), 2)
                for band_name, level_db in basic_metrics.get("band_energy", {}).items()
            },
            "mix_indexes": {
                "sub_index": round(float(getattr(mix_indexes, "sub_index", 0.0)), 2),
                "bass_index": round(float(getattr(mix_indexes, "bass_index", 0.0)), 2),
                "body_index": round(float(getattr(mix_indexes, "body_index", 0.0)), 2),
                "mud_index": round(float(getattr(mix_indexes, "mud_index", 0.0)), 2),
                "presence_index": round(float(getattr(mix_indexes, "presence_index", 0.0)), 2),
                "harshness_index": round(float(getattr(mix_indexes, "harshness_index", 0.0)), 2),
                "sibilance_index": round(float(getattr(mix_indexes, "sibilance_index", 0.0)), 2),
                "air_index": round(float(getattr(mix_indexes, "air_index", 0.0)), 2),
            } if mix_indexes is not None else {},
        })

    band_hierarchy = []
    if contribution_matrix is not None:
        for band_name, row in sorted(contribution_matrix.band_contributions.items()):
            ordered = sorted(row.items(), key=lambda item: item[1], reverse=True)
            dominant_stem, dominant_share = ordered[0] if ordered else ("", 0.0)
            runner_up_stem, runner_up_share = ordered[1] if len(ordered) > 1 else ("", 0.0)
            band_hierarchy.append({
                "band": band_name,
                "dominant_stem": dominant_stem,
                "dominant_share": round(float(dominant_share), 4),
                "runner_up_stem": runner_up_stem,
                "runner_up_share": round(float(runner_up_share), 4),
            })

    master_mix_indexes = {}
    for item in stems_report:
        if item["stem"] == "MASTER":
            master_mix_indexes = dict(item.get("mix_indexes", {}))
            break

    slope_conformity = {
        "reference_tilt_db_per_octave": 4.5,
        "master_mix_indexes": master_mix_indexes,
        "sub_deficit_db": round(max(0.0, -float(master_mix_indexes.get("sub_index", 0.0))), 2),
        "bass_deficit_db": round(max(0.0, -float(master_mix_indexes.get("bass_index", 0.0))), 2),
        "body_deficit_db": round(max(0.0, -float(master_mix_indexes.get("body_index", 0.0))), 2),
    }

    kick_focus = {}
    kick_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "kick"),
        None,
    )
    if kick_entry:
        kick_channel, kick_plan = kick_entry
        kick_audio = rendered_channels.get(kick_channel)
        drums_audio = stem_audio.get("DRUMS")
        master_audio = stem_audio.get("MASTER")
        if kick_audio is not None and drums_audio is not None and master_audio is not None:
            activity = kick_plan.event_activity or _event_activity_ranges(mono_sum(kick_audio), sr, "kick") or {}
            ranges = activity.get("ranges") or []
            kick_focus_audio = _activity_window_audio(kick_audio, ranges, sr)
            drums_focus_audio = _activity_window_audio(drums_audio, ranges, sr)
            master_focus_audio = _activity_window_audio(master_audio, ranges, sr)
            kick_metrics = metrics_for(kick_focus_audio, sr, instrument="kick")
            kick_punch_power = _band_power(kick_focus_audio, sr, 55.0, 95.0)
            kick_click_power = _band_power(kick_focus_audio, sr, 2500.0, 4500.0)
            drums_punch_power = _band_power(drums_focus_audio, sr, 55.0, 95.0)
            drums_click_power = _band_power(drums_focus_audio, sr, 2500.0, 4500.0)
            master_click_power = _band_power(master_focus_audio, sr, 2500.0, 4500.0)
            kick_focus = {
                "channel": kick_channel,
                "file": kick_plan.path.name,
                "analysis_ranges": len(ranges),
                "active_sec": round(float((activity.get("active_samples") or 0) / sr), 3) if sr else 0.0,
                "kick_dynamic_range_db": round(float(kick_metrics.get("dynamic_range_db", 0.0)), 2),
                "kick_punch_db": round(_band_rms_db(kick_focus_audio, sr, 55.0, 95.0), 2),
                "kick_click_db": round(_band_rms_db(kick_focus_audio, sr, 2500.0, 4500.0), 2),
                "kick_box_db": round(_band_rms_db(kick_focus_audio, sr, 220.0, 420.0), 2),
                "kick_click_minus_punch_db": round(
                    _band_rms_db(kick_focus_audio, sr, 2500.0, 4500.0)
                    - _band_rms_db(kick_focus_audio, sr, 55.0, 95.0),
                    2,
                ),
                "kick_click_share_in_drums": round(float(kick_click_power / max(drums_click_power, 1e-12)), 4),
                "kick_punch_share_in_drums": round(float(kick_punch_power / max(drums_punch_power, 1e-12)), 4),
                "kick_click_share_in_master": round(float(kick_click_power / max(master_click_power, 1e-12)), 4),
            }

    tilt_conformity = {
        stem_name: _ltas_tilt_profile(audio, sr)
        for stem_name, audio in stem_audio.items()
        if stem_name in {"MASTER", "DRUMS", "MUSIC", "LEAD", "BASS", "KICK"}
    }

    return {
        "rendered_channels": rendered_channels,
        "stem_audio": stem_audio,
        "stems": stems_report,
        "band_hierarchy": band_hierarchy,
        "slope_conformity": slope_conformity,
        "tilt_conformity": tilt_conformity,
        "kick_focus": kick_focus,
    }


def apply_stem_mix_verification(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    genre: str | None = None,
) -> dict[str, Any]:
    snapshot_before = _stem_mix_snapshot(plans, target_len, sr)
    genre_token = str(genre or "").strip().lower()
    kick_focus_before = snapshot_before.get("kick_focus") or {}
    kick_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "kick"),
        None,
    )
    if not kick_entry:
        return {
            "enabled": False,
            "reason": "kick_missing",
            "before": {
                "stems": snapshot_before["stems"],
                "band_hierarchy": snapshot_before["band_hierarchy"],
                "slope_conformity": snapshot_before["slope_conformity"],
                "tilt_conformity": snapshot_before["tilt_conformity"],
                "kick_focus": kick_focus_before,
            },
        }

    _, kick_plan = kick_entry
    desired_click_share_in_drums = 0.14 if genre_token == "rock" else 0.09
    desired_click_share_in_master = 0.045 if genre_token == "rock" else 0.03
    desired_dynamic_range_max_db = 16.5 if genre_token == "rock" else 18.0
    desired_click_minus_punch_db = -17.0 if genre_token == "rock" else -19.0
    desired_box_minus_click_db = -1.5 if genre_token == "rock" else 1.5
    slope_before = snapshot_before.get("slope_conformity") or {}
    master_mix_indexes_before = slope_before.get("master_mix_indexes") or {}
    master_tilt_before = (snapshot_before.get("tilt_conformity") or {}).get("MASTER", {})
    master_tilt_regions_before = master_tilt_before.get("regions_db") or {}

    click_share_drums = float(kick_focus_before.get("kick_click_share_in_drums", 0.0))
    click_share_master = float(kick_focus_before.get("kick_click_share_in_master", 0.0))
    kick_dynamic_range_db = float(kick_focus_before.get("kick_dynamic_range_db", 0.0))
    click_minus_punch_db = float(kick_focus_before.get("kick_click_minus_punch_db", -100.0))
    box_minus_click_db = float(kick_focus_before.get("kick_box_db", -100.0) - kick_focus_before.get("kick_click_db", -100.0))
    weight_vs_plateau_db = float(master_tilt_before.get("weight_60_120_vs_plateau_db", 0.0))
    sub_vs_plateau_db = float(master_tilt_before.get("sub_35_60_vs_plateau_db", 0.0))
    high_vs_plateau_db = float(master_tilt_before.get("high_4500_12000_vs_plateau_db", 0.0))
    plateau_spread_db = float(master_tilt_before.get("plateau_spread_db", 0.0))
    sub_index_before = float(master_mix_indexes_before.get("sub_index", 0.0))
    bass_index_before = float(master_mix_indexes_before.get("bass_index", 0.0))
    body_index_before = float(master_mix_indexes_before.get("body_index", 0.0))
    low_end_deficit = max(
        0.0,
        (-sub_index_before) - (0.8 if genre_token == "rock" else 1.1),
        (-bass_index_before) - (0.6 if genre_token == "rock" else 0.9),
        (-body_index_before) - (0.35 if genre_token == "rock" else 0.6),
    )
    target_weight_vs_plateau_db = 2.0 if genre_token == "rock" else 1.0
    target_sub_vs_plateau_db = 0.5 if genre_token == "rock" else -0.25
    tilt_weight_shortage = max(0.0, target_weight_vs_plateau_db - weight_vs_plateau_db)
    tilt_sub_shortage = max(0.0, target_sub_vs_plateau_db - sub_vs_plateau_db)
    tilt_brightness_excess = max(0.0, high_vs_plateau_db + 1.5)
    plateau_unevenness = max(0.0, plateau_spread_db - 6.0)

    click_shortage = max(
        0.0,
        desired_click_share_in_drums - click_share_drums,
        (desired_click_share_in_master - click_share_master) * 1.8,
    )
    dynamic_shortage = max(0.0, kick_dynamic_range_db - desired_dynamic_range_max_db)
    click_tone_shortage = max(0.0, desired_click_minus_punch_db - click_minus_punch_db)
    box_excess = max(0.0, box_minus_click_db - desired_box_minus_click_db)

    actions: list[dict[str, Any]] = []
    if click_shortage > 0.0 or dynamic_shortage > 0.0 or click_tone_shortage > 0.0 or box_excess > 0.0:
        before_comp = {
            "threshold_db": round(float(kick_plan.comp_threshold_db), 2),
            "ratio": round(float(kick_plan.comp_ratio), 2),
            "attack_ms": round(float(kick_plan.comp_attack_ms), 2),
            "release_ms": round(float(kick_plan.comp_release_ms), 2),
        }
        kick_plan.comp_threshold_db = float(min(kick_plan.comp_threshold_db, -23.5 if genre_token == "rock" else -22.0))
        kick_plan.comp_ratio = float(max(kick_plan.comp_ratio, 5.4 if genre_token == "rock" else 4.8))
        kick_plan.comp_attack_ms = float(np.clip(max(kick_plan.comp_attack_ms, 10.0), 10.0, 12.0))
        kick_plan.comp_release_ms = float(np.clip(kick_plan.comp_release_ms, 80.0, 95.0))
        actions.append({
            "type": "kick_compressor_reseat",
            "before": before_comp,
            "after": {
                "threshold_db": round(float(kick_plan.comp_threshold_db), 2),
                "ratio": round(float(kick_plan.comp_ratio), 2),
                "attack_ms": round(float(kick_plan.comp_attack_ms), 2),
                "release_ms": round(float(kick_plan.comp_release_ms), 2),
            },
            "reason": "Kick stem stays too spiky relative to the drum/master stems and needs firmer body control without choking the click.",
        })

    if click_shortage > 0.0 or click_tone_shortage > 0.0:
        click_gain_db = float(np.clip(0.8 + click_shortage * 12.0 + click_tone_shortage * 0.08, 0.8, 2.1))
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=3500.0,
            gain_db=click_gain_db,
            q=1.6,
        )
        if action is not None:
            actions.append({
                "type": "kick_click_boost",
                "target": {"freq_hz": 3500.0, "gain_db": round(click_gain_db, 2), "q": 1.6},
                "result": action,
                "reason": "Kick click share inside the drum and master stems is too low.",
            })

    if click_shortage > 0.015:
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=2200.0,
            gain_db=0.6,
            q=1.4,
        )
        if action is not None:
            actions.append({
                "type": "kick_upper_attack_support",
                "target": {"freq_hz": 2200.0, "gain_db": 0.6, "q": 1.4},
                "result": action,
                "reason": "Kick needs more upper attack definition so the beater survives inside the drum stem.",
            })

    if box_excess > 0.0:
        box_cut_db = float(np.clip(-0.6 - box_excess * 0.15, -1.4, -0.6))
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=320.0,
            gain_db=box_cut_db,
            q=1.25,
        )
        if action is not None:
            actions.append({
                "type": "kick_box_cleanup",
                "target": {"freq_hz": 320.0, "gain_db": round(box_cut_db, 2), "q": 1.25},
                "result": action,
                "reason": "Kick box energy is overtaking the click band and keeps the drum from sitting forward in the mix.",
            })

    if low_end_deficit > 0.0 or tilt_weight_shortage > 0.0 or tilt_sub_shortage > 0.0:
        low_end_support = max(low_end_deficit * 0.12, tilt_weight_shortage * 0.35, tilt_sub_shortage * 0.25)
        kick_low_gain_db = float(np.clip(0.75 + low_end_support, 0.75, 2.4))
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=62.0,
            gain_db=kick_low_gain_db,
            q=0.95,
        )
        if action is not None:
            actions.append({
                "type": "master_slope_low_end_support_kick",
                "target": {"freq_hz": 62.0, "gain_db": round(kick_low_gain_db, 2), "q": 0.95},
                "result": action,
                "reason": "AutoFOH slope-conformity check found a low-end deficit versus the -4.5 dB/oct target line.",
            })

        bass_entry = next(
            ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "bass_guitar"),
            None,
        )
        if bass_entry is not None:
            _, bass_plan = bass_entry
            bass_low_gain_db = float(np.clip(0.55 + max(low_end_deficit * 0.08, tilt_weight_shortage * 0.25), 0.55, 1.8))
            action = _merge_reference_eq_adjustment(
                bass_plan,
                freq=78.0,
                gain_db=bass_low_gain_db,
                q=1.0,
            )
            if action is not None:
                actions.append({
                    "type": "master_slope_low_end_support_bass",
                    "target": {"freq_hz": 78.0, "gain_db": round(bass_low_gain_db, 2), "q": 1.0},
                    "result": action,
                    "reason": "Bass stem gets bounded support so the final mix better conforms to the -4.5 dB/oct slope target without giving away kick priority.",
                })
            if max(low_end_deficit, tilt_weight_shortage) > 4.0 and bass_plan.fader_db < -0.6:
                before_fader = float(bass_plan.fader_db)
                bass_plan.fader_db = float(np.clip(bass_plan.fader_db + 0.55, -30.0, 0.0))
                if bass_plan.fader_db != before_fader:
                    actions.append({
                        "type": "master_slope_bass_fader_recovery",
                        "before": {"fader_db": round(before_fader, 2)},
                        "after": {"fader_db": round(float(bass_plan.fader_db), 2)},
                        "reason": "Kick remains the low-end leader, but the bass fader recovers a little because the master slope still lacks weight.",
                    })

        if max(low_end_deficit, tilt_weight_shortage, tilt_sub_shortage) > 4.0 and kick_plan.trim_db < 3.0:
            before_trim = float(kick_plan.trim_db)
            kick_plan.trim_db = float(np.clip(kick_plan.trim_db + 0.6, -18.0, 12.0))
            if kick_plan.trim_db != before_trim:
                actions.append({
                    "type": "master_slope_kick_trim_support",
                    "before": {"trim_db": round(before_trim, 2)},
                    "after": {"trim_db": round(float(kick_plan.trim_db), 2)},
                    "reason": "Kick gets a small pre-compression lift because the master still sits below the target low-end slope.",
                })

    if tilt_brightness_excess > 0.0 or plateau_unevenness > 0.0:
        cymbal_cut_db = float(np.clip(0.45 + tilt_brightness_excess * 0.18 + plateau_unevenness * 0.05, 0.45, 1.2))
        for channel, plan in plans.items():
            if plan.muted or plan.instrument not in {"overhead", "hi_hat", "ride"}:
                continue
            action = _merge_reference_eq_adjustment(
                plan,
                freq=8500.0,
                gain_db=-cymbal_cut_db,
                q=1.0,
            )
            if action is None:
                continue
            actions.append({
                "type": "tilt_high_band_trim",
                "channel": channel,
                "target": {"freq_hz": 8500.0, "gain_db": round(-cymbal_cut_db, 2), "q": 1.0},
                "result": action,
                "reason": "The compensated LTAS stays too bright above 4.5 kHz, so cymbal/top-end stems are trimmed to restore the intended post-5 kHz roll-off.",
            })

    if not actions:
        return {
            "enabled": True,
            "genre": genre_token,
            "applied": False,
            "notes": [
                "Stem mix verification assembled the mix by stems and checked spectral balance, dynamics, and band hierarchy.",
                "Kick stem already has enough click share and controlled dynamics inside the drum/master stems.",
            ],
            "before": {
                "stems": snapshot_before["stems"],
                "band_hierarchy": snapshot_before["band_hierarchy"],
                "slope_conformity": snapshot_before["slope_conformity"],
                "tilt_conformity": snapshot_before["tilt_conformity"],
                "kick_focus": kick_focus_before,
            },
            "after": {
                "stems": snapshot_before["stems"],
                "band_hierarchy": snapshot_before["band_hierarchy"],
                "slope_conformity": snapshot_before["slope_conformity"],
                "tilt_conformity": snapshot_before["tilt_conformity"],
                "kick_focus": kick_focus_before,
            },
            "actions": [],
        }

    snapshot_after = _stem_mix_snapshot(plans, target_len, sr)
    return {
        "enabled": True,
        "genre": genre_token,
        "applied": True,
        "notes": [
            "Stem mix verification assembles rendered channels into stems before the final print.",
            "The pass checks stem AChH, dynamics, and band hierarchy, then reseats the kick only if its click/dynamics are weak inside the drum or master stems.",
        ],
        "before": {
            "stems": snapshot_before["stems"],
            "band_hierarchy": snapshot_before["band_hierarchy"],
            "slope_conformity": snapshot_before["slope_conformity"],
            "tilt_conformity": snapshot_before["tilt_conformity"],
            "kick_focus": kick_focus_before,
        },
        "after": {
            "stems": snapshot_after["stems"],
            "band_hierarchy": snapshot_after["band_hierarchy"],
            "slope_conformity": snapshot_after["slope_conformity"],
            "tilt_conformity": snapshot_after["tilt_conformity"],
            "kick_focus": snapshot_after["kick_focus"],
        },
        "actions": actions,
    }


def apply_kick_bass_hierarchy(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    desired_kick_advantage_db: float = 1.5,
) -> dict[str, Any]:
    kick_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "kick"),
        None,
    )
    bass_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "bass_guitar"),
        None,
    )
    if not kick_entry or not bass_entry:
        return {"enabled": False, "reason": "kick_or_bass_missing"}

    kick_channel, kick_plan = kick_entry
    bass_channel, bass_plan = bass_entry
    kick_audio = render_channel(kick_plan.path, kick_plan, target_len, sr)
    bass_audio = render_channel(bass_plan.path, bass_plan, target_len, sr)

    kick_block, kick_meta = _analysis_signal_for_metrics(mono_sum(kick_audio), sr, "kick")
    bass_block, bass_meta = _analysis_signal_for_metrics(mono_sum(bass_audio), sr, "bass_guitar")
    kick_anchor_db = _band_rms_db(kick_block, sr, 55.0, 95.0)
    bass_anchor_db = _band_rms_db(bass_block, sr, 55.0, 95.0)
    kick_click_db = _band_rms_db(kick_block, sr, 2500.0, 4500.0)
    bass_low_mid_db = _band_rms_db(bass_block, sr, 90.0, 160.0)
    measured_advantage = kick_anchor_db - bass_anchor_db
    shortage_db = desired_kick_advantage_db - measured_advantage

    if shortage_db <= 0.25:
        return {
            "enabled": False,
            "reason": "hierarchy_already_satisfied",
            "kick_channel": kick_channel,
            "bass_channel": bass_channel,
            "kick_anchor_db": round(kick_anchor_db, 2),
            "bass_anchor_db": round(bass_anchor_db, 2),
            "measured_advantage_db": round(measured_advantage, 2),
            "desired_kick_advantage_db": round(desired_kick_advantage_db, 2),
        }

    kick_fader_before = float(kick_plan.fader_db)
    bass_fader_before = float(bass_plan.fader_db)
    kick_boost_db = float(np.clip(0.8 + shortage_db * 0.45, 0.75, 1.8))
    bass_cut_db = float(np.clip(0.45 + shortage_db * 0.4, 0.35, 1.4))
    kick_plan.fader_db = float(np.clip(kick_plan.fader_db + kick_boost_db, -30.0, 0.0))
    bass_plan.fader_db = float(np.clip(bass_plan.fader_db - bass_cut_db, -30.0, 0.0))

    kick_eq_changes: list[tuple[float, float, float]] = []
    bass_eq_changes: list[tuple[float, float, float]] = []

    if kick_anchor_db < bass_anchor_db + desired_kick_advantage_db:
        kick_eq_changes.append((68.0, float(np.clip(0.8 + shortage_db * 0.35, 0.8, 1.6)), 0.95))
        kick_eq_changes.append((3200.0, 0.6, 1.6))
    bass_overlap_db = max(bass_anchor_db, bass_low_mid_db)
    if shortage_db > 0.5 or bass_overlap_db > kick_anchor_db - 1.0:
        bass_eq_changes.append((82.0, float(np.clip(-0.7 - shortage_db * 0.2, -1.3, -0.7)), 1.0))
        bass_eq_changes.append((125.0, -0.6, 1.1))

    kick_plan.eq_bands.extend(kick_eq_changes)
    bass_plan.eq_bands.extend(bass_eq_changes)

    return {
        "enabled": True,
        "kick_channel": kick_channel,
        "bass_channel": bass_channel,
        "kick_file": kick_plan.path.name,
        "bass_file": bass_plan.path.name,
        "kick_analysis_mode": kick_meta.get("analysis_mode"),
        "bass_analysis_mode": bass_meta.get("analysis_mode"),
        "kick_anchor_db": round(kick_anchor_db, 2),
        "bass_anchor_db": round(bass_anchor_db, 2),
        "kick_click_db": round(kick_click_db, 2),
        "bass_low_mid_db": round(bass_low_mid_db, 2),
        "measured_advantage_db": round(measured_advantage, 2),
        "desired_kick_advantage_db": round(desired_kick_advantage_db, 2),
        "shortage_db": round(shortage_db, 2),
        "kick_fader_before_db": round(kick_fader_before, 2),
        "kick_fader_after_db": round(float(kick_plan.fader_db), 2),
        "bass_fader_before_db": round(bass_fader_before, 2),
        "bass_fader_after_db": round(float(bass_plan.fader_db), 2),
        "kick_eq_added": [
            {"freq_hz": round(freq, 2), "gain_db": round(gain, 2), "q": round(q, 2)}
            for freq, gain, q in kick_eq_changes
        ],
        "bass_eq_added": [
            {"freq_hz": round(freq, 2), "gain_db": round(gain, 2), "q": round(q, 2)}
            for freq, gain, q in bass_eq_changes
        ],
        "notes": [
            "Kick is protected as the low-end anchor and must stay ahead of bass in the punch band.",
            "Bass is trimmed only in the overlapping 55-125 Hz area so the groove stays intact while the kick leads.",
        ],
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
    reference_context: ReferenceMixContext | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Console-like 2-bus cleanup and glue.
    for ch in range(2):
        mix[:, ch] = highpass(mix[:, ch], sr, 28.0)
        mix[:, ch] = compressor(mix[:, ch], sr, threshold_db=-11.0, ratio=1.6, attack_ms=25, release_ms=250)

    meter = pyln.Meter(sr)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    peak_db = amp_to_db(peak)
    pre_lufs = None
    try:
        pre_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass

    reference_mastering: dict[str, Any] = {
        "requested": bool(reference_context is not None),
        "enabled": False,
    }

    if not final_limiter:
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
        if reference_context is not None:
            reference_mastering.update({
                "reason": "final_limiter_disabled",
                "reference_path": str(reference_context.path),
                "source_type": reference_context.source_type,
                "reference_sources": [str(path) for path in reference_context.source_paths],
            })
        return np.asarray(mix, dtype=np.float32), {
            "final_limiter": False,
            "soft_limiter": False,
            "static_master_gain_db": round(static_master_gain_db, 2),
            "live_peak_ceiling_dbfs": round(live_peak_ceiling_db, 2),
            "pre_master_peak_dbfs": round(peak_db, 2),
            "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
            "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
            "reference_mastering": reference_mastering,
            "note": "No final limiting or clipping stage; only static master trim is used for live-style headroom.",
        }

    if reference_context is not None:
        reference_input_mix = np.asarray(mix, dtype=np.float32).copy()
        reference_mastering.update({
            "reference_path": str(reference_context.path),
            "source_type": reference_context.source_type,
            "reference_sources": [str(path) for path in reference_context.source_paths],
        })
        if reference_context.audio is None or reference_context.sample_rate is None:
            reference_mastering["reason"] = "reference_has_no_audio_stream"
        else:
            ref_audio = reference_context.audio
            ref_sr = int(reference_context.sample_rate)
            if ref_sr != sr:
                ref_audio = resample_audio(ref_audio, ref_sr, sr)
            auto_master = AutoMaster(sample_rate=sr, target_lufs=target_lufs, true_peak_limit=-1.0)
            try:
                mastered = auto_master.master(mix, reference=ref_audio, sample_rate=sr)
                if isinstance(mastered, MasteringResult):
                    mastered_audio = np.asarray(mastered.audio, dtype=np.float32)
                    reference_mastering.update({
                        "enabled": bool(mastered.success),
                        "backend": "matchering_result",
                        "peak_dbfs": round(float(mastered.peak_db), 2),
                        "lufs": round(float(mastered.lufs), 2),
                        "gain_applied_db": round(float(mastered.gain_applied_db), 2),
                        "limiter_reduction_db": round(float(mastered.limiter_reduction_db), 2),
                        "error": mastered.error,
                    })
                else:
                    mastered_audio = np.asarray(mastered, dtype=np.float32)
                    post_lufs = None
                    try:
                        post_lufs = float(meter.integrated_loudness(mastered_audio))
                    except Exception:
                        pass
                    reference_mastering.update({
                        "enabled": True,
                        "backend": "reference_audio_fallback",
                        "peak_dbfs": round(amp_to_db(float(np.max(np.abs(mastered_audio))) if len(mastered_audio) else 0.0), 2),
                        "lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
                    })
                post_lufs = None
                try:
                    post_lufs = float(meter.integrated_loudness(mastered_audio))
                except Exception:
                    pass
                expected_floor = None
                style_lufs = float(reference_context.style_profile.loudness_lufs)
                if post_lufs is not None and np.isfinite(post_lufs):
                    comparison_points = [style_lufs, target_lufs]
                    if pre_lufs is not None and np.isfinite(pre_lufs):
                        comparison_points.append(float(pre_lufs))
                    expected_floor = min(comparison_points) - 8.0
                if (
                    post_lufs is None
                    or not np.isfinite(post_lufs)
                    or (expected_floor is not None and post_lufs < expected_floor)
                ):
                    reference_mastering.update({
                        "enabled": False,
                        "reason": "reference_mastering_rejected_low_loudness",
                        "rejected_post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
                        "expected_floor_lufs": round(expected_floor, 2) if expected_floor is not None else None,
                    })
                    mix = reference_input_mix
                else:
                    mix = mastered_audio
                    return np.asarray(mix, dtype=np.float32), {
                        "final_limiter": True,
                        "soft_limiter": False,
                        "ceiling_dbfs": -1.0,
                        "pre_master_peak_dbfs": round(peak_db, 2),
                        "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
                        "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
                        "reference_mastering": reference_mastering,
                    }
            except Exception as exc:
                reference_mastering.update({
                    "enabled": False,
                    "reason": "reference_mastering_failed",
                    "error": str(exc),
                })

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
        "pre_master_peak_dbfs": round(peak_db, 2),
        "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
        "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
        "reference_mastering": reference_mastering,
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
    parser.add_argument("--no-autofoh-analyzer-pass", action="store_true")
    parser.add_argument("--codex-correction-pass", action="store_true")
    parser.add_argument("--codex-orchestrator", action="store_true")
    parser.add_argument("--codex-orchestrator-dry-run", action="store_true")
    parser.add_argument("--codex-orchestrator-allow-llm", action="store_true")
    parser.add_argument("--codex-orchestrator-max-actions", type=int, default=5)
    parser.add_argument("--soft-master", action="store_true")
    parser.add_argument("--master-target-lufs", type=float, default=-16.0)
    parser.add_argument("--reference", default="", help="Path to an external reference track or saved style preset JSON")
    parser.add_argument("--genre", default="", help="Optional genre focus for bounded mix voicing, for example 'rock'")
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
    autofoh_config = config.get("autofoh", {})
    reference_context = prepare_reference_mix_context(args.reference)
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
            "vocal_target_delta_db": 7.0 if str(args.genre).strip().lower() == "rock" else 6.0,
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

    codex_corrections: dict[str, Any] = {
        "enabled": False,
        "actions": [],
        "requested": bool(args.codex_correction_pass),
        "notes": [
            "Legacy codex heuristic correction passes are disabled.",
            "Analyzer-only policy is active: additional corrections must come from measured AutoFOH detectors.",
        ] if args.codex_correction_pass else [],
    }
    codex_bleed_control = {
        "enabled": False,
        "requested": bool(args.codex_correction_pass),
        "notes": [
            "Legacy codex bleed-control heuristics are disabled.",
            "Use the AutoFOH analyzer pass and measured event-based dynamics instead.",
        ] if args.codex_correction_pass else [],
    }
    event_based_dynamics = apply_event_based_dynamics(plans)
    autofoh_analyzer_pass = (
        {"enabled": False, "notes": ["AutoFOH analyzer pass explicitly disabled by CLI flag."]}
        if args.no_autofoh_analyzer_pass
        else apply_autofoh_analyzer_pass(plans, target_len, sr, autofoh_config)
    )
    bass_drum_push = apply_bass_drum_push(plans, args.bass_drum_boost_db)
    kick_presence_boost = apply_kick_presence_boost(plans, args.kick_presence_boost_db)
    cymbal_focus_cleanup = apply_cymbal_cleanup_for_kick_focus(plans, args.kick_focus_cymbal_cut_db)
    vocal_bed_balance = {
        "enabled": False,
        "notes": [
            "Static vocal bed attenuation is disabled.",
            "Vocal space must come from priority EQ and measured analyzer corrections.",
        ],
    }
    cross_adaptive_eq = apply_cross_adaptive_eq(plans, target_len, sr)
    reference_mix_guidance = apply_reference_mix_guidance(plans, sr, reference_context)
    genre_mix_profile = apply_genre_mix_profile(plans, args.genre)
    kick_bass_hierarchy = apply_kick_bass_hierarchy(
        plans,
        target_len,
        sr,
        desired_kick_advantage_db=1.8 if str(args.genre).strip().lower() == "rock" else 1.5,
    )
    stem_mix_verification = apply_stem_mix_verification(
        plans,
        target_len,
        sr,
        genre=args.genre,
    )
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
            "autofoh_actions": plan.autofoh_actions,
            "cross_adaptive_eq": plan.cross_adaptive_eq,
            "peak_db": round(plan.metrics["peak_db"], 2),
            "rms_db": round(plan.metrics["rms_db"], 2),
            "analysis_mode": plan.metrics.get("analysis_mode"),
            "analysis_active_sec": plan.metrics.get("analysis_active_sec"),
            "analysis_active_ratio": plan.metrics.get("analysis_active_ratio"),
            "analysis_threshold_db": plan.metrics.get("analysis_threshold_db"),
        })

    dynamic_vocal_priority = {
        "enabled": False,
        "requested_disable_flag": bool(args.no_drum_vocal_duck),
        "notes": [
            "Vocal ducking is disabled.",
            "Vocal space must be created by priority EQ and measured analyzer corrections.",
        ],
    }
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
        reference_context=reference_context,
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
        "reference": str(reference_context.path) if reference_context is not None else "",
        "reference_sources": [str(path) for path in reference_context.source_paths] if reference_context is not None else [],
        "genre": str(args.genre or ""),
        "sample_rate": sr,
        "codex_orchestrator": orchestration_report,
        "duration_sec": round(target_len / sr, 3),
        "final_peak_dbfs": round(amp_to_db(float(np.max(np.abs(mix)))), 2),
        "final_lufs": round(final_lufs, 2) if final_lufs is not None and np.isfinite(final_lufs) else None,
        "music_bed_lufs": round(_music_bed_lufs(plans), 2),
        "vocal_bed_balance": vocal_bed_balance,
        "bass_drum_boost": bass_drum_push,
        "cross_adaptive_eq": cross_adaptive_eq,
        "reference_mix_guidance": reference_mix_guidance,
        "genre_mix_profile": genre_mix_profile,
        "kick_bass_hierarchy": kick_bass_hierarchy,
        "stem_mix_verification": stem_mix_verification,
        "dynamic_vocal_priority": dynamic_vocal_priority,
        "kick_presence_boost": kick_presence_boost,
        "kick_focus_cymbal_cut": cymbal_focus_cleanup,
        "soft_master": args.soft_master,
        "master_target_lufs": round(args.master_target_lufs, 2),
        "autofoh_analyzer_pass": autofoh_analyzer_pass,
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
