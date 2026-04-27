#!/usr/bin/env python3
"""Offline multitrack mix using only the shared-chat balancing logic.

This route intentionally avoids the existing rule engine, agent passes,
ducking, and master-EQ workflows. It follows the logic extracted from:
https://chatgpt.com/share/69e7cc2e-2a50-832a-88f6-bf6c2164e7cb

Core principles implemented here:
- The master spectrum is a balance meter, not a master-EQ target.
- Use +4.5 dB/oct compensated LTAS over the densest musical section.
- Prefer source/stem fixes over master processing.
- Build around three anchors: kick+bass, lead/vocal, snare/rhythmic attack.
- Add layers in order and react to broad sustained deviations only.
- Free vocal space by EQ on competing stems, not by ducking.
- Keep all moves small and bounded.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, lfilter


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from output_paths import ai_logs_dir, ai_logs_path, ai_mixing_dir, ai_mixing_path, ensure_ai_output_dirs, ensure_parent_dir  # noqa: E402


ANALYSIS_WINDOW_SEC = 18.0
COMPENSATION_DB_PER_OCTAVE = 4.5

BAND_SPECS = {
    "50_100": (50.0, 100.0),
    "100_200": (100.0, 200.0),
    "200_500": (200.0, 500.0),
    "500_1000": (500.0, 1000.0),
    "1000_2500": (1000.0, 2500.0),
    "1500_4000": (1500.0, 4000.0),
    "2500_5000": (2500.0, 5000.0),
    "5000_8000": (5000.0, 8000.0),
    "6000_10000": (6000.0, 10000.0),
    "700_2000": (700.0, 2000.0),
    "8000_12000": (8000.0, 12000.0),
}

# Targets are relative to the compensated analyzer zero-line. The zero-line is
# normalized to the average level in the 100 Hz - 5 kHz musical core.
DISPLAY_CORRIDOR = {
    "50_100": {"min": 3.0, "max": 8.0, "target": 5.5},
    "100_200": {"min": 0.0, "max": 4.0, "target": 2.0},
    "200_500": {"min": -2.0, "max": 2.0, "target": 0.5},
    "500_1000": {"min": -1.5, "max": 1.0, "target": -0.2},
    "1000_2500": {"min": -1.0, "max": 1.0, "target": 0.0},
    "2500_5000": {"min": -2.0, "max": 0.5, "target": -0.8},
    "5000_8000": {"min": -4.0, "max": 0.0, "target": -1.8},
    "8000_12000": {"min": -6.0, "max": -1.0, "target": -3.0},
}

MIDLINE_BANDS = ("100_200", "200_500", "500_1000", "1000_2500", "2500_5000")


@dataclass
class EQMove:
    freq_hz: float
    gain_db: float
    q: float
    reason: str


@dataclass
class MixChannel:
    name: str
    path: Path
    role: str
    stems: tuple[str, ...]
    priority: int
    pan: float
    audio: np.ndarray
    sample_rate: int
    fader_db: float = 0.0
    hpf_hz: float = 0.0
    eq_moves: list[EQMove] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)


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


def highpass(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    if freq <= 0.0:
        return x.astype(np.float32)
    b, a = butter(2, freq / (sr * 0.5), btype="highpass")
    return lfilter(b, a, x).astype(np.float32)


def peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    if abs(gain_db) < 1e-5:
        return x.astype(np.float32)
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


def equal_power_gains(pan: float) -> tuple[float, float]:
    pan = float(np.clip(pan, -100.0, 100.0))
    theta = (pan + 100.0) / 200.0 * (math.pi / 2.0)
    return float(math.cos(theta)), float(math.sin(theta))


def pan_mono(x: np.ndarray, pan: float) -> np.ndarray:
    left, right = equal_power_gains(pan)
    return np.column_stack((x * left, x * right)).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return amp_to_db(float(np.sqrt(np.mean(np.square(x))) + 1e-12))


def analysis_window_start(x: np.ndarray, sr: int, window_sec: float = ANALYSIS_WINDOW_SEC) -> int:
    window = max(4096, int(window_sec * sr))
    if len(x) <= window:
        return 0
    hop = max(2048, window // 4)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(x) - window + 1, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return best_start


def ltas_spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([0.0], dtype=np.float32), np.array([1e-12], dtype=np.float32)
    n_fft = min(len(x), 16384)
    if n_fft < 1024:
        n_fft = len(x)
    if n_fft <= 1:
        return np.array([0.0], dtype=np.float32), np.array([1e-12], dtype=np.float32)
    block = x[:n_fft]
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)).astype(np.float32) + 1e-12
    freqs = np.fft.rfftfreq(len(block), 1.0 / sr).astype(np.float32)
    return freqs, spec


def compensated_band_levels(x: np.ndarray, sr: int) -> dict[str, float]:
    freqs, spec = ltas_spectrum(x, sr)
    if len(spec) <= 1:
        return {band: -120.0 for band in DISPLAY_CORRIDOR}
    compensation_db = COMPENSATION_DB_PER_OCTAVE * np.log2(np.maximum(freqs, 1.0) / 100.0)
    weighted = spec * np.power(10.0, compensation_db / 20.0)
    levels = {}
    for band_name, (lo, hi) in BAND_SPECS.items():
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            levels[band_name] = -120.0
            continue
        # Use per-bin average so wide bands do not appear louder only because
        # they contain more FFT bins than narrower bands.
        levels[band_name] = amp_to_db(float(np.mean(weighted[idx])))

    # The shared-chat logic treats 100 Hz - 5 kHz as a calm zero-line, but not
    # as a hard target. Median is more stable than mean when one midrange band
    # briefly dominates the densest section.
    reference = float(np.median([levels[name] for name in MIDLINE_BANDS]))
    return {
        band_name: float(level_db - reference)
        for band_name, level_db in levels.items()
        if band_name in DISPLAY_CORRIDOR or band_name in {"1500_4000", "6000_10000", "700_2000"}
    }


def raw_band_energy(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    freqs, spec = ltas_spectrum(x, sr)
    idx = (freqs >= lo) & (freqs < hi)
    if not np.any(idx):
        return 0.0
    return float(np.sum(spec[idx]))


def stem_of(channel: MixChannel, preferred: set[str] | None = None) -> str:
    if preferred:
        for stem in channel.stems:
            if stem in preferred:
                return stem
    return channel.stems[0]


def default_pan(name: str) -> float:
    lower = name.lower()
    if "playback l" in lower:
        return -70.0
    if "playback r" in lower:
        return 70.0
    if "backs l" in lower:
        return -55.0
    if "backs r" in lower:
        return 55.0
    if "guitar l" in lower:
        return -55.0
    if "guitar r" in lower:
        return 55.0
    if "oh l" in lower:
        return -65.0
    if "oh r" in lower:
        return 65.0
    if "room dr l" in lower:
        return -75.0
    if "room dr r" in lower:
        return 75.0
    if "hi-hat" in lower:
        return 25.0
    if "ride" in lower:
        return -25.0
    if lower == "tom.wav":
        return -12.0
    if "f tom" in lower:
        return 12.0
    return 0.0


def classify_track(path: Path) -> tuple[str, tuple[str, ...], int]:
    name = path.name.lower()
    if "kick" in name:
        return "kick", ("KICK", "DRUMS"), 2
    if "snare" in name:
        return "snare", ("SNARE", "DRUMS"), 2
    if "f tom" in name or name == "tom.wav":
        return "toms", ("TOMS", "DRUMS"), 3
    if "hi-hat" in name:
        return "hi_hat", ("CYMBALS", "DRUMS"), 4
    if "ride" in name:
        return "ride", ("CYMBALS", "DRUMS"), 4
    if "oh " in name or "room dr" in name:
        return "overheads_room", ("CYMBALS", "DRUMS"), 4
    if "bass" in name:
        return "bass", ("BASS", "MUSIC"), 2
    if "guitar" in name:
        return "guitars", ("GUITARS", "MUSIC"), 3
    if "playback" in name:
        return "playback", ("PLAYBACK", "MUSIC"), 3
    if "backs" in name:
        return "bgv", ("BGV", "VOCALS"), 3
    if "vox" in name:
        return "lead_vocal", ("LEAD", "VOCALS"), 1
    return "unknown", ("UNKNOWN",), 5


def load_channels(input_dir: Path) -> tuple[list[MixChannel], int, int]:
    channels: list[MixChannel] = []
    sample_rate = None
    target_len = 0
    for path in sorted(input_dir.glob("*.wav")):
        audio, sr = read_mono(path)
        role, stems, priority = classify_track(path)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise RuntimeError(f"Mismatched sample rate for {path.name}: {sr} != {sample_rate}")
        target_len = max(target_len, len(audio))
        channels.append(
            MixChannel(
                name=path.name,
                path=path,
                role=role,
                stems=stems,
                priority=priority,
                pan=default_pan(path.name),
                audio=audio,
                sample_rate=sr,
            )
        )
    if not channels or sample_rate is None:
        raise RuntimeError(f"No WAV files found in {input_dir}")

    for channel in channels:
        if len(channel.audio) < target_len:
            channel.audio = np.pad(channel.audio, (0, target_len - len(channel.audio))).astype(np.float32)
    return channels, sample_rate, target_len


def bounded_fader_move(channel: MixChannel, delta_db: float, reason: str, max_step: float = 1.0, total_limit: float = 1.5):
    allowed_delta = float(np.clip(delta_db, -max_step, max_step))
    new_value = float(np.clip(channel.fader_db + allowed_delta, -total_limit, total_limit))
    actual_delta = new_value - channel.fader_db
    if abs(actual_delta) < 0.1:
        return
    before = channel.fader_db
    channel.fader_db = new_value
    channel.notes.append({
        "type": "fader",
        "reason": reason,
        "before_db": round(before, 2),
        "after_db": round(new_value, 2),
        "delta_db": round(actual_delta, 2),
    })


def bounded_eq_move(
    channel: MixChannel,
    freq_hz: float,
    gain_db: float,
    q: float,
    reason: str,
    max_step: float = 1.0,
    max_total: float = 2.0,
):
    requested = float(np.clip(gain_db, -max_step, max_step))
    current_total = sum(
        move.gain_db
        for move in channel.eq_moves
        if abs(math.log2(max(move.freq_hz, 20.0) / max(freq_hz, 20.0))) < 0.33
    )
    new_total = float(np.clip(current_total + requested, -max_total, max_total))
    actual = new_total - current_total
    if abs(actual) < 0.1:
        return
    channel.eq_moves.append(EQMove(freq_hz=float(freq_hz), gain_db=float(actual), q=float(q), reason=reason))
    channel.notes.append({
        "type": "eq",
        "reason": reason,
        "freq_hz": round(float(freq_hz), 1),
        "gain_db": round(float(actual), 2),
        "q": round(float(q), 2),
    })


def set_hpf_if_higher(channel: MixChannel, freq_hz: float, reason: str):
    freq_hz = float(freq_hz)
    if freq_hz <= channel.hpf_hz + 5.0:
        return
    before = channel.hpf_hz
    channel.hpf_hz = freq_hz
    channel.notes.append({
        "type": "hpf",
        "reason": reason,
        "before_hz": round(before, 1),
        "after_hz": round(freq_hz, 1),
    })


def process_mono(channel: MixChannel, start: int = 0, end: int | None = None) -> np.ndarray:
    x = channel.audio[start:end].astype(np.float32)
    if channel.hpf_hz > 0.0:
        x = highpass(x, channel.sample_rate, channel.hpf_hz)
    for move in channel.eq_moves:
        x = peaking_eq(x, channel.sample_rate, move.freq_hz, move.gain_db, move.q)
    if abs(channel.fader_db) > 1e-4:
        x = (x * db_to_amp(channel.fader_db)).astype(np.float32)
    return x.astype(np.float32)


def render_mix(channels: list[MixChannel], target_len: int, subset: set[str] | None = None) -> np.ndarray:
    mix = np.zeros((target_len, 2), dtype=np.float32)
    for channel in channels:
        if subset is not None and not any(stem in subset for stem in channel.stems):
            continue
        mono = process_mono(channel)
        mix += pan_mono(mono, channel.pan)
    return mix


def analysis_mix(channels: list[MixChannel], start: int, end: int, include_names: set[str] | None = None) -> np.ndarray:
    segment = np.zeros(end - start, dtype=np.float32)
    for channel in channels:
        if include_names is not None and channel.name not in include_names:
            continue
        segment += process_mono(channel, start, end)
    return segment


def band_shares(channels: list[MixChannel], start: int, end: int, lo: float, hi: float) -> dict[str, float]:
    energies = {}
    total = 0.0
    for channel in channels:
        energy = raw_band_energy(process_mono(channel, start, end), channel.sample_rate, lo, hi)
        energies[channel.name] = energy
        total += energy
    if total <= 1e-12:
        return {channel.name: 0.0 for channel in channels}
    return {name: float(value / total) for name, value in energies.items()}


def stem_shares(channels: list[MixChannel], start: int, end: int, lo: float, hi: float) -> dict[str, float]:
    energies: dict[str, float] = {}
    total = 0.0
    for channel in channels:
        stem = stem_of(channel, {"KICK", "SNARE", "TOMS", "CYMBALS", "BASS", "GUITARS", "PLAYBACK", "BGV", "LEAD"})
        energy = raw_band_energy(process_mono(channel, start, end), channel.sample_rate, lo, hi)
        energies[stem] = energies.get(stem, 0.0) + energy
        total += energy
    if total <= 1e-12:
        return {stem: 0.0 for stem in energies}
    return {name: float(value / total) for name, value in energies.items()}


def top_culprits(
    channels: list[MixChannel],
    start: int,
    end: int,
    lo: float,
    hi: float,
    *,
    allowed_roles: set[str] | None = None,
    exclude_roles: set[str] | None = None,
    limit: int = 3,
) -> list[tuple[MixChannel, float]]:
    shares = band_shares(channels, start, end, lo, hi)
    candidates = []
    for channel in channels:
        if allowed_roles is not None and channel.role not in allowed_roles:
            continue
        if exclude_roles is not None and channel.role in exclude_roles:
            continue
        candidates.append((channel, shares.get(channel.name, 0.0)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:limit]


def lead_channels(channels: list[MixChannel]) -> list[MixChannel]:
    return [channel for channel in channels if channel.role == "lead_vocal"]


def lead_masking_state(channels: list[MixChannel], start: int, end: int) -> dict[str, float]:
    lead_energy = 0.0
    accompaniment_energy = 0.0
    lead_bright = 0.0
    total_bright = 0.0
    for channel in channels:
        energy = raw_band_energy(process_mono(channel, start, end), channel.sample_rate, 1500.0, 4000.0)
        if channel.role == "lead_vocal":
            lead_energy += energy
        else:
            accompaniment_energy += energy
    total = lead_energy + accompaniment_energy + 1e-12
    for channel in channels:
        bright = raw_band_energy(process_mono(channel, start, end), channel.sample_rate, 6000.0, 10000.0)
        total_bright += bright
        if channel.role == "lead_vocal":
            lead_bright += bright
    return {
        "lead_share_1500_4000": float(lead_energy / total),
        "accompaniment_share_1500_4000": float(accompaniment_energy / total),
        "lead_sibilance_share_6000_10000": float(
            lead_bright / (total_bright + 1e-12)
        ),
    }


def band_deviation_summary(channels: list[MixChannel], start: int, end: int) -> dict[str, float]:
    mix = analysis_mix(channels, start, end)
    levels = compensated_band_levels(mix, channels[0].sample_rate)
    return {band: round(float(levels.get(band, -120.0)), 2) for band in DISPLAY_CORRIDOR}


def equalize_lead_levels(channels: list[MixChannel], start: int, end: int):
    leads = lead_channels(channels)
    if len(leads) < 2:
        return
    levels = {channel.name: rms_db(process_mono(channel, start, end)) for channel in leads}
    target = float(np.median(list(levels.values())))
    for channel in leads:
        shortfall = target - levels[channel.name]
        if shortfall > 0.9:
            bounded_fader_move(channel, min(1.0, shortfall), "Lead layer parity from chat-only vocal anchor logic", max_step=1.0, total_limit=1.5)


def apply_low_end_anchor_rules(channels: list[MixChannel], start: int, end: int):
    summary = band_deviation_summary(channels, start, end)
    if summary["50_100"] > DISPLAY_CORRIDOR["50_100"]["max"]:
        culprits = top_culprits(channels, start, end, 50.0, 100.0, allowed_roles={"kick", "bass"}, limit=2)
        for channel, share in culprits:
            if share < 0.25:
                continue
            if channel.role == "kick":
                bounded_eq_move(channel, 78.0, -1.0, 1.0, "50-100 Hz overweight: split kick/bass roles")
            else:
                bounded_eq_move(channel, 82.0, -1.0, 1.0, "50-100 Hz overweight: split kick/bass roles")
            bounded_fader_move(channel, -0.5, "50-100 Hz overweight: keep low-end anchor controlled", max_step=0.5)
            break
    if summary["100_200"] > DISPLAY_CORRIDOR["100_200"]["max"]:
        culprits = top_culprits(channels, start, end, 100.0, 200.0, allowed_roles={"kick", "bass", "toms", "guitars"}, limit=3)
        for channel, share in culprits:
            if share < 0.18:
                continue
            bounded_eq_move(channel, 150.0, -1.0, 0.9, "100-200 Hz body excess on low-end anchor phase")
            break
    bass = next((channel for channel in channels if channel.role == "bass"), None)
    if bass is not None:
        bass_low = raw_band_energy(process_mono(bass, start, end), bass.sample_rate, 60.0, 120.0)
        bass_audibility = raw_band_energy(process_mono(bass, start, end), bass.sample_rate, 700.0, 2000.0)
        if bass_low > 0.0 and (bass_audibility / bass_low) < 0.14:
            bounded_eq_move(bass, 1200.0, 1.0, 1.0, "Bass audibility support from chat-only 700 Hz - 2 kHz rule")


def apply_lead_anchor_rules(channels: list[MixChannel], start: int, end: int):
    summary = band_deviation_summary(channels, start, end)
    equalize_lead_levels(channels, start, end)
    mask = lead_masking_state(channels, start, end)
    if mask["lead_share_1500_4000"] < 0.34:
        culprits = top_culprits(
            channels,
            start,
            end,
            1500.0,
            4000.0,
            allowed_roles={"guitars", "playback", "bgv", "snare", "hi_hat", "ride", "overheads_room"},
            limit=3,
        )
        for channel, share in culprits:
            if share < 0.1:
                continue
            freq = 2500.0 if channel.role in {"guitars", "playback", "bgv"} else 3200.0
            bounded_eq_move(channel, freq, -1.0, 1.0, "Free 1.5-4 kHz space around lead instead of master EQ")
        if mask["lead_share_1500_4000"] < 0.28:
            for channel in lead_channels(channels):
                bounded_fader_move(channel, 0.5, "Small lead support after competitor EQ in 1.5-4 kHz", max_step=0.5, total_limit=1.5)
    if summary["1000_2500"] < DISPLAY_CORRIDOR["1000_2500"]["min"]:
        for channel in lead_channels(channels):
            bounded_eq_move(channel, 2200.0, 0.8, 1.0, "Lead anchor support when 1-2.5 kHz drops below the chat-only corridor")
            bounded_fader_move(channel, 0.5, "Small lead anchor lift for underfilled 1-2.5 kHz zone", max_step=0.5, total_limit=1.5)


def apply_rhythm_anchor_rules(channels: list[MixChannel], start: int, end: int):
    summary = band_deviation_summary(channels, start, end)
    if summary["2500_5000"] > DISPLAY_CORRIDOR["2500_5000"]["max"]:
        snare = next((channel for channel in channels if channel.role == "snare"), None)
        if snare is not None:
            bounded_eq_move(snare, 3500.0, -0.8, 1.2, "2.5-5 kHz aggression after adding rhythmic attack")
    if summary["100_200"] > DISPLAY_CORRIDOR["100_200"]["max"]:
        for channel in channels:
            if channel.role == "toms":
                bounded_eq_move(channel, 180.0, -0.8, 1.0, "Tom body excess in rhythmic anchor phase")


def apply_music_layer_rules(channels: list[MixChannel], start: int, end: int):
    summary = band_deviation_summary(channels, start, end)
    if summary["200_500"] > DISPLAY_CORRIDOR["200_500"]["max"]:
        culprits = top_culprits(
            channels,
            start,
            end,
            200.0,
            500.0,
            allowed_roles={"guitars", "playback", "bgv", "overheads_room", "lead_vocal"},
            exclude_roles={"kick", "bass"},
            limit=4,
        )
        for channel, share in culprits:
            if share < 0.12:
                continue
            bounded_eq_move(channel, 320.0, -1.0, 0.9, "200-500 Hz blanket: clean the culprit stem, not master EQ")
            if channel.role in {"guitars", "playback", "bgv", "overheads_room"}:
                target_hpf = 120.0 if channel.role in {"guitars", "playback"} else 150.0
                set_hpf_if_higher(channel, target_hpf, "Secondary layer HPF from 200-500 Hz buildup rule")
    apply_lead_anchor_rules(channels, start, end)


def apply_cymbal_air_rules(channels: list[MixChannel], start: int, end: int):
    summary = band_deviation_summary(channels, start, end)
    if summary["5000_8000"] > DISPLAY_CORRIDOR["5000_8000"]["max"] or summary["8000_12000"] > DISPLAY_CORRIDOR["8000_12000"]["max"]:
        culprits = top_culprits(
            channels,
            start,
            end,
            6000.0,
            10000.0,
            allowed_roles={"hi_hat", "ride", "overheads_room", "bgv", "playback", "lead_vocal"},
            limit=5,
        )
        for channel, share in culprits:
            if share < 0.08:
                continue
            if channel.role == "hi_hat":
                bounded_eq_move(channel, 7800.0, -1.5, 1.4, "Cymbals must not dominate 6-10 kHz brightness")
                bounded_fader_move(channel, -0.5, "Reduce hi-hat dominance in shared-chat cymbal rule", max_step=0.5)
            elif channel.role == "ride":
                bounded_eq_move(channel, 6800.0, -1.0, 1.3, "Ride dominance in 6-10 kHz")
                bounded_fader_move(channel, -0.4, "Reduce ride dominance in brightness band", max_step=0.4)
            elif channel.role == "overheads_room":
                bounded_eq_move(channel, 7600.0, -1.0, 1.0, "Overheads/room should not build a 6-10 kHz plate")
                set_hpf_if_higher(channel, 180.0, "Keep overhead/room low-mid from building a 200-500 Hz blanket")
            elif channel.role in {"bgv", "lead_vocal"}:
                bounded_eq_move(channel, 7200.0, -0.6, 1.6, "Tame vocal sibilance only if it dominates the 6-10 kHz band")
            elif channel.role == "playback":
                bounded_eq_move(channel, 7000.0, -0.8, 1.1, "Playback brightness must leave room for cymbals and lead")


def apply_final_chat_only_pass(channels: list[MixChannel], start: int, end: int):
    for _ in range(2):
        summary = band_deviation_summary(channels, start, end)
        changed_before = sum(len(channel.notes) for channel in channels)
        if summary["50_100"] < DISPLAY_CORRIDOR["50_100"]["min"]:
            kick = next((channel for channel in channels if channel.role == "kick"), None)
            bass = next((channel for channel in channels if channel.role == "bass"), None)
            if kick is not None:
                bounded_eq_move(kick, 72.0, 1.0, 1.0, "50-100 Hz support via kick anchor instead of master EQ")
                bounded_fader_move(kick, 0.5, "Small kick anchor lift when the full mix underfills 50-100 Hz", max_step=0.5, total_limit=1.5)
            if bass is not None:
                bounded_fader_move(bass, 0.5, "Return some bass anchor weight when 50-100 Hz drops below corridor", max_step=0.5, total_limit=1.5)
        if summary["200_500"] > DISPLAY_CORRIDOR["200_500"]["max"]:
            culprits = top_culprits(
                channels,
                start,
                end,
                200.0,
                500.0,
                allowed_roles={"guitars", "playback", "bgv", "overheads_room", "lead_vocal", "toms"},
                exclude_roles={"kick", "bass"},
                limit=3,
            )
            for channel, share in culprits:
                if share < 0.1:
                    continue
                bounded_eq_move(channel, 320.0, -0.8, 1.0, "Final broad mud cleanup on culprit stem")
        if summary["2500_5000"] > DISPLAY_CORRIDOR["2500_5000"]["max"]:
            culprits = top_culprits(
                channels,
                start,
                end,
                2500.0,
                5000.0,
                allowed_roles={"guitars", "playback", "bgv", "snare", "hi_hat", "ride", "overheads_room"},
                limit=3,
            )
            for channel, share in culprits:
                if share < 0.08:
                    continue
                freq = 3200.0 if channel.role in {"guitars", "playback", "bgv"} else 4200.0
                bounded_eq_move(channel, freq, -0.8, 1.0, "Final 2.5-5 kHz aggression cleanup by priority")
        if summary["5000_8000"] > DISPLAY_CORRIDOR["5000_8000"]["max"] or summary["8000_12000"] > DISPLAY_CORRIDOR["8000_12000"]["max"]:
            apply_cymbal_air_rules(channels, start, end)
        apply_lead_anchor_rules(channels, start, end)
        changed_after = sum(len(channel.notes) for channel in channels)
        if changed_after == changed_before:
            break


def peak_dbfs(stereo: np.ndarray) -> float:
    peak = float(np.max(np.abs(stereo))) if len(stereo) else 0.0
    return amp_to_db(max(peak, 1e-12))


def normalize_peak(stereo: np.ndarray, target_peak_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    current_peak = peak_dbfs(stereo)
    gain_db = float(target_peak_dbfs - current_peak)
    out = (stereo * db_to_amp(gain_db)).astype(np.float32)
    return out, gain_db


def integrated_lufs(stereo: np.ndarray, sr: int) -> float | None:
    if len(stereo) < sr:
        return None
    meter = pyln.Meter(sr)
    try:
        return float(meter.integrated_loudness(stereo))
    except Exception:
        return None


def phase_log(label: str, channels: list[MixChannel], start: int, end: int) -> dict[str, Any]:
    summary = band_deviation_summary(channels, start, end)
    mask = lead_masking_state(channels, start, end)
    return {
        "phase": label,
        "band_deviation_db": summary,
        "lead_masking": {key: round(value, 3) for key, value in mask.items()},
    }


def names_for_roles(channels: list[MixChannel], roles: set[str]) -> set[str]:
    return {channel.name for channel in channels if channel.role in roles}


def export_mp3(wav_path: Path, mp3_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP3 export")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_chat_only_mix(input_dir: Path, output_path: Path, report_path: Path):
    channels, sr, target_len = load_channels(input_dir)
    ensure_ai_output_dirs()
    output_path = ensure_parent_dir(output_path)
    report_path = ensure_parent_dir(report_path)

    rough_mix = render_mix(channels, target_len)
    analysis_start = analysis_window_start(np.mean(rough_mix, axis=1), sr)
    analysis_end = min(target_len, analysis_start + int(ANALYSIS_WINDOW_SEC * sr))

    report: dict[str, Any] = {
        "source_chat_url": "https://chatgpt.com/share/69e7cc2e-2a50-832a-88f6-bf6c2164e7cb",
        "mode": "chat_only_shared_logic",
        "artifact_policy": {
            "audio_dir": str(ai_mixing_dir()),
            "logs_dir": str(ai_logs_dir()),
            "report": str(report_path),
        },
        "analysis_window_sec": ANALYSIS_WINDOW_SEC,
        "analysis_window_start_sec": round(analysis_start / sr, 2),
        "analysis_window_end_sec": round(analysis_end / sr, 2),
        "rules": [
            "master spectrum used as balance meter only",
            "+4.5 dB/oct compensated LTAS",
            "anchor order: kick+bass -> lead -> rhythmic attack -> music -> cymbals/air",
            "source/stem fixes only; no master EQ",
            "vocal space created by EQ on competitors",
            "small bounded fader/EQ moves only",
            "react to broad sustained deviations, not momentary peaks",
        ],
        "analysis_before": phase_log("before", channels, analysis_start, analysis_end),
        "phases": [],
    }

    # Phase 1: kick + bass.
    low_end_names = names_for_roles(channels, {"kick", "bass"})
    apply_low_end_anchor_rules([channel for channel in channels if channel.name in low_end_names], analysis_start, analysis_end)
    report["phases"].append(phase_log("kick_bass_anchor", channels, analysis_start, analysis_end))

    # Phase 2: add lead.
    lead_names = low_end_names | names_for_roles(channels, {"lead_vocal"})
    apply_lead_anchor_rules([channel for channel in channels if channel.name in lead_names], analysis_start, analysis_end)
    report["phases"].append(phase_log("lead_anchor", channels, analysis_start, analysis_end))

    # Phase 3: add snare/toms.
    rhythm_names = lead_names | names_for_roles(channels, {"snare", "toms"})
    apply_rhythm_anchor_rules([channel for channel in channels if channel.name in rhythm_names], analysis_start, analysis_end)
    report["phases"].append(phase_log("rhythm_anchor", channels, analysis_start, analysis_end))

    # Phase 4: add guitars/playback/BGV.
    music_names = rhythm_names | names_for_roles(channels, {"guitars", "playback", "bgv"})
    apply_music_layer_rules([channel for channel in channels if channel.name in music_names], analysis_start, analysis_end)
    report["phases"].append(phase_log("music_layer", channels, analysis_start, analysis_end))

    # Phase 5: add cymbals and ambience.
    cymbal_names = music_names | names_for_roles(channels, {"hi_hat", "ride", "overheads_room"})
    apply_cymbal_air_rules([channel for channel in channels if channel.name in cymbal_names], analysis_start, analysis_end)
    report["phases"].append(phase_log("cymbal_air_layer", channels, analysis_start, analysis_end))

    apply_final_chat_only_pass(channels, analysis_start, analysis_end)
    report["analysis_after"] = phase_log("after", channels, analysis_start, analysis_end)

    stereo = render_mix(channels, target_len)
    stereo, master_gain_db = normalize_peak(stereo, target_peak_dbfs=-1.0)

    output_wav = output_path.with_suffix(".wav")
    sf.write(output_wav, stereo, sr)
    export_mp3(output_wav, output_path)

    report["master_gain_db"] = round(master_gain_db, 2)
    report["final_peak_dbfs"] = round(peak_dbfs(stereo), 2)
    lufs_value = integrated_lufs(stereo, sr)
    report["final_lufs"] = round(lufs_value, 2) if lufs_value is not None else None
    report["channels"] = {
        channel.name: {
            "role": channel.role,
            "stems": list(channel.stems),
            "priority": channel.priority,
            "pan": channel.pan,
            "fader_db": round(channel.fader_db, 2),
            "hpf_hz": round(channel.hpf_hz, 1),
            "eq_moves": [
                {
                    "freq_hz": round(move.freq_hz, 1),
                    "gain_db": round(move.gain_db, 2),
                    "q": round(move.q, 2),
                    "reason": move.reason,
                }
                for move in channel.eq_moves
            ],
            "notes": channel.notes,
        }
        for channel in channels
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)

    return output_path, report_path, output_wav


def main():
    parser = argparse.ArgumentParser(description="Mix multitrack using only the shared chat balancing logic")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ai_mixing_path("CHAT_ONLY_SHARED_MIX.mp3"))
    parser.add_argument("--report", type=Path, default=ai_logs_path("CHAT_ONLY_SHARED_MIX_report.json"))
    args = parser.parse_args()
    ensure_ai_output_dirs()
    run_chat_only_mix(args.input_dir, args.output, args.report)


if __name__ == "__main__":
    main()
