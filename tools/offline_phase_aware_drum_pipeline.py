#!/usr/bin/env python3
"""Render Song Repa drums through the phase-aware EQ guard pipeline.

The render is fully offline. It reads local drum stems, asks
phase_aware_eq_guard for dry-run phase proposals, applies concrete
delay/polarity candidates to the offline audio, and exports raw plus processed
drum-only MP3 files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase_aware_eq_guard import (  # noqa: E402
    GroupEnableMode,
    PhaseAwareEQGuardConfig,
    PhaseChannelInput,
    analyze_phase_aware_eq_guard,
)


@dataclass(frozen=True)
class DrumStem:
    file_name: str
    channel_id: int
    label: str
    weight: float
    bleed_ratio: float
    pan: float


DRUM_STEMS: Tuple[DrumStem, ...] = (
    DrumStem("KICK.wav", 1, "KICK", 3.2, 0.18, 0.0),
    DrumStem("SNARE T.wav", 2, "SNARE T", 2.2, 0.25, 0.0),
    DrumStem("Snare B.wav", 3, "SNARE B", 1.1, 0.36, 0.0),
    DrumStem("Tom.wav", 4, "TOM", 1.2, 0.34, -0.28),
    DrumStem("F Tom.wav", 5, "F TOM", 1.2, 0.34, 0.32),
    DrumStem("OH L.wav", 6, "OH L", 1.4, 0.68, -0.86),
    DrumStem("OH R.wav", 7, "OH R", 1.4, 0.68, 0.86),
)


def db_to_amp(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def decode_pcm(data: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        decoded = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        decoded = (decoded - 128.0) / 128.0
    elif sample_width == 2:
        decoded = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(data, dtype=np.uint8)
        triples = raw.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        decoded = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        decoded = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    return decoded.reshape(-1, channels)


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    decoded = decode_pcm(frames, sample_width, channels)
    if channels > 1:
        mono = np.mean(decoded, axis=1)
    else:
        mono = decoded[:, 0]
    return np.asarray(mono, dtype=np.float32), int(sample_rate)


def write_wav_stereo(path: Path, stereo: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(stereo, -1.0, 1.0)
    encoded = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(encoded.tobytes())


def select_analysis_window(
    tracks: Mapping[str, np.ndarray],
    sample_rate: int,
    duration_sec: float,
    block_sec: float = 0.5,
) -> Tuple[int, int]:
    max_len = max(audio.size for audio in tracks.values())
    window_samples = max(1, min(int(duration_sec * sample_rate), max_len))
    block_samples = max(1, int(block_sec * sample_rate))
    block_count = int(math.ceil(max_len / block_samples))
    energy = np.zeros(block_count, dtype=np.float64)

    for audio in tracks.values():
        padded_len = block_count * block_samples
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[: audio.size] = audio
        blocks = padded.reshape(block_count, block_samples)
        energy += np.mean(np.square(blocks), axis=1)

    window_blocks = max(1, int(math.ceil(window_samples / block_samples)))
    if block_count <= window_blocks:
        return 0, max_len

    rolling = np.convolve(energy, np.ones(window_blocks), mode="valid")
    start_block = int(np.argmax(rolling))
    start = start_block * block_samples
    end = min(start + window_samples, max_len)
    return start, end


def pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.size == length:
        return audio
    padded = np.zeros(length, dtype=np.float32)
    padded[: min(length, audio.size)] = audio[:length]
    return padded


def apply_delay_correction(audio: np.ndarray, delay_ms: float, sample_rate: int) -> Tuple[np.ndarray, int]:
    samples = int(round(delay_ms * sample_rate / 1000.0))
    if samples == 0:
        return np.asarray(audio, dtype=np.float32).copy(), 0

    corrected = np.zeros_like(audio, dtype=np.float32)
    if samples > 0:
        corrected[:-samples] = audio[samples:]
    else:
        delay = abs(samples)
        corrected[delay:] = audio[:-delay]
    return corrected, samples


def apply_phase_actions(
    tracks: Mapping[str, np.ndarray],
    channel_to_name: Mapping[int, str],
    actions: Sequence[Mapping[str, object]],
    sample_rate: int,
    min_confidence: float,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    processed = {name: audio.copy() for name, audio in tracks.items()}
    applied: List[Dict[str, object]] = []

    for action in actions:
        candidate_channel = int(action["candidate_channel"])
        name = channel_to_name.get(candidate_channel)
        if not name:
            continue
        confidence = float(action.get("confidence", 0.0))
        concrete_actions = tuple(str(item) for item in action.get("actions", ()))
        if confidence < min_confidence:
            applied.append(
                {
                    "track": name,
                    "skipped": True,
                    "reason": "confidence below offline apply threshold",
                    "confidence": confidence,
                    "actions": list(concrete_actions),
                }
            )
            continue

        audio = processed[name]
        polarity_applied = False
        delay_samples = 0

        if "polarity_candidate" in concrete_actions:
            audio = -audio
            polarity_applied = True
        if "delay_candidate" in concrete_actions:
            audio, delay_samples = apply_delay_correction(audio, float(action.get("delay_ms", 0.0)), sample_rate)

        processed[name] = audio
        applied.append(
            {
                "track": name,
                "skipped": False,
                "actions": list(concrete_actions),
                "polarity_inverted": polarity_applied,
                "delay_ms": float(action.get("delay_ms", 0.0)),
                "delay_correction_samples": int(delay_samples),
                "confidence": confidence,
                "reason": str(action.get("reason", "")),
            }
        )

    return processed, applied


def pan_mono(audio: np.ndarray, pan: float) -> np.ndarray:
    safe_pan = max(-1.0, min(1.0, float(pan)))
    angle = (safe_pan + 1.0) * math.pi / 4.0
    left = math.cos(angle)
    right = math.sin(angle)
    return np.column_stack((audio * left, audio * right)).astype(np.float32)


def mix_drums(tracks: Mapping[str, np.ndarray], stems: Iterable[DrumStem]) -> np.ndarray:
    length = max(audio.size for audio in tracks.values())
    mix = np.zeros((length, 2), dtype=np.float32)
    for stem in stems:
        audio = pad_to_length(tracks[stem.label], length)
        mix += pan_mono(audio, stem.pan)
    return mix


def peak_normalize(stereo: np.ndarray, ceiling_db: float) -> Tuple[np.ndarray, float, float]:
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak <= 1e-12:
        return stereo.astype(np.float32), 0.0, peak
    target = db_to_amp(ceiling_db)
    gain = target / peak
    return (stereo * gain).astype(np.float32), float(20.0 * math.log10(gain)), peak


def encode_mp3(wav_path: Path, mp3_path: Path, encoder_path: str, bitrate: str) -> Dict[str, object]:
    encoder_name = Path(encoder_path).name.lower()
    if "lame" in encoder_name:
        bitrate_value = bitrate[:-1] if bitrate.endswith("k") else bitrate
        command = [encoder_path, "--quiet", "-b", bitrate_value, str(wav_path), str(mp3_path)]
    else:
        command = [
            encoder_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ]

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MP3 encoder timed out while encoding {mp3_path}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"MP3 encoder failed while encoding {mp3_path}: {stderr}") from exc

    return {
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def resolve_mp3_encoder(preferred: str) -> str:
    candidates = [
        preferred,
        "/opt/homebrew/bin/lame",
        "/usr/local/bin/lame",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    for candidate in candidates:
        if candidate == "ffmpeg":
            return candidate
        if candidate and Path(candidate).exists():
            return candidate
    return "ffmpeg"


def build_report(
    *,
    input_dir: Path,
    output_dir: Path,
    sample_rate: int,
    tracks: Mapping[str, np.ndarray],
    analysis_start: int,
    analysis_end: int,
    guard_result: object,
    applied_actions: Sequence[Mapping[str, object]],
    raw_gain_db: float,
    phased_gain_db: float,
    raw_peak: float,
    phased_peak: float,
    encode_logs: Sequence[Mapping[str, object]],
    raw_mp3: Path,
    phased_mp3: Path,
) -> Dict[str, object]:
    duration = max(audio.size for audio in tracks.values()) / float(sample_rate)
    return {
        "pipeline_id": "offline_phase_aware_drum_pipeline_v1",
        "source": str(input_dir),
        "output_dir": str(output_dir),
        "drum_tracks": list(tracks.keys()),
        "sample_rate": sample_rate,
        "duration_sec": duration,
        "analysis_window": {
            "start_sec": analysis_start / float(sample_rate),
            "end_sec": analysis_end / float(sample_rate),
            "duration_sec": (analysis_end - analysis_start) / float(sample_rate),
        },
        "phase_module": guard_result.to_dict(),
        "offline_actions_applied": list(applied_actions),
        "raw_render": {
            "mp3": str(raw_mp3),
            "pre_normalize_peak": raw_peak,
            "peak_normalize_gain_db": raw_gain_db,
        },
        "phase_aware_render": {
            "mp3": str(phased_mp3),
            "pre_normalize_peak": phased_peak,
            "peak_normalize_gain_db": phased_gain_db,
        },
        "encode_logs": list(encode_logs),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(Path.home() / "Desktop" / "SONG  REPA"))
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Desktop" / "SONG_REPA_DRUM_PHASE_AWARE_20260603"),
    )
    parser.add_argument("--analysis-duration-sec", type=float, default=30.0)
    parser.add_argument("--mp3-bitrate", default="320k")
    parser.add_argument("--ceiling-db", type=float, default=-1.0)
    parser.add_argument("--min-apply-confidence", type=float, default=0.0)
    parser.add_argument("--mp3-encoder", default="/opt/homebrew/bin/lame")
    parser.add_argument("--keep-wav", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks: Dict[str, np.ndarray] = {}
    sample_rate = None
    for stem in DRUM_STEMS:
        path = input_dir / stem.file_name
        if not path.exists():
            raise FileNotFoundError(f"missing drum stem: {path}")
        audio, rate = read_wav_mono(path)
        if sample_rate is None:
            sample_rate = rate
        elif sample_rate != rate:
            raise ValueError(f"sample-rate mismatch: {path} is {rate}, expected {sample_rate}")
        tracks[stem.label] = audio

    assert sample_rate is not None
    analysis_start, analysis_end = select_analysis_window(tracks, sample_rate, args.analysis_duration_sec)
    channel_inputs = [
        PhaseChannelInput(
            channel_id=stem.channel_id,
            audio=tracks[stem.label][analysis_start:analysis_end],
            group_id="DRUMS",
            weight=stem.weight,
            label=stem.label,
            bleed_ratio=stem.bleed_ratio,
        )
        for stem in DRUM_STEMS
    ]

    config = PhaseAwareEQGuardConfig(
        sample_rate=sample_rate,
        group_id="DRUMS",
        group_enable_mode=GroupEnableMode.GROUP_PHASE_LOCK,
        low_freq_focus=True,
        lookahead_buffer_ms=5.0,
        report_latency=True,
        max_delay_ms=10.0,
        delay_candidate_threshold_ms=0.35,
        coherence_ok_threshold=0.72,
        coherence_risk_threshold=0.42,
        low_freq_loss_threshold_db=1.5,
        bleed_risk_threshold=0.55,
        dry_run=True,
    )
    guard_result = analyze_phase_aware_eq_guard(channel_inputs, config)
    channel_to_name = {stem.channel_id: stem.label for stem in DRUM_STEMS}
    processed_tracks, applied_actions = apply_phase_actions(
        tracks,
        channel_to_name,
        guard_result.suggested_actions,
        sample_rate,
        args.min_apply_confidence,
    )

    raw_mix = mix_drums(tracks, DRUM_STEMS)
    phased_mix = mix_drums(processed_tracks, DRUM_STEMS)
    raw_mix, raw_gain_db, raw_peak = peak_normalize(raw_mix, args.ceiling_db)
    phased_mix, phased_gain_db, phased_peak = peak_normalize(phased_mix, args.ceiling_db)

    raw_wav = output_dir / "SONG_REPA_DRUMS_RAW.wav"
    phased_wav = output_dir / "SONG_REPA_DRUMS_PHASE_AWARE.wav"
    raw_mp3 = output_dir / "SONG_REPA_DRUMS_RAW.mp3"
    phased_mp3 = output_dir / "SONG_REPA_DRUMS_PHASE_AWARE.mp3"
    report_path = output_dir / "SONG_REPA_DRUMS_PHASE_AWARE_report.json"

    write_wav_stereo(raw_wav, raw_mix, sample_rate)
    write_wav_stereo(phased_wav, phased_mix, sample_rate)

    encoder_path = resolve_mp3_encoder(args.mp3_encoder)
    encode_logs = [
        encode_mp3(raw_wav, raw_mp3, encoder_path, args.mp3_bitrate),
        encode_mp3(phased_wav, phased_mp3, encoder_path, args.mp3_bitrate),
    ]
    if not args.keep_wav:
        raw_wav.unlink(missing_ok=True)
        phased_wav.unlink(missing_ok=True)

    report = build_report(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_rate=sample_rate,
        tracks=tracks,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        guard_result=guard_result,
        applied_actions=applied_actions,
        raw_gain_db=raw_gain_db,
        phased_gain_db=phased_gain_db,
        raw_peak=raw_peak,
        phased_peak=phased_peak,
        encode_logs=encode_logs,
        raw_mp3=raw_mp3,
        phased_mp3=phased_mp3,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "raw_mp3": str(raw_mp3),
            "phase_aware_mp3": str(phased_mp3),
            "report": str(report_path),
            "actions_applied": applied_actions,
            "analysis_window_sec": report["analysis_window"],
        },
        indent=2,
        ensure_ascii=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
