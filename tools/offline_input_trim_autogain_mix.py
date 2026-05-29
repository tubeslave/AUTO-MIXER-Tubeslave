#!/usr/bin/env python3
"""Offline multitrack render driven by the canonical input-trim autogain method.

This runner is intentionally parallel to the live SafeGain path: it imports
SignalStats.calculate_input_trim_recommendation(...) and renders the resulting
trim decisions into audio files without touching any mixer client or OSC path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from lufs_gain_staging import (  # noqa: E402
    INPUT_TRIM_AUTOGAIN_METHOD_ID,
    LocalInputGainProfile,
    LUFSMeter,
    SignalStats,
    TruePeakMeter,
)


DEFAULT_INPUT_DIR = Path.home() / "Desktop" / "SONG  REPA"
DEFAULT_OUTPUT = Path.home() / "Desktop" / "SONG_REPA_INPUT_TRIM_AUTOGAIN.wav"
DEFAULT_REPORT = Path.home() / "Desktop" / "SONG_REPA_INPUT_TRIM_AUTOGAIN_report.json"


@dataclass(frozen=True)
class TrackRole:
    preset: str
    pan: float


def db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def amp_to_db(value: float) -> float:
    return float(20.0 * math.log10(max(float(value), 1e-12)))


def mono_sum(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1).astype(np.float32)


def equal_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * (math.pi / 4.0)
    left = math.cos(theta)
    right = math.sin(theta)
    return np.column_stack((mono * left, mono * right)).astype(np.float32)


def infer_track_role(path: Path) -> TrackRole:
    name = " ".join(path.stem.lower().replace("_", " ").replace("-", " ").split())

    def side_pan(default: float = 0.0) -> float:
        tokens = name.split()
        if tokens and tokens[-1] == "l":
            return -1.0
        if tokens and tokens[-1] == "r":
            return 1.0
        if " left" in f" {name} ":
            return -1.0
        if " right" in f" {name} ":
            return 1.0
        return default

    if "kick" in name:
        return TrackRole("kick", 0.0)
    if "snare" in name or "sn t" in name or "snare t" in name or "snare b" in name:
        return TrackRole("snare", 0.0)
    if name.startswith("f tom") or "floor tom" in name:
        return TrackRole("tom", 0.28)
    if "tom" in name:
        return TrackRole("tom", -0.22)
    if name.startswith("oh l") or name == "ohl" or "overhead l" in name:
        return TrackRole("overheads", -1.0)
    if name.startswith("oh r") or name == "ohr" or "overhead r" in name:
        return TrackRole("overheads", 1.0)
    if "overhead" in name or name.startswith("oh"):
        return TrackRole("overheads", side_pan(0.0))
    if "back vox" in name or "back vocal" in name or "backing vocal" in name:
        pan = side_pan(0.0)
        if pan == 0.0:
            pan = -0.62 if " l" in f" {name} " else 0.62 if " r" in f" {name} " else 0.0
        return TrackRole("backingVocal", float(np.clip(pan, -0.75, 0.75)))
    if "vocal" in name or name == "vox":
        return TrackRole("leadVocal", 0.0)
    if "bass" in name:
        return TrackRole("bass", 0.0)
    if "guitar" in name or "gtr" in name:
        return TrackRole("electricGuitar", float(np.clip(side_pan(0.0), -0.85, 0.85)))
    if "accordion" in name:
        return TrackRole("accordion", 0.0)
    if "playback" in name:
        return TrackRole("playback", side_pan(0.0))
    if "keys" in name or "keyboard" in name or "synth" in name:
        return TrackRole("keyboard", side_pan(0.0))
    return TrackRole("custom", side_pan(0.0))


def iter_wavs(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def calibration_config(config: Dict[str, Any]) -> Dict[str, Any]:
    automation = config.get("automation")
    if not isinstance(automation, dict):
        return {}
    safe_gain = automation.get("safe_gain_calibration")
    return safe_gain if isinstance(safe_gain, dict) else {}


def preset_metrics(cal_config: Dict[str, Any], preset: str) -> Dict[str, float]:
    targets = cal_config.get("preset_targets")
    if not isinstance(targets, dict):
        targets = {}
    raw = targets.get(preset) or targets.get("custom") or {}
    return {
        "target_lufs": float(raw.get("target_lufs", cal_config.get("target_lufs", -18.0))),
        "max_peak_limit": float(raw.get("max_peak_limit", cal_config.get("max_peak_limit", -3.0))),
        "target_peak_db": float(raw.get("target_peak_db", cal_config.get("target_peak_db", -10.0))),
    }


def spectral_bands_for_block(block: np.ndarray, sr: int) -> Dict[str, float]:
    if len(block) < 32:
        return {}
    x = np.asarray(block, dtype=np.float32)
    window = np.hanning(len(x)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(x * window)).astype(np.float64)
    freqs = np.fft.rfftfreq(len(x), 1.0 / float(sr))
    bands = {
        "low": (40.0, 120.0),
        "fundamental": (80.0, 240.0),
        "low_mid_body": (180.0, 500.0),
        "mid": (500.0, 2000.0),
        "presence": (2000.0, 5000.0),
        "high": (5000.0, 10000.0),
        "air": (10000.0, min(20000.0, sr / 2.0)),
    }
    result: Dict[str, float] = {}
    norm = max(float(len(x)) / 2.0, 1.0)
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band_rms = float(np.sqrt(np.mean(np.square(spectrum[mask] / norm))) + 1e-12)
        result[name] = amp_to_db(band_rms)
    return result


def collect_stats(
    mono: np.ndarray,
    sr: int,
    channel_id: int,
    *,
    noise_gate_threshold_db: float,
    chunk_sec: float,
) -> SignalStats:
    stats = SignalStats(
        channel_id=channel_id,
        noise_gate_threshold_db=float(noise_gate_threshold_db),
    )
    lufs_meter = LUFSMeter(sample_rate=sr, window_ms=400.0)
    peak_meter = TruePeakMeter(sample_rate=sr)
    chunk_len = max(1, int(round(float(sr) * max(0.05, float(chunk_sec)))))
    for start in range(0, len(mono), chunk_len):
        block = mono[start : start + chunk_len].astype(np.float32, copy=False)
        if len(block) == 0:
            continue
        lufs = lufs_meter.process(block)
        true_peak = peak_meter.process(block)
        rms = float(np.sqrt(np.mean(np.square(block))) + 1e-12)
        stats.update_sample(
            true_peak,
            lufs,
            amp_to_db(rms),
            spectral_bands_db=spectral_bands_for_block(block, sr),
        )
    return stats


def calculate_autogain_report(
    mono: np.ndarray,
    sr: int,
    channel_id: int,
    preset: str,
    cal_config: Dict[str, Any],
    preamp_profile: LocalInputGainProfile,
    *,
    mode: str,
    current_trim_db: float,
    previous_report: Optional[Dict[str, Any]],
    analysis_timestamp_sec: float,
    chunk_sec: float,
) -> Dict[str, Any]:
    stats = collect_stats(
        mono,
        sr,
        channel_id,
        noise_gate_threshold_db=float(cal_config.get("noise_gate_threshold", -40.0)),
        chunk_sec=chunk_sec,
    )
    targets = preset_metrics(cal_config, preset)
    stats.calculate_input_trim_recommendation(
        current_trim_db=current_trim_db,
        target_peak_db=targets["target_peak_db"],
        max_peak_limit=targets["max_peak_limit"],
        target_lufs=targets["target_lufs"],
        min_signal_presence=float(cal_config.get("min_signal_presence", 0.05)),
        preamp_profile=preamp_profile,
        mode=mode,
        mixer_channel=channel_id,
        preset=preset,
        previous_recommendation=previous_report,
        analysis_timestamp_sec=float(analysis_timestamp_sec),
        analysis_window_sec=max(float(len(mono)) / float(sr), float(chunk_sec)),
    )
    report = stats.get_report()
    report["preset"] = preset
    report["preset_targets"] = targets
    return report


def iterative_offline_report(
    mono: np.ndarray,
    sr: int,
    channel_id: int,
    preset: str,
    cal_config: Dict[str, Any],
    preamp_profile: LocalInputGainProfile,
    *,
    mode: str,
    analysis_passes: int,
    chunk_sec: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    previous: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    cumulative_gain_db = 0.0
    working_mono = mono.astype(np.float32, copy=False)
    for pass_index in range(max(1, int(analysis_passes))):
        report = calculate_autogain_report(
            working_mono,
            sr,
            channel_id,
            preset,
            cal_config,
            preamp_profile,
            mode=mode,
            current_trim_db=cumulative_gain_db,
            previous_report=previous,
            analysis_timestamp_sec=float(pass_index),
            chunk_sec=chunk_sec,
        )
        report["offline_analysis_pass"] = pass_index + 1
        report["offline_cumulative_gain_before_db"] = round(cumulative_gain_db, 3)
        history.append(report)
        previous = report
        step_gain_db = float(report.get("suggested_gain_db", 0.0) or 0.0)
        if abs(step_gain_db) < 0.25:
            wait_reasons = {
                "gain_up_stability_wait",
                "behavior_change_stability_wait",
                "source_identity_change_wait",
                "channel_profile_locked_review",
                "channel_profile_stability_wait",
                "decision_confidence_gain_up_hold",
                "decision_confidence_stability_wait",
                "anti_oscillation_hold",
            }
            if str(report.get("limited_by")) not in wait_reasons:
                break
            continue
        cumulative_gain_db += step_gain_db
        working_mono = (mono * db_to_amp(cumulative_gain_db)).astype(np.float32)

    final_report = dict(history[-1])
    final_report["offline_cumulative_gain_db"] = round(cumulative_gain_db, 3)
    return final_report, history, cumulative_gain_db


def pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) == target_len:
        return audio.astype(np.float32, copy=False)
    if len(audio) > target_len:
        return audio[:target_len].astype(np.float32, copy=False)
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: len(audio)] = audio.astype(np.float32, copy=False)
    return padded


def render_track(audio: np.ndarray, pan: float, gain_db: float, target_len: int) -> np.ndarray:
    gain = db_to_amp(gain_db)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        mono = pad_or_trim(arr, target_len)
        return (equal_power_pan(mono, pan) * gain).astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        if len(arr) > target_len:
            stereo = arr[:target_len]
        elif len(arr) < target_len:
            stereo = np.zeros((target_len, 2), dtype=np.float32)
            stereo[: len(arr), :] = arr
        else:
            stereo = arr
        return (stereo.astype(np.float32, copy=False) * gain).astype(np.float32)
    mono = pad_or_trim(mono_sum(arr), target_len)
    return (equal_power_pan(mono, pan) * gain).astype(np.float32)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def run_offline_input_trim_autogain(
    input_dir: Path,
    output_path: Path,
    report_path: Path,
    *,
    config_path: Path,
    mode: str = "soundcheck",
    analysis_passes: int = 4,
    chunk_sec: float = 0.4,
    master_ceiling_dbfs: float = -1.0,
    min_signal_presence: Optional[float] = None,
) -> Dict[str, Any]:
    wavs = iter_wavs(input_dir)
    if not wavs:
        raise RuntimeError(f"No WAV files found in {input_dir}")

    config = load_config(config_path)
    cal_config = dict(calibration_config(config))
    if min_signal_presence is not None:
        cal_config["min_signal_presence"] = float(min_signal_presence)
    preamp_profile = LocalInputGainProfile.from_config(cal_config.get("local_input_preamp", {}))

    infos = [sf.info(str(path)) for path in wavs]
    sample_rates = {int(info.samplerate) for info in infos}
    if len(sample_rates) != 1:
        raise RuntimeError(f"Sample-rate mismatch: {sorted(sample_rates)}")
    sr = sample_rates.pop()
    target_len = max(int(info.frames) for info in infos)
    mix = np.zeros((target_len, 2), dtype=np.float32)
    tracks: List[Dict[str, Any]] = []

    for channel_id, path in enumerate(wavs, start=1):
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if int(file_sr) != sr:
            raise RuntimeError(f"{path.name}: sample-rate mismatch {file_sr} != {sr}")
        mono = mono_sum(audio)
        role = infer_track_role(path)
        report, pass_history, cumulative_gain_db = iterative_offline_report(
            mono,
            sr,
            channel_id,
            role.preset,
            cal_config,
            preamp_profile,
            mode=mode,
            analysis_passes=analysis_passes,
            chunk_sec=chunk_sec,
        )
        applied_gain_db = float(cumulative_gain_db)
        mix += render_track(audio, role.pan, applied_gain_db, target_len)
        tracks.append({
            "channel": channel_id,
            "file": path.name,
            "preset": role.preset,
            "pan": role.pan,
            "applied_gain_db": round(applied_gain_db, 3),
            "method_report": report,
            "analysis_passes": pass_history,
        })

    pre_master_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    pre_master_peak_db = amp_to_db(pre_master_peak)
    master_trim_db = min(0.0, float(master_ceiling_dbfs) - pre_master_peak_db)
    if master_trim_db < 0.0:
        mix = (mix * db_to_amp(master_trim_db)).astype(np.float32)
    final_peak_db = amp_to_db(float(np.max(np.abs(mix))) if mix.size else 0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix, sr, subtype="PCM_24")

    summary = {
        "type": "offline_input_trim_autogain_render",
        "method_id": INPUT_TRIM_AUTOGAIN_METHOD_ID,
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "config_path": str(config_path),
        "mode": mode,
        "sample_rate": sr,
        "duration_sec": round(float(target_len) / float(sr), 3),
        "track_count": len(tracks),
        "analysis_passes": max(1, int(analysis_passes)),
        "chunk_sec": float(chunk_sec),
        "master_ceiling_dbfs": float(master_ceiling_dbfs),
        "min_signal_presence": float(cal_config.get("min_signal_presence", 0.05)),
        "pre_master_peak_dbfs": round(pre_master_peak_db, 3),
        "master_safety_trim_db": round(master_trim_db, 3),
        "final_peak_dbfs": round(final_peak_db, 3),
        "tracks": tracks,
    }
    report_path.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default_config.json")
    parser.add_argument("--mode", choices=("soundcheck", "live"), default="soundcheck")
    parser.add_argument("--analysis-passes", type=int, default=4)
    parser.add_argument("--chunk-sec", type=float, default=0.4)
    parser.add_argument("--master-ceiling-dbfs", type=float, default=-1.0)
    parser.add_argument(
        "--min-signal-presence",
        type=float,
        default=None,
        help="Optional offline override for sparse isolated stems; default uses config.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_offline_input_trim_autogain(
        args.input_dir.expanduser(),
        args.output.expanduser(),
        args.report.expanduser(),
        config_path=args.config.expanduser(),
        mode=args.mode,
        analysis_passes=args.analysis_passes,
        chunk_sec=args.chunk_sec,
        master_ceiling_dbfs=args.master_ceiling_dbfs,
        min_signal_presence=args.min_signal_presence,
    )
    print(json.dumps({
        "output_path": summary["output_path"],
        "report_path": summary["report_path"],
        "track_count": summary["track_count"],
        "final_peak_dbfs": summary["final_peak_dbfs"],
        "master_safety_trim_db": summary["master_safety_trim_db"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
