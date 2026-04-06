#!/usr/bin/env python3
"""
Offline multitrack mix: gain staging → HPF → RBJ EQ → Pedalboard Compressor → LUFS balance
→ sum (pan) → AutoMaster (bus glue, LUFS norm, true-peak limiter).

Uses pedalboard (JUCE) dynamics, backend/auto_mastering.AutoMaster,
docs/CONVENTIONS (K-weight/TP via auto_mastering), mixing_rules-inspired chains.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Project backend on path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

from auto_mastering import AutoMaster, measure_lufs, measure_true_peak  # noqa: E402

import pyloudnorm as pyln  # noqa: E402
import soundfile as sf  # noqa: E402
from pedalboard import Compressor, Pedalboard  # noqa: E402
from scipy.signal import butter, lfilter, sosfilt  # noqa: E402

logger = logging.getLogger("full_mix_stems")

# --- RBJ biquads (Audio EQ Cookbook), peaking EQ ---
def _rbj_peak(w0: float, gain_db: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Peaking EQ. w0 = 2*pi*f/fs."""
    a = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * max(0.1, q))
    b0 = 1.0 + alpha * a
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_biquad_chain(x: np.ndarray, sr: int, bands: Sequence[Tuple[str, Any]]) -> np.ndarray:
    """bands: ('peak', f_hz, gain_db, q) only (RBJ peaking EQ)."""
    y = np.asarray(x, dtype=np.float64)
    for band in bands:
        kind = band[0]
        f0 = float(band[1])
        g = float(band[2])
        q = float(band[3])
        w0 = 2.0 * np.pi * f0 / sr
        if kind != "peak":
            raise ValueError("Only peak EQ supported in chain")
        b, a = _rbj_peak(w0, g, q)
        y = lfilter(b, a, y)
    return y


def apply_highpass(x: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    if fc <= 0:
        return x
    sos = butter(order, fc, btype="high", fs=sr, output="sos")
    return sosfilt(sos, np.asarray(x, dtype=np.float64))


@dataclass
class StemPreset:
    hpf_hz: float
    eq_bands: List[Tuple[str, float, float, float]] = field(default_factory=list)
    comp: Dict[str, float] = field(default_factory=dict)
    target_lufs: float = -20.0


# Target LUFS after processing (balance) — aligned with level_balance_stems.py
TARGET_BY_NAME: Dict[str, float] = {
    "Vox1.wav": -17.0,
    "Vox2.wav": -17.5,
    "Vox3.wav": -17.5,
    "Kick.wav": -19.5,
    "Snare.wav": -20.5,
    "Floor Tom.wav": -26.0,
    "Mid Tom.wav": -26.0,
    "Hi-Hat.wav": -22.0,
    "Ride.wav": -26.0,
    "Overhead L.wav": -23.5,
    "Overhead R.wav": -23.5,
    "Drum Room L.wav": -25.0,
    "Drum Room R.wav": -25.0,
    "Guitar L.wav": -21.0,
    "Guitar R.wav": -21.0,
    "Accordion.wav": -18.5,
    "Playback L.wav": -22.5,
    "Playback R.wav": -22.5,
}

CENTER = {
    "Kick.wav",
    "Snare.wav",
    "Floor Tom.wav",
    "Mid Tom.wav",
    "Hi-Hat.wav",
    "Ride.wav",
    "Accordion.wav",
    "Vox1.wav",
    "Vox2.wav",
    "Vox3.wav",
}
LEFT_ONLY = {
    "Drum Room L.wav",
    "Overhead L.wav",
    "Guitar L.wav",
    "Playback L.wav",
}
RIGHT_ONLY = {
    "Drum Room R.wav",
    "Overhead R.wav",
    "Guitar R.wav",
    "Playback R.wav",
}


def stem_preset(name: str) -> StemPreset:
    """mixing_rules-inspired chains (subset)."""
    t = TARGET_BY_NAME[name]
    if name.startswith("Vox"):
        return StemPreset(
            hpf_hz=80.0,
            eq_bands=[
                ("peak", 250.0, -2.5, 1.4),
                ("peak", 3000.0, 2.0, 1.2),
                ("peak", 12000.0, 1.5, 0.75),
            ],
            comp={
                "threshold_db": -20.0,
                "ratio": 3.5,
                "attack_ms": 10.0,
                "release_ms": 110.0,
                "knee_db": 6.0,
                "makeup_db": 1.5,
            },
            target_lufs=t,
        )
    if name == "Kick.wav":
        return StemPreset(
            hpf_hz=35.0,
            eq_bands=[
                ("peak", 62.0, 3.0, 1.0),
                ("peak", 350.0, -3.5, 1.2),
                ("peak", 4000.0, 2.5, 1.5),
            ],
            comp={
                "threshold_db": -16.0,
                "ratio": 4.5,
                "attack_ms": 22.0,
                "release_ms": 62.0,
                "knee_db": 3.0,
                "makeup_db": 1.0,
            },
            target_lufs=t,
        )
    if name == "Snare.wav":
        return StemPreset(
            hpf_hz=80.0,
            eq_bands=[
                ("peak", 200.0, 2.0, 1.2),
                ("peak", 750.0, -2.0, 1.5),
                ("peak", 5200.0, 2.5, 1.3),
            ],
            comp={
                "threshold_db": -14.0,
                "ratio": 3.2,
                "attack_ms": 8.0,
                "release_ms": 100.0,
                "knee_db": 4.0,
                "makeup_db": 1.0,
            },
            target_lufs=t,
        )
    if "Tom" in name:
        return StemPreset(
            hpf_hz=70.0,
            eq_bands=[
                ("peak", 160.0, 2.0, 1.1),
                ("peak", 420.0, -3.0, 1.3),
                ("peak", 4000.0, 2.0, 1.4),
            ],
            comp={
                "threshold_db": -22.0,
                "ratio": 3.0,
                "attack_ms": 14.0,
                "release_ms": 85.0,
                "knee_db": 4.0,
                "makeup_db": 2.0,
            },
            target_lufs=t,
        )
    if name == "Hi-Hat.wav":
        return StemPreset(
            hpf_hz=200.0,
            eq_bands=[("peak", 10000.0, 1.5, 0.9)],
            comp={
                "threshold_db": -18.0,
                "ratio": 2.0,
                "attack_ms": 5.0,
                "release_ms": 80.0,
                "knee_db": 4.0,
                "makeup_db": 0.5,
            },
            target_lufs=t,
        )
    if name == "Ride.wav":
        return StemPreset(
            hpf_hz=200.0,
            eq_bands=[("peak", 8000.0, 1.0, 0.9)],
            comp={
                "threshold_db": -24.0,
                "ratio": 2.0,
                "attack_ms": 8.0,
                "release_ms": 120.0,
                "knee_db": 4.0,
                "makeup_db": 2.0,
            },
            target_lufs=t,
        )
    if name.startswith("Overhead"):
        return StemPreset(
            hpf_hz=150.0,
            eq_bands=[
                ("peak", 400.0, -2.0, 1.2),
                ("peak", 10000.0, 1.5, 0.85),
            ],
            comp={
                "threshold_db": -18.0,
                "ratio": 2.0,
                "attack_ms": 22.0,
                "release_ms": 150.0,
                "knee_db": 4.0,
                "makeup_db": 0.5,
            },
            target_lufs=t,
        )
    if name.startswith("Drum Room"):
        return StemPreset(
            hpf_hz=100.0,
            eq_bands=[("peak", 5000.0, 1.0, 0.8)],
            comp={
                "threshold_db": -20.0,
                "ratio": 2.0,
                "attack_ms": 25.0,
                "release_ms": 200.0,
                "knee_db": 6.0,
                "makeup_db": 0.0,
            },
            target_lufs=t,
        )
    if name.startswith("Guitar"):
        return StemPreset(
            hpf_hz=80.0,
            eq_bands=[
                ("peak", 280.0, -2.0, 1.3),
                ("peak", 2600.0, 1.8, 1.1),
            ],
            comp={
                "threshold_db": -16.0,
                "ratio": 2.2,
                "attack_ms": 22.0,
                "release_ms": 140.0,
                "knee_db": 5.0,
                "makeup_db": 0.5,
            },
            target_lufs=t,
        )
    if name == "Accordion.wav":
        return StemPreset(
            hpf_hz=80.0,
            eq_bands=[
                ("peak", 1200.0, 1.0, 1.0),
                ("peak", 3500.0, 1.2, 1.0),
            ],
            comp={
                "threshold_db": -17.0,
                "ratio": 2.8,
                "attack_ms": 15.0,
                "release_ms": 120.0,
                "knee_db": 5.0,
                "makeup_db": 0.5,
            },
            target_lufs=t,
        )
    if name.startswith("Playback"):
        return StemPreset(
            hpf_hz=40.0,
            eq_bands=[("peak", 350.0, -1.0, 1.0)],
            comp={
                "threshold_db": -15.0,
                "ratio": 2.0,
                "attack_ms": 30.0,
                "release_ms": 180.0,
                "knee_db": 6.0,
                "makeup_db": 0.0,
            },
            target_lufs=t,
        )
    raise KeyError(name)


def gain_stage_trim(x: np.ndarray, peak_dbfs: float) -> np.ndarray:
    """Digital trim so peak = peak_dbfs (mixing_rules −18…−12 corridor, use −14 nominal)."""
    peak = float(np.max(np.abs(x))) + 1e-12
    target_lin = 10.0 ** (peak_dbfs / 20.0)
    return x * (target_lin / peak)


def process_stem_mono(x: np.ndarray, sr: int, name: str) -> Tuple[np.ndarray, StemPreset]:
    preset = stem_preset(name)
    y = gain_stage_trim(x.astype(np.float64), -14.0)
    y = apply_highpass(y, sr, preset.hpf_hz, order=4 if preset.hpf_hz >= 50 else 2)
    y = apply_biquad_chain(y, sr, preset.eq_bands)
    mk = float(preset.comp.get("makeup_db", 0.0))
    comp_pb = Pedalboard(
        [
            Compressor(
                threshold_db=preset.comp["threshold_db"],
                ratio=preset.comp["ratio"],
                attack_ms=preset.comp["attack_ms"],
                release_ms=preset.comp["release_ms"],
            )
        ]
    )
    y = comp_pb(y.astype(np.float32), sr).astype(np.float64) * (10.0 ** (mk / 20.0))
    # Loudness balance to target integrated LUFS (pyloudnorm)
    meter = pyln.Meter(sr)
    cur = meter.integrated_loudness(y.reshape(-1, 1))
    tgt = preset.target_lufs
    if cur > -70.0:
        y = pyln.normalize.loudness(y.reshape(-1, 1), cur, tgt).flatten()
    return y.astype(np.float32), preset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mix_dir",
        nargs="?",
        default="/Users/dmitrijvolkov/Desktop/MIX",
        help="Folder with WAV stems",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/Users/dmitrijvolkov/Desktop/MIX_FullPipeline.wav",
        help="Output mastered stereo WAV",
    )
    parser.add_argument(
        "--target-master-lufs",
        type=float,
        default=-13.0,
        help="Integrated LUFS target after AutoMaster",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    mix_dir = os.path.abspath(args.mix_dir)
    names = sorted(f for f in os.listdir(mix_dir) if f.lower().endswith(".wav"))
    for n in TARGET_BY_NAME:
        if n not in names:
            print(f"Missing stem: {n}", file=sys.stderr)
            return 1

    first_path = os.path.join(mix_dir, names[0])
    _, sr = sf.read(first_path)
    n_samples = None
    out = None

    report_lines: List[str] = [
        "Full pipeline: trim -14 dBFS peak → HPF → RBJ EQ → Pedalboard compressor → LUFS stem balance → sum → AutoMaster",
        f"Master target LUFS={args.target_master_lufs}, TP limit -1.0 dBTP",
        "",
    ]

    for name in sorted(TARGET_BY_NAME.keys()):
        path = os.path.join(mix_dir, name)
        x, r = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if r != sr:
            print("Sample rate mismatch", name, file=sys.stderr)
            return 1
        if n_samples is None:
            n_samples = len(x)
            out = np.zeros((n_samples, 2), dtype=np.float32)
        elif len(x) != n_samples:
            print("Length mismatch", name, file=sys.stderr)
            return 1

        y, preset = process_stem_mono(x, sr, name)
        m = pyln.Meter(sr).integrated_loudness(y.reshape(-1, 1))
        report_lines.append(f"{name}: post-chain I={m:+.2f} LUFS (target {preset.target_lufs:+.1f})")

        if name in CENTER:
            out[:, 0] += y
            out[:, 1] += y
        elif name in LEFT_ONLY:
            out[:, 0] += y
        elif name in RIGHT_ONLY:
            out[:, 1] += y
        else:
            print("Pan?", name, file=sys.stderr)
            return 1

    pre_lufs = measure_lufs(out, sr)
    pre_tp = measure_true_peak(out, sr)
    report_lines.extend(
        [
            "",
            f"Pre-master: LUFS={pre_lufs:.2f}, TruePeak={pre_tp:.2f} dBTP",
        ]
    )

    master = AutoMaster(
        sample_rate=sr,
        target_lufs=args.target_master_lufs,
        true_peak_limit=-1.0,
    )
    result = master.master(out.astype(np.float32))
    if not result.success:
        print("Mastering failed:", result.error, file=sys.stderr)
        return 1

    mastered = result.audio.astype(np.float32)
    if mastered.ndim == 1:
        mastered = np.column_stack([mastered, mastered])

    out_path = os.path.abspath(args.output)
    sf.write(out_path, mastered, sr, subtype="PCM_24")

    post_lufs = measure_lufs(mastered, sr)
    post_tp = measure_true_peak(mastered, sr)
    report_lines.extend(
        [
            f"Post-master: LUFS={post_lufs:.2f}, TruePeak={post_tp:.2f} dBTP",
            f"Limiter reduction (reported): {result.limiter_reduction_db:.2f} dB",
        ]
    )

    rep_path = os.path.splitext(out_path)[0] + "_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("Wrote:", out_path)
    print("Report:", rep_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
