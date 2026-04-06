#!/usr/bin/env python3
"""
Sum multitrack WAV stems with gain derived from integrated LUFS targets only
(no EQ, compression, or other processing). Mono stems panned L/C/R as noted.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# Target integrated LUFS per file (relative balance only; genre-agnostic defaults).
TARGET_LUFS: Dict[str, float] = {
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


def load_mono(path: str) -> Tuple[np.ndarray, int]:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    if data.shape[1] != 1:
        data = np.mean(data, axis=1, keepdims=True)
    return data, rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mix_dir",
        nargs="?",
        default=".",
        help="Folder containing WAV stems",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output stereo WAV path",
    )
    args = parser.parse_args()
    mix_dir = os.path.abspath(args.mix_dir)
    wavs = sorted(f for f in os.listdir(mix_dir) if f.lower().endswith(".wav"))
    missing = [f for f in TARGET_LUFS if f not in wavs]
    if missing:
        print("Missing expected stems:", missing, file=sys.stderr)
        return 1

    rate = None
    n_samples = None
    meter = None
    gains_db: List[Tuple[str, float, float, float]] = []

    for name in wavs:
        if name not in TARGET_LUFS:
            print(f"Skip (no target): {name}", file=sys.stderr)
            continue
        path = os.path.join(mix_dir, name)
        data, r = load_mono(path)
        if rate is None:
            rate = r
            meter = pyln.Meter(rate)
            n_samples = data.shape[0]
        elif r != rate or data.shape[0] != n_samples:
            print(f"Length/rate mismatch: {name}", file=sys.stderr)
            return 1

        measured = float(meter.integrated_loudness(data))
        target = TARGET_LUFS[name]
        delta_db = target - measured
        gains_db.append((name, measured, target, delta_db))

    out = np.zeros((n_samples, 2), dtype=np.float64)

    for name, _, _, delta_db in gains_db:
        path = os.path.join(mix_dir, name)
        data, _ = load_mono(path)
        g = float(10.0 ** (delta_db / 20.0))
        x = data[:, 0] * g
        if name in CENTER:
            out[:, 0] += x
            out[:, 1] += x
        elif name in LEFT_ONLY:
            out[:, 0] += x
        elif name in RIGHT_ONLY:
            out[:, 1] += x
        else:
            print(f"Pan rule missing: {name}", file=sys.stderr)
            return 1

    peak = float(np.max(np.abs(out)))
    peak_target = 10.0 ** (-1.0 / 20.0)
    if peak > 0:
        out *= peak_target / peak

    sf.write(
        args.output,
        out.astype(np.float32),
        rate,
        subtype="PCM_24",
    )

    report_path = os.path.splitext(args.output)[0] + "_gains.txt"
    lines = [
        "Level-balance mix (integrated LUFS targets + pan, peak -1 dBFS).",
        f"Sample rate: {rate} Hz, samples: {n_samples}",
        "stem | measured_I | target_I | delta_dB",
    ]
    for name, m, t, d in gains_db:
        lines.append(f"{name} | {m:+.2f} | {t:+.2f} | {d:+.2f}")
    lines.append(f"normalize_peak_before_dBFS: {20*np.log10(peak):.2f}")
    lines.append(f"normalize_scale: {peak_target/peak:.6f}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:", args.output)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
