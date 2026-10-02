#!/usr/bin/env python3
"""
Mix one MUSDB18-7 track using VirtualWingMixer gain staging + simple pan law.

Dependencies (install if missing)::
    pip install musdb numpy soundfile pyloudnorm prometheus-client

On first run, ``musdb.DB(download=True)`` fetches the ~140 MB MUSDB18-7 sample
to ``~/MUSDB18/MUSDB18-7`` (7 s clips, stems: drums, bass, vocals, other).

Outputs WAV, Prometheus textfile metrics, and a short loudness report.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python virtual_mixer/mix_musdb7_demo.py` (repo root must be on path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Prepend repo root so `virtual_mixer` resolves as a package, not `virtual_mixer.py`
# when the interpreter adds this script's directory to sys.path[0].
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf
from prometheus_client import CollectorRegistry, Counter, write_to_textfile

from virtual_mixer.virtual_mixer import Channel, VirtualWingMixer

try:
    import pyloudnorm as pyln
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install pyloudnorm: pip install pyloudnorm") from exc

try:
    import musdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install musdb: pip install musdb") from exc


def _pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan for stereo stem."""
    p = float(np.clip(pan, -1.0, 1.0))
    ang = (p + 1.0) * (np.pi / 4.0)
    return float(np.cos(ang)), float(np.sin(ang))


def _stem_gain_db(vm: VirtualWingMixer, ch: int) -> float:
    c = vm.input_channels[ch]
    fd = Channel.fader_to_db(c.fader)
    if not np.isfinite(fd):
        fd = -120.0
    return float(fd + c.gain_db)


def _apply_stem(
    vm: VirtualWingMixer, stereo: np.ndarray, ch: int
) -> np.ndarray:
    g_db = _stem_gain_db(vm, ch)
    g = 10.0 ** (g_db / 20.0)
    gl, gr = _pan_gains(vm.input_channels[ch].pan)
    out = np.zeros_like(stereo)
    out[:, 0] = stereo[:, 0] * g * gl
    out[:, 1] = stereo[:, 1] * g * gr
    return out


def _default_scene(vm: VirtualWingMixer) -> None:
    """Simple rock-ish balance on channels 1–4."""
    vm.input_channels[1].fader = 0.82
    vm.input_channels[1].pan = -0.05
    vm.input_channels[1].gain_db = 0.0

    vm.input_channels[2].fader = 0.78
    vm.input_channels[2].pan = 0.1
    vm.input_channels[2].gain_db = 1.5

    vm.input_channels[3].fader = 0.85
    vm.input_channels[3].pan = 0.0
    vm.input_channels[3].gain_db = 2.0

    vm.input_channels[4].fader = 0.72
    vm.input_channels[4].pan = 0.15
    vm.input_channels[4].gain_db = -1.0

    vm.master.fader = 0.76
    vm.master.on = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/virtual_mix_demo"),
        help="Output directory for WAV, metrics, and report",
    )
    parser.add_argument(
        "--track-index",
        type=int,
        default=0,
        help="Index in MUSDB18-7 (default: first track)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = os.path.expanduser("~/MUSDB18/MUSDB18-7")
    if not os.path.isdir(root):
        musdb.DB(download=True)

    mus = musdb.DB(root=root)
    track = mus[args.track_index]

    stems = [
        (1, track.sources["drums"].audio.astype(np.float64)),
        (2, track.sources["bass"].audio.astype(np.float64)),
        (3, track.sources["vocals"].audio.astype(np.float64)),
        (4, track.sources["other"].audio.astype(np.float64)),
    ]

    vm = VirtualWingMixer()
    _default_scene(vm)

    mix = np.zeros_like(stems[0][1])
    for ch, audio in stems:
        mix = mix + _apply_stem(vm, audio, ch)

    m_db = Channel.fader_to_db(vm.master.fader)
    if not np.isfinite(m_db):
        m_db = -120.0
    mix = mix * (10.0 ** (m_db / 20.0))

    peak = float(np.max(np.abs(mix)))
    target_peak = 10.0 ** (-1.0 / 20.0)
    if peak > target_peak:
        mix = mix * (target_peak / peak)

    sr = 44100
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(mix))

    reg = CollectorRegistry()
    Counter(
        "virtual_mixer_musdb7_mixes_total",
        "MUSDB18-7 demo mixes rendered",
        registry=reg,
    ).inc()
    metrics_path = args.out_dir / "metrics.prom"
    write_to_textfile(str(metrics_path), reg)

    safe_name = track.name.replace(" ", "_").replace("/", "-")
    wav_path = args.out_dir / f"{safe_name}_virtual_mix.wav"
    sf.write(str(wav_path), mix, sr, subtype="PCM_24")

    report_path = args.out_dir / "mix_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"track: {track.name}\n")
        f.write(f"integrated_lufs: {lufs:.3f}\n")
        f.write(f"sample_peak_after_limit: {float(np.max(np.abs(mix))):.6f}\n")

    print(wav_path)
    print(f"integrated_lufs={lufs:.3f}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
