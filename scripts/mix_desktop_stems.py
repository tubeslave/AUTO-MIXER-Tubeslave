#!/usr/bin/env python3
"""
Mix mono live stems from Desktop/MIX: pan L/R pairs, rough fader balance,
then EBU R128-style loudnorm (true peak aware).

Usage:
  python scripts/mix_desktop_stems.py
  python scripts/mix_desktop_stems.py --mix-dir /path/to/MIX
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# (filename, pan c|l|r, volume_adjust_db)
# Pan: center, hard left, hard right (mono -> stereo).
# Levels: conservative live-style; vocals slightly forward.
DEFAULT_STEMS: list[tuple[str, str, float]] = [
    ("Kick.wav", "c", 0.0),
    ("Snare.wav", "c", 0.0),
    ("Hi-Hat.wav", "c", -1.5),
    ("Ride.wav", "c", -1.0),
    ("Floor Tom.wav", "c", -0.5),
    ("Mid Tom.wav", "c", -0.5),
    ("Overhead L.wav", "l", -2.0),
    ("Overhead R.wav", "r", -2.0),
    ("Drum Room L.wav", "l", -4.5),
    ("Drum Room R.wav", "r", -4.5),
    ("Guitar L.wav", "l", -2.5),
    ("Guitar R.wav", "r", -2.5),
    ("Accordion.wav", "c", -2.0),
    ("Playback L.wav", "l", -3.5),
    ("Playback R.wav", "r", -3.5),
    ("Vox1.wav", "c", 1.5),
    ("Vox2.wav", "c", 0.5),
    ("Vox3.wav", "c", 0.5),
]


def _pan_expr(pan: str) -> str:
    if pan == "c":
        return "c0=c0|c1=c0"
    if pan == "l":
        return "c0=1.0*c0|c1=0.0*c0"
    if pan == "r":
        return "c0=0.0*c0|c1=1.0*c0"
    raise ValueError(pan)


def build_filter_complex(stems: list[tuple[str, str, float]]) -> str:
    chains: list[str] = []
    for i, (_, pan, vol_db) in enumerate(stems):
        p = _pan_expr(pan)
        vol = f"{vol_db:.2f}dB"
        chains.append(f"[{i}:a]pan=stereo|{p},volume={vol}[s{i}]")
    n = len(stems)
    ins = "".join(f"[s{i}]" for i in range(n))
    # normalize=1 avoids excessive sum before loudnorm; duration=first matches stems
    chains.append(
        f"{ins}amix=inputs={n}:duration=first:dropout_transition=0:normalize=1[mx]"
    )
    chains.append(
        "[mx]loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:print_format=summary[out]"
    )
    return ";".join(chains)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mix mono WAV stems with ffmpeg.")
    parser.add_argument(
        "--mix-dir",
        type=Path,
        default=Path.home() / "Desktop" / "MIX",
        help="Folder containing stems (default: ~/Desktop/MIX)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output WAV (default: <mix-dir>/Mixed_AutoMaster.wav)",
    )
    args = parser.parse_args()
    mix_dir: Path = args.mix_dir.expanduser().resolve()
    if not mix_dir.is_dir():
        print(f"Mix directory not found: {mix_dir}", file=sys.stderr)
        return 1

    out = args.output or (mix_dir / "Mixed_AutoMaster.wav")
    out = out.expanduser().resolve()

    stems = DEFAULT_STEMS
    for fn, _, _ in stems:
        p = mix_dir / fn
        if not p.is_file():
            print(f"Missing stem: {p}", file=sys.stderr)
            return 1

    fc = build_filter_complex(stems)
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for fn, _, _ in stems:
        cmd.extend(["-i", str(mix_dir / fn)])
    cmd.extend(
        [
            "-filter_complex",
            fc,
            "-map",
            "[out]",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s24le",
            str(out),
        ]
    )
    print("Running ffmpeg (this may take a minute)...")
    print(" ".join(cmd[:6]), "... -filter_complex ... ->", out)
    subprocess.run(cmd, check=True)
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
