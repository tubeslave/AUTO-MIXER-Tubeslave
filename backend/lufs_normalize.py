#!/usr/bin/env python3
"""
Standalone LUFS normalization utility.

Replaces ffmpeg loudnorm for precise LUFS targeting using pyloudnorm
(ITU-R BS.1770-4 compliant).

Usage:
    python3 lufs_normalize.py input.wav -o output.wav --target -14
    python3 lufs_normalize.py input.wav --measure
    python3 lufs_normalize.py *.wav --target -8 --true-peak -1.0 --suffix _normalized
    python3 lufs_normalize.py input.wav -o output.wav --target -14 --bit-depth 24

Requires: pyloudnorm, soundfile, numpy, scipy
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import pyloudnorm as pyln
except ImportError:
    print("ERROR: pyloudnorm not installed. Run: pip install pyloudnorm", file=sys.stderr)
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile", file=sys.stderr)
    sys.exit(1)


def measure_true_peak(audio: np.ndarray, sample_rate: int) -> float:
    """Measure true peak with 4x oversampling (ITU-R BS.1770-4)."""
    from scipy.signal import resample_poly
    if audio.ndim == 1:
        oversampled = resample_poly(audio, 4, 1)
    else:
        oversampled = np.column_stack(
            [resample_poly(audio[:, ch], 4, 1) for ch in range(audio.shape[1])]
        )
    peak_lin = np.max(np.abs(oversampled))
    if peak_lin > 0:
        return float(20 * np.log10(peak_lin))
    return -100.0


def measure_loudness_range(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate Loudness Range (LRA) using short-term LUFS distribution.

    Simplified LRA: difference between 95th and 10th percentile of
    short-term (3s) loudness values, per EBU R128.
    """
    meter = pyln.Meter(sample_rate, block_size=3.0)  # 3-second blocks for LRA
    # Manually compute short-term loudness over sliding window
    block_samples = int(3.0 * sample_rate)
    hop_samples = int(0.1 * sample_rate)  # 100ms hop

    if audio.ndim == 1:
        audio_2d = audio.reshape(-1, 1)
    else:
        audio_2d = audio

    loudness_values = []
    for start in range(0, len(audio_2d) - block_samples, hop_samples):
        block = audio_2d[start:start + block_samples]
        try:
            l = meter.integrated_loudness(block)
            if l > -70.0:  # Absolute gate
                loudness_values.append(l)
        except Exception:
            continue

    if len(loudness_values) < 2:
        return 0.0

    loudness_values = np.array(loudness_values)

    # Relative gate: -20 LU below ungated mean
    ungated_mean = np.mean(loudness_values)
    relative_gate = ungated_mean - 20.0
    gated = loudness_values[loudness_values >= relative_gate]

    if len(gated) < 2:
        return 0.0

    p10 = np.percentile(gated, 10)
    p95 = np.percentile(gated, 95)
    return float(p95 - p10)


def measure_file(filepath: str) -> dict:
    """Measure all loudness metrics for an audio file."""
    audio, sr = sf.read(filepath, dtype='float32')

    meter = pyln.Meter(sr)
    if audio.ndim == 1:
        integrated = meter.integrated_loudness(audio.reshape(-1, 1))
    else:
        integrated = meter.integrated_loudness(audio)

    true_peak = measure_true_peak(audio, sr)
    sample_peak = float(20 * np.log10(np.max(np.abs(audio)) + 1e-10))
    lra = measure_loudness_range(audio, sr)
    duration = len(audio) / sr

    return {
        'file': filepath,
        'sample_rate': sr,
        'channels': 1 if audio.ndim == 1 else audio.shape[1],
        'duration_sec': round(duration, 2),
        'integrated_lufs': round(integrated, 2),
        'true_peak_dbtp': round(true_peak, 2),
        'sample_peak_dbfs': round(sample_peak, 2),
        'loudness_range_lu': round(lra, 1),
    }


def normalize_file(
    input_path: str,
    output_path: str,
    target_lufs: float = -14.0,
    true_peak_limit: float = -1.0,
    bit_depth: int = 24,
) -> dict:
    """Normalize an audio file to target LUFS with true peak limiting.

    Returns dict with before/after measurements.
    """
    audio, sr = sf.read(input_path, dtype='float64')  # float64 for precision

    meter = pyln.Meter(sr)
    if audio.ndim == 1:
        audio_for_meter = audio.reshape(-1, 1)
    else:
        audio_for_meter = audio

    # Measure input
    input_lufs = meter.integrated_loudness(audio_for_meter)
    input_true_peak = measure_true_peak(audio.astype(np.float32), sr)

    if input_lufs < -70.0:
        print(f"  WARNING: Audio too quiet ({input_lufs:.1f} LUFS), skipping.", file=sys.stderr)
        return {
            'input_lufs': input_lufs,
            'output_lufs': input_lufs,
            'gain_db': 0.0,
            'true_peak_limited': False,
            'skipped': True,
        }

    # Normalize to target LUFS
    normalized = pyln.normalize.loudness(audio_for_meter, input_lufs, target_lufs)
    if audio.ndim == 1:
        normalized = normalized.flatten()

    # True peak limiting (iterative for precision)
    tp_limited = False
    for _ in range(3):  # Max 3 iterations
        tp = measure_true_peak(normalized.astype(np.float32), sr)
        if tp > true_peak_limit:
            reduction_db = tp - true_peak_limit + 0.1  # +0.1 dB safety margin
            reduction_linear = 10 ** (-reduction_db / 20.0)
            normalized = normalized * reduction_linear
            tp_limited = True
        else:
            break

    # Measure output
    if audio.ndim == 1:
        output_for_meter = normalized.reshape(-1, 1)
    else:
        output_for_meter = normalized
    output_lufs = meter.integrated_loudness(output_for_meter)
    output_true_peak = measure_true_peak(normalized.astype(np.float32), sr)

    # Write output
    subtype_map = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
    subtype = subtype_map.get(bit_depth, 'PCM_24')

    # Clip to prevent wrapping in integer formats
    if bit_depth in (16, 24):
        normalized = np.clip(normalized, -1.0, 1.0 - 1e-7)

    sf.write(output_path, normalized, sr, subtype=subtype)

    gain_db = target_lufs - input_lufs

    return {
        'input_file': input_path,
        'output_file': output_path,
        'sample_rate': sr,
        'bit_depth': bit_depth,
        'input_lufs': round(input_lufs, 2),
        'output_lufs': round(output_lufs, 2),
        'target_lufs': target_lufs,
        'deviation_lu': round(output_lufs - target_lufs, 2),
        'gain_db': round(gain_db, 2),
        'input_true_peak_dbtp': round(input_true_peak, 2),
        'output_true_peak_dbtp': round(output_true_peak, 2),
        'true_peak_limit': true_peak_limit,
        'true_peak_limited': tp_limited,
        'skipped': False,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Precise LUFS normalization (ITU-R BS.1770-4)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure loudness of a file
  %(prog)s input.wav --measure

  # Normalize to -14 LUFS (streaming standard)
  %(prog)s input.wav -o output.wav --target -14

  # Normalize to -8 LUFS for live sound, 24-bit
  %(prog)s input.wav -o output.wav --target -8 --true-peak -1.0 --bit-depth 24

  # Batch normalize all WAV files
  %(prog)s *.wav --target -14 --suffix _normalized
        """,
    )
    parser.add_argument('input', nargs='+', help='Input audio file(s)')
    parser.add_argument('-o', '--output', help='Output file (single file mode)')
    parser.add_argument('--target', type=float, default=-14.0,
                        help='Target integrated LUFS (default: -14.0)')
    parser.add_argument('--true-peak', type=float, default=-1.0,
                        help='True peak limit in dBTP (default: -1.0)')
    parser.add_argument('--bit-depth', type=int, default=24, choices=[16, 24, 32],
                        help='Output bit depth (default: 24)')
    parser.add_argument('--suffix', default='_normalized',
                        help='Suffix for batch output files (default: _normalized)')
    parser.add_argument('--measure', action='store_true',
                        help='Measure only, do not normalize')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    results = []

    if args.measure:
        for filepath in args.input:
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}", file=sys.stderr)
                continue
            metrics = measure_file(filepath)
            results.append(metrics)

            if not args.json:
                print(f"\n{'='*60}")
                print(f"  File:           {metrics['file']}")
                print(f"  Duration:       {metrics['duration_sec']:.1f} s")
                print(f"  Sample Rate:    {metrics['sample_rate']} Hz")
                print(f"  Channels:       {metrics['channels']}")
                print(f"  Integrated:     {metrics['integrated_lufs']:.2f} LUFS")
                print(f"  True Peak:      {metrics['true_peak_dbtp']:.2f} dBTP")
                print(f"  Sample Peak:    {metrics['sample_peak_dbfs']:.2f} dBFS")
                print(f"  Loudness Range: {metrics['loudness_range_lu']:.1f} LU")
                print(f"{'='*60}")
    else:
        for i, filepath in enumerate(args.input):
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}", file=sys.stderr)
                continue

            # Determine output path
            if args.output and len(args.input) == 1:
                output_path = args.output
            else:
                p = Path(filepath)
                output_path = str(p.parent / f"{p.stem}{args.suffix}{p.suffix}")

            print(f"\nProcessing: {filepath}")
            print(f"  Target: {args.target:.1f} LUFS, True Peak: {args.true_peak:.1f} dBTP")

            result = normalize_file(
                filepath, output_path,
                target_lufs=args.target,
                true_peak_limit=args.true_peak,
                bit_depth=args.bit_depth,
            )
            results.append(result)

            if not args.json:
                if result['skipped']:
                    print(f"  SKIPPED (too quiet)")
                else:
                    print(f"  Input:   {result['input_lufs']:.2f} LUFS / {result['input_true_peak_dbtp']:.2f} dBTP")
                    print(f"  Output:  {result['output_lufs']:.2f} LUFS / {result['output_true_peak_dbtp']:.2f} dBTP")
                    print(f"  Gain:    {result['gain_db']:+.2f} dB")
                    print(f"  Deviation: {result['deviation_lu']:+.2f} LU from target")
                    if result['true_peak_limited']:
                        print(f"  ⚠ True peak limited to {result['true_peak_limit']:.1f} dBTP")
                    print(f"  Saved:   {result['output_file']}")

    if args.json:
        import json
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
