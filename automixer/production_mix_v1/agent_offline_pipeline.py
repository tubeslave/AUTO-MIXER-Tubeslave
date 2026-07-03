#!/usr/bin/env python3
"""
Agent Offline Pipeline — unified mixing for SONG REPA.
Branch: agent-offline-pipeline

Uses production_mix_v1 ayaic_offline_pipeline with full features:
- Bleed-aware primary signal detection
- Arrangement-aware input levels (Ayaic)
- Phase/delay alignment (GCC-PHAT)
- HPF/LPF correction
- Corrective EQ (contextual_deep_eq)
- Contextual compression
- Musical panning
- Musical output balance
- Snare pair coherence
- Snare top/bottom balance
- Panorama width restoration
- Mastering (loudnorm)

Run from project root:
  cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
  PYTHONPATH=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main:/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/backend python3 automixer/production_mix_v1/agent_offline_pipeline.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from automixer.production_mix_v1.ayaic_offline_pipeline import run_ayaic_offline_pipeline


def main():
    parser = argparse.ArgumentParser(description="Agent Offline Pipeline")
    parser.add_argument("--input-dir", default="/Users/dmitrijvolkov/Desktop/SONG  REPA")
    parser.add_argument("--output-dir", default="/Users/dmitrijvolkov/Desktop/SONG_REPA_AGENT_MIX")
    parser.add_argument("--output-name", default="SONG_REPA_AGENT_MIX")
    parser.add_argument("--master-target-lufs", type=float, default=-20.0)
    parser.add_argument("--master-ceiling-dbfs", type=float, default=-1.0)
    parser.add_argument("--corrective-eq-method", default="contextual_deep_eq",
                        choices=["none", "profile", "bleed_control", "cross_adaptive",
                                 "frequency_window", "frequency_cross_profile",
                                 "project_corrective", "project_corrective_hybrid",
                                 "contextual_deep_eq"])
    parser.add_argument("--compression-enabled", action="store_true", default=True)
    parser.add_argument("--compression-disabled", action="store_true")
    parser.add_argument("--panning-enabled", action="store_true", default=True)
    parser.add_argument("--panning-disabled", action="store_true")
    parser.add_argument("--output-balance-enabled", action="store_true", default=True)
    parser.add_argument("--output-balance-disabled", action="store_true")
    parser.add_argument("--write-mp3", action="store_true", default=True)
    parser.add_argument("--no-mp3", action="store_true")
    parser.add_argument("--skip-group-levels", action="store_true")
    parser.add_argument("--skip-master-level", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use generated test tones")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_name = args.output_name

    compression_enabled = args.compression_enabled and not args.compression_disabled
    panning_enabled = args.panning_enabled and not args.panning_disabled
    output_balance_enabled = args.output_balance_enabled and not args.output_balance_disabled
    write_mp3 = args.write_mp3 and not args.no_mp3

    print("=" * 70)
    print("  Agent Offline Pipeline — SONG REPA")
    print("  Branch: agent-offline-pipeline")
    print("=" * 70)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Name:   {output_name}")
    print(f"  Master: {args.master_target_lufs} LUFS / {args.master_ceiling_dbfs} dBFS ceiling")
    print()
    print("  Modules:")
    print(f"    Phase alignment:     ON")
    print(f"    HPF/LPF correction:  ON")
    print(f"    Arrangement levels:  ON")
    print(f"    Corrective EQ:       {args.corrective_eq_method}")
    print(f"    Contextual compression: {'ON' if compression_enabled else 'OFF'}")
    print(f"    Musical panning:     {'ON' if panning_enabled else 'OFF'}")
    print(f"    Output balance:      {'ON' if output_balance_enabled else 'OFF'}")
    print(f"    Snare coherence:     ON")
    print(f"    Snare balance:       ON")
    print(f"    Panorama width:      ON")
    print(f"    Group levels:        {'OFF' if args.skip_group_levels else 'ON'}")
    print(f"    Master level:        {'OFF' if args.skip_master_level else 'ON'}")
    print(f"    Write MP3:           {'ON' if write_mp3 else 'OFF'}")
    print("=" * 70)
    print()

    result = run_ayaic_offline_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        output_name=output_name,
        master_target_lufs=args.master_target_lufs,
        master_ceiling_dbfs=args.master_ceiling_dbfs,
        corrective_eq_method=args.corrective_eq_method,
        compression_enabled=compression_enabled,
        panning_enabled=panning_enabled,
        output_balance_enabled=output_balance_enabled,
        write_mp3=write_mp3,
        skip_group_levels=args.skip_group_levels,
        skip_master_level=args.skip_master_level,
    )

    print()
    print("=" * 70)
    print("  DONE")
    print(f"  Output: {getattr(result, 'final_mix_path', 'N/A')}")
    print(f"  Report: {getattr(result, 'report_path', 'N/A')}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    main()
