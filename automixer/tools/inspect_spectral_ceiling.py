"""Inspect spectral ceiling EQ proposals for one audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from backend.heuristics.spectral_ceiling_eq import (
    SpectralCeilingEQAnalyzer,
    SpectralCeilingEQConfig,
    format_spectral_ceiling_log,
)


def _load_config(path: str | None) -> SpectralCeilingEQConfig:
    config_path = Path(path).expanduser() if path else Path("config/automixer.yaml")
    if not config_path.exists():
        return SpectralCeilingEQConfig()
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return SpectralCeilingEQConfig()
    section = payload.get("spectral_ceiling_eq", {}) if isinstance(payload, dict) else {}
    return SpectralCeilingEQConfig.from_mapping(section)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m automixer.tools.inspect_spectral_ceiling"
    )
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--role", required=True, help="Instrument role, e.g. lead_vocal")
    parser.add_argument("--config", default="", help="Optional automixer.yaml path")
    parser.add_argument("--track-id", default="", help="Optional display name")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run for inspection")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human log")
    args = parser.parse_args()

    audio, sample_rate = sf.read(
        str(Path(args.input).expanduser()),
        always_2d=True,
        dtype="float32",
    )
    config = _load_config(args.config or None)
    if args.dry_run:
        config.dry_run = True
    proposal = SpectralCeilingEQAnalyzer(config).analyze(
        audio,
        instrument_role=args.role,
        sample_rate=int(sample_rate),
        track_id=args.track_id or Path(args.input).stem,
        role_confidence=1.0,
    )
    if args.json:
        print(json.dumps(proposal.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_spectral_ceiling_log(proposal))
        if proposal.warnings:
            print("safety_warnings:")
            for warning in proposal.warnings:
                print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
