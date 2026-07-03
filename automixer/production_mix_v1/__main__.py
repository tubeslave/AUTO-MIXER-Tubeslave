"""CLI for `python -m automixer.production_mix_v1`."""

from __future__ import annotations

import argparse
import json

from .ayaic_offline_pipeline import CORRECTIVE_EQ_METHODS, DEFAULT_CORRECTIVE_EQ_METHOD, DEFAULT_MASTER_CEILING_DBFS, DEFAULT_MASTER_TARGET_LUFS, DEFAULT_MAX_PHASE_DELAY_MS, run_ayaic_offline_pipeline
from .contextual_compression import DEFAULT_CONTEXTUAL_COMPRESSION_STYLE, STYLE_MODIFIERS
from .musical_panning import DEFAULT_MUSICAL_PANNING_STYLE, STYLE_MODIFIERS as PANNING_STYLE_MODIFIERS
from .musical_output_balance import DEFAULT_MUSICAL_BALANCE_STYLE, STYLE_MODIFIERS as BALANCE_STYLE_MODIFIERS
from .config import load_production_mix_config
from .models import MODE_LIVE, MODE_OFFLINE
from .pipeline import ProductionMixPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production_mix_v1 safely")
    parser.add_argument("--scheme", choices=["production-mix-v1", "ayaic-clean"], default="production-mix-v1")
    parser.add_argument("--config", default="")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--output-dir", default="production_mix_v1_out")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--mode", choices=[MODE_OFFLINE, MODE_LIVE], default=MODE_OFFLINE)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live-apply", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-optional-critics", action="store_true")
    parser.add_argument("--master-target-lufs", type=float, default=DEFAULT_MASTER_TARGET_LUFS)
    parser.add_argument("--master-ceiling-dbfs", type=float, default=DEFAULT_MASTER_CEILING_DBFS)
    parser.add_argument("--max-phase-delay-ms", type=float, default=DEFAULT_MAX_PHASE_DELAY_MS)
    parser.add_argument("--no-mp3", action="store_true")
    parser.add_argument("--drums-only", action="store_true")
    parser.add_argument("--use-group-levels", action="store_true")
    parser.add_argument("--skip-group-levels", action="store_true")
    parser.add_argument("--skip-master-level", action="store_true")
    parser.add_argument("--corrective-eq-method", choices=CORRECTIVE_EQ_METHODS, default=DEFAULT_CORRECTIVE_EQ_METHOD)
    parser.add_argument("--enable-autoeq", action="store_true")
    parser.add_argument("--autoeq-mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--autoeq-report-only", action="store_true")
    parser.add_argument("--autoeq-disable-dynamic", action="store_true")
    parser.add_argument("--autoeq-disable-group-eq", action="store_true")
    parser.add_argument("--autoeq-enable-master-eq", action="store_true")
    parser.add_argument("--autoeq-osc-apply", action="store_true")
    parser.add_argument("--disable-contextual-compression", action="store_true")
    parser.add_argument("--compression-report-only", action="store_true")
    parser.add_argument("--compression-style", choices=sorted(STYLE_MODIFIERS), default=DEFAULT_CONTEXTUAL_COMPRESSION_STYLE)
    parser.add_argument("--compression-bpm", type=float, default=0.0)
    parser.add_argument("--disable-musical-panning", action="store_true")
    parser.add_argument("--panning-report-only", action="store_true")
    parser.add_argument("--panning-style", choices=sorted(PANNING_STYLE_MODIFIERS), default=DEFAULT_MUSICAL_PANNING_STYLE)
    parser.add_argument("--disable-musical-output-balance", action="store_true")
    parser.add_argument("--output-balance-report-only", action="store_true")
    parser.add_argument("--output-balance-style", choices=sorted(BALANCE_STYLE_MODIFIERS), default=DEFAULT_MUSICAL_BALANCE_STYLE)
    args = parser.parse_args(argv)
    if args.scheme == "ayaic-clean":
        if not args.input_dir:
            parser.error("--input-dir is required for --scheme ayaic-clean")
        result = run_ayaic_offline_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_name=args.output_name or "ayaic_clean_mix",
            master_target_lufs=args.master_target_lufs,
            master_ceiling_dbfs=args.master_ceiling_dbfs,
            max_phase_delay_ms=args.max_phase_delay_ms,
            write_mp3=not args.no_mp3,
            drums_only=bool(args.drums_only),
            skip_group_levels=bool(args.skip_group_levels or not args.use_group_levels),
            skip_master_level=bool(args.skip_master_level),
            corrective_eq_method=args.corrective_eq_method,
            autoeq_enabled=bool(args.enable_autoeq),
            autoeq_mode=args.autoeq_mode,
            autoeq_report_only=bool(args.autoeq_report_only),
            autoeq_config={
                "apply_dynamic_eq": not bool(args.autoeq_disable_dynamic),
                "apply_group_eq": not bool(args.autoeq_disable_group_eq),
                "apply_master_eq": bool(args.autoeq_enable_master_eq),
                "osc_dry_run": not bool(args.autoeq_osc_apply),
                "osc_apply": bool(args.autoeq_osc_apply),
            },
            compression_enabled=not bool(args.disable_contextual_compression),
            compression_style=args.compression_style,
            compression_bpm=float(args.compression_bpm) if args.compression_bpm > 0 else None,
            compression_report_only=bool(args.compression_report_only),
            panning_enabled=not bool(args.disable_musical_panning),
            panning_style=args.panning_style,
            panning_report_only=bool(args.panning_report_only),
            output_balance_enabled=not bool(args.disable_musical_output_balance),
            output_balance_style=args.output_balance_style,
            output_balance_report_only=bool(args.output_balance_report_only),
        )
        print(json.dumps({"wav": result.output_wav, "mp3": result.output_mp3, "report": result.report}, ensure_ascii=True))
        return 0
    pipeline = ProductionMixPipeline(
        load_production_mix_config(args.config or None),
        enable_optional_critics=bool(args.enable_optional_critics),
    )
    result = pipeline.run(
        input_dir=args.input_dir or None,
        output_dir=args.output_dir,
        mode=args.mode,
        dry_run=not bool(args.live_apply),
        smoke=bool(args.smoke),
    )
    print(json.dumps({"run_id": result.run_id, "selected": result.decision.selected.candidate.name, "accepted": result.decision.accepted, "artifacts": result.artifacts, "warnings": result.warnings}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
