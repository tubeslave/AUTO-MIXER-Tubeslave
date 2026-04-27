"""Command-line interface for the mix-agent facade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mix_agent.agent import apply_conservative, suggest
from mix_agent.reporting import render_markdown_report, write_json_report, write_markdown_report


def _add_common_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stems", default="", help="Directory containing stem audio files")
    parser.add_argument("--mix", default="", help="Stereo mix file")
    parser.add_argument("--reference", default="", help="Optional reference track")
    parser.add_argument("--genre", default="neutral", help="Genre profile")
    parser.add_argument("--target-platform", default="streaming", help="Target playback context")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m mix_agent")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_parser = sub.add_parser("analyze", help="Analyze mix/stems and write a report")
    _add_common_audio_args(analyze_parser)
    analyze_parser.add_argument("--out", required=True, help="Output .md or .json report")

    suggest_parser = sub.add_parser("suggest", help="Write machine-readable suggestions")
    _add_common_audio_args(suggest_parser)
    suggest_parser.add_argument("--out", required=True, help="Output suggestions JSON")

    apply_parser = sub.add_parser("apply", help="Apply conservative offline actions")
    apply_parser.add_argument("--stems", required=True, help="Directory containing stem audio files")
    apply_parser.add_argument("--suggestions", default="", help="Suggestions JSON from analyze/suggest")
    apply_parser.add_argument("--reference", default="", help="Optional reference track")
    apply_parser.add_argument("--genre", default="neutral", help="Genre profile")
    apply_parser.add_argument("--target-platform", default="streaming", help="Target playback context")
    apply_parser.add_argument("--out", required=True, help="Output rendered WAV")
    apply_parser.add_argument("--report", default="", help="Optional apply report .md or .json")

    args = parser.parse_args()
    if args.command in {"analyze", "suggest"}:
        plan = suggest(
            stems=args.stems or None,
            mix=args.mix or None,
            reference=args.reference or None,
            genre=args.genre,
            target_platform=args.target_platform,
            out=args.out,
        )
        if args.command == "analyze" and str(args.out).lower().endswith(".json"):
            write_json_report(plan, args.out)
        elif args.command == "analyze" and str(args.out).lower().endswith(".md"):
            write_markdown_report(plan, args.out)
        print(json.dumps({"issues": len(plan.issues), "actions": len(plan.actions), "out": args.out}, ensure_ascii=False))
        return 0

    if args.command == "apply":
        plan = apply_conservative(
            stems=args.stems,
            out=args.out,
            suggestions=args.suggestions or None,
            reference=args.reference or None,
            genre=args.genre,
            target_platform=args.target_platform,
            report=args.report or None,
        )
        if not args.report:
            report_path = str(Path(args.out).with_suffix(".mix_agent.md"))
            Path(report_path).write_text(render_markdown_report(plan), encoding="utf-8")
        print(json.dumps({"applied_actions": len(plan.applied_actions), "out": args.out}, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
