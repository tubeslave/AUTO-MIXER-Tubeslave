#!/usr/bin/env python3
"""Run the current Automixer soundcheck logic on the REPA Song multitrack.

This is an offline demonstration pipeline. It does not talk to a real mixer and
does not grant live write authority. The rendered WAV comes from the current
channel-triggered soundcheck renderer, while the smart-DSP section is captured
as proposal-side evidence for today's correction-method logic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_INPUT_DIR = Path.home() / "Desktop" / "REPA Song"
DEFAULT_AUDIO_DIR = Path.home() / "Desktop" / "Ai MIXING"
DEFAULT_LOG_DIR = Path.home() / "Desktop" / "Ai LOGS"
PIPELINE_NAME = "repa_song_current_soundcheck"
RENDER_TIMEOUT_SEC = 3600.0


def timestamp_slug(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%dT%H%M%S")


def default_artifact_stem(now: datetime | None = None) -> str:
    return f"REPA_Song_AUTOMIXER_SOUNDCHECK_CURRENT_{timestamp_slug(now)}"


def default_output_path(now: datetime | None = None) -> Path:
    return DEFAULT_AUDIO_DIR / f"{default_artifact_stem(now)}.wav"


def default_report_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.report.json"


def default_summary_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.summary.md"


def default_smart_dsp_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.smart_dsp.json"


def default_run_log_path(output_path: Path) -> Path:
    return DEFAULT_LOG_DIR / f"{output_path.stem}.render.log"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def has_audio_files(input_dir: Path) -> bool:
    return input_dir.is_dir() and any(
        item.is_file() and item.suffix.lower() in {".wav", ".wave"}
        for item in input_dir.iterdir()
    )


def resolve_input_dir(input_dir: Path) -> Path:
    if has_audio_files(input_dir):
        return input_dir
    raise SystemExit(f"No WAV files found in {input_dir}")


def build_render_command(
    *,
    input_dir: Path,
    output_path: Path,
    report_path: Path,
    tempo_bpm: float,
    genre: str,
    reference: str,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "channel_triggered_soundcheck_mix.py"),
        "--input-dir",
        str(input_dir),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--tempo-bpm",
        str(float(tempo_bpm)),
    ]
    if genre:
        command.extend(["--genre", genre])
    if reference:
        command.extend(["--reference", reference])
    return command


def subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    paths = [str(REPO_ROOT), str(BACKEND_DIR)]
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_current_soundcheck_renderer(
    *,
    input_dir: Path,
    output_path: Path,
    report_path: Path,
    run_log_path: Path,
    tempo_bpm: float,
    genre: str,
    reference: str,
) -> None:
    command = build_render_command(
        input_dir=input_dir,
        output_path=output_path,
        report_path=report_path,
        tempo_bpm=tempo_bpm,
        genre=genre,
        reference=reference,
    )
    ensure_parent(output_path)
    ensure_parent(report_path)
    ensure_parent(run_log_path)
    with run_log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=subprocess_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=RENDER_TIMEOUT_SEC,
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_channel_triggered_module():
    spec = importlib.util.spec_from_file_location(
        "channel_triggered_soundcheck_mix_module",
        REPO_ROOT / "tools" / "channel_triggered_soundcheck_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_target_plane_lufs(instrument: str) -> float:
    if instrument == "lead_vocal":
        return -22.0
    if instrument == "backing_vocal":
        return -27.0
    if instrument in {"accordion", "playback"}:
        return -30.0
    if instrument in {"kick", "snare", "rack_tom", "floor_tom", "bass_guitar", "electric_guitar"}:
        return -25.0
    if instrument in {"overhead", "room"}:
        return -30.0
    return -25.0


def _compact_recommendation(proposal: dict[str, Any]) -> dict[str, Any]:
    selected = proposal.get("selected_strategy") or {}
    evidence = proposal.get("evidence") or {}
    audio_evidence = evidence.get("audio_evidence") or {}
    return {
        "target": proposal.get("target"),
        "source_role": evidence.get("source_role") or proposal.get("source_role"),
        "status": proposal.get("reason"),
        "confidence": proposal.get("confidence"),
        "selected_strategy_id": selected.get("strategy_id"),
        "method_family": selected.get("method_family"),
        "problem": selected.get("problem"),
        "parameters": selected.get("parameters"),
        "blockers": proposal.get("blockers") or [],
        "audio_evidence": {
            "problem_hints": audio_evidence.get("problem_hints") or [],
            "peak_dbfs": audio_evidence.get("peak_dbfs"),
            "estimated_lufs": audio_evidence.get("estimated_lufs"),
            "true_peak_headroom_db": audio_evidence.get("true_peak_headroom_db"),
            "muddiness_score": audio_evidence.get("muddiness_score"),
            "harshness_score": audio_evidence.get("harshness_score"),
            "sibilance_score": audio_evidence.get("sibilance_score"),
            "feedback_risk": audio_evidence.get("feedback_risk"),
            "phase_correlation": audio_evidence.get("phase_correlation"),
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_smart_dsp_sidecar(
    *,
    input_dir: Path,
    soundcheck_report: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        from smart_dsp_evidence_extractor import analyze_smart_dsp_audio  # noqa: PLC0415
    except ImportError as exc:
        return {
            "pipeline": PIPELINE_NAME,
            "mode": "soundcheck",
            "status": "smart_dsp_unavailable",
            "rendered_audio": str(output_path),
            "notes": [
                "The WAV is rendered by the current channel-triggered soundcheck pipeline.",
                f"Smart-DSP proposal sidecar was skipped: {exc}",
                "No WING or live mixer writes are performed by this offline pipeline.",
            ],
            "recommendation_count": 0,
            "recommendations": [],
            "operator_proposals": [],
        }

    final_channels = list(soundcheck_report.get("final_channels") or [])
    recommendations: list[dict[str, Any]] = []
    full_proposals: list[dict[str, Any]] = []
    for index, channel_report in enumerate(final_channels, start=1):
        file_name = str(channel_report.get("file") or "")
        if not file_name:
            continue
        audio_path = input_dir / file_name
        if not audio_path.exists():
            continue
        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        instrument = str(channel_report.get("instrument") or "custom")
        recommendation = analyze_smart_dsp_audio(
            samples,
            target_id=f"ch{index}:{Path(file_name).stem}",
            source_role=instrument,
            source_label=file_name,
            mode="soundcheck",
            sample_rate=int(sample_rate),
            source_identity_confirmed=True,
            source_identity_confidence=0.86,
            current_state_readback_available=False,
            rollback_plan_available=False,
            allow_inferred_source_identity=True,
            target_plane_lufs=_source_target_plane_lufs(instrument),
            active_performance=False,
        )
        proposal = _json_safe(recommendation.to_operator_proposal())
        full_proposals.append(proposal)
        recommendations.append(_compact_recommendation(proposal))

    return {
        "pipeline": PIPELINE_NAME,
        "mode": "soundcheck",
        "status": "proposal_sidecar_only",
        "rendered_audio": str(output_path),
        "notes": [
            "The WAV is rendered by the current channel-triggered soundcheck pipeline.",
            "Smart-DSP recommendations are captured as proposal-only evidence.",
            "No WING or live mixer writes are performed by this offline pipeline.",
        ],
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "operator_proposals": full_proposals,
    }


def write_smart_dsp_sidecar(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path).write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _format_stage_list() -> list[str]:
    return [
        "channel-triggered source detection",
        "bleed-aware trim and source classification",
        "drum phase alignment and overhead-anchored pan",
        "rule-agent fader/compressor decisions",
        "event-based dynamics for drums and vocal gaps",
        "AutoFOH analyzer corrections",
        "cross-adaptive EQ",
        "kick/bass hierarchy",
        "frequency-window and stem mix verification",
        "snare top/bottom pair safety",
        "offline FX sends/returns",
        "master bus compression and final limiter",
        "smart-DSP proposal sidecar",
    ]


def write_summary(
    *,
    summary_path: Path,
    output_path: Path,
    report_path: Path,
    smart_dsp_path: Path | None,
    soundcheck_report: dict[str, Any],
    smart_dsp_report: dict[str, Any] | None,
    run_log_path: Path,
) -> None:
    final_channels = soundcheck_report.get("final_channels") or []
    snare_safety = soundcheck_report.get("snare_pair_safety") or {}
    lines = [
        "# REPA Song Automixer Soundcheck Pipeline",
        "",
        "This offline render demonstrates the current Automixer soundcheck logic.",
        "It does not connect to a real mixer and does not perform live writes.",
        "",
        f"- output: `{output_path}`",
        f"- report: `{report_path}`",
        f"- render_log: `{run_log_path}`",
        f"- smart_dsp_report: `{smart_dsp_path}`" if smart_dsp_path else "- smart_dsp_report: `disabled`",
        f"- final_lufs: `{soundcheck_report.get('final_lufs')}`",
        f"- final_peak_dbfs: `{soundcheck_report.get('final_peak_dbfs')}`",
        f"- duration_sec: `{soundcheck_report.get('duration_sec')}`",
        f"- sample_rate: `{soundcheck_report.get('sample_rate')}`",
        f"- sha256: `{sha256_file(output_path) if output_path.exists() else ''}`",
        "",
        "## Current Soundcheck Stages",
        "",
    ]
    lines.extend(f"- {stage}" for stage in _format_stage_list())
    lines.extend(["", "## Applied Channel State", ""])
    lines.append("| Ch | File | Instrument | Trim | Fader | Pan | HPF | Phase | Delay | Expander |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|---:|---|")
    for index, channel in enumerate(final_channels, start=1):
        expander = channel.get("event_expander") or {}
        expander_label = "on" if expander.get("enabled") else "off"
        lines.append(
            "| {ch} | {file} | {instrument} | {trim} | {fader} | {pan} | {hpf} | "
            "{phase} | {delay} | {expander} |".format(
                ch=index,
                file=channel.get("file"),
                instrument=channel.get("instrument"),
                trim=channel.get("trim_db"),
                fader=channel.get("fader_db"),
                pan=channel.get("pan"),
                hpf=channel.get("hpf"),
                phase=channel.get("phase_invert"),
                delay=channel.get("delay_ms"),
                expander=expander_label,
            )
        )

    lines.extend(["", "## Safety Highlights", ""])
    if snare_safety.get("enabled"):
        for action in snare_safety.get("actions", []):
            lines.append(
                "- snare {role} {file}: trim {before_trim} -> {after_trim}, "
                "fader {before_fader} -> {after_fader}".format(
                    role=action.get("role"),
                    file=action.get("file"),
                    before_trim=(action.get("before") or {}).get("trim_db"),
                    after_trim=(action.get("after") or {}).get("trim_db"),
                    before_fader=(action.get("before") or {}).get("fader_db"),
                    after_fader=(action.get("after") or {}).get("fader_db"),
                )
            )
    else:
        lines.append("- snare pair safety: not triggered")

    if smart_dsp_report:
        lines.extend(["", "## Smart-DSP Proposal Sidecar", ""])
        for item in smart_dsp_report.get("recommendations", []):
            lines.append(
                "- {target}: {strategy} ({family}) confidence={confidence}".format(
                    target=item.get("target"),
                    strategy=item.get("selected_strategy_id"),
                    family=item.get("method_family"),
                    confidence=item.get("confidence"),
                )
            )

    ensure_parent(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = resolve_input_dir(Path(args.input_dir).expanduser())
    output_path = Path(args.output).expanduser() if args.output else default_output_path()
    report_path = Path(args.report).expanduser() if args.report else default_report_path(output_path)
    summary_path = Path(args.summary).expanduser() if args.summary else default_summary_path(output_path)
    smart_dsp_path = (
        None
        if args.no_smart_dsp
        else Path(args.smart_dsp_report).expanduser()
        if args.smart_dsp_report
        else default_smart_dsp_path(output_path)
    )
    run_log_path = (
        Path(args.run_log).expanduser() if args.run_log else default_run_log_path(output_path)
    )

    run_current_soundcheck_renderer(
        input_dir=input_dir,
        output_path=output_path,
        report_path=report_path,
        run_log_path=run_log_path,
        tempo_bpm=args.tempo_bpm,
        genre=args.genre,
        reference=args.reference,
    )

    soundcheck_report = load_json(report_path)
    smart_dsp_report = None
    if smart_dsp_path is not None:
        smart_dsp_report = build_smart_dsp_sidecar(
            input_dir=input_dir,
            soundcheck_report=soundcheck_report,
            output_path=output_path,
        )
        write_smart_dsp_sidecar(smart_dsp_path, smart_dsp_report)

    write_summary(
        summary_path=summary_path,
        output_path=output_path,
        report_path=report_path,
        smart_dsp_path=smart_dsp_path,
        soundcheck_report=soundcheck_report,
        smart_dsp_report=smart_dsp_report,
        run_log_path=run_log_path,
    )

    manifest = {
        "pipeline": PIPELINE_NAME,
        "input_dir": str(input_dir),
        "output": str(output_path),
        "report": str(report_path),
        "summary": str(summary_path),
        "smart_dsp_report": str(smart_dsp_path) if smart_dsp_path else "",
        "run_log": str(run_log_path),
        "final_lufs": soundcheck_report.get("final_lufs"),
        "final_peak_dbfs": soundcheck_report.get("final_peak_dbfs"),
        "sample_rate": soundcheck_report.get("sample_rate"),
        "duration_sec": soundcheck_report.get("duration_sec"),
        "sha256": sha256_file(output_path),
    }
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render REPA Song through the current Automixer soundcheck pipeline.",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--smart-dsp-report", default="")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--genre", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--no-smart-dsp", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = run_pipeline(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
