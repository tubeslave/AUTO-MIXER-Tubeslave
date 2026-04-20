#!/usr/bin/env python3
"""Render an online-soundcheck style mix from local multitrack stems.

The track starts from a conservative baseline and then crossfades through
the same built-in correction stages used by the project so the finished file
lets you hear the mix "settling in" during playback.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))


def load_offline_mix_module():
    spec = importlib.util.spec_from_file_location(
        "offline_agent_mix_runtime",
        REPO_ROOT / "tools" / "offline_agent_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clone_plans(plans: dict[int, Any]) -> dict[int, Any]:
    return {channel: copy.deepcopy(plan) for channel, plan in plans.items()}


def build_online_schedule(duration_sec: float) -> list[dict[str, Any]]:
    online_window_sec = min(52.0, max(28.0, duration_sec * 0.34))
    warmup_sec = min(6.0, max(4.0, online_window_sec * 0.12))
    remaining = max(12.0, online_window_sec - warmup_sec)
    fade_one = remaining * 0.34
    fade_two = remaining * 0.33
    fade_three = max(6.0, remaining - fade_one - fade_two)

    start_one = warmup_sec
    start_two = start_one + fade_one
    start_three = start_two + fade_two
    return [
        {
            "from_stage": "baseline",
            "to_stage": "gain_phase_agent",
            "start_sec": round(start_one, 2),
            "fade_sec": round(fade_one, 2),
        },
        {
            "from_stage": "gain_phase_agent",
            "to_stage": "adaptive_cleanup",
            "start_sec": round(start_two, 2),
            "fade_sec": round(fade_two, 2),
        },
        {
            "from_stage": "adaptive_cleanup",
            "to_stage": "live_finish",
            "start_sec": round(start_three, 2),
            "fade_sec": round(fade_three, 2),
        },
    ]


def crossfade_stage_sequence(stage_mixes: list[np.ndarray], schedule: list[dict[str, Any]], sr: int) -> np.ndarray:
    if not stage_mixes:
        return np.zeros((0, 2), dtype=np.float32)
    out = stage_mixes[0].copy()
    total_len = len(out)
    for idx, transition in enumerate(schedule, start=1):
        if idx >= len(stage_mixes):
            break
        target = stage_mixes[idx]
        start = int(float(transition["start_sec"]) * sr)
        fade = max(1, int(float(transition["fade_sec"]) * sr))
        if start >= total_len:
            continue
        end = min(total_len, start + fade)
        if end > start:
            ramp = np.linspace(0.0, 1.0, end - start, endpoint=False, dtype=np.float32)[:, None]
            out[start:end] = (out[start:end] * (1.0 - ramp) + target[start:end] * ramp).astype(np.float32)
        out[end:] = target[end:]
    return out.astype(np.float32)


def build_initial_plans(mixmod, input_dir: Path) -> tuple[list[Path], int, int, dict[int, Any]]:
    wavs = sorted(p for p in input_dir.glob("*.wav") if p.is_file())
    if not wavs:
        raise SystemExit(f"No WAV files found in {input_dir}")

    info = sf.info(str(wavs[0]))
    sr = int(info.samplerate)
    target_len = int(info.frames)
    plans: dict[int, Any] = {}

    for idx, path in enumerate(wavs, start=1):
        instrument, pan, hpf, target_rms, eq, comp, phase = mixmod.classify_track(path)
        mono, file_sr = mixmod.read_mono(path)
        if file_sr != sr:
            raise ValueError(f"{path.name}: sample rate mismatch {file_sr} != {sr}")
        metrics = mixmod.metrics_for(mono, sr, instrument=instrument)
        trim = float(np.clip(target_rms - metrics["rms_db"], -18.0, 12.0))
        threshold, ratio, attack, release = comp
        plans[idx] = mixmod.ChannelPlan(
            path=path,
            name=path.stem,
            instrument=instrument,
            pan=pan,
            hpf=hpf,
            target_rms_db=target_rms,
            trim_db=trim,
            eq_bands=list(eq),
            comp_threshold_db=threshold,
            comp_ratio=ratio,
            comp_attack_ms=attack,
            comp_release_ms=release,
            phase_invert=phase,
            event_activity=mixmod._event_activity_ranges(mono, sr, instrument) or {},
            metrics=metrics,
        )
        del mono

    return wavs, sr, target_len, plans


def make_llm_client(mixmod, ai_config: dict[str, Any], use_llm: bool):
    if not use_llm:
        return None
    return mixmod.LLMClient(
        backend=ai_config.get("llm_backend", "auto"),
        model=ai_config.get("llm_model", "gpt-5.4"),
        ollama_url=ai_config.get("ollama_url", "http://localhost:11434"),
        model_fallbacks=ai_config.get("model_fallbacks") or None,
        kimi_timeout_sec=float(ai_config.get("kimi_timeout_sec", 120)),
        kimi_cli_path=ai_config.get("kimi_cli_path") or None,
        kimi_work_dir=ai_config.get("kimi_work_dir") or None,
    )


def build_agent_states(mixmod, plans: dict[int, Any]) -> dict[int, dict[str, Any]]:
    music_bed_lufs = mixmod._music_bed_lufs(plans)
    states: dict[int, dict[str, Any]] = {}
    for channel, plan in plans.items():
        state = dict(plan.metrics)
        level_offset = plan.trim_db + plan.fader_db
        for key in ("peak_db", "true_peak_db", "rms_db", "lufs_momentary"):
            if key in state:
                state[key] = float(state[key]) + level_offset
        state.update({
            "channel_id": channel,
            "name": plan.name,
            "instrument": plan.instrument,
            "is_muted": False,
            "mix_lufs": music_bed_lufs,
            "vocal_target_delta_db": 6.0,
        })
        states[channel] = state
    return states


def run_agent_stage(mixmod, plans: dict[int, Any], ai_config: dict[str, Any], use_llm: bool):
    llm = make_llm_client(mixmod, ai_config, use_llm)
    console = mixmod.VirtualConsole(plans)
    agent = mixmod.MixingAgent(
        knowledge_base=mixmod.KnowledgeBase(
            knowledge_dir=ai_config.get("knowledge_dir") or None,
            use_vector_db=False,
            allowed_categories=mixmod.KnowledgeBase.AGENT_RUNTIME_CATEGORIES,
        ),
        rule_engine=mixmod.RuleEngine(),
        llm_client=llm,
        mixer_client=console,
        mode=mixmod.AgentMode.AUTO,
        cycle_interval=0.5,
    )
    agent.configure_safety_limits(max_fader_step_db=3.0, max_fader_db=6.0, max_eq_step_db=2.0, max_comp_threshold_step_db=3.0)
    states = build_agent_states(mixmod, plans)
    agent.update_channel_states_batch(states)
    actions = agent._prepare_actions(agent._decide({"channels": states}))
    asyncio.run(agent._act(actions))
    return console, agent, actions


def render_stage_mix(
    mixmod,
    plans: dict[int, Any],
    target_len: int,
    sr: int,
    tempo_bpm: float,
    enable_dynamic_vocal_priority: bool,
    enable_fx: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    rendered_channels: dict[int, np.ndarray] = {}
    for channel, plan in plans.items():
        if plan.muted:
            continue
        rendered_channels[channel] = mixmod.render_channel(plan.path, plan, target_len, sr)

    dynamic_vocal_priority = (
        mixmod.apply_dynamic_vocal_priority(rendered_channels, plans, sr)
        if enable_dynamic_vocal_priority
        else {"enabled": False}
    )
    fx_returns, fx_report = (
        mixmod.apply_offline_fx_plan(rendered_channels, plans, sr, tempo_bpm=tempo_bpm)
        if enable_fx
        else ({}, {"enabled": False})
    )

    mix = np.zeros((target_len, 2), dtype=np.float32)
    for rendered in rendered_channels.values():
        mix += rendered
    for rendered in fx_returns.values():
        mix += rendered

    return mix.astype(np.float32), {
        "dynamic_vocal_priority": dynamic_vocal_priority,
        "fx": fx_report,
        "pre_master_peak_dbfs": round(mixmod.amp_to_db(float(np.max(np.abs(mix))) if len(mix) else 0.0), 2),
    }


def channel_report_for_plan(plan: Any) -> dict[str, Any]:
    return {
        "file": plan.path.name,
        "instrument": plan.instrument,
        "trim_db": round(plan.trim_db, 2),
        "fader_db": round(plan.fader_db, 2),
        "pan": round(plan.pan, 3),
        "hpf": plan.hpf,
        "lpf": plan.lpf,
        "phase_invert": plan.phase_invert,
        "delay_ms": round(plan.delay_ms, 3),
        "event_expander": plan.expander_report,
        "analysis_mode": plan.metrics.get("analysis_mode"),
        "analysis_active_ratio": plan.metrics.get("analysis_active_ratio"),
        "analysis_threshold_db": plan.metrics.get("analysis_threshold_db"),
    }


def export_mp3(output_path: Path, audio: np.ndarray, sr: int):
    tmp_wav = output_path.with_suffix(".wav")
    sf.write(tmp_wav, audio, sr, subtype="PCM_24")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(tmp_wav),
        "-codec:a", "libmp3lame",
        "-b:a", "320k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    try:
        tmp_wav.unlink()
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(Path.home() / "Desktop" / "MIX"))
    parser.add_argument("--output", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT_online_soundcheck.mp3"))
    parser.add_argument("--report", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT_online_soundcheck_report.json"))
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-codex-correction-pass", action="store_true")
    parser.add_argument("--no-final-limiter", action="store_true")
    args = parser.parse_args()

    mixmod = load_offline_mix_module()
    input_dir = Path(args.input_dir)
    _, sr, target_len, initial_plans = build_initial_plans(mixmod, input_dir)
    duration_sec = target_len / sr
    schedule = build_online_schedule(duration_sec)

    config = yaml.safe_load((REPO_ROOT / "config" / "automixer.yaml").read_text(encoding="utf-8"))
    ai_config = config.get("ai", {})

    baseline_plans = clone_plans(initial_plans)

    working_plans = clone_plans(initial_plans)
    phase_report = mixmod.apply_drum_phase_alignment(working_plans, sr)
    drum_pan_rule = mixmod.apply_overhead_anchored_drum_panning(working_plans, sr)
    console, agent, actions = run_agent_stage(mixmod, working_plans, ai_config, use_llm=args.use_llm)
    gain_phase_agent_plans = clone_plans(working_plans)

    codex_corrections = {"enabled": False, "actions": []}
    if not args.no_codex_correction_pass:
        raw_codex_actions = mixmod.codex_correction_actions(working_plans)
        prepared_codex_actions = agent._prepare_actions(raw_codex_actions)
        asyncio.run(agent._act(prepared_codex_actions))
        codex_corrections = {
            "enabled": True,
            "actions": [action.__dict__ for action in prepared_codex_actions],
            "notes": [
                "Codex correction pass was applied after the rule engine to emulate online AI refinement.",
            ],
        }

    codex_bleed_control = mixmod.apply_codex_bleed_control(working_plans) if not args.no_codex_correction_pass else {"enabled": False}
    event_based_dynamics = mixmod.apply_event_based_dynamics(working_plans)
    vocal_bed_balance = mixmod.apply_vocal_bed_balance(working_plans)
    cross_adaptive_eq = mixmod.apply_cross_adaptive_eq(working_plans, target_len, sr)
    adaptive_cleanup_plans = clone_plans(working_plans)

    stage_names = ["baseline", "gain_phase_agent", "adaptive_cleanup", "live_finish"]
    stage_mixes = []
    stage_reports: dict[str, Any] = {}

    stage_mix, stage_report = render_stage_mix(
        mixmod,
        baseline_plans,
        target_len,
        sr,
        args.tempo_bpm,
        enable_dynamic_vocal_priority=False,
        enable_fx=False,
    )
    stage_mixes.append(stage_mix)
    stage_reports[stage_names[0]] = stage_report

    stage_mix, stage_report = render_stage_mix(
        mixmod,
        gain_phase_agent_plans,
        target_len,
        sr,
        args.tempo_bpm,
        enable_dynamic_vocal_priority=False,
        enable_fx=False,
    )
    stage_mixes.append(stage_mix)
    stage_reports[stage_names[1]] = stage_report

    stage_mix, stage_report = render_stage_mix(
        mixmod,
        adaptive_cleanup_plans,
        target_len,
        sr,
        args.tempo_bpm,
        enable_dynamic_vocal_priority=False,
        enable_fx=False,
    )
    stage_mixes.append(stage_mix)
    stage_reports[stage_names[2]] = stage_report

    final_stage_mix, final_stage_report = render_stage_mix(
        mixmod,
        adaptive_cleanup_plans,
        target_len,
        sr,
        args.tempo_bpm,
        enable_dynamic_vocal_priority=True,
        enable_fx=True,
    )
    stage_mixes.append(final_stage_mix)
    stage_reports[stage_names[3]] = final_stage_report

    online_mix = crossfade_stage_sequence(stage_mixes, schedule, sr)
    online_mix, master_report = mixmod.master_process(
        online_mix,
        sr,
        final_limiter=not args.no_final_limiter,
        live_peak_ceiling_db=-3.0,
    )

    output = Path(args.output)
    export_mp3(output, online_mix, sr)

    meter = pyln.Meter(sr)
    final_lufs = None
    try:
        final_lufs = float(meter.integrated_loudness(online_mix))
    except Exception:
        pass

    report = {
        "input_dir": str(input_dir),
        "output": str(output),
        "sample_rate": sr,
        "duration_sec": round(duration_sec, 3),
        "online_soundcheck": {
            "enabled": True,
            "stage_order": stage_names,
            "schedule": schedule,
            "description": (
                "The file starts from a conservative baseline and then crossfades through "
                "gain/phase/agent, adaptive cleanup, and live-finish stages to emulate "
                "online soundcheck corrections being applied during playback."
            ),
        },
        "final_peak_dbfs": round(mixmod.amp_to_db(float(np.max(np.abs(online_mix))) if len(online_mix) else 0.0), 2),
        "final_lufs": round(final_lufs, 2) if final_lufs is not None and np.isfinite(final_lufs) else None,
        "master_processing": master_report,
        "phase_alignment": phase_report,
        "drum_pan_rule": drum_pan_rule,
        "agent_actions": agent.get_action_history(100),
        "agent_audit": agent.get_action_audit_log(100),
        "virtual_console_calls": console.calls,
        "codex_corrections": codex_corrections,
        "codex_bleed_control": codex_bleed_control,
        "event_based_dynamics": event_based_dynamics,
        "vocal_bed_balance": vocal_bed_balance,
        "cross_adaptive_eq": cross_adaptive_eq,
        "stage_reports": stage_reports,
        "final_channels": [channel_report_for_plan(plan) for _, plan in sorted(adaptive_cleanup_plans.items())],
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
