#!/usr/bin/env python3
"""Render a channel-triggered soundcheck mix from local multitrack stems.

Each channel waits for relevant non-bleed signal, collects enough active
material for analysis, and only then crossfades into the corrected channel
state. This emulates a practical soundcheck where channels are processed
after their intended source has actually played.
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


def _first_signal_segment(
    mixmod,
    x: np.ndarray,
    sr: int,
    window_sec: float,
    detect_hpf_hz: float = 40.0,
    detect_lpf_hz: float | None = None,
) -> tuple[int, int]:
    detect = mixmod.highpass(x, sr, detect_hpf_hz)
    if detect_lpf_hz and detect_lpf_hz > 0.0:
        detect = mixmod.lowpass(detect, sr, detect_lpf_hz)

    frame = max(256, int(0.24 * sr))
    hop = max(64, int(0.06 * sr))
    starts, rms_db = mixmod._frame_rms_db(detect, frame, hop)
    if len(rms_db) == 0:
        end = min(len(x), max(1, int(window_sec * sr)))
        return 0, end

    peak_db = mixmod.amp_to_db(float(np.max(np.abs(detect))) if len(detect) else 0.0)
    floor_db = float(np.percentile(rms_db, 45))
    threshold_db = max(float(np.percentile(rms_db, 68)), floor_db + 4.0, peak_db - 22.0, -52.0)
    active_idx = np.flatnonzero(rms_db >= threshold_db)
    if len(active_idx) == 0:
        end = min(len(x), max(1, int(window_sec * sr)))
        return 0, end

    pad = int(0.12 * sr)
    start = max(0, int(starts[int(active_idx[0])]) - pad)
    end = min(len(x), start + max(1, int(window_sec * sr)))
    return start, end


def _collect_analysis_audio(
    mixmod,
    x: np.ndarray,
    sr: int,
    instrument: str,
    ranges_report: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    duration_sec = len(x) / sr if sr else 0.0
    target_active_sec = analysis_target_active_sec(instrument, duration_sec)
    target_active_samples = max(1, int(round(target_active_sec * sr)))
    ranges = list(ranges_report.get("ranges", []))

    if not ranges:
        start, end = _first_signal_segment(
            mixmod,
            x,
            sr,
            window_sec=max(1.0, target_active_sec),
            detect_hpf_hz=40.0,
            detect_lpf_hz=8000.0,
        )
        block = x[start:end]
        return block, {
            "analysis_audio_mode": "first_signal_fallback",
            "analysis_audio_start_sec": round(start / sr, 3) if sr else 0.0,
            "analysis_audio_end_sec": round(end / sr, 3) if sr else 0.0,
            "analysis_audio_sec": round(len(block) / sr, 3) if sr else 0.0,
            "analysis_target_active_sec": round(target_active_sec, 3),
        }

    chunks: list[np.ndarray] = []
    collected = 0
    analysis_end = int(ranges[0][0])
    for start, end in ranges:
        remaining = target_active_samples - collected
        if remaining <= 0:
            break
        seg_end = min(int(end), int(start) + remaining)
        if seg_end <= int(start):
            continue
        chunks.append(x[int(start):seg_end])
        collected += seg_end - int(start)
        analysis_end = seg_end

    if not chunks:
        start, end = ranges[0]
        block = x[int(start):int(end)]
        analysis_start = int(start)
        analysis_end = int(end)
    else:
        block = np.concatenate(chunks, axis=0)
        analysis_start = int(ranges[0][0])

    return block, {
        "analysis_audio_mode": "accumulated_relevant_signal",
        "analysis_audio_start_sec": round(analysis_start / sr, 3) if sr else 0.0,
        "analysis_audio_end_sec": round(analysis_end / sr, 3) if sr else 0.0,
        "analysis_audio_sec": round(len(block) / sr, 3) if sr else 0.0,
        "analysis_target_active_sec": round(target_active_sec, 3),
    }


def build_initial_plans(mixmod, input_dir: Path) -> tuple[int, int, dict[int, Any]]:
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
        ranges_report = detect_relevant_signal_ranges(
            mixmod,
            mono,
            sr,
            instrument,
            cached_event_activity=mixmod._event_activity_ranges(mono, sr, instrument) or {},
        )
        analysis_audio, analysis_meta = _collect_analysis_audio(mixmod, mono, sr, instrument, ranges_report)
        full_track_metrics = mixmod.metrics_for(mono, sr, instrument=instrument)
        analysis_metrics = mixmod.metrics_for(analysis_audio, sr, instrument=instrument)
        trim = float(np.clip(target_rms - analysis_metrics["rms_db"], -18.0, 12.0))
        metrics = dict(full_track_metrics)
        for key in ("rms_db", "lufs_momentary", "dynamic_range_db", "band_energy", "channel_armed", "needs_attention"):
            metrics[key] = analysis_metrics[key]
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
            metrics={**metrics, **analysis_meta},
        )
        del mono

    return sr, target_len, plans


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


def render_channels(mixmod, plans: dict[int, Any], target_len: int, sr: int) -> dict[int, np.ndarray]:
    rendered: dict[int, np.ndarray] = {}
    for channel, plan in plans.items():
        if plan.muted:
            continue
        rendered[channel] = mixmod.render_channel(plan.path, plan, target_len, sr)
    return rendered


def detector_profile_for_instrument(instrument: str) -> dict[str, float] | None:
    if instrument == "bass_guitar":
        return {
            "frame_ms": 320.0,
            "hop_ms": 80.0,
            "pad_ms": 180.0,
            "detect_hpf_hz": 35.0,
            "detect_lpf_hz": 260.0,
            "percentile": 78.0,
            "peak_offset_db": 18.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -46.0,
        }
    if instrument in {"electric_guitar", "acoustic_guitar", "accordion", "custom"}:
        return {
            "frame_ms": 320.0,
            "hop_ms": 80.0,
            "pad_ms": 180.0,
            "detect_hpf_hz": 110.0,
            "detect_lpf_hz": 4800.0,
            "percentile": 76.0,
            "peak_offset_db": 18.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -46.0,
        }
    if instrument in {"playback", "synth"}:
        return {
            "frame_ms": 420.0,
            "hop_ms": 100.0,
            "pad_ms": 220.0,
            "detect_hpf_hz": 40.0,
            "detect_lpf_hz": 10000.0,
            "percentile": 72.0,
            "peak_offset_db": 20.0,
            "floor_margin_db": 5.0,
            "min_threshold_db": -48.0,
        }
    if instrument == "ride":
        return {
            "frame_ms": 170.0,
            "hop_ms": 40.0,
            "pad_ms": 120.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 94.0,
            "peak_offset_db": 18.0,
            "peak_percentile": 99.9,
            "floor_margin_db": 5.0,
            "min_threshold_db": -50.0,
        }
    if instrument == "overhead":
        return {
            "frame_ms": 220.0,
            "hop_ms": 60.0,
            "pad_ms": 140.0,
            "detect_hpf_hz": 55.0,
            "detect_lpf_hz": 9000.0,
            "percentile": 84.0,
            "peak_offset_db": 17.0,
            "floor_margin_db": 5.0,
            "min_threshold_db": -44.0,
        }
    if instrument == "room":
        return {
            "frame_ms": 360.0,
            "hop_ms": 90.0,
            "pad_ms": 220.0,
            "detect_hpf_hz": 60.0,
            "detect_lpf_hz": 6500.0,
            "percentile": 80.0,
            "peak_offset_db": 18.0,
            "floor_margin_db": 5.0,
            "min_threshold_db": -46.0,
        }
    return None


def detect_relevant_signal_ranges(mixmod, x: np.ndarray, sr: int, instrument: str, cached_event_activity: dict[str, Any] | None = None) -> dict[str, Any]:
    if cached_event_activity and cached_event_activity.get("ranges"):
        active_samples = int(cached_event_activity.get("active_samples", sum(end - start for start, end in cached_event_activity["ranges"])))
        return {
            "mode": "cached_event_activity",
            "ranges": list(cached_event_activity["ranges"]),
            "threshold_db": cached_event_activity.get("threshold_db"),
            "active_samples": active_samples,
        }

    profile = detector_profile_for_instrument(instrument)
    if not profile:
        window_sec = 5.0 if instrument not in {"kick", "snare"} else 2.0
        start, end = _first_signal_segment(
            mixmod,
            x,
            sr,
            window_sec=window_sec,
            detect_hpf_hz=40.0,
            detect_lpf_hz=8000.0,
        )
        return {
            "mode": "fallback_active_segment",
            "ranges": [(start, end)] if end > start else [],
            "threshold_db": None,
            "active_samples": max(0, end - start),
        }

    detect = mixmod.highpass(x, sr, profile["detect_hpf_hz"])
    detect = mixmod.lowpass(detect, sr, profile["detect_lpf_hz"])
    frame = max(256, int(profile["frame_ms"] * sr / 1000.0))
    hop = max(64, int(profile["hop_ms"] * sr / 1000.0))
    starts, rms_db = mixmod._frame_rms_db(detect, frame, hop)
    if len(rms_db) == 0:
        start, end = _first_signal_segment(
            mixmod,
            x,
            sr,
            window_sec=4.0,
            detect_hpf_hz=profile["detect_hpf_hz"],
            detect_lpf_hz=profile["detect_lpf_hz"],
        )
        return {
            "mode": "fallback_active_segment",
            "ranges": [(start, end)] if end > start else [],
            "threshold_db": None,
            "active_samples": max(0, end - start),
        }

    peak_percentile = float(profile.get("peak_percentile", 100.0))
    if len(detect):
        if peak_percentile >= 100.0:
            detect_peak = float(np.max(np.abs(detect)))
        else:
            detect_peak = float(np.percentile(np.abs(detect), peak_percentile))
    else:
        detect_peak = 0.0
    detect_peak_db = mixmod.amp_to_db(detect_peak)
    noise_floor_db = float(np.percentile(rms_db, 50))
    threshold_db = max(
        float(np.percentile(rms_db, profile["percentile"])),
        detect_peak_db - profile["peak_offset_db"],
        noise_floor_db + profile["floor_margin_db"],
        profile["min_threshold_db"],
    )
    active_idx = np.flatnonzero(rms_db >= threshold_db)
    if len(active_idx) == 0:
        start, end = _first_signal_segment(
            mixmod,
            x,
            sr,
            window_sec=4.0,
            detect_hpf_hz=profile["detect_hpf_hz"],
            detect_lpf_hz=profile["detect_lpf_hz"],
        )
        return {
            "mode": "fallback_active_segment",
            "ranges": [(start, end)] if end > start else [],
            "threshold_db": round(threshold_db, 2),
            "active_samples": max(0, end - start),
        }

    pad = int(profile["pad_ms"] * sr / 1000.0)
    ranges = []
    for idx in active_idx:
        start = max(0, int(starts[idx]) - pad)
        end = min(len(x), int(starts[idx]) + frame + pad)
        if end > start:
            ranges.append((start, end))
    merged = mixmod._merge_ranges(ranges, gap=pad // 2)
    return {
        "mode": "detector_event_activity",
        "ranges": merged,
        "threshold_db": round(threshold_db, 2),
        "active_samples": sum(end - start for start, end in merged),
    }


def analysis_target_active_sec(instrument: str, duration_sec: float) -> float:
    defaults = {
        "lead_vocal": 1.6,
        "backing_vocal": 1.4,
        "kick": 0.18,
        "snare": 0.3,
        "rack_tom": 0.22,
        "floor_tom": 0.28,
        "hi_hat": 1.2,
        "ride": 1.2,
        "percussion": 1.0,
        "bass_guitar": 2.2,
        "electric_guitar": 2.0,
        "acoustic_guitar": 2.0,
        "accordion": 2.0,
        "playback": 1.6,
        "synth": 1.6,
        "overhead": 1.4,
        "room": 1.8,
        "custom": 1.8,
    }
    default = defaults.get(instrument, 1.8)
    return float(min(default, max(0.2, duration_sec * 0.08)))


def adaptive_analysis_target_active_sec(
    instrument: str,
    base_target_sec: float,
    detected_active_sec: float,
) -> float:
    if detected_active_sec <= 0.0:
        return float(base_target_sec)

    if instrument in {"lead_vocal", "backing_vocal"}:
        return float(min(base_target_sec, max(0.9, min(1.6, detected_active_sec * 0.45))))
    if instrument == "kick":
        return float(min(base_target_sec, max(0.12, min(0.18, detected_active_sec * 0.35))))
    if instrument in {"snare", "rack_tom", "floor_tom"}:
        return float(min(base_target_sec, max(0.16, min(0.24, detected_active_sec * 0.35))))
    return float(base_target_sec)


def analysis_compute_latency_sec(instrument: str) -> float:
    if instrument in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
        return 0.45
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.65
    return 0.75


def correction_fade_sec(instrument: str) -> float:
    if instrument in {"kick", "snare", "rack_tom", "floor_tom"}:
        return 1.2
    if instrument in {"hi_hat", "ride", "percussion"}:
        return 1.5
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 1.8
    return 2.2


def build_channel_analysis_timeline(ranges_report: dict[str, Any], instrument: str, duration_sec: float, sr: int) -> dict[str, Any]:
    ranges = list(ranges_report.get("ranges", []))
    base_target_active_sec = analysis_target_active_sec(instrument, duration_sec)
    compute_latency_sec = analysis_compute_latency_sec(instrument)
    fade_sec = correction_fade_sec(instrument)
    detected_active_sec = float(ranges_report.get("active_samples", 0)) / sr
    target_active_sec = adaptive_analysis_target_active_sec(
        instrument,
        base_target_active_sec,
        detected_active_sec,
    )

    if not ranges:
        apply_sec = min(duration_sec - 0.25, max(0.0, compute_latency_sec))
        return {
            "detection_mode": ranges_report.get("mode", "none"),
            "trigger_start_sec": None,
            "analysis_ready_sec": None,
            "apply_start_sec": round(apply_sec, 3),
            "fade_sec": round(fade_sec, 3),
            "target_active_sec": round(target_active_sec, 3),
            "base_target_active_sec": round(base_target_active_sec, 3),
            "detected_active_sec": 0.0,
            "threshold_db": ranges_report.get("threshold_db"),
            "ranges_count": 0,
            "used_fallback": True,
        }

    active_accum_sec = 0.0
    ready_sample = int(ranges[-1][1])
    for start, end in ranges:
        event_sec = max(0.0, float(end - start) / sr)
        active_accum_sec += event_sec
        if active_accum_sec >= target_active_sec:
            over_sec = active_accum_sec - target_active_sec
            ready_sample = max(int(start), int(end - (over_sec * sr)))
            break

    trigger_start_sec = float(ranges[0][0]) / sr
    ready_sec = float(ready_sample) / sr
    return {
        "detection_mode": ranges_report.get("mode", "unknown"),
        "trigger_start_sec": round(trigger_start_sec, 3),
        "analysis_ready_sec": round(ready_sec, 3),
        "apply_start_sec": round(min(duration_sec, ready_sec + compute_latency_sec), 3),
        "fade_sec": round(fade_sec, 3),
        "target_active_sec": round(target_active_sec, 3),
        "base_target_active_sec": round(base_target_active_sec, 3),
        "detected_active_sec": round(detected_active_sec, 3),
        "threshold_db": ranges_report.get("threshold_db"),
        "ranges_count": len(ranges),
        "used_fallback": ranges_report.get("mode", "").startswith("fallback"),
    }


def envelope_from_timeline(target_len: int, sr: int, timeline: dict[str, Any]) -> np.ndarray:
    apply_start = float(timeline.get("apply_start_sec") or 0.0)
    fade_sec = max(0.05, float(timeline.get("fade_sec") or 0.0))
    start = int(apply_start * sr)
    fade = max(1, int(fade_sec * sr))
    env = np.zeros(target_len, dtype=np.float32)
    if start >= target_len:
        return env
    end = min(target_len, start + fade)
    if end > start:
        env[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False, dtype=np.float32)
    env[end:] = 1.0
    return env


def blend_rendered_channels(
    baseline_channels: dict[int, np.ndarray],
    final_channels: dict[int, np.ndarray],
    timelines: dict[int, dict[str, Any]],
    target_len: int,
    sr: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    blended: dict[int, np.ndarray] = {}
    envelope_reports: dict[int, dict[str, Any]] = {}
    for channel, baseline in baseline_channels.items():
        final = final_channels[channel]
        timeline = timelines[channel]
        env = envelope_from_timeline(target_len, sr, timeline)[:, None]
        blended[channel] = (baseline * (1.0 - env) + final * env).astype(np.float32)
        envelope_reports[channel] = {
            **timeline,
            "envelope_mode": "channel_triggered_correction",
        }
    return blended, envelope_reports


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
    parser.add_argument("--output", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT_channel_triggered_soundcheck.mp3"))
    parser.add_argument("--report", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT_channel_triggered_soundcheck_report.json"))
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-codex-correction-pass", action="store_true")
    parser.add_argument("--no-final-limiter", action="store_true")
    parser.add_argument("--reference", default="", help="Path to an external reference track or saved style preset JSON")
    parser.add_argument("--genre", default="", help="Optional genre focus for bounded mix voicing, for example 'rock'")
    args = parser.parse_args()

    mixmod = load_offline_mix_module()
    input_dir = Path(args.input_dir)
    sr, target_len, initial_plans = build_initial_plans(mixmod, input_dir)
    duration_sec = target_len / sr

    config = yaml.safe_load((REPO_ROOT / "config" / "automixer.yaml").read_text(encoding="utf-8"))
    ai_config = config.get("ai", {})
    autofoh_config = config.get("autofoh", {})
    reference_context = mixmod.prepare_reference_mix_context(args.reference)

    baseline_plans = clone_plans(initial_plans)
    working_plans = clone_plans(initial_plans)

    phase_report = mixmod.apply_drum_phase_alignment(working_plans, sr)
    drum_pan_rule = mixmod.apply_overhead_anchored_drum_panning(working_plans, sr)
    console, agent, actions = run_agent_stage(mixmod, working_plans, ai_config, use_llm=args.use_llm)

    codex_corrections = {
        "enabled": False,
        "actions": [],
        "requested_disable_flag": bool(args.no_codex_correction_pass),
        "notes": [
            "Legacy codex heuristic corrections are disabled in analyzer-only mode.",
        ],
    }
    codex_bleed_control = {
        "enabled": False,
        "requested_disable_flag": bool(args.no_codex_correction_pass),
        "notes": [
            "Legacy codex bleed-control heuristics are disabled in analyzer-only mode.",
        ],
    }
    event_based_dynamics = mixmod.apply_event_based_dynamics(working_plans)
    autofoh_analyzer_pass = (
        {"enabled": False, "notes": ["AutoFOH analyzer pass disabled via legacy --no-codex-correction-pass flag."]}
        if args.no_codex_correction_pass
        else mixmod.apply_autofoh_analyzer_pass(working_plans, target_len, sr, autofoh_config)
    )
    vocal_bed_balance = {
        "enabled": False,
        "notes": [
            "Static vocal bed attenuation is disabled.",
            "Vocal space must come from priority EQ and measured analyzer corrections.",
        ],
    }
    cross_adaptive_eq = mixmod.apply_cross_adaptive_eq(working_plans, target_len, sr)
    reference_mix_guidance = mixmod.apply_reference_mix_guidance(working_plans, sr, reference_context)
    genre_mix_profile = mixmod.apply_genre_mix_profile(working_plans, args.genre)
    kick_bass_hierarchy = mixmod.apply_kick_bass_hierarchy(
        working_plans,
        target_len,
        sr,
        desired_kick_advantage_db=1.8 if str(args.genre).strip().lower() == "rock" else 1.5,
    )
    stem_mix_verification = mixmod.apply_stem_mix_verification(
        working_plans,
        target_len,
        sr,
        genre=args.genre,
    )
    final_plans = clone_plans(working_plans)

    baseline_rendered = render_channels(mixmod, baseline_plans, target_len, sr)
    final_rendered = render_channels(mixmod, final_plans, target_len, sr)

    timelines: dict[int, dict[str, Any]] = {}
    channel_analysis_reports = []
    for channel, plan in final_plans.items():
        mono, file_sr = mixmod.read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        ranges_report = detect_relevant_signal_ranges(
            mixmod,
            mono,
            sr,
            plan.instrument,
            cached_event_activity=plan.event_activity,
        )
        timeline = build_channel_analysis_timeline(ranges_report, plan.instrument, duration_sec, sr)
        timelines[channel] = timeline
        channel_analysis_reports.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            **timeline,
        })

    blended_channels, envelope_reports = blend_rendered_channels(
        baseline_rendered,
        final_rendered,
        timelines,
        target_len,
        sr,
    )

    dynamic_vocal_priority = {
        "enabled": False,
        "notes": [
            "Vocal ducking is disabled.",
            "Vocal space must be created by priority EQ and measured analyzer corrections.",
        ],
    }
    fx_returns, fx_report = mixmod.apply_offline_fx_plan(blended_channels, final_plans, sr, tempo_bpm=args.tempo_bpm)

    mix = np.zeros((target_len, 2), dtype=np.float32)
    for rendered in blended_channels.values():
        mix += rendered
    for rendered in fx_returns.values():
        mix += rendered

    mix, master_report = mixmod.master_process(
        mix,
        sr,
        final_limiter=not args.no_final_limiter,
        live_peak_ceiling_db=-3.0,
        reference_context=reference_context,
    )

    output = Path(args.output)
    export_mp3(output, mix, sr)

    meter = pyln.Meter(sr)
    final_lufs = None
    try:
        final_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass

    report = {
        "input_dir": str(input_dir),
        "output": str(output),
        "reference": str(reference_context.path) if reference_context is not None else "",
        "reference_sources": [str(path) for path in reference_context.source_paths] if reference_context is not None else [],
        "genre": str(args.genre or ""),
        "sample_rate": sr,
        "duration_sec": round(duration_sec, 3),
        "channel_triggered_soundcheck": {
            "enabled": True,
            "description": (
                "Each channel waits for relevant signal instead of bleed, accumulates enough "
                "active dry material for analysis, then crossfades from baseline to corrected state."
            ),
            "channels_with_triggered_analysis": len(channel_analysis_reports),
        },
        "final_peak_dbfs": round(mixmod.amp_to_db(float(np.max(np.abs(mix))) if len(mix) else 0.0), 2),
        "final_lufs": round(final_lufs, 2) if final_lufs is not None and np.isfinite(final_lufs) else None,
        "master_processing": master_report,
        "phase_alignment": phase_report,
        "drum_pan_rule": drum_pan_rule,
        "agent_actions": agent.get_action_history(100),
        "agent_audit": agent.get_action_audit_log(100),
        "virtual_console_calls": console.calls,
        "autofoh_analyzer_pass": autofoh_analyzer_pass,
        "codex_corrections": codex_corrections,
        "codex_bleed_control": codex_bleed_control,
        "event_based_dynamics": event_based_dynamics,
        "vocal_bed_balance": vocal_bed_balance,
        "cross_adaptive_eq": cross_adaptive_eq,
        "reference_mix_guidance": reference_mix_guidance,
        "genre_mix_profile": genre_mix_profile,
        "kick_bass_hierarchy": kick_bass_hierarchy,
        "stem_mix_verification": stem_mix_verification,
        "dynamic_vocal_priority": dynamic_vocal_priority,
        "fx": fx_report,
        "channel_analysis": channel_analysis_reports,
        "channel_envelopes": envelope_reports,
        "final_channels": [channel_report_for_plan(plan) for _, plan in sorted(final_plans.items())],
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
