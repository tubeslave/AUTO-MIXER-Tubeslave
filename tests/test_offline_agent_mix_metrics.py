import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_offline_agent_mix():
    spec = importlib.util.spec_from_file_location(
        "offline_agent_mix_metrics_module",
        Path(__file__).resolve().parents[1] / "tools" / "offline_agent_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_based_metrics_raise_tom_rms_above_bleed_only_average():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.012 * np.sin(2.0 * np.pi * 6500.0 * t)).astype(np.float32)
    audio = bleed.copy()
    for start_sec in (1.0, 3.4, 5.8):
        start = int(start_sec * sr)
        length = int(0.16 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += (0.42 * np.sin(2.0 * np.pi * 145.0 * burst_t)).astype(np.float32) * env

    full_metrics = mod.metrics_for(audio, sr, instrument="custom")
    tom_metrics = mod.metrics_for(audio, sr, instrument="rack_tom")

    assert tom_metrics["analysis_mode"] == "event_based"
    assert tom_metrics["analysis_active_ratio"] < 0.25
    assert tom_metrics["rms_db"] > full_metrics["rms_db"] + 8.0


def test_event_based_metrics_focus_vocal_phrases_not_between_phrase_bleed():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 10.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.01 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
    audio = bleed.copy()
    for start_sec in (0.8, 3.2, 6.4):
        start = int(start_sec * sr)
        length = int(0.9 * sr)
        phrase_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        phrase = (
            0.06 * np.sin(2.0 * np.pi * 220.0 * phrase_t)
            + 0.03 * np.sin(2.0 * np.pi * 880.0 * phrase_t)
        ).astype(np.float32)
        audio[start:start + length] += phrase * env

    full_metrics = mod.metrics_for(audio, sr, instrument="custom")
    vocal_metrics = mod.metrics_for(audio, sr, instrument="lead_vocal")

    assert vocal_metrics["analysis_mode"] == "event_based"
    assert 0.15 < vocal_metrics["analysis_active_ratio"] < 0.5
    assert vocal_metrics["rms_db"] > full_metrics["rms_db"] + 4.0


def test_event_based_expander_reduces_tom_bleed_between_hits():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.015 * np.sin(2.0 * np.pi * 6200.0 * t)).astype(np.float32)
    audio = bleed.copy()
    hit_ranges = []
    for start_sec in (1.0, 3.4, 5.8):
        start = int(start_sec * sr)
        length = int(0.18 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += (0.46 * np.sin(2.0 * np.pi * 155.0 * burst_t)).astype(np.float32) * env
        hit_ranges.append((start, start + length))

    metrics = mod.metrics_for(audio, sr, instrument="rack_tom")
    plan = mod.ChannelPlan(
        path=Path("TOM.wav"),
        name="TOM",
        instrument="rack_tom",
        pan=0.0,
        hpf=65.0,
        target_rms_db=-25.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    processed, report = mod.apply_event_based_expander(audio, sr, plan)

    hit_mask = np.zeros(len(audio), dtype=bool)
    for start, end in hit_ranges:
        hit_mask[start:end] = True
    bleed_mask = ~hit_mask

    before_bleed_rms = np.sqrt(np.mean(np.square(audio[bleed_mask]))).item()
    after_bleed_rms = np.sqrt(np.mean(np.square(processed[bleed_mask]))).item()
    before_hit_rms = np.sqrt(np.mean(np.square(audio[hit_mask]))).item()
    after_hit_rms = np.sqrt(np.mean(np.square(processed[hit_mask]))).item()

    assert summary["enabled"] is True
    assert report["enabled"] is True
    assert report["mode"] == "event_based_expander"
    assert mod.amp_to_db(after_bleed_rms) < mod.amp_to_db(before_bleed_rms) - 5.0
    assert mod.amp_to_db(after_hit_rms) > mod.amp_to_db(before_hit_rms) - 1.5


def test_event_based_expander_softly_closes_vocal_mic_between_phrases():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 10.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.012 * np.sin(2.0 * np.pi * 6800.0 * t)).astype(np.float32)
    audio = bleed.copy()
    phrase_ranges = []
    for start_sec in (0.8, 3.2, 6.4):
        start = int(start_sec * sr)
        length = int(0.9 * sr)
        phrase_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        phrase = (
            0.07 * np.sin(2.0 * np.pi * 220.0 * phrase_t)
            + 0.035 * np.sin(2.0 * np.pi * 880.0 * phrase_t)
        ).astype(np.float32)
        audio[start:start + length] += phrase * env
        phrase_ranges.append((start, start + length))

    metrics = mod.metrics_for(audio, sr, instrument="lead_vocal")
    plan = mod.ChannelPlan(
        path=Path("Lead.wav"),
        name="Lead",
        instrument="lead_vocal",
        pan=0.0,
        hpf=90.0,
        target_rms_db=-20.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    processed, report = mod.apply_event_based_expander(audio, sr, plan)

    phrase_mask = np.zeros(len(audio), dtype=bool)
    for start, end in phrase_ranges:
        phrase_mask[start:end] = True
    gap_mask = ~phrase_mask

    before_gap_rms = np.sqrt(np.mean(np.square(audio[gap_mask]))).item()
    after_gap_rms = np.sqrt(np.mean(np.square(processed[gap_mask]))).item()
    before_phrase_rms = np.sqrt(np.mean(np.square(audio[phrase_mask]))).item()
    after_phrase_rms = np.sqrt(np.mean(np.square(processed[phrase_mask]))).item()

    assert summary["enabled"] is True
    assert report["enabled"] is True
    assert mod.amp_to_db(after_gap_rms) < mod.amp_to_db(before_gap_rms) - 2.5
    assert mod.amp_to_db(after_phrase_rms) > mod.amp_to_db(before_phrase_rms) - 1.2


def test_event_based_expander_uses_cached_raw_event_windows_after_trim():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.014 * np.sin(2.0 * np.pi * 6100.0 * t)).astype(np.float32)
    audio = bleed.copy()
    for start_sec in (1.2, 4.1):
        start = int(start_sec * sr)
        length = int(0.16 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += (0.35 * np.sin(2.0 * np.pi * 130.0 * burst_t)).astype(np.float32) * env

    metrics = mod.metrics_for(audio, sr, instrument="floor_tom")
    plan = mod.ChannelPlan(
        path=Path("F TOM.wav"),
        name="F TOM",
        instrument="floor_tom",
        pan=0.0,
        hpf=55.0,
        target_rms_db=-24.5,
        trim_db=-9.0,
        fader_db=-1.0,
        lpf=5800.0,
        metrics=metrics,
        event_activity=mod._event_activity_ranges(audio, sr, "floor_tom") or {},
    )
    mod.apply_event_based_dynamics({1: plan})

    processed_input = audio * mod.db_to_amp(plan.trim_db + plan.fader_db)
    processed_input = mod.highpass(processed_input, sr, plan.hpf)
    processed_input = mod.lowpass(processed_input, sr, plan.lpf)
    _, report = mod.apply_event_based_expander(processed_input, sr, plan)

    assert report["enabled"] is True
    assert report["mode"] == "event_based_expander"


def test_event_based_expander_reduces_snare_bleed_between_hits():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.014 * np.sin(2.0 * np.pi * 7600.0 * t)).astype(np.float32)
    audio = bleed.copy()
    hit_ranges = []
    for start_sec in (0.9, 2.4, 4.8, 6.6):
        start = int(start_sec * sr)
        length = int(0.15 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        tone = (
            0.17 * np.sin(2.0 * np.pi * 210.0 * burst_t)
            + 0.08 * np.sin(2.0 * np.pi * 1800.0 * burst_t)
        ).astype(np.float32)
        audio[start:start + length] += tone * env
        hit_ranges.append((start, start + length))

    metrics = mod.metrics_for(audio, sr, instrument="snare")
    plan = mod.ChannelPlan(
        path=Path("SNARE.wav"),
        name="SNARE",
        instrument="snare",
        pan=0.0,
        hpf=110.0,
        target_rms_db=-23.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    processed, report = mod.apply_event_based_expander(audio, sr, plan)

    hit_mask = np.zeros(len(audio), dtype=bool)
    for start, end in hit_ranges:
        hit_mask[start:end] = True
    bleed_mask = ~hit_mask

    before_bleed_rms = np.sqrt(np.mean(np.square(audio[bleed_mask]))).item()
    after_bleed_rms = np.sqrt(np.mean(np.square(processed[bleed_mask]))).item()
    before_hit_rms = np.sqrt(np.mean(np.square(audio[hit_mask]))).item()
    after_hit_rms = np.sqrt(np.mean(np.square(processed[hit_mask]))).item()

    assert metrics["analysis_mode"] == "event_based"
    assert summary["enabled"] is True
    assert report["enabled"] is True
    assert mod.amp_to_db(after_bleed_rms) < mod.amp_to_db(before_bleed_rms) - 3.8
    assert mod.amp_to_db(after_hit_rms) > mod.amp_to_db(before_hit_rms) - 1.2


def test_ambient_mics_do_not_get_event_based_expander():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 6.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    audio = (
        0.04 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 4200.0 * t)
    ).astype(np.float32)

    metrics = mod.metrics_for(audio, sr, instrument="overhead")
    plan = mod.ChannelPlan(
        path=Path("OH L.wav"),
        name="OH L",
        instrument="overhead",
        pan=-0.7,
        hpf=150.0,
        target_rms_db=-27.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    _, report = mod.apply_event_based_expander(audio, sr, plan)

    assert metrics["analysis_mode"] == "windowed_full_track"
    assert summary["enabled"] is False
    assert report["enabled"] is False


def test_ride_event_detection_survives_single_clipped_peak():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 6.0
    audio = np.zeros(int(sr * duration_sec), dtype=np.float32)

    audio[10] = 1.0

    for start_sec in (2.0, 2.8, 3.6):
        start = int(start_sec * sr)
        length = int(0.42 * sr)
        t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        burst = (
            0.11 * np.sin(2.0 * np.pi * 3200.0 * t)
            + 0.09 * np.sin(2.0 * np.pi * 6400.0 * t)
            + 0.06 * np.sin(2.0 * np.pi * 8800.0 * t)
        ).astype(np.float32)
        audio[start:start + length] += burst * env

    activity = mod._event_activity_ranges(audio, sr, "ride")

    assert activity is not None
    assert activity["ranges"]
    first_start_sec = activity["ranges"][0][0] / sr
    assert 1.7 < first_start_sec < 2.3
