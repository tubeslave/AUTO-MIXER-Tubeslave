import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_channel_triggered_module():
    spec = importlib.util.spec_from_file_location(
        "channel_triggered_soundcheck_mix_module",
        Path(__file__).resolve().parents[1] / "tools" / "channel_triggered_soundcheck_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_channel_analysis_timeline_reaches_target_before_last_range_end():
    mod = load_channel_triggered_module()
    sr = 48_000
    ranges_report = {
        "mode": "cached_event_activity",
        "ranges": [
            (int(1.0 * sr), int(1.6 * sr)),
            (int(3.0 * sr), int(3.9 * sr)),
            (int(5.0 * sr), int(6.2 * sr)),
        ],
        "threshold_db": -18.0,
        "active_samples": int(2.7 * sr),
    }

    timeline = mod.build_channel_analysis_timeline(ranges_report, "lead_vocal", 20.0, sr)

    assert timeline["detection_mode"] == "cached_event_activity"
    assert timeline["trigger_start_sec"] == 1.0
    assert timeline["analysis_ready_sec"] > 5.0
    assert timeline["apply_start_sec"] > timeline["analysis_ready_sec"]
    assert timeline["target_active_sec"] > 0.0
    assert timeline["detected_active_sec"] == 2.7


def test_envelope_from_timeline_ramps_after_apply_start():
    mod = load_channel_triggered_module()
    sr = 10
    timeline = {
        "apply_start_sec": 2.0,
        "fade_sec": 1.0,
    }

    env = mod.envelope_from_timeline(50, sr, timeline)

    assert np.allclose(env[:20], 0.0)
    assert 0.0 < env[25] < 1.0
    assert np.allclose(env[35:], 1.0)


def test_ride_detector_ignores_single_clipped_peak():
    mod = load_channel_triggered_module()
    mixmod = mod.load_offline_mix_module()
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

    report = mod.detect_relevant_signal_ranges(mixmod, audio, sr, "ride")

    assert report["mode"] == "detector_event_activity"
    assert report["ranges"]
    first_start_sec = report["ranges"][0][0] / sr
    assert 1.7 < first_start_sec < 2.3
