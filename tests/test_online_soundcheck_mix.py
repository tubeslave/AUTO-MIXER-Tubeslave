import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_online_soundcheck_module():
    spec = importlib.util.spec_from_file_location(
        "online_soundcheck_mix_module",
        Path(__file__).resolve().parents[1] / "tools" / "online_soundcheck_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_online_schedule_is_ordered_and_positive():
    mod = load_online_soundcheck_module()
    schedule = mod.build_online_schedule(148.5)

    assert len(schedule) == 3
    assert schedule[0]["to_stage"] == "gain_phase_agent"
    assert schedule[1]["to_stage"] == "adaptive_cleanup"
    assert schedule[2]["to_stage"] == "live_finish"
    assert schedule[0]["start_sec"] < schedule[1]["start_sec"] < schedule[2]["start_sec"]
    assert all(item["fade_sec"] > 0.0 for item in schedule)


def test_crossfade_stage_sequence_applies_stages_in_order():
    mod = load_online_soundcheck_module()
    sr = 10
    stage0 = np.zeros((80, 2), dtype=np.float32)
    stage1 = np.ones((80, 2), dtype=np.float32)
    stage2 = np.full((80, 2), 2.0, dtype=np.float32)
    stage3 = np.full((80, 2), 3.0, dtype=np.float32)
    schedule = [
        {"start_sec": 1.0, "fade_sec": 1.0},
        {"start_sec": 3.0, "fade_sec": 1.0},
        {"start_sec": 5.0, "fade_sec": 1.0},
    ]

    mixed = mod.crossfade_stage_sequence([stage0, stage1, stage2, stage3], schedule, sr)

    assert np.allclose(mixed[5], [0.0, 0.0], atol=1e-5)
    assert np.allclose(mixed[25], [1.0, 1.0], atol=1e-5)
    assert np.allclose(mixed[45], [2.0, 2.0], atol=1e-5)
    assert np.allclose(mixed[70], [3.0, 3.0], atol=1e-5)
