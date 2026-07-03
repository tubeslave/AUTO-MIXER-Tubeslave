import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def load_online_soundcheck_module():
    spec = importlib.util.spec_from_file_location(
        "online_soundcheck_mix_module",
        Path(__file__).resolve().parents[1] / "tools" / "online_soundcheck_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_channel_triggered_soundcheck_module():
    spec = importlib.util.spec_from_file_location(
        "channel_triggered_soundcheck_mix_module",
        Path(__file__).resolve().parents[1] / "tools" / "channel_triggered_soundcheck_mix.py",
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


def test_online_soundcheck_final_limiter_is_explicit_opt_in():
    mod = load_online_soundcheck_module()
    args = SimpleNamespace(final_limiter=False, no_final_limiter=False)

    assert mod._final_limiter_enabled(args) is False

    args.final_limiter = True
    assert mod._final_limiter_enabled(args) is True

    args.no_final_limiter = True
    assert mod._final_limiter_enabled(args) is False


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


def test_channel_triggered_defaults_to_repa_song_desktop_folder():
    mod = load_channel_triggered_soundcheck_module()

    assert mod.default_repa_song_input_dir() == Path.home() / "Desktop" / "REPA Song"
    assert mod.default_repa_song_output_path().name == "REPA Song_ONLINE_SOUNDCHECK_MIX.wav"


def test_channel_triggered_mastering_flags_are_explicit_opt_in():
    mod = load_channel_triggered_soundcheck_module()
    args = SimpleNamespace(
        final_limiter=False,
        no_final_limiter=False,
        reference_mastering=False,
    )

    assert mod._final_limiter_enabled(args) is False
    assert mod._reference_mastering_enabled(args, final_limiter=False) is False

    args.final_limiter = True
    args.reference_mastering = True
    assert mod._final_limiter_enabled(args) is True
    assert mod._reference_mastering_enabled(args, final_limiter=True) is True

    args.no_final_limiter = True
    assert mod._final_limiter_enabled(args) is False
    assert mod._reference_mastering_enabled(args, final_limiter=False) is False


def test_channel_triggered_export_audio_writes_pcm_wav(tmp_path):
    mod = load_channel_triggered_soundcheck_module()
    output = tmp_path / "render.wav"
    audio = np.zeros((128, 2), dtype=np.float32)
    audio[12, 0] = 0.25

    mod.export_audio(output, audio, 48000)

    info = sf.info(output)
    assert info.samplerate == 48000
    assert info.channels == 2
    assert info.subtype == "PCM_24"


def test_channel_triggered_snare_pair_safety_tucks_bottom_mic(tmp_path):
    mod = load_channel_triggered_soundcheck_module()

    @dataclass
    class Plan:
        path: Path
        name: str
        instrument: str
        trim_db: float
        fader_db: float

    plans = {
        13: Plan(tmp_path / "SNARE T.wav", "SNARE T", "snare", 7.0, 0.0),
        14: Plan(tmp_path / "Snare B.wav", "Snare B", "snare", 7.0, 0.0),
    }

    report = mod.apply_snare_pair_mix_safety(plans)

    assert report["enabled"] is True
    assert plans[13].trim_db == 0.0
    assert plans[13].fader_db == -1.0
    assert plans[14].trim_db == -2.0
    assert plans[14].fader_db == -12.0
