import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "repa_song_soundcheck_pipeline_module",
        Path(__file__).resolve().parents[1] / "tools" / "repa_song_soundcheck_pipeline.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_output_path_uses_timestamped_current_soundcheck_name():
    mod = load_pipeline_module()

    output = mod.default_output_path(datetime(2026, 6, 26, 12, 3, 4))

    assert output.name == "REPA_Song_AUTOMIXER_SOUNDCHECK_CURRENT_20260626T120304.wav"
    assert output.parent == Path.home() / "Desktop" / "Ai MIXING"


def test_build_render_command_routes_to_channel_triggered_renderer(tmp_path):
    mod = load_pipeline_module()

    command = mod.build_render_command(
        input_dir=tmp_path / "input",
        output_path=tmp_path / "mix.wav",
        report_path=tmp_path / "report.json",
        tempo_bpm=130.0,
        genre="rock",
        reference="",
    )

    assert command[0] == sys.executable
    assert command[1].endswith("tools/channel_triggered_soundcheck_mix.py")
    assert "--input-dir" in command
    assert "--output" in command
    assert "--report" in command
    assert "--tempo-bpm" in command
    assert "--genre" in command
    assert "rock" in command


@pytest.mark.parametrize("signal", ["silence", "tone", "noise"])
def test_source_envelopes_serialize_in_cli_manifest(tmp_path, signal):
    mod = load_pipeline_module()
    sample_rate = 48000
    time = np.arange(sample_rate) / sample_rate
    audio = {
        "silence": np.zeros(sample_rate),
        "tone": 0.2 * np.sin(2 * np.pi * 60 * time),
        "noise": 0.05 * np.random.default_rng(7).standard_normal(sample_rate),
    }[signal]
    sf.write(tmp_path / "vocal.wav", audio, sample_rate)

    manifest = {"source_envelopes": mod.build_source_envelopes(tmp_path)}
    encoded = json.dumps(manifest, allow_nan=False)
    envelope = json.loads(encoded)["source_envelopes"]["vocal.wav"]

    assert type(envelope["crest_ok"]) is bool
    low, high = envelope["crest_target_db"]
    assert envelope["crest_ok"] == (low <= envelope["measured_crest_db"] <= high)
    assert envelope["crest_ok"] is (signal == "noise")


def test_write_summary_includes_soundcheck_stages_and_channels(tmp_path):
    mod = load_pipeline_module()
    output = tmp_path / "mix.wav"
    output.write_bytes(b"abc")
    report = {
        "final_lufs": -15.6,
        "final_peak_dbfs": -1.0,
        "duration_sec": 12.3,
        "sample_rate": 48000,
        "snare_pair_safety": {
            "enabled": True,
            "actions": [
                {
                    "role": "bottom",
                    "file": "Snare B.wav",
                    "before": {"trim_db": 7.0, "fader_db": 0.0},
                    "after": {"trim_db": -2.0, "fader_db": -12.0},
                }
            ],
        },
        "final_channels": [
            {
                "file": "Vocal.wav",
                "instrument": "lead_vocal",
                "trim_db": -0.3,
                "fader_db": 0.0,
                "pan": 0.0,
                "hpf": 90.0,
                "phase_invert": False,
                "delay_ms": 0.0,
                "event_expander": {"enabled": True},
            }
        ],
    }
    smart = {
        "recommendations": [
            {
                "target": "ch1:Vocal",
                "selected_strategy_id": "presence_lift",
                "method_family": "eq",
                "confidence": 0.7,
            }
        ]
    }

    summary = tmp_path / "summary.md"
    mod.write_summary(
        summary_path=summary,
        output_path=output,
        report_path=tmp_path / "report.json",
        smart_dsp_path=tmp_path / "smart.json",
        soundcheck_report=report,
        smart_dsp_report=smart,
        run_log_path=tmp_path / "run.log",
    )

    text = summary.read_text(encoding="utf-8")
    assert "channel-triggered source detection" in text
    assert "Vocal.wav" in text
    assert "Snare B.wav" in text
    assert "presence_lift" in text
