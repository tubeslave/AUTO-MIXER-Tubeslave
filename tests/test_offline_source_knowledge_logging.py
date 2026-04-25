"""Tests for source-grounded candidate logging in the offline mix pass."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from source_knowledge import SourceKnowledgeLayer, iter_jsonl_events


def load_offline_agent_mix():
    spec = importlib.util.spec_from_file_location(
        "offline_agent_mix_source_logging_module",
        Path(__file__).resolve().parents[1] / "tools" / "offline_agent_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def enabled_layer(tmp_path):
    layer = SourceKnowledgeLayer({
        "source_knowledge": {
            "enabled": True,
            "log_path": str(tmp_path / "source_decisions.jsonl"),
            "min_rule_confidence": 0.0,
            "log_retrievals": False,
        }
    })
    return layer


def test_record_source_candidate_logs_rules_metrics_and_feedback(tmp_path):
    mod = load_offline_agent_mix()
    layer = enabled_layer(tmp_path)
    sr = 48000
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr // 8) / sr)).astype(np.float32)
    matches = layer.retrieve(
        "sharp vocal harshness",
        domains=["eq"],
        instruments=["lead_vocal"],
        problems=["harshness"],
        action_types=["eq_candidate"],
        limit=3,
    )

    layer.start()
    assert mod._record_source_candidate(
        layer,
        session_id="session-1",
        decision_id="decision-1",
        channel="1:Vocal.wav",
        instrument="lead_vocal",
        category="eq",
        problem="harshness",
        matches=matches,
        action={"action_type": "eq_candidate", "freq_hz": 3500.0, "gain_db": -2.0, "q": 1.2},
        before_audio=audio,
        after_audio=audio * 0.8,
        sr=sr,
        context={"stage": "unit_test"},
    )
    layer.stop()

    rows = list(iter_jsonl_events(layer.logger.path))
    decision = next(row for row in rows if row["event_type"] == "source_decision")
    feedback = next(row for row in rows if row["event_type"] == "source_feedback")

    assert decision["selected_rule_ids"]
    assert decision["source_ids"]
    assert "rms_db" in decision["before_metrics"]
    assert "rms_db" in decision["after_metrics"]
    assert decision["context"]["listening_feedback"]["listener"] == "codex"
    assert decision["selected_action"]["listening_feedback"]
    assert feedback["listener"] == "codex"
    assert feedback["rating"].startswith("codex_")


def test_trace_channel_source_candidates_logs_eq_comp_pan_without_console_calls(tmp_path):
    mod = load_offline_agent_mix()
    layer = enabled_layer(tmp_path)
    sr = 48000
    samples = np.arange(sr // 4)
    audio = (
        0.08 * np.sin(2 * np.pi * 180 * samples / sr)
        + 0.03 * np.sin(2 * np.pi * 3600 * samples / sr)
    ).astype(np.float32)
    wav_path = tmp_path / "Vocal.wav"
    sf.write(wav_path, audio, sr)
    plan = mod.ChannelPlan(
        path=wav_path,
        name="Vocal",
        instrument="lead_vocal",
        pan=-0.25,
        hpf=80.0,
        target_rms_db=-18.0,
        eq_bands=[(320.0, -2.0, 1.2)],
        comp_threshold_db=-24.0,
        comp_ratio=3.0,
        comp_attack_ms=8.0,
        comp_release_ms=90.0,
    )
    console = mod.VirtualConsole({1: plan})

    layer.start()
    logged = mod.trace_channel_source_candidates(
        layer,
        session_id="offline-session",
        channel=1,
        plan=plan,
        target_len=len(audio),
        sr=sr,
    )
    layer.stop()

    rows = list(iter_jsonl_events(layer.logger.path))
    action_types = [
        row["selected_action"]["action_type"]
        for row in rows
        if row["event_type"] == "source_decision"
    ]
    assert logged == 3
    assert {"eq_candidate", "compressor_candidate", "pan_candidate"} <= set(action_types)
    assert all(
        row["selected_rule_ids"] and row["source_ids"]
        for row in rows
        if row["event_type"] == "source_decision"
    )
    assert console.calls == []


def test_apply_offline_fx_plan_logs_fx_candidates(tmp_path):
    mod = load_offline_agent_mix()
    layer = enabled_layer(tmp_path)
    sr = 48000
    samples = np.arange(sr // 3)
    mono = (0.05 * np.sin(2 * np.pi * 440 * samples / sr)).astype(np.float32)
    stereo = mod.pan_mono(mono, 0.0)
    wav_path = tmp_path / "Vocal.wav"
    sf.write(wav_path, mono, sr)
    plan = mod.ChannelPlan(
        path=wav_path,
        name="Vocal",
        instrument="lead_vocal",
        pan=0.0,
        hpf=80.0,
        target_rms_db=-18.0,
    )

    layer.start()
    returns, report = mod.apply_offline_fx_plan(
        {1: stereo},
        {1: plan},
        sr,
        source_layer=layer,
        source_session_id="offline-session",
    )
    layer.stop()

    rows = list(iter_jsonl_events(layer.logger.path))
    fx_decisions = [
        row for row in rows
        if row["event_type"] == "source_decision"
        and row["selected_action"]["action_type"] in {"fx_candidate", "fx_send_candidate"}
    ]
    assert returns
    assert report["enabled"] is True
    assert fx_decisions
    assert all(row["selected_rule_ids"] and row["source_ids"] for row in fx_decisions)
