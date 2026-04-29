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


def test_source_rules_mert_only_plan_uses_metrics_and_disables_project_passes(tmp_path):
    mod = load_offline_agent_mix()
    layer = enabled_layer(tmp_path)
    sr = 48000
    target_len = sr
    vocal = mod.ChannelPlan(
        path=tmp_path / "Vocal.wav",
        name="Vocal",
        instrument="lead_vocal",
        pan=0.4,
        hpf=40.0,
        target_rms_db=-18.0,
        trim_db=5.0,
        fader_db=-6.0,
        phase_invert=True,
        eq_bands=[(1000.0, 4.0, 1.0)],
        comp_threshold_db=-10.0,
        comp_ratio=1.0,
        metrics={
            "rms_db": -26.0,
            "peak_db": -7.0,
            "dynamic_range_db": 19.0,
            "analysis_active_ratio": 0.72,
            "band_energy": {
                "sub": -36.0,
                "bass": -22.0,
                "low_mid": -14.0,
                "mid": -20.0,
                "presence": -28.0,
            },
        },
    )
    guitar = mod.ChannelPlan(
        path=tmp_path / "Guitar L.wav",
        name="Guitar L",
        instrument="electric_guitar",
        pan=0.0,
        hpf=40.0,
        target_rms_db=-18.0,
        metrics={
            "rms_db": -24.0,
            "peak_db": -8.0,
            "dynamic_range_db": 16.0,
            "analysis_active_ratio": 0.9,
            "band_energy": {
                "sub": -42.0,
                "bass": -25.0,
                "low_mid": -22.0,
                "mid": -18.0,
                "presence": -15.0,
            },
        },
    )

    report = mod.apply_source_rules_mert_only_plan(
        {1: vocal, 2: guitar},
        sr,
        target_len,
        layer,
        "source-only-test",
    )

    assert report["enabled"] is True
    assert report["mode"] == "source_rules_mert_only"
    assert "mixing_agent_rule_engine" in report["disabled_project_passes"]
    assert vocal.fader_db == 0.0
    assert vocal.phase_invert is False
    assert vocal.expander_enabled is False
    assert vocal.eq_bands
    assert vocal.comp_ratio > 1.0
    assert vocal.trim_analysis["mode"] == "source_rules_mert_only"
    assert guitar.pan < 0.0
    assert guitar.eq_bands


def test_source_rules_only_fx_plan_is_manual_and_filtered(tmp_path):
    mod = load_offline_agent_mix()
    plans = {
        1: mod.ChannelPlan(
            path=tmp_path / "Vocal.wav",
            name="Vocal",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
        ),
        2: mod.ChannelPlan(
            path=tmp_path / "Back Vox L.wav",
            name="Back Vox L",
            instrument="backing_vocal",
            pan=-0.3,
            hpf=110.0,
            target_rms_db=-24.0,
        ),
    }

    fx_plan = mod.build_source_rules_only_fx_plan(plans, tempo_bpm=120.0)

    assert "source_rules_mert_only" in fx_plan.notes
    assert {bus.bus_id for bus in fx_plan.buses} == {13, 14, 15, 16}
    assert all(bus.hpf_hz >= 180.0 for bus in fx_plan.buses)
    lead_sends = [send for send in fx_plan.sends if send.channel_id == 1]
    bgv_sends = [send for send in fx_plan.sends if send.channel_id == 2]
    assert {send.bus_id for send in lead_sends} == {13, 15}
    assert {send.bus_id for send in bgv_sends} == {13, 16}


def test_layer_group_corrections_balance_snare_bottom_from_summed_group(tmp_path):
    mod = load_offline_agent_mix()
    sr = 48000
    samples = np.arange(sr, dtype=np.float32)
    hit = (0.18 * np.sin(2 * np.pi * 220 * samples / sr)).astype(np.float32)
    top_audio = mod.pan_mono(hit, 0.0)
    bottom_audio = mod.pan_mono(hit * 0.9, 0.0)
    top_plan = mod.ChannelPlan(
        path=tmp_path / "SNARE T.wav",
        name="SNARE T",
        instrument="snare",
        pan=0.0,
        hpf=90.0,
        target_rms_db=-29.0,
        metrics={"rms_db": -30.0, "band_energy": {"low_mid": -14.0}},
    )
    bottom_plan = mod.ChannelPlan(
        path=tmp_path / "Snare B.wav",
        name="Snare B",
        instrument="snare",
        pan=0.0,
        hpf=90.0,
        target_rms_db=-36.0,
        metrics={"rms_db": -30.0, "band_energy": {"low_mid": -15.0}},
    )
    plans = {1: top_plan, 2: bottom_plan}
    rendered = {1: top_audio.copy(), 2: bottom_audio.copy()}

    report = mod.apply_layer_group_mix_corrections(
        rendered,
        plans,
        sr,
        base_target_rms_db=-28.0,
        band_medians={"low_mid": -20.0, "presence": -22.0},
        source_layer=None,
        source_session_id="test-layer-group",
    )

    assert report["enabled"] is True
    assert report["groups"][0]["group_id"] == "snare"
    top_db = mod._audio_rms_db(rendered[1])
    bottom_db = mod._audio_rms_db(rendered[2])
    assert abs((top_db - bottom_db) - 10.0) < 0.7
    assert any(action["type"] == "group_level" for action in report["groups"][0]["actions"])
    assert top_plan.pan == 0.0
    assert bottom_plan.pan == 0.0
    assert top_plan.trim_analysis["layer_group_id"] == "snare"
    assert bottom_plan.trim_analysis["layer_group_id"] == "snare"


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
