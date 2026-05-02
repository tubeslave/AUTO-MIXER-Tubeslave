from __future__ import annotations

import numpy as np
import pytest

from automixer.production_mix_v1.ayaic_offline_pipeline import (
    DEFAULT_MASTER_TARGET_LUFS,
    AyaicStem,
    CORRECTIVE_EQ_METHODS,
    DEFAULT_CORRECTIVE_EQ_METHOD,
    _analyze_bleed_activity,
    _autoeq_instrument,
    _bleed_analysis_instrument,
    _apply_corrective_eq,
    _apply_hpf_lpf_correction,
    _apply_phase_delay_from_overheads,
    _apply_group_levels,
    _apply_pair_levels,
    _apply_track_levels,
    _ayaic_bus_target_lufs,
    _corrective_eq_instrument,
    _drum_instrument,
    _instrument_group,
    _lufs,
    _lufs_method,
    _offline_hpf_lpf_plan,
    _project_channel_role,
    _normalize_corrective_eq_method,
    _sum_audio,
    _track_target_lufs,
    run_ayaic_offline_pipeline,
)


def test_track_targets_include_ayaic_summed_stem_rules():
    assert _track_target_lufs("SNARE T.wav") == -26.0
    assert _track_target_lufs("Snare B.wav") == -35.0
    assert _track_target_lufs("OH L.wav") == -38.0
    assert _track_target_lufs("Guitar L.wav") == -28.0
    assert _track_target_lufs("Playback R.wav") == -28.0
    assert _track_target_lufs("Accordion.wav") == -28.0
    assert _track_target_lufs("BACKS L.wav") == -28.0
    assert _track_target_lufs("ROOM DR L.wav") == -45.0


def test_master_target_and_loudness_method_are_integrated_lufs():
    assert DEFAULT_MASTER_TARGET_LUFS == -20.0
    assert "Integrated LUFS" in _lufs_method()


def test_group_targets_use_ayaic_bus_level_planes():
    assert _ayaic_bus_target_lufs("vocals", [_stem("Vocal", np.ones(1024), 48_000, 1)]) == -22.0
    assert _ayaic_bus_target_lufs("back_vocals", [_stem("Back Vox L", np.ones(1024), 48_000, 1)]) == -28.0
    assert _ayaic_bus_target_lufs("drums", [_stem("KICK", np.ones(1024), 48_000, 1)]) == -25.0
    assert _ayaic_bus_target_lufs("bass", [_stem("Bass", np.ones(1024), 48_000, 1)]) == -25.0
    assert _ayaic_bus_target_lufs("playback", [_stem("Playback L", np.ones(1024), 48_000, 1)]) == -25.0
    assert _ayaic_bus_target_lufs("accordion", [_stem("Accordion", np.ones(1024), 48_000, 1)]) == -25.0
    assert _ayaic_bus_target_lufs("guitars", [_stem("Guitar L", np.ones(1024), 48_000, 1)]) == -25.0


def test_instrument_group_uses_user_defined_bus_layout():
    assert _instrument_group("OH L.wav") == "drums"
    assert _instrument_group("KICK.wav") == "drums"
    assert _instrument_group("Bass.wav") == "bass"
    assert _instrument_group("Playback R.wav") == "playback"
    assert _instrument_group("Accordion.wav") == "accordion"
    assert _instrument_group("Guitar L.wav") == "guitars"
    assert _instrument_group("Vocal.wav") == "vocals"
    assert _instrument_group("Back Vox L.wav") == "back_vocals"
    assert _instrument_group("BACKS L.wav") == "back_vocals"
    assert _instrument_group("ROOM DR L.wav") == "drums"


def test_mix_folder_names_map_to_expected_ayaic_profiles():
    backs_plan = _offline_hpf_lpf_plan("BACKS L.wav")
    room_plan = _offline_hpf_lpf_plan("ROOM DR L.wav")

    assert backs_plan["instrument"] == "backing_vocal"
    assert backs_plan["hpf_hz"] == 110.0
    assert backs_plan["lpf_hz"] == 9500.0
    assert _corrective_eq_instrument("BACKS L.wav") == "backing_vocal"
    assert _autoeq_instrument("BACKS L.wav") == "back_vocal"
    assert _project_channel_role("BACKS L.wav") == "back_vocal"
    assert _bleed_analysis_instrument("BACKS L.wav") == "backing_vocal"

    assert _drum_instrument("ROOM DR L.wav") == "room"
    assert room_plan["instrument"] == "room"
    assert room_plan["hpf_hz"] == 80.0
    assert room_plan["lpf_hz"] == 6500.0
    assert _corrective_eq_instrument("ROOM DR L.wav") == "room"
    assert _autoeq_instrument("ROOM DR L.wav") == "room"
    assert _project_channel_role("ROOM DR L.wav") == "room"


def test_group_level_correction_reaches_target_without_gain_clip():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    quiet = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32) * 0.001
    stems = [
        _stem("KICK", quiet, sample_rate, 1),
        _stem("SNARE T", quiet, sample_rate, 2),
    ]
    for stem in stems:
        stem.group = "drums"

    report = _apply_group_levels(stems)
    summed = _sum_audio([stem.audio for stem in stems])

    assert report[0]["target_lufs"] == -25.0
    assert report[0]["gain_db"] > 3.0
    assert _lufs(summed) == pytest.approx(-25.0, abs=0.01)


def test_track_levels_measure_primary_signal_windows_before_bleed_floor():
    sample_rate = 48_000
    length = sample_rate * 4
    time = np.arange(length, dtype=np.float32) / sample_rate
    bleed = np.sin(2.0 * np.pi * 900.0 * time).astype(np.float32) * 0.004
    snare = bleed.copy()
    snare[sample_rate:sample_rate + sample_rate // 4] += 0.12 * np.sin(
        2.0 * np.pi * 900.0 * time[: sample_rate // 4]
    ).astype(np.float32)
    snare[sample_rate * 2:sample_rate * 2 + sample_rate // 4] += 0.12 * np.sin(
        2.0 * np.pi * 900.0 * time[: sample_rate // 4]
    ).astype(np.float32)
    stem = _stem("SNARE T", snare, sample_rate, 1)
    stem.target_lufs = -26.0
    stems = [stem]

    bleed_report = _analyze_bleed_activity(stems)
    track_report = _apply_track_levels(stems)

    assert bleed_report[0]["analysis_mode"] == "event_based_primary_signal"
    assert bleed_report[0]["analysis_active_ratio"] < 0.5
    assert track_report[0]["measurement_scope"] == "event_based_primary_signal"
    assert track_report[0]["post_lufs"] == pytest.approx(-26.0, abs=0.02)
    assert track_report[0]["post_full_track_lufs"] < -26.0
    assert not any(item.get("type") == "summed_stem_validation" for item in track_report)


def test_pair_levels_are_separate_from_channel_levels():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32) * 0.04
    right = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32) * 0.02
    stems = [
        _stem("Guitar L", left, sample_rate, 1),
        _stem("Guitar R", right, sample_rate, 2),
    ]

    channel_report = _apply_track_levels(stems)
    pair_report = _apply_pair_levels(stems)

    assert len(channel_report) == 2
    assert pair_report[0]["type"] == "summed_stem_validation"
    assert pair_report[0]["stem"] == "Guitar"
    assert pair_report[0]["post_sum_lufs"] == pytest.approx(-25.0, abs=0.02)


def test_pair_lufs_uses_drums_lead_vocal_and_backs_buses():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32) * 0.04
    stems = [
        _stem("KICK", sine, sample_rate, 1),
        _stem("SNARE", sine * 0.9, sample_rate, 2),
        _stem("OH L", sine * 0.6, sample_rate, 3),
        _stem("OH R", sine * 0.6, sample_rate, 4),
        _stem("Hi-Hat", sine * 0.3, sample_rate, 5),
        _stem("Ride", sine * 0.25, sample_rate, 6),
        _stem("ROOM DR L", sine * 0.1, sample_rate, 7),
        _stem("ROOM DR R", sine * 0.1, sample_rate, 8),
        _stem("Katya VOX", sine * 0.5, sample_rate, 9),
        _stem("Sergey Vox", sine * 0.4, sample_rate, 10),
        _stem("Slava Vox", sine * 0.35, sample_rate, 11),
        _stem("BACKS L", sine * 0.25, sample_rate, 12),
        _stem("BACKS R", sine * 0.25, sample_rate, 13),
    ]

    report = _apply_pair_levels(stems)
    by_stem = {item["stem"]: item for item in report}

    assert "Overheads" not in by_stem
    assert "Snare" not in by_stem
    assert by_stem["Drums Bus"]["target_lufs"] == -25.0
    assert set(by_stem["Drums Bus"]["members"]) == {
        "KICK.wav",
        "SNARE.wav",
        "OH L.wav",
        "OH R.wav",
        "Hi-Hat.wav",
        "Ride.wav",
        "ROOM DR L.wav",
        "ROOM DR R.wav",
    }
    assert by_stem["Drums Bus"]["post_sum_lufs"] == pytest.approx(-25.0, abs=0.02)
    assert by_stem["Lead Vocal Bus"]["target_lufs"] == -22.0
    assert set(by_stem["Lead Vocal Bus"]["members"]) == {"Katya VOX.wav", "Sergey Vox.wav", "Slava Vox.wav"}
    assert by_stem["Lead Vocal Bus"]["post_sum_lufs"] == pytest.approx(-22.0, abs=0.02)
    assert by_stem["Backs Bus"]["target_lufs"] == -28.0
    assert set(by_stem["Backs Bus"]["members"]) == {"BACKS L.wav", "BACKS R.wav"}
    assert by_stem["Backs Bus"]["post_sum_lufs"] == pytest.approx(-28.0, abs=0.02)


def test_hpf_lpf_correction_uses_offline_filter_targets():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    kick = (
        0.08 * np.sin(2.0 * np.pi * 50.0 * time)
        + 0.02 * np.sin(2.0 * np.pi * 11_000.0 * time)
    ).astype(np.float32)
    stem = _stem("KICK", kick, sample_rate, 1)
    stems = [stem]

    assert _offline_hpf_lpf_plan("KICK.wav") == {"instrument": "kick", "hpf_hz": 35.0, "lpf_hz": 6800.0}
    report = _apply_hpf_lpf_correction(stems)

    assert report[0]["hpf_hz"] == 35.0
    assert report[0]["lpf_hz"] == 6800.0
    assert report[0]["source"]["source_path"] == "tools/offline_agent_mix.py"
    assert report[0]["post_full_track_lufs"] < report[0]["pre_full_track_lufs"]


def test_corrective_eq_methods_are_ported_from_offline_pipeline():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    snare = (
        0.05 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.02 * np.sin(2.0 * np.pi * 6500.0 * time)
    ).astype(np.float32)
    stem = _stem("SNARE T", snare, sample_rate, 1)

    assert CORRECTIVE_EQ_METHODS == (
        "none",
        "profile",
        "bleed_control",
        "cross_adaptive",
        "frequency_window",
        "frequency_cross_profile",
        "project_corrective",
        "project_corrective_hybrid",
        "contextual_deep_eq",
    )
    assert _normalize_corrective_eq_method("classify-track") == "profile"
    assert _normalize_corrective_eq_method("4-3-1") == "frequency_cross_profile"
    assert _normalize_corrective_eq_method("project-corrective-eq") == "project_corrective"
    assert _normalize_corrective_eq_method("project-hybrid") == "project_corrective_hybrid"
    assert _normalize_corrective_eq_method("deep-eq") == "contextual_deep_eq"
    assert DEFAULT_CORRECTIVE_EQ_METHOD == "contextual_deep_eq"
    profile = _apply_corrective_eq([stem], method="profile")

    assert profile["method"] == "profile"
    assert profile["source"]["source_function"] == "classify_track"
    assert profile["applied_band_count"] == 3
    assert stem.corrective_eq_notes[0]["bands"][0]["frequency_hz"] == 200.0


def test_contextual_deep_eq_allows_large_moves_with_critic_notes():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = (
        0.12 * np.sin(2.0 * np.pi * 260.0 * time)
        + 0.01 * np.sin(2.0 * np.pi * 3000.0 * time)
    ).astype(np.float32)
    guitar = (
        0.12 * np.sin(2.0 * np.pi * 320.0 * time)
        + 0.10 * np.sin(2.0 * np.pi * 3100.0 * time)
    ).astype(np.float32)
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("Guitar L", guitar, sample_rate, 2),
    ]
    stems[0].group = "vocals"
    stems[1].group = "guitars"

    report = _apply_corrective_eq(stems, method="contextual_deep_eq")

    assert report["method"] == "contextual_deep_eq"
    assert report["source"]["name"] == "Contextual Deep EQ"
    assert report["analysis"]["critic_notes"]
    assert any(
        abs(band["gain_db"]) > 2.2
        for action in report["actions"]
        for band in action["bands"]
    )
    assert any(
        "regret_score" in band
        for action in report["actions"]
        for band in action["bands"]
    )
    assert report["analysis"]["after"]["channel_diagnostics"]


def test_project_corrective_eq_uses_primary_role_and_group_diagnostics():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = np.sin(2.0 * np.pi * 3000.0 * time).astype(np.float32) * 0.03
    guitar = np.sin(2.0 * np.pi * 3200.0 * time).astype(np.float32) * 0.08
    bass = np.sin(2.0 * np.pi * 85.0 * time).astype(np.float32) * 0.05
    kick = np.sin(2.0 * np.pi * 60.0 * time).astype(np.float32) * 0.035
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("Guitar L", guitar, sample_rate, 2),
        _stem("Bass", bass, sample_rate, 3),
        _stem("KICK", kick, sample_rate, 4),
    ]
    stems[0].group = "vocals"
    stems[1].group = "guitars"
    stems[2].group = "bass"
    stems[3].group = "drums"

    report = _apply_corrective_eq(stems, method="project_corrective")

    assert report["source"]["name"] == "Project Corrective EQ"
    assert report["analysis"]["channel_diagnostics"]
    assert {item["group"] for item in report["analysis"]["group_diagnostics"]} >= {"vocals", "guitars", "bass", "drums"}
    assert report["analysis"]["detected_conflicts"]
    assert report["applied_band_count"] <= 5 * len(stems)
    assert all(band["gain_db"] <= 0.0 for action in report["actions"] for band in action["bands"])
    guitar_action = next(action for action in report["actions"] if action["file"] == "Guitar L.wav")
    assert any(2500.0 <= band["frequency_hz"] <= 4000.0 for band in guitar_action["bands"])
    assert report["analysis"]["after"]["channel_diagnostics"]


def test_project_corrective_hybrid_merges_evidence_with_safety_limits():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = np.sin(2.0 * np.pi * 3000.0 * time).astype(np.float32) * 0.03
    guitar = np.sin(2.0 * np.pi * 3200.0 * time).astype(np.float32) * 0.08
    snare = (
        0.04 * np.sin(2.0 * np.pi * 750.0 * time)
        + 0.035 * np.sin(2.0 * np.pi * 6500.0 * time)
    ).astype(np.float32)
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("Guitar L", guitar, sample_rate, 2),
        _stem("SNARE T", snare, sample_rate, 3),
    ]
    stems[0].group = "vocals"
    stems[1].group = "guitars"
    stems[2].group = "drums"

    report = _apply_corrective_eq(stems, method="project_corrective_hybrid")

    assert report["source"]["name"] == "Hybrid Project Corrective EQ"
    assert report["analysis"]["components"]["project_corrective"]["candidate_band_count"] > 0
    assert report["analysis"]["components"]["frequency_window_diagnostic"]["candidate_band_count"] >= 0
    assert report["analysis"]["dynamic_candidates_report_only"]
    assert report["applied_band_count"] <= 4 * len(stems)
    assert all(band["gain_db"] <= 0.0 for action in report["actions"] for band in action["bands"])
    assert all(abs(band["gain_db"]) <= 1.6 for action in report["actions"] for band in action["bands"])
    guitar_action = next(action for action in report["actions"] if action["file"] == "Guitar L.wav")
    assert any(band.get("component") == "project_corrective" for band in guitar_action["bands"])


def test_bleed_control_corrective_eq_uses_only_eq_cuts():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    kick = np.sin(2.0 * np.pi * 60.0 * time).astype(np.float32) * 0.04
    stem = _stem("KICK", kick, sample_rate, 1)

    report = _apply_corrective_eq([stem], method="bleed_control")

    assert report["source"]["source_function"] == "apply_codex_bleed_control"
    assert report["analysis"]["excluded_changes"] == ["fader_db", "compressor", "lpf"]
    assert [band["frequency_hz"] for band in report["actions"][0]["bands"]] == [6500.0, 9500.0]


def test_pipeline_skips_group_levels_by_default(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = 0.02 * np.sin(2.0 * np.pi * 120.0 * time).astype(np.float32)
    sf.write(tmp_path / "Guitar L.wav", sine, sample_rate)
    sf.write(tmp_path / "Guitar R.wav", sine * 0.8, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="fixed_clean",
        write_mp3=False,
    )

    assert result.summary["pipeline_stages"] == [
        "bleed_primary_signal_detection",
        "arrangement_input_levels_pre_phase",
        "phase_delay_from_overhead_pair",
        "offline_hpf_lpf_correction",
        "corrective_eq_contextual_deep_eq",
        "ayaic_channel_pair_levels_post_corrective_eq",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_method"] == DEFAULT_CORRECTIVE_EQ_METHOD
    assert result.summary["settings"]["input_leveling_method"] == "arrangement_aware"
    assert result.summary["settings"]["skip_group_levels"] is True
    assert result.summary["pre_phase_channel_levels"]
    assert result.summary["pre_phase_channel_levels"][0]["type"] == "arrangement_aware_input_level"
    assert result.summary["pre_phase_channel_levels"][0]["replacement_for"] == "ayaic_channel_levels_pre_phase"
    assert "arrangement_density_index" in result.summary["pre_phase_channel_levels"][0]
    assert result.summary["hpf_lpf_correction"]
    assert result.summary["settings"]["autoeq_enabled"] is False
    assert result.summary["autoeq_settings"]["enabled"] is False
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"]
    assert result.summary["pair_levels"]
    assert result.summary["group_levels"][0]["enabled"] is False
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_pipeline_can_enable_autoeq_before_final_channel_pair_levels(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    vocal = 0.04 * np.sin(2.0 * np.pi * 3000.0 * time).astype(np.float32)
    guitar = 0.08 * np.sin(2.0 * np.pi * 3200.0 * time).astype(np.float32)
    sf.write(tmp_path / "Vocal.wav", vocal, sample_rate)
    sf.write(tmp_path / "Guitar L.wav", guitar, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="with_autoeq",
        write_mp3=False,
        corrective_eq_method="none",
        autoeq_enabled=True,
        autoeq_mode="offline",
    )

    assert "autoeq_engine" in result.summary["pipeline_stages"]
    assert result.summary["pipeline_stages"].index("offline_hpf_lpf_correction") < result.summary["pipeline_stages"].index("autoeq_engine")
    assert result.summary["pipeline_stages"].index("autoeq_engine") < result.summary["pipeline_stages"].index("ayaic_channel_pair_levels_post_phase_filter")
    assert result.summary["autoeq_settings"]["enabled"] is True
    assert result.summary["autoeq_channel_analysis"]
    assert result.summary["autoeq_decisions"] or result.summary["autoeq_rejected_decisions"]
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_pipeline_can_insert_corrective_eq_and_repeat_channel_pair_levels(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = 0.02 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    sf.write(tmp_path / "Guitar L.wav", sine, sample_rate)
    sf.write(tmp_path / "Guitar R.wav", sine * 0.8, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="with_corrective_eq",
        write_mp3=False,
        corrective_eq_method="profile",
    )

    assert "corrective_eq_profile" in result.summary["pipeline_stages"]
    assert result.summary["pipeline_stages"].index("corrective_eq_profile") < result.summary["pipeline_stages"].index("ayaic_channel_pair_levels_post_corrective_eq")
    assert "ayaic_channel_pair_levels_post_phase_filter" not in result.summary["pipeline_stages"]
    assert result.summary["corrective_eq"]["applied_band_count"] == 6
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"]
    assert result.summary["post_corrective_eq_pair_levels"]
    assert result.summary["pair_levels"] == result.summary["post_corrective_eq_pair_levels"]
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_pipeline_can_apply_431_corrective_eq_chain_before_final_levels(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = 0.02 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    sf.write(tmp_path / "Accordion.wav", sine, sample_rate)
    sf.write(tmp_path / "Vocal.wav", sine * 0.8, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="with_431_eq_chain",
        write_mp3=False,
        corrective_eq_method="4-3-1",
    )

    assert result.summary["pipeline_stages"] == [
        "bleed_primary_signal_detection",
        "arrangement_input_levels_pre_phase",
        "phase_delay_from_overhead_pair",
        "offline_hpf_lpf_correction",
        "corrective_eq_frequency_cross_profile",
        "ayaic_channel_pair_levels_post_corrective_eq",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_sequence"] == ["frequency_window", "cross_adaptive", "profile"]
    assert [stage["method"] for stage in result.summary["corrective_eq"]["stages"]] == ["frequency_window", "cross_adaptive", "profile"]
    assert result.summary["pre_corrective_eq_pair_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"]
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_pipeline_can_apply_project_corrective_eq_before_final_levels(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = 0.03 * np.sin(2.0 * np.pi * 3000.0 * time).astype(np.float32)
    guitar = 0.08 * np.sin(2.0 * np.pi * 3200.0 * time).astype(np.float32)
    sf.write(tmp_path / "Vocal.wav", vocal, sample_rate)
    sf.write(tmp_path / "Guitar L.wav", guitar, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="with_project_corrective_eq",
        write_mp3=False,
        corrective_eq_method="project_corrective",
    )

    assert result.summary["pipeline_stages"] == [
        "bleed_primary_signal_detection",
        "arrangement_input_levels_pre_phase",
        "phase_delay_from_overhead_pair",
        "offline_hpf_lpf_correction",
        "corrective_eq_project_corrective",
        "ayaic_channel_pair_levels_post_corrective_eq",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_sequence"] == ["project_corrective"]
    assert result.summary["corrective_eq"]["analysis"]["channel_diagnostics"]
    assert result.summary["corrective_eq"]["analysis"]["group_diagnostics"]
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"]
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_pipeline_can_apply_project_corrective_hybrid_eq_before_final_levels(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = 0.03 * np.sin(2.0 * np.pi * 3000.0 * time).astype(np.float32)
    guitar = 0.08 * np.sin(2.0 * np.pi * 3200.0 * time).astype(np.float32)
    sf.write(tmp_path / "Vocal.wav", vocal, sample_rate)
    sf.write(tmp_path / "Guitar L.wav", guitar, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="with_project_corrective_hybrid_eq",
        write_mp3=False,
        corrective_eq_method="project_corrective_hybrid",
    )

    assert result.summary["pipeline_stages"] == [
        "bleed_primary_signal_detection",
        "arrangement_input_levels_pre_phase",
        "phase_delay_from_overhead_pair",
        "offline_hpf_lpf_correction",
        "corrective_eq_project_corrective_hybrid",
        "ayaic_channel_pair_levels_post_corrective_eq",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_sequence"] == ["project_corrective_hybrid"]
    assert result.summary["corrective_eq"]["analysis"]["components"]["project_corrective"]
    assert result.summary["corrective_eq"]["analysis"]["dynamic_candidates_report_only"]
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"]
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_pipeline_can_opt_into_group_levels(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = 0.02 * np.sin(2.0 * np.pi * 120.0 * time).astype(np.float32)
    sf.write(tmp_path / "KICK.wav", sine, sample_rate)
    sf.write(tmp_path / "Bass.wav", sine, sample_rate)

    out = tmp_path / "out"
    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        output_name="with_group",
        write_mp3=False,
        corrective_eq_method="none",
        skip_group_levels=False,
    )

    assert "ayayc_instrument_group_levels" in result.summary["pipeline_stages"]
    assert result.summary["pipeline_stages"].index("ayaic_channel_pair_levels_post_phase_filter") < result.summary["pipeline_stages"].index("ayayc_instrument_group_levels")
    assert result.summary["settings"]["skip_group_levels"] is False
    assert any(item.get("group") == "drums" for item in result.summary["group_levels"])
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_phase_delay_uses_overhead_pair_as_farthest_reference():
    sample_rate = 48_000
    length = sample_rate
    reference = np.zeros(length, dtype=np.float32)
    close = np.zeros(length, dtype=np.float32)
    reference[10_000] = 1.0
    close[9_760] = 1.0
    stems = [
        _stem("KICK", close, sample_rate, 1),
        _stem("OH L", reference, sample_rate, 2),
        _stem("OH R", reference, sample_rate, 3),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=8.0)
    actions = result["phase_alignment"]

    assert result["order"] == ["apply_overhead_pair_phase_alignment", "apply_drum_phase_alignment"]
    assert result["drum_pan_rule"]["enabled"] is False
    assert len(actions) == 1
    assert actions[0]["reference"] == "overhead_pair"
    assert actions[0]["applied_delay_ms"] == 5.0
    assert stems[0].delay_ms == pytest.approx(5.0)


def test_overhead_pair_alignment_keeps_latest_overhead_at_zero_ms():
    sample_rate = 48_000
    length = sample_rate
    left = np.zeros(length, dtype=np.float32)
    right = np.zeros(length, dtype=np.float32)
    left[10_000] = 1.0
    right[9_904] = 1.0
    stems = [
        _stem("OH L", left, sample_rate, 1),
        _stem("OH R", right, sample_rate, 2),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=10.0)
    overhead = result["overhead_pair_alignment"]

    assert overhead["zero_ms_source"] == "OH L.wav"
    assert overhead["applied_left_delay_ms"] == 0.0
    assert overhead["applied_right_delay_ms"] == pytest.approx(2.0)
    assert stems[0].delay_ms == 0.0
    assert stems[1].delay_ms == pytest.approx(2.0)


def test_overhead_pair_can_invert_polarity_after_delay_alignment():
    sample_rate = 48_000
    length = sample_rate
    left = np.zeros(length, dtype=np.float32)
    right = np.zeros(length, dtype=np.float32)
    left[10_000] = 1.0
    right[9_904] = -1.0
    stems = [
        _stem("OH L", left, sample_rate, 1),
        _stem("OH R", right, sample_rate, 2),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=10.0)
    overhead = result["overhead_pair_alignment"]

    assert overhead["phase_invert_applied"] is True
    assert overhead["phase_inverted_file"] == "OH R.wav"
    assert overhead["applied_right_delay_ms"] == pytest.approx(2.0)
    assert stems[1].phase_invert is True


def test_close_mic_can_invert_polarity_after_delay_alignment():
    sample_rate = 48_000
    length = sample_rate
    overhead = np.zeros(length, dtype=np.float32)
    snare = np.zeros(length, dtype=np.float32)
    overhead[10_000] = 1.0
    snare[9_760] = -1.0
    stems = [
        _stem("OH L", overhead, sample_rate, 1),
        _stem("OH R", overhead, sample_rate, 2),
        _stem("SNARE T", snare, sample_rate, 3),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=10.0)
    snare_action = result["phase_alignment"][0]

    assert snare_action["phase_invert"] is True
    assert snare_action["applied_delay_ms"] == pytest.approx(5.0)
    assert stems[2].phase_invert is True


def _stem(name: str, mono: np.ndarray, sample_rate: int, channel_id: int) -> AyaicStem:
    return AyaicStem(
        name=name,
        path=f"{name}.wav",
        audio=np.column_stack([mono, mono]).astype(np.float32),
        sample_rate=sample_rate,
        channel_id=channel_id,
        target_lufs=-25.0,
        group="drums",
    )
