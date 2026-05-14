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
    _band_energy_for_rendered,
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
    _peak_dbfs,
    _sum_audio,
    _track_target_lufs,
    run_ayaic_offline_pipeline,
)
from automixer.production_mix_v1.contextual_compression import (
    apply_contextual_compression,
    estimate_tempo_bpm,
)
from automixer.production_mix_v1.musical_panning import apply_musical_panning
from automixer.production_mix_v1.musical_output_balance import (
    _limit_mix_peak_growth,
    apply_musical_output_balance,
)
from automixer.production_mix_v1.musical_panorama_width import apply_musical_panorama_width
from automixer.production_mix_v1.snare_top_bottom_balance import apply_snare_top_bottom_balance
from automixer.production_mix_v1.snare_pair_coherence import apply_snare_pair_coherence
from automixer.production_mix_v1.arrangement_input_levels import _offline_input_role_guard


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
        "global_bleed_phase_alignment",
        "offline_hpf_lpf_correction",
        "corrective_eq_contextual_deep_eq",
        "contextual_compression",
        "musical_panning",
        "musical_output_balance",
        "musical_panorama_width",
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
    assert "global_phase_alignment" in result.summary
    assert result.summary["contextual_compression"]["enabled"] is True
    assert result.summary["musical_panning"]["enabled"] is True
    assert result.summary["musical_output_balance"]["enabled"] is True
    assert result.summary["snare_top_bottom_balance"]["enabled"] is False
    assert result.summary["musical_panorama_width"]["enabled"] is True
    assert result.summary["musical_panning"]["channel_pans"]
    assert result.summary["post_compression_channel_levels"]
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"] == []
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
    assert result.summary["pipeline_stages"].index("autoeq_engine") < result.summary["pipeline_stages"].index("contextual_compression")
    assert result.summary["pipeline_stages"].index("contextual_compression") < result.summary["pipeline_stages"].index("musical_panning")
    assert result.summary["pipeline_stages"].index("musical_panning") < result.summary["pipeline_stages"].index("musical_output_balance")
    assert result.summary["pipeline_stages"].index("musical_output_balance") < result.summary["pipeline_stages"].index("musical_panorama_width")
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
    assert result.summary["pipeline_stages"].index("corrective_eq_profile") < result.summary["pipeline_stages"].index("contextual_compression")
    assert result.summary["pipeline_stages"].index("contextual_compression") < result.summary["pipeline_stages"].index("musical_panning")
    assert result.summary["pipeline_stages"].index("musical_panning") < result.summary["pipeline_stages"].index("musical_output_balance")
    assert result.summary["pipeline_stages"].index("musical_output_balance") < result.summary["pipeline_stages"].index("musical_panorama_width")
    assert "ayaic_channel_pair_levels_post_phase_filter" not in result.summary["pipeline_stages"]
    assert result.summary["corrective_eq"]["applied_band_count"] == 6
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_corrective_eq_channel_levels"] == []
    assert result.summary["post_corrective_eq_pair_levels"] == []
    assert result.summary["pair_levels"] == result.summary["post_compression_pair_levels"]
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
        "global_bleed_phase_alignment",
        "offline_hpf_lpf_correction",
        "corrective_eq_frequency_cross_profile",
        "contextual_compression",
        "musical_panning",
        "musical_output_balance",
        "musical_panorama_width",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_sequence"] == ["frequency_window", "cross_adaptive", "profile"]
    assert [stage["method"] for stage in result.summary["corrective_eq"]["stages"]] == ["frequency_window", "cross_adaptive", "profile"]
    assert result.summary["pre_corrective_eq_pair_levels"] == []
    assert result.summary["post_compression_channel_levels"]
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
        "global_bleed_phase_alignment",
        "offline_hpf_lpf_correction",
        "corrective_eq_project_corrective",
        "contextual_compression",
        "musical_panning",
        "musical_output_balance",
        "musical_panorama_width",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_sequence"] == ["project_corrective"]
    assert result.summary["corrective_eq"]["analysis"]["channel_diagnostics"]
    assert result.summary["corrective_eq"]["analysis"]["group_diagnostics"]
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_compression_channel_levels"]
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
        "global_bleed_phase_alignment",
        "offline_hpf_lpf_correction",
        "corrective_eq_project_corrective_hybrid",
        "contextual_compression",
        "musical_panning",
        "musical_output_balance",
        "musical_panorama_width",
        "ayaic_master_level",
    ]
    assert result.summary["settings"]["corrective_eq_sequence"] == ["project_corrective_hybrid"]
    assert result.summary["corrective_eq"]["analysis"]["components"]["project_corrective"]
    assert result.summary["corrective_eq"]["analysis"]["dynamic_candidates_report_only"]
    assert result.summary["post_phase_channel_levels"] == []
    assert result.summary["post_compression_channel_levels"]
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
    assert result.summary["pipeline_stages"].index("musical_panorama_width") < result.summary["pipeline_stages"].index("ayayc_instrument_group_levels")
    assert result.summary["settings"]["skip_group_levels"] is False
    assert any(item.get("group") == "drums" for item in result.summary["group_levels"])
    assert result.summary["master_level"]["final_lufs"] == pytest.approx(-20.0, abs=0.02)


def test_contextual_compression_reduces_vocal_crest_and_logs_params():
    sample_rate = 48_000
    duration = 3.0
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    envelope = 0.03 + 0.11 * (np.sin(2.0 * np.pi * 1.5 * time) > 0.55).astype(np.float32)
    vocal = (envelope * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    stem = _stem("Vocal", vocal, sample_rate, 1)
    before_range = _frame_dynamic_range_for_test(stem.audio)

    report = apply_contextual_compression(
        [stem],
        style="live_pop_rock",
        bpm=120.0,
        role_fn=lambda _name: "lead_vocal",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
    )
    after_range = _frame_dynamic_range_for_test(stem.audio)
    channel = report["channels"][0]

    assert report["applied"] is True
    assert channel["parameters"]["ratio"] > 1.0
    assert channel["gain_reduction"]["avg_gr_db"] > 0.5
    assert after_range < before_range
    assert stem.compression_notes


def test_musical_panning_centers_anchors_and_links_stereo_pairs():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = (0.04 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    kick = (0.08 * np.sin(2.0 * np.pi * 70.0 * time)).astype(np.float32)
    bass = (0.06 * np.sin(2.0 * np.pi * 90.0 * time)).astype(np.float32)
    guitar = (0.05 * np.sin(2.0 * np.pi * 880.0 * time)).astype(np.float32)
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("KICK", kick, sample_rate, 2),
        _stem("Bass", bass, sample_rate, 3),
        _stem("Guitar L", guitar, sample_rate, 4),
        _stem("Guitar R", guitar * 0.85, sample_rate, 5),
    ]

    report = apply_musical_panning(
        stems,
        style="live_pop_rock",
        role_fn=lambda name: _project_channel_role(name),
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    by_file = {item["file"]: item for item in report["channel_pans"]}

    assert report["applied"] is True
    assert by_file["Vocal.wav"]["final_pan"] == pytest.approx(0.0, abs=0.001)
    assert by_file["KICK.wav"]["final_pan"] == pytest.approx(0.0, abs=0.001)
    assert by_file["Bass.wav"]["final_pan"] == pytest.approx(0.0, abs=0.001)
    assert by_file["Guitar L.wav"]["final_pan"] < -0.25
    assert by_file["Guitar R.wav"]["final_pan"] > 0.25
    assert by_file["Guitar L.wav"]["final_pan"] == pytest.approx(-by_file["Guitar R.wav"]["final_pan"], abs=0.02)
    assert report["pair_pans"][0]["base"] == "guitar"
    assert stems[0].pan_notes


def test_musical_panning_limits_low_end_dominant_non_anchor():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    guitar_low = (0.12 * np.sin(2.0 * np.pi * 82.0 * time)).astype(np.float32)
    stem = _stem("Guitar", guitar_low, sample_rate, 1)

    report = apply_musical_panning(
        [stem],
        role_fn=lambda _name: "guitar",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    channel = report["channel_pans"][0]

    assert abs(channel["final_pan"]) <= 0.18
    assert any("low_end_energy_center_guard" in item["reasons"] for item in report["blocked_pans"])


def test_musical_panning_keeps_low_end_stereo_bed_wider_than_mono_guard():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    guitar_low = (0.12 * np.sin(2.0 * np.pi * 82.0 * time)).astype(np.float32)
    stems = [
        _stem("Guitar L", guitar_low, sample_rate, 1),
        _stem("Guitar R", guitar_low * 0.9, sample_rate, 2),
    ]

    report = apply_musical_panning(
        stems,
        role_fn=lambda _name: "guitar",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    by_file = {item["file"]: item for item in report["channel_pans"]}

    assert by_file["Guitar L.wav"]["final_pan"] == pytest.approx(-0.36)
    assert by_file["Guitar R.wav"]["final_pan"] == pytest.approx(0.36)
    assert any("low_end_energy_center_guard" in item["reasons"] for item in report["blocked_pans"])


def test_musical_output_balance_lowers_loud_music_bed_under_vocal():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = (0.035 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    guitar = (0.18 * np.sin(2.0 * np.pi * 660.0 * time)).astype(np.float32)
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("Guitar L", guitar, sample_rate, 2),
        _stem("Guitar R", guitar * 0.8, sample_rate, 3),
    ]

    report = apply_musical_output_balance(
        stems,
        style="live_pop_rock",
        role_fn=lambda name: "lead_vocal" if "Vocal" in name else "guitar",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    by_file = {item["file"]: item for item in report["channel_offsets"]}

    assert report["primary_source"]["file"] == "Vocal.wav"
    assert by_file["Guitar L.wav"]["final_offset_db"] < -4.0
    assert by_file["Guitar L.wav"]["final_offset_db"] == pytest.approx(by_file["Guitar R.wav"]["final_offset_db"], abs=0.01)
    assert by_file["Vocal.wav"]["final_offset_db"] >= -0.1
    assert stems[0].output_balance_notes


def test_musical_output_balance_limits_unknown_channel_moves():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    unknown = (0.25 * np.sin(2.0 * np.pi * 500.0 * time)).astype(np.float32)
    vocal = (0.02 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("Mystery", unknown, sample_rate, 2),
    ]

    report = apply_musical_output_balance(
        stems,
        role_fn=lambda name: "lead_vocal" if "Vocal" in name else "unknown",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    mystery = next(item for item in report["channel_offsets"] if item["file"] == "Mystery.wav")

    assert mystery["final_offset_db"] >= -2.5
    assert any(item["file"] == "Mystery.wav" for item in report["blocked_offsets"])


def test_arrangement_input_keeps_overhead_from_being_premixed_away():
    gain, reason = _offline_input_role_guard("overhead", -2.4)

    assert gain == pytest.approx(-0.75)
    assert reason == "offline_overhead_air_preservation_floor"


def test_musical_output_balance_can_restore_quiet_overheads():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = (0.07 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    overhead = (0.002 * np.sin(2.0 * np.pi * 8000.0 * time)).astype(np.float32)
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("OH L", overhead, sample_rate, 2),
        _stem("OH R", overhead * 0.9, sample_rate, 3),
    ]

    report = apply_musical_output_balance(
        stems,
        style="live_pop_rock",
        role_fn=lambda name: "lead_vocal" if "Vocal" in name else "overhead",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    by_file = {item["file"]: item for item in report["channel_offsets"]}

    assert by_file["OH L.wav"]["final_offset_db"] > 1.0
    assert by_file["OH R.wav"]["final_offset_db"] == pytest.approx(
        by_file["OH L.wav"]["final_offset_db"],
        abs=0.01,
    )


def test_output_peak_guard_keeps_cuts_while_scaling_boosts():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    loud_bed = (0.25 * np.sin(2.0 * np.pi * 500.0 * time)).astype(np.float32)
    quiet_air = (0.22 * np.sin(2.0 * np.pi * 500.0 * time)).astype(np.float32)
    stems = [
        _stem("Guitar", loud_bed, sample_rate, 1),
        _stem("OH L", quiet_air, sample_rate, 2),
    ]

    safe, blocked = _limit_mix_peak_growth(
        stems,
        {1: -5.0, 2: 3.5},
        before_peak=-12.0,
        peak_fn=_peak_dbfs,
        max_growth_db=0.25,
    )

    assert safe[1] == pytest.approx(-5.0)
    assert 0.0 <= safe[2] < 3.5
    assert any("positive_offset_scale" in item["reasons"] for item in blocked)


def test_panorama_width_stage_raises_side_without_mono_level_change():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    vocal = 0.08 * np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    guitar = 0.04 * np.sin(2.0 * np.pi * 880.0 * time).astype(np.float32)
    left = _stem("Guitar L", guitar, sample_rate, 1)
    right = _stem("Guitar R", guitar * 0.8, sample_rate, 2)
    vocal_stem = _stem("Vocal", vocal, sample_rate, 3)
    left.pan = -0.36
    right.pan = 0.36
    left.audio = np.column_stack([guitar, guitar * 0.65]).astype(np.float32)
    right.audio = np.column_stack([guitar * 0.52, guitar * 0.8]).astype(np.float32)
    stems = [left, right, vocal_stem]
    before = _sum_audio([stem.audio for stem in stems])
    before_side_mid = _side_mid_db_for_test(before)
    before_mono_lufs = _lufs(np.column_stack([np.mean(before, axis=1), np.mean(before, axis=1)]), sample_rate)

    report = apply_musical_panorama_width(
        stems,
        role_fn=lambda name: "lead_vocal" if "Vocal" in name else "guitar",
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
    )
    after = _sum_audio([stem.audio for stem in stems])
    after_side_mid = _side_mid_db_for_test(after)
    after_mono_lufs = _lufs(np.column_stack([np.mean(after, axis=1), np.mean(after, axis=1)]), sample_rate)

    assert report["applied"] is True
    assert after_side_mid > before_side_mid + 2.0
    assert after_mono_lufs == pytest.approx(before_mono_lufs, abs=0.05)


def test_snare_top_bottom_balance_trims_bottom_below_top():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    top_audio = (0.04 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    bottom_audio = (0.08 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    top = _stem("SNARE T", top_audio, sample_rate, 1)
    bottom = _stem("Snare B", bottom_audio, sample_rate, 2)

    report = apply_snare_top_bottom_balance(
        [top, bottom],
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
    )
    pair = report["pairs"][0]

    assert report["applied"] is True
    assert pair["primary_relative_before_db"] > 0.0
    assert pair["primary_relative_after_db"] == pytest.approx(-12.0, abs=0.15)
    assert pair["applied_trim_db"] < -17.0
    assert top.track_gain_db == pytest.approx(0.0)
    assert bottom.track_gain_db == pytest.approx(pair["applied_trim_db"], abs=0.001)


def test_snare_pair_coherence_groups_centers_and_sets_opposite_phase():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    top = _stem("SNARE T", 0.04 * np.sin(2.0 * np.pi * 220.0 * time), sample_rate, 1)
    bottom = _stem("Snare B", 0.04 * np.sin(2.0 * np.pi * 220.0 * time), sample_rate, 2)
    top.group = "drums"
    bottom.group = "drums"
    top.pan = -0.22
    bottom.pan = 0.18
    top.phase_invert = False
    bottom.phase_invert = False

    report = apply_snare_pair_coherence([top, bottom])
    pair = report["pairs"][0]

    assert report["applied"] is True
    assert top.group == "snare"
    assert bottom.group == "snare"
    assert top.pan == pytest.approx(0.0)
    assert bottom.pan == pytest.approx(0.0)
    assert top.phase_invert != bottom.phase_invert
    assert pair["polarity_relation"] == "opposite_phase_flags"
    assert "bottom_phase_inverted_relative_to_top" in pair["actions"]


def test_musical_panning_center_locks_snare_role():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.04 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    top = _stem("SNARE T", audio, sample_rate, 1)
    bottom = _stem("Snare B", audio, sample_rate, 2)
    top.pan = -0.35
    bottom.pan = 0.35

    report = apply_musical_panning(
        [top, bottom],
        role_fn=lambda _name: "snare",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        band_energy_fn=_band_energy_for_rendered,
    )
    by_file = {item["file"]: item for item in report["channel_pans"]}

    assert top.pan == pytest.approx(0.0)
    assert bottom.pan == pytest.approx(0.0)
    assert by_file["SNARE T.wav"]["final_pan"] == pytest.approx(0.0)
    assert by_file["Snare B.wav"]["final_pan"] == pytest.approx(0.0)
    assert report["critic"]["decision"] == "ok"


def test_pipeline_applies_snare_bottom_relative_balance(tmp_path):
    import soundfile as sf

    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    snare_top = (0.04 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    snare_bottom = (0.08 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    sf.write(tmp_path / "SNARE T.wav", snare_top, sample_rate)
    sf.write(tmp_path / "Snare B.wav", snare_bottom, sample_rate)

    result = run_ayaic_offline_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        output_name="snare_pair",
        write_mp3=False,
        corrective_eq_method="none",
    )
    report = result.summary["snare_top_bottom_balance"]
    pair = report["pairs"][0]
    coherence = result.summary["snare_pair_coherence"]["pairs"][0]

    assert "snare_pair_coherence" in result.summary["pipeline_stages"]
    assert "snare_top_bottom_balance" in result.summary["pipeline_stages"]
    assert coherence["top_after"]["group"] == "snare"
    assert coherence["bottom_after"]["group"] == "snare"
    assert coherence["top_after"]["pan"] == 0.0
    assert coherence["bottom_after"]["pan"] == 0.0
    assert coherence["polarity_relation"] == "opposite_phase_flags"
    assert report["applied"] is True
    assert -14.0 <= pair["primary_relative_after_db"] <= -10.0


def test_contextual_compression_release_follows_tempo():
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    bass = (0.08 * np.sin(2.0 * np.pi * 80.0 * time) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 2.0 * time))).astype(np.float32)
    slow = _stem("Bass", bass.copy(), sample_rate, 1)
    fast = _stem("Bass", bass.copy(), sample_rate, 1)

    slow_report = apply_contextual_compression(
        [slow],
        style="live_pop_rock",
        bpm=80.0,
        report_only=True,
        role_fn=lambda _name: "bass",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
    )
    fast_report = apply_contextual_compression(
        [fast],
        style="live_pop_rock",
        bpm=160.0,
        report_only=True,
        role_fn=lambda _name: "bass",
        primary_audio_fn=lambda item: item.audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
    )

    assert fast_report["channels"][0]["parameters"]["release_ms"] < slow_report["channels"][0]["parameters"]["release_ms"]


def test_tempo_estimator_finds_synthetic_120_bpm_click():
    sample_rate = 48_000
    length = sample_rate * 8
    click = np.zeros(length, dtype=np.float32)
    for start in range(0, length, sample_rate // 2):
        click[start:start + 64] = 1.0
    stem = _stem("KICK", click, sample_rate, 1)

    bpm, confidence = estimate_tempo_bpm([stem], sample_rate=sample_rate)

    assert bpm == pytest.approx(120.0, abs=3.0)
    assert confidence > 0.15


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

    assert result["order"][:1] == ["snare_bleed_phase_alignment_fallback"]
    assert result["order"][1:] == [
        "build_global_bleed_correlation_graph",
        "solve_global_delay_offsets",
        "apply_fractional_delay_and_polarity",
        "validate_global_phase_alignment",
    ]
    assert result["snare_bleed_alignment"]["enabled"] is False
    assert result["global_phase_alignment"]["enabled"] is True
    assert result["drum_pan_rule"]["enabled"] is False
    assert len(actions) == 1
    assert actions[0]["reference"] == "global_bleed_correlation_graph"
    assert actions[0]["applied_delay_ms"] == 5.0
    assert stems[0].delay_ms == pytest.approx(5.0)


def test_snare_bleed_alignment_keeps_latest_snare_arrival_at_zero_ms():
    sample_rate = 48_000
    length = sample_rate
    snare = _snare_burst(length, 10_000)
    overhead = _snare_burst(length, 10_240, 0.5)
    guitar_bleed = _snare_burst(length, 10_120, 0.4)
    kick_bleed = _snare_burst(length, 9_900, 0.6)
    stems = [
        _stem("SNARE T", snare, sample_rate, 1),
        _stem("OH L", overhead, sample_rate, 2),
        _stem("Guitar L", guitar_bleed, sample_rate, 3),
        _stem("KICK", kick_bleed, sample_rate, 4),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=10.0)
    snare_report = result["snare_bleed_alignment"]

    assert result["order"][0] == "detect_snare_transients"
    assert snare_report["enabled"] is True
    assert snare_report["zero_ms_source"] == "OH L.wav"
    assert snare_report["latest_arrival_ms"] == pytest.approx(5.0, abs=0.03)
    assert result["global_phase_alignment"]["enabled"] is False
    assert stems[1].delay_ms == pytest.approx(0.0, abs=0.001)
    assert stems[0].delay_ms == pytest.approx(5.0, abs=0.05)
    assert stems[2].delay_ms == pytest.approx(2.5, abs=0.05)
    assert stems[3].delay_ms == pytest.approx(7.083, abs=0.05)
    assert {action["file"] for action in result["phase_alignment"]} == {"SNARE T.wav", "Guitar L.wav", "KICK.wav"}


def test_snare_bleed_alignment_skips_channels_without_reliable_snare_bleed():
    sample_rate = 48_000
    length = sample_rate
    time = np.arange(length, dtype=np.float32) / sample_rate
    guitar = (0.05 * np.sin(2.0 * np.pi * 913.0 * time)).astype(np.float32)
    stems = [
        _stem("SNARE T", _snare_burst(length, 10_000), sample_rate, 1),
        _stem("OH L", _snare_burst(length, 10_240, 0.5), sample_rate, 2),
        _stem("Guitar L", guitar, sample_rate, 3),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=10.0)
    snare_report = result["snare_bleed_alignment"]

    assert snare_report["enabled"] is True
    assert stems[0].delay_ms == pytest.approx(5.0, abs=0.05)
    assert stems[1].delay_ms == pytest.approx(0.0, abs=0.001)
    assert stems[2].delay_ms == pytest.approx(0.0, abs=0.001)
    assert any(item["file"] == "Guitar L.wav" for item in snare_report["rejected_mics"])


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


def test_drum_bleed_reference_does_not_delay_overhead_for_negative_close_offset():
    sample_rate = 48_000
    length = sample_rate
    close = np.zeros(length, dtype=np.float32)
    overhead = np.zeros(length, dtype=np.float32)
    close[10_000] = 1.0
    overhead[9_880] = 1.0
    stems = [
        _stem("F Tom", close, sample_rate, 1),
        _stem("OH R", overhead, sample_rate, 2),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=8.0)

    assert result["phase_alignment"] == []
    assert stems[0].delay_ms == pytest.approx(0.0)
    assert stems[1].delay_ms == pytest.approx(0.0)
    assert result["global_phase_alignment"]["enabled"] is False


def test_global_phase_alignment_handles_non_drum_bleed_pair():
    sample_rate = 48_000
    length = sample_rate * 2
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.04, length).astype(np.float32)
    base[:1024] *= np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    base[-1024:] *= np.linspace(1.0, 0.0, 1024, dtype=np.float32)
    guitar_bleed = _advance_signal(base, 2.35 * sample_rate / 1000.0)
    stems = [
        _stem("Vocal", base, sample_rate, 1),
        _stem("Guitar L", guitar_bleed, sample_rate, 2),
    ]
    stems[0].group = "vocals"
    stems[1].group = "guitars"

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=8.0)
    guitar_action = next(action for action in result["phase_alignment"] if action["file"] == "Guitar L.wav")

    assert result["global_phase_alignment"]["enabled"] is True
    assert guitar_action["applied_incremental_delay_ms"] == pytest.approx(2.35, abs=0.15)
    assert stems[1].delay_ms == pytest.approx(2.35, abs=0.15)


def test_global_phase_alignment_rejects_low_correlation_pairs():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    vocal = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32) * 0.05
    guitar = np.sin(2.0 * np.pi * 913.0 * time).astype(np.float32) * 0.05
    stems = [
        _stem("Vocal", vocal, sample_rate, 1),
        _stem("Guitar L", guitar, sample_rate, 2),
    ]

    result = _apply_phase_delay_from_overheads(stems, max_delay_ms=8.0)

    assert result["phase_alignment"] == []
    assert result["global_phase_alignment"]["enabled"] is False
    assert result["global_phase_alignment"]["reason"] == "no_reliable_correlation_edges"


def _advance_signal(signal: np.ndarray, samples: float) -> np.ndarray:
    positions = np.arange(len(signal), dtype=np.float64) + float(samples)
    base = np.arange(len(signal), dtype=np.float64)
    return np.interp(positions, base, signal, left=0.0, right=0.0).astype(np.float32)


def _snare_burst(length: int, start: int, amp: float = 1.0) -> np.ndarray:
    burst = np.zeros(length, dtype=np.float32)
    shape_len = min(128, max(8, length - start))
    shape = np.hanning(shape_len).astype(np.float32) * float(amp)
    burst[start:start + shape_len] += shape
    return burst


def _rms_db_for_test(audio: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))) + 1e-12)


def _frame_dynamic_range_for_test(audio: np.ndarray) -> float:
    arr = np.mean(audio, axis=1) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)
    frame = 2400
    hop = 1200
    values = []
    for start in range(0, max(1, len(arr) - frame + 1), hop):
        block = arr[start:start + frame]
        if block.size:
            values.append(_rms_db_for_test(block))
    if not values:
        return 0.0
    levels = np.asarray(values, dtype=np.float64)
    return float(np.percentile(levels, 95) - np.percentile(levels, 10))


def _side_mid_db_for_test(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    mid = (arr[:, 0] + arr[:, 1]) * 0.5
    side = (arr[:, 0] - arr[:, 1]) * 0.5
    return 20.0 * np.log10(
        (float(np.sqrt(np.mean(side * side))) + 1e-12)
        / (float(np.sqrt(np.mean(mid * mid))) + 1e-12)
    )


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
