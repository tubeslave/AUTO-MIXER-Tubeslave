"""Tests for AutoFOH soundcheck profile storage and learned corridors."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_analysis import build_stem_contribution_matrix, extract_analysis_features
from autofoh_detectors import aggregate_stem_features
from autofoh_profiles import (
    AutoFOHSoundcheckProfileStore,
    build_phase_learning_snapshot,
    build_soundcheck_profile,
)


def _sine(
    frequency_hz: float,
    amplitude: float = 1.0,
    sample_rate: int = 48000,
    duration_sec: float = 0.25,
):
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def test_soundcheck_profile_round_trips_with_learned_corridor(tmp_path):
    silence_channel_features = {
        1: extract_analysis_features(np.zeros(12000, dtype=np.float32)),
        2: extract_analysis_features(np.zeros(12000, dtype=np.float32)),
    }
    source_channel_features = {
        1: extract_analysis_features(_sine(320.0, amplitude=0.8)),
        2: extract_analysis_features(_sine(500.0, amplitude=0.2)),
    }
    channel_features = {
        1: extract_analysis_features(_sine(320.0, amplitude=0.2)),
        2: extract_analysis_features(_sine(3200.0, amplitude=0.8)),
    }
    channel_metadata = {
        1: {
            "name": "Gtr 1",
            "source_role": "guitar",
            "stem_roles": ["GUITARS", "MUSIC"],
            "allowed_controls": ["eq", "fader"],
            "priority": 0.6,
        },
        2: {
            "name": "Lead Vox",
            "source_role": "lead_vocal",
            "stem_roles": ["LEAD"],
            "allowed_controls": ["eq", "fader"],
            "priority": 1.0,
        },
    }
    silence_stem_features = aggregate_stem_features(
        silence_channel_features,
        {1: ["GUITARS", "MUSIC"], 2: ["LEAD"]},
    )
    source_stem_features = aggregate_stem_features(
        source_channel_features,
        {1: ["GUITARS", "MUSIC"], 2: ["LEAD"]},
    )
    stem_features = aggregate_stem_features(
        channel_features,
        {1: ["GUITARS", "MUSIC"], 2: ["LEAD"]},
    )
    contribution_matrix = build_stem_contribution_matrix(
        {
            stem_name: feature
            for stem_name, feature in stem_features.items()
            if stem_name != "MASTER"
        }
    )
    phase_snapshots = {
        "SILENCE_CAPTURE": build_phase_learning_snapshot(
            phase_name="SILENCE_CAPTURE",
            runtime_state="SILENCE_CAPTURE",
            channel_features=silence_channel_features,
            stem_features=silence_stem_features,
            metadata={"reset_count": 2},
            notes="Unit test silence capture",
        ),
        "LINE_CHECK": build_phase_learning_snapshot(
            phase_name="LINE_CHECK",
            runtime_state="LINE_CHECK",
            channel_features=source_channel_features,
            stem_features=source_stem_features,
            metadata={"detected_channel_ids": [1, 2]},
            notes="Unit test line check",
        ),
        "SOURCE_LEARNING": build_phase_learning_snapshot(
            phase_name="SOURCE_LEARNING",
            runtime_state="SOURCE_LEARNING",
            channel_features=source_channel_features,
            stem_features=source_stem_features,
            metadata={"active_roles": ["guitar", "lead_vocal"]},
            notes="Unit test source learning",
        ),
        "FULL_BAND_LEARNING": build_phase_learning_snapshot(
            phase_name="FULL_BAND_LEARNING",
            runtime_state="FULL_BAND_LEARNING",
            channel_features=channel_features,
            stem_features=stem_features,
            metadata={"stem_names": ["GUITARS", "LEAD", "MUSIC"]},
            notes="Unit test full-band phase",
        ),
    }
    profile = build_soundcheck_profile(
        channel_features=channel_features,
        channel_metadata=channel_metadata,
        stem_features=stem_features,
        stem_contributions=contribution_matrix.band_contributions,
        sample_rate=48000,
        profile_name="unit_soundcheck",
        phase_snapshots=phase_snapshots,
    )

    store = AutoFOHSoundcheckProfileStore(tmp_path / "profile.json")
    store.save(profile)
    loaded = store.load()

    assert loaded.name == "unit_soundcheck"
    assert loaded.channel_count == 2
    assert loaded.channels[2].source_role == "lead_vocal"
    assert loaded.master.target_corridor.name == "learned_soundcheck"
    assert "MUD" in loaded.target_corridor.target_median_db
    assert "LINE_CHECK" in loaded.phase_snapshots
    assert loaded.phase_snapshots["LINE_CHECK"].runtime_state == "LINE_CHECK"
    assert loaded.phase_snapshots["FULL_BAND_LEARNING"].notes == "Unit test full-band phase"
    assert "SOURCE_LEARNING" in loaded.phase_targets
    assert loaded.phase_targets["FULL_BAND_LEARNING"].target_corridor.source == "phase:full_band_learning"
    assert (
        loaded.phase_targets["FULL_BAND_LEARNING"].expected_source_role_rms_db["lead_vocal"]
        == loaded.phase_snapshots["FULL_BAND_LEARNING"].channel_features[2].rms_db
    )
    assert (
        loaded.phase_targets["FULL_BAND_LEARNING"].noise_floor_db_by_channel[1]
        == loaded.phase_targets["SILENCE_CAPTURE"].expected_channel_rms_db[1]
    )
    assert (
        loaded.phase_targets["SOURCE_LEARNING"].target_corridor.target_for_band("PRESENCE")
        != loaded.phase_targets["FULL_BAND_LEARNING"].target_corridor.target_for_band("PRESENCE")
    )
    assert loaded.phase_targets["FULL_BAND_LEARNING"].hpf_frequency_range_hz_by_channel[1]["max_hz"] > 0.0
    assert (
        loaded.phase_targets["FULL_BAND_LEARNING"].compressor_threshold_range_db_by_channel[2]["max_db"]
        <= -5.0
    )
    assert (
        loaded.phase_targets["FULL_BAND_LEARNING"].compressor_ratio_range_by_channel[2]["max_ratio"]
        >= 1.5
    )
    assert (
        loaded.phase_targets["FULL_BAND_LEARNING"].fx_send_level_range_db_by_channel[2]["max_db"]
        <= -5.0
    )


def test_learned_target_corridor_recenters_mix_indexes_for_same_material():
    samples = _sine(320.0, amplitude=0.8)
    default_features = extract_analysis_features(samples)
    stem_features = {"MASTER": default_features}
    profile = build_soundcheck_profile(
        channel_features={1: default_features},
        channel_metadata={
            1: {
                "name": "Gtr 1",
                "source_role": "guitar",
                "stem_roles": ["GUITARS"],
            }
        },
        stem_features=stem_features,
        stem_contributions={},
        sample_rate=48000,
    )

    learned_features = extract_analysis_features(
        samples,
        target_corridor=profile.target_corridor,
    )

    assert abs(learned_features.mix_indexes.mud_index) < abs(default_features.mix_indexes.mud_index)
