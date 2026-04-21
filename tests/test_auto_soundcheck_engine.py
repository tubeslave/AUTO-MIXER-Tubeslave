"""Tests for AutoSoundcheckEngine safety wiring."""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo
from autofoh_analysis import extract_analysis_features
from autofoh_detectors import LeadMaskingAnalyzer, aggregate_stem_features
from autofoh_models import RuntimeState
from autofoh_profiles import build_phase_learning_snapshot, build_soundcheck_profile
from autofoh_safety import AutoFOHSafetyController, ChannelEQMove
from observation_mixer import ObservationMixerClient


class FakeMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def set_channel_gain(self, channel, value):
        raise AssertionError("Observation mode must intercept writes")


def test_selected_channels_drive_engine_channel_iterator():
    engine = AutoSoundcheckEngine(
        selected_channels=[4, 2, 4],
        num_channels=8,
        auto_discover=False,
    )

    assert engine._iter_channels() == [2, 4]
    assert engine._configured_channel_count() == 2


def test_observation_mode_wraps_mixer_and_intercepts_writes():
    observed = []
    engine = AutoSoundcheckEngine(
        observe_only=True,
        auto_discover=False,
        on_observation=observed.append,
    )
    engine.mixer_client = FakeMixer()

    engine._activate_observation_mode()

    assert isinstance(engine.mixer_client, ObservationMixerClient)
    assert engine.mixer_client.set_channel_gain(1, 5.0) is True
    assert any("operation" in item for item in observed)


class ResetTrackingMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def __init__(self):
        self.names = {
            1: "Mystery Source",
            2: "LD VOX",
        }
        self.reset_calls = []

    def get_channel_name(self, channel):
        return self.names[channel]

    def reset_channel_processing(self, channel):
        self.reset_calls.append(channel)


class DetectorMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def __init__(self):
        self.fader = {1: -6.0, 2: -8.0}
        self.eq_gain = {}
        self.eq_freq = {}
        self.hpf = {}
        self.compressor = {}
        self.send_level = {}
        self.calls = []

    def get_fader(self, channel):
        return self.fader.get(channel, -10.0)

    def get_eq_band_gain(self, channel, band):
        band_num = int(str(band).replace("g", ""))
        return self.eq_gain.get((channel, band_num), 0.0)

    def get_eq_band_frequency(self, channel, band):
        band_num = int(str(band).replace("f", ""))
        return self.eq_freq.get((channel, band_num), 350.0)

    def set_eq_band(self, channel, band, freq, gain, q):
        self.eq_gain[(channel, band)] = gain
        self.eq_freq[(channel, band)] = freq
        self.calls.append(("set_eq_band", channel, band, freq, gain, q))
        return True

    def set_hpf(self, channel, freq_hz, enabled=True):
        self.hpf[channel] = (freq_hz, enabled)
        self.calls.append(("set_hpf", channel, freq_hz, enabled))
        return True

    def set_compressor(
        self,
        channel,
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        makeup_db=0.0,
        enabled=True,
    ):
        self.compressor[channel] = (
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            enabled,
        )
        self.calls.append(
            (
                "set_compressor",
                channel,
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                makeup_db,
                enabled,
            )
        )
        return True

    def set_send_level(self, channel, send_bus, level_db):
        self.send_level[(channel, send_bus)] = level_db
        self.calls.append(("set_send_level", channel, send_bus, level_db))
        return True


class DetectorAudioCapture:
    def __init__(self, buffers):
        self.buffers = buffers

    def get_buffer(self, channel, size):
        return self.buffers[channel][:size]


class SequenceAudioCapture:
    def __init__(self, buffers):
        self.buffers = {channel: list(items) for channel, items in buffers.items()}

    def get_buffer(self, channel, size):
        queue = self.buffers[channel]
        if len(queue) > 1:
            return queue.pop(0)[:size]
        return queue[0][:size]


def _build_single_channel_phase_profile(
    samples,
    *,
    source_role,
    stem_roles,
    allowed_controls,
    name="Ch 1",
):
    channel_features = {1: extract_analysis_features(samples)}
    stem_features = aggregate_stem_features(channel_features, {1: stem_roles})
    return build_soundcheck_profile(
        channel_features=channel_features,
        channel_metadata={
            1: {
                "name": name,
                "source_role": source_role,
                "stem_roles": stem_roles,
                "allowed_controls": allowed_controls,
                "priority": 0.6,
            }
        },
        stem_features=stem_features,
        stem_contributions={},
        phase_snapshots={
            "SNAPSHOT_LOCK": build_phase_learning_snapshot(
                phase_name="SNAPSHOT_LOCK",
                runtime_state="SNAPSHOT_LOCK",
                channel_features=channel_features,
                stem_features=stem_features,
            ),
        },
    )


def test_unknown_channels_are_skipped_for_auto_reset_by_default():
    mixer = ResetTrackingMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.mixer_client = mixer

    engine._scan_channels()
    engine._reset_channels()

    assert mixer.reset_calls == [2]
    assert engine.channels[1].auto_corrections_enabled is False
    assert engine.channels[2].auto_corrections_enabled is True


def test_monitor_analysis_routes_lead_masking_action_through_safety_layer():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.PRE_SHOW_CHECK
    engine.audio_capture = DetectorAudioCapture(
        {
            1: sine(2200.0, 0.2),
            2: sine(2800.0, 0.9),
        }
    )
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            source_role="lead_vocal",
            stem_roles=["LEAD"],
            allowed_controls=["fader", "eq"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            fader_db=-6.0,
            priority=1.0,
        ),
        2: ChannelInfo(
            channel=2,
            name="Gtr 1",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["eq", "fader"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            fader_db=-8.0,
            priority=0.6,
        ),
    }
    engine.lead_masking_analyzer = LeadMaskingAnalyzer(
        masking_threshold_db=3.0,
        persistence_required_cycles=1,
    )

    decisions = engine._run_autofoh_monitor_analysis()

    assert len(decisions) == 1
    assert decisions[0].sent is True
    assert mixer.calls[-1][0] == "set_eq_band"
    assert mixer.calls[-1][1] == 2


def test_proxy_audio_rollback_restores_previous_console_state():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.PRE_SHOW_CHECK
    engine.evaluation_policy.allow_proxy_audio_evaluation_for_testing = True
    engine.evaluation_policy.allow_proxy_audio_rollback_for_testing = True
    engine.evaluation_policy.worsening_tolerance_db = 0.1
    engine.audio_capture = SequenceAudioCapture(
        {
            1: [
                sine(320.0, 1.0),
            ]
        }
    )
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Gtr 1",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["eq"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            priority=0.6,
        ),
    }

    pre_features = extract_analysis_features(sine(320.0, 0.8))
    decision = engine._execute_action(
        ChannelEQMove(
            channel_id=1,
            band=2,
            freq_hz=350.0,
            gain_db=-1.0,
            q=1.3,
            reason="Mud cleanup",
        ),
        evaluation_context={
            "band_name": "MUD",
            "pre_features": pre_features,
            "expected_effect": "Reduce sustained mud",
        },
    )

    assert decision is not None and decision.sent is True

    outcomes = engine._evaluate_pending_actions(force=True)

    assert len(outcomes) == 1
    assert outcomes[0].worsened is True
    assert outcomes[0].should_rollback is True
    assert mixer.calls[-1] == ("set_eq_band", 1, 2, 350.0, 0.0, 1.3)


def test_engine_can_save_and_load_soundcheck_profile(tmp_path):
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    profile_path = tmp_path / "autofoh_profile.json"

    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.soundcheck_profile_path = str(profile_path)
    engine.soundcheck_profile_enabled = True
    engine.audio_capture = DetectorAudioCapture(
        {
            1: sine(320.0, 0.8),
            2: sine(2200.0, 0.2),
        }
    )
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Gtr 1",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["eq", "fader"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            priority=0.6,
        ),
        2: ChannelInfo(
            channel=2,
            name="Lead Vox",
            source_role="lead_vocal",
            stem_roles=["LEAD"],
            allowed_controls=["eq", "fader"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            priority=1.0,
        ),
    }
    engine._capture_learning_phase(
        "SILENCE_CAPTURE",
        RuntimeState.SILENCE_CAPTURE,
        include_inactive=True,
        metadata={"reset_count": 2},
        notes="Test silence capture",
    )
    engine._capture_learning_phase(
        "LINE_CHECK",
        RuntimeState.LINE_CHECK,
        include_inactive=True,
        metadata={"detected_channel_ids": [1, 2]},
        notes="Test line check",
    )
    engine._capture_learning_phase(
        "SOURCE_LEARNING",
        RuntimeState.SOURCE_LEARNING,
        metadata={"channel_count": 2},
        notes="Test source learning",
    )
    engine._capture_learning_phase(
        "STEM_LEARNING",
        RuntimeState.STEM_LEARNING,
        metadata={"stem_names": ["GUITARS", "LEAD", "MUSIC"]},
        notes="Test stem learning",
    )
    engine._capture_learning_phase(
        "FULL_BAND_LEARNING",
        RuntimeState.FULL_BAND_LEARNING,
        metadata={"learned_target_corridor_preview": {"MUD": 1.0}},
        notes="Test full-band learning",
    )
    engine._capture_learning_phase(
        "SNAPSHOT_LOCK",
        RuntimeState.SNAPSHOT_LOCK,
        metadata={"applied_channel_ids": []},
        notes="Test snapshot lock",
    )

    profile = engine._build_soundcheck_profile_from_live_buffers()
    saved_path = engine._save_soundcheck_profile(profile)

    assert profile is not None
    assert saved_path == profile_path
    assert profile_path.exists()

    engine2 = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine2.soundcheck_profile_path = str(profile_path)
    engine2.soundcheck_profile_enabled = True
    engine2.soundcheck_profile_use_loaded_target_corridor = True

    loaded = engine2._load_soundcheck_profile()

    assert loaded is not None
    assert loaded.channel_count == 2
    assert loaded.channels[2].source_role == "lead_vocal"
    assert "SILENCE_CAPTURE" in loaded.phase_snapshots
    assert "SNAPSHOT_LOCK" in loaded.phase_snapshots
    assert loaded.phase_snapshots["LINE_CHECK"].metadata["detected_channel_ids"] == [1, 2]
    assert (
        engine2.current_target_corridor.target_for_band("MUD")
        == loaded.target_corridor.target_for_band("MUD")
    )
    assert "FULL_BAND_LEARNING" in loaded.phase_targets
    assert loaded.phase_targets["SNAPSHOT_LOCK"].runtime_state == "SNAPSHOT_LOCK"


def test_runtime_state_uses_phase_specific_target_corridor_for_analysis():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    source_channel_features = {
        1: extract_analysis_features(sine(320.0, 0.8)),
    }
    source_stem_features = aggregate_stem_features(
        source_channel_features,
        {1: ["GUITARS", "MUSIC"]},
    )
    full_band_channel_features = {
        1: extract_analysis_features(sine(3200.0, 0.8)),
    }
    full_band_stem_features = aggregate_stem_features(
        full_band_channel_features,
        {1: ["GUITARS", "MUSIC"]},
    )
    profile = build_soundcheck_profile(
        channel_features=full_band_channel_features,
        channel_metadata={
            1: {
                "name": "Gtr 1",
                "source_role": "guitar",
                "stem_roles": ["GUITARS", "MUSIC"],
                "allowed_controls": ["eq", "fader"],
                "priority": 0.6,
            }
        },
        stem_features=full_band_stem_features,
        stem_contributions={},
        phase_snapshots={
            "LINE_CHECK": build_phase_learning_snapshot(
                phase_name="LINE_CHECK",
                runtime_state="LINE_CHECK",
                channel_features=source_channel_features,
                stem_features=source_stem_features,
            ),
            "FULL_BAND_LEARNING": build_phase_learning_snapshot(
                phase_name="FULL_BAND_LEARNING",
                runtime_state="FULL_BAND_LEARNING",
                channel_features=full_band_channel_features,
                stem_features=full_band_stem_features,
            ),
            "SNAPSHOT_LOCK": build_phase_learning_snapshot(
                phase_name="SNAPSHOT_LOCK",
                runtime_state="SNAPSHOT_LOCK",
                channel_features=full_band_channel_features,
                stem_features=full_band_stem_features,
            ),
        },
    )

    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.audio_capture = DetectorAudioCapture({1: sine(3200.0, 0.8)})
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Gtr 1",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["eq", "fader"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            fader_db=-8.0,
            priority=0.6,
        ),
    }
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_loaded_target_corridor = True

    line_check_features = engine._capture_live_analysis_features(1, RuntimeState.LINE_CHECK)
    chorus_features = engine._capture_live_analysis_features(1, RuntimeState.CHORUS)
    _, line_stem_features, _, _ = engine._collect_autofoh_monitor_features(
        runtime_state=RuntimeState.LINE_CHECK
    )
    _, chorus_stem_features, _, _ = engine._collect_autofoh_monitor_features(
        runtime_state=RuntimeState.CHORUS
    )

    assert line_check_features is not None
    assert chorus_features is not None
    assert (
        abs(float(chorus_features.mix_indexes.presence_index))
        < abs(float(line_check_features.mix_indexes.presence_index))
    )
    assert (
        abs(float(chorus_stem_features["MASTER"].mix_indexes.presence_index))
        < abs(float(line_stem_features["MASTER"].mix_indexes.presence_index))
    )
    assert (
        engine._target_corridor_for_runtime_state(RuntimeState.CHORUS).target_for_band("PRESENCE")
        == profile.phase_targets["FULL_BAND_LEARNING"].target_corridor.target_for_band("PRESENCE")
    )


def test_phase_target_guard_blocks_detector_eq_when_stem_is_below_learned_baseline():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.PRE_SHOW_CHECK
    engine.audio_capture = DetectorAudioCapture(
        {
            1: sine(2200.0, 0.2),
            2: sine(2800.0, 0.9),
        }
    )
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            source_role="lead_vocal",
            stem_roles=["LEAD"],
            allowed_controls=["fader", "eq"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            fader_db=-6.0,
            priority=1.0,
        ),
        2: ChannelInfo(
            channel=2,
            name="Gtr 1",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["eq", "fader"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            fader_db=-8.0,
            priority=0.6,
        ),
    }
    engine.lead_masking_analyzer = LeadMaskingAnalyzer(
        masking_threshold_db=3.0,
        persistence_required_cycles=1,
    )
    profile = build_soundcheck_profile(
        channel_features={
            1: extract_analysis_features(sine(2200.0, 0.2)),
            2: extract_analysis_features(sine(2800.0, 0.9)),
        },
        channel_metadata={
            1: {
                "name": "Lead Vox",
                "source_role": "lead_vocal",
                "stem_roles": ["LEAD"],
                "allowed_controls": ["eq", "fader"],
                "priority": 1.0,
            },
            2: {
                "name": "Gtr 1",
                "source_role": "guitar",
                "stem_roles": ["GUITARS", "MUSIC"],
                "allowed_controls": ["eq", "fader"],
                "priority": 0.6,
            },
        },
        stem_features=aggregate_stem_features(
            {
                1: extract_analysis_features(sine(2200.0, 0.2)),
                2: extract_analysis_features(sine(2800.0, 0.9)),
            },
            {1: ["LEAD"], 2: ["GUITARS", "MUSIC"]},
        ),
        stem_contributions={},
        phase_snapshots={
            "SNAPSHOT_LOCK": build_phase_learning_snapshot(
                phase_name="SNAPSHOT_LOCK",
                runtime_state="SNAPSHOT_LOCK",
                channel_features={
                    1: extract_analysis_features(sine(2200.0, 0.2)),
                    2: extract_analysis_features(sine(2800.0, 0.9)),
                },
                stem_features=aggregate_stem_features(
                    {
                        1: extract_analysis_features(sine(2200.0, 0.2)),
                        2: extract_analysis_features(sine(2800.0, 0.9)),
                    },
                    {1: ["LEAD"], 2: ["GUITARS", "MUSIC"]},
                ),
            ),
        },
    )
    profile.phase_targets["SNAPSHOT_LOCK"].expected_stem_rms_db["GUITARS"] = 3.0
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_loaded_target_corridor = True
    engine.soundcheck_profile_use_phase_target_action_guards = True

    decisions = engine._run_autofoh_monitor_analysis()

    assert decisions == []
    assert mixer.calls == []


def test_phase_target_guard_blocks_initial_fader_move_without_mutating_channel_state():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.audio_capture = DetectorAudioCapture({1: sine(320.0, 0.9)})
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Gtr 1",
            preset="electricGuitar",
            source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"],
            allowed_controls=["fader"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            fader_db=-20.0,
            priority=0.6,
        ),
    }
    profile = build_soundcheck_profile(
        channel_features={1: extract_analysis_features(sine(320.0, 0.9))},
        channel_metadata={
            1: {
                "name": "Gtr 1",
                "source_role": "guitar",
                "stem_roles": ["GUITARS", "MUSIC"],
                "allowed_controls": ["fader"],
                "priority": 0.6,
            }
        },
        stem_features=aggregate_stem_features(
            {1: extract_analysis_features(sine(320.0, 0.9))},
            {1: ["GUITARS", "MUSIC"]},
        ),
        stem_contributions={},
        phase_snapshots={
            "SNAPSHOT_LOCK": build_phase_learning_snapshot(
                phase_name="SNAPSHOT_LOCK",
                runtime_state="SNAPSHOT_LOCK",
                channel_features={1: extract_analysis_features(sine(320.0, 0.9))},
                stem_features=aggregate_stem_features(
                    {1: extract_analysis_features(sine(320.0, 0.9))},
                    {1: ["GUITARS", "MUSIC"]},
                ),
            ),
        },
    )
    profile.phase_targets["SNAPSHOT_LOCK"].expected_channel_rms_db[1] = -30.0
    profile.phase_targets["SNAPSHOT_LOCK"].channel_level_tolerance_db = 2.0
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_loaded_target_corridor = True
    engine.soundcheck_profile_use_phase_target_action_guards = True

    phase_guard_context = engine._build_phase_target_guard_context(
        runtime_state=RuntimeState.SNAPSHOT_LOCK
    )
    original_fader_db = engine.channels[1].fader_db

    engine._apply_fader(
        1,
        engine.channels[1],
        "electricGuitar",
        phase_guard_context=phase_guard_context,
    )

    assert mixer.calls == []
    assert engine.channels[1].fader_db == original_fader_db


def test_phase_target_guard_blocks_initial_hpf_above_learned_max():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    samples = sine(8000.0, 0.4)
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.audio_capture = DetectorAudioCapture({1: samples})
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="HH",
            preset="hihat",
            source_role="hihat",
            stem_roles=["CYMBALS", "DRUMS"],
            allowed_controls=["hpf"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            priority=0.4,
        ),
    }
    profile = _build_single_channel_phase_profile(
        samples,
        source_role="hihat",
        stem_roles=["CYMBALS", "DRUMS"],
        allowed_controls=["hpf"],
        name="HH",
    )
    profile.phase_targets["SNAPSHOT_LOCK"].hpf_frequency_range_hz_by_channel[1] = {
        "min_hz": 80.0,
        "max_hz": 180.0,
    }
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_phase_target_action_guards = True

    engine._apply_hpf(
        1,
        "hihat",
        phase_guard_context=engine._build_phase_target_guard_context(
            runtime_state=RuntimeState.SNAPSHOT_LOCK
        ),
    )

    assert mixer.calls == []


def test_phase_target_guard_blocks_initial_compressor_outside_learned_range():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    samples = sine(2200.0, 0.5)
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.audio_capture = DetectorAudioCapture({1: samples})
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            preset="leadVocal",
            source_role="lead_vocal",
            stem_roles=["LEAD"],
            allowed_controls=["compressor"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            priority=1.0,
        ),
    }
    profile = _build_single_channel_phase_profile(
        samples,
        source_role="lead_vocal",
        stem_roles=["LEAD"],
        allowed_controls=["compressor"],
        name="Lead Vox",
    )
    profile.phase_targets["SNAPSHOT_LOCK"].compressor_threshold_range_db_by_channel[1] = {
        "min_db": -14.0,
        "max_db": -10.0,
    }
    profile.phase_targets["SNAPSHOT_LOCK"].compressor_ratio_range_by_channel[1] = {
        "min_ratio": 1.5,
        "max_ratio": 2.0,
    }
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_phase_target_action_guards = True

    engine._apply_compressor(
        1,
        engine.channels[1],
        "leadVocal",
        phase_guard_context=engine._build_phase_target_guard_context(
            runtime_state=RuntimeState.SNAPSHOT_LOCK
        ),
    )

    assert mixer.calls == []
    assert engine.channels[1].compressor_applied is False


def test_phase_target_guard_blocks_initial_fx_send_above_learned_max():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    samples = sine(2200.0, 0.5)
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.audio_capture = DetectorAudioCapture({1: samples})
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            preset="leadVocal",
            source_role="lead_vocal",
            stem_roles=["LEAD"],
            allowed_controls=["fx_send"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=True,
            priority=1.0,
        ),
    }
    profile = _build_single_channel_phase_profile(
        samples,
        source_role="lead_vocal",
        stem_roles=["LEAD"],
        allowed_controls=["fx_send"],
        name="Lead Vox",
    )
    profile.phase_targets["SNAPSHOT_LOCK"].fx_send_level_range_db_by_channel[1] = {
        "min_db": -32.0,
        "max_db": -20.0,
    }
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_phase_target_action_guards = True

    engine._apply_fx_sends(
        1,
        engine.channels[1],
        "leadVocal",
        phase_guard_context=engine._build_phase_target_guard_context(
            runtime_state=RuntimeState.SNAPSHOT_LOCK
        ),
    )

    assert mixer.calls == []


def test_engine_writes_autofoh_session_report_on_stop(tmp_path):
    log_path = tmp_path / "autofoh.jsonl"
    report_path = tmp_path / "autofoh_report.json"

    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.autofoh_logging_enabled = True
    engine.autofoh_write_session_report_on_stop = True
    engine.autofoh_log_path = str(log_path)
    engine.autofoh_report_path = str(report_path)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK

    engine._log_autofoh_event(
        "phase_target_guard_blocked",
        channel_id=1,
        action={"action_type": "ChannelFaderMove"},
        message="phase target guard blocked boost; channel input already above learned baseline",
        metadata={"phase_name": "SNAPSHOT_LOCK"},
    )
    engine._log_autofoh_event(
        "action_decision",
        channel_id=1,
        requested_action={"action_type": "ChannelFaderMove"},
        applied_action={"action_type": "ChannelFaderMove"},
        requested_runtime_state=RuntimeState.SNAPSHOT_LOCK.value,
        sent=False,
        allowed=False,
        supported=True,
        bounded=False,
        rate_limited=False,
        message="phase target guard blocked boost; channel input already above learned baseline",
    )

    engine.stop()

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["guard_block_count"] >= 1
    assert payload["guard_blocks_by_action_type"]["ChannelFaderMove"] >= 1
    assert payload["channels_with_guard_blocks"] == [1]
    assert "guard_blocks=1" in engine.autofoh_session_report_summary
    status = engine.get_status()
    assert "autofoh_session_report_summary" in status
    assert "ChannelFaderMove" in status["autofoh_session_report_summary"]
