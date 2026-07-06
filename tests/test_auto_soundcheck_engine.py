"""Tests for AutoSoundcheckEngine safety wiring."""

import os
import sys
import json
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo, ChannelSnapshot
from autofoh_analysis import extract_analysis_features
from autofoh_detectors import LeadMaskingAnalyzer, aggregate_stem_features
from autofoh_models import RuntimeState
from autofoh_profiles import build_phase_learning_snapshot, build_soundcheck_profile
from autofoh_safety import (AutoFOHSafetyConfig, AutoFOHSafetyController,
                            ChannelEQMove, ChannelFaderMove)
from correction_action_executor import CorrectionActionExecutorConfig, GatedCorrectionActionExecutor
from feedback_detector import FeedbackEvent
from observation_mixer import ObservationMixerClient
from signal_metrics import ChannelMetrics, InterChannelMetrics, LevelMetrics


def _blocked_history(engine, cmd):
    return [
        decision
        for decision in engine.safety_controller.history
        if getattr(decision, "blocked", False)
        and any(call["cmd"] == cmd for call in decision.payload.get("blocked_calls", []))
    ]


def _blocked_calls(engine, cmd):
    return [
        call
        for decision in _blocked_history(engine, cmd)
        for call in decision.payload.get("blocked_calls", [])
        if call["cmd"] == cmd
    ]


def _blocked_console_calls(engine, cmd):
    return [
        call
        for payload in engine.blocked_console_write_history
        for call in payload.get("blocked_calls", [])
        if call["cmd"] == cmd
    ]


def test_auto_soundcheck_source_has_no_direct_mixer_writes():
    source = Path("backend/auto_soundcheck_engine.py").read_text()

    assert "self.mixer_client.set_" not in source
    assert '"set_fader"' in source
    assert "self.mixer_client.set_eq_on(" not in source
    assert "self.mixer_client.set_compressor_threshold(" not in source
    assert "self.mixer_client.set_mute(" not in source
    assert '"set_eq_on"' in source
    assert '"set_compressor_threshold"' in source
    assert '"set_mute"' in source


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


class GatedEngineMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def __init__(self):
        self.fader = {1: -10.0}
        self.calls = []

    def get_fader(self, channel):
        return self.fader.get(channel)

    def set_fader(self, channel, value_db):
        self.fader[channel] = value_db
        self.calls.append(("set_fader", channel, value_db))
        return True


def test_engine_routes_actions_through_gated_executor_and_exports_bundle():
    mixer = GatedEngineMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
        observe_only=True,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(
        mixer,
        config=engine.action_safety_config,
    )
    engine.correction_executor = GatedCorrectionActionExecutor(
        engine.safety_controller,
        config=CorrectionActionExecutorConfig.from_config({
            "promoted_action_types": {"ChannelFaderMove": True},
        }),
    )

    decision = engine._execute_action(
        ChannelFaderMove(channel_id=1, target_db=-9.5, reason="observe-only fader proof"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    bundle = engine.get_soundcheck_recommendation_bundle()

    assert decision is not None
    assert decision.sent is False
    assert decision.message == "dry_run_only"
    assert mixer.calls == []
    assert bundle["recommendation_count"] == 1
    assert bundle["proposals"][0]["action_type"] == "set_fader_db"


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


class GainTrackingMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def __init__(self):
        self.gain_calls = []

    def set_gain(self, channel, value):
        self.gain_calls.append((channel, value))
        return True


class DetectorMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def __init__(self):
        self.fader = {1: -6.0, 2: -8.0}
        self.main_fader = {1: -2.0}
        self.delay = {}
        self.polarity = {}
        self.bus_fader = {}
        self.eq_gain = {}
        self.eq_freq = {}
        self.hpf = {}
        self.compressor = {}
        self.compressor_threshold = {}
        self.compressor_gr = {}
        self.gate = {}
        self.send_level = {}
        self.bus_settings = {
            5: {
                "bus": 5,
                "name": "VOC BUS",
                "fader_db": -4.0,
                "muted": False,
                "eq_bands": [],
                "compressor_enabled": True,
                "dca_assignments": [6],
                "tags": "#D6",
            }
        }
        self.dca_settings = {
            2: {"dca": 2, "name": "Lead DCA", "fader_db": -3.0, "muted": False},
            6: {"dca": 6, "name": "Vocal DCA", "fader_db": -2.0, "muted": False},
        }
        self.calls = []

    def get_fader(self, channel):
        return self.fader.get(channel, -10.0)

    def get_dca_fader(self, dca):
        return getattr(self, "dca_fader", {}).get(dca, 0.0)

    def set_dca_fader(self, dca, value_db):
        if not hasattr(self, "dca_fader"):
            self.dca_fader = {}
        self.dca_fader[dca] = value_db
        self.calls.append(("set_dca_fader", dca, value_db))
        return True

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

    def get_main_fader(self, main=1):
        return self.main_fader.get(main, 0.0)

    def set_main_fader(self, main, value_db):
        self.main_fader[main] = value_db
        self.calls.append(("set_main_fader", main, value_db))
        return True

    def get_delay(self, channel):
        return self.delay.get(channel, 0.0)

    def set_delay(self, channel, delay_ms, enabled=True):
        self.delay[channel] = delay_ms
        self.calls.append(("set_delay", channel, delay_ms, enabled))
        return True

    def get_polarity(self, channel):
        return self.polarity.get(channel, False)

    def set_polarity(self, channel, inverted):
        self.polarity[channel] = inverted
        self.calls.append(("set_polarity", channel, inverted))
        return True

    def get_bus_fader(self, bus):
        if bus in self.bus_fader:
            return self.bus_fader[bus]
        return self.bus_settings.get(bus, {}).get("fader_db", -100.0)

    def set_bus_fader(self, bus, value_db):
        self.bus_fader[bus] = value_db
        self.calls.append(("set_bus_fader", bus, value_db))
        return True

    def get_channel_settings(self, channel):
        return {
            "channel": channel,
            "name": f"Ch {channel}",
            "fader_db": self.fader.get(channel, -10.0),
            "muted": False,
            "gain_db": 0.0,
            "hpf_freq": 20.0,
            "hpf_enabled": False,
            "eq_on": True,
            "eq_bands": [
                (90.0, self.eq_gain.get((channel, 1), 0.0), 1.0),
                (320.0, self.eq_gain.get((channel, 2), 0.0), 1.0),
                (2500.0, self.eq_gain.get((channel, 3), 0.0), 1.0),
                (7600.0, self.eq_gain.get((channel, 4), 0.0), 1.0),
            ],
            "input_routing": {
                "main_group": "MOD",
                "main_channel": channel,
                "alt_group": None,
                "alt_channel": None,
            },
            "main_send": {"on": 1, "level_db": 0.0, "pre": 0},
            "active_bus_sends": [],
            "dca_assignments": [],
        }

    def get_bus_settings(self, bus):
        self.calls.append(("get_bus_settings", bus))
        return dict(self.bus_settings.get(bus, {"bus": bus, "fader_db": -100.0, "muted": False}))

    def get_dca_settings(self, dca):
        self.calls.append(("get_dca_settings", dca))
        return dict(self.dca_settings.get(dca, {"dca": dca, "fader_db": -100.0, "muted": False}))

    def set_eq_on(self, channel, on):
        self.calls.append(("set_eq_on", channel, on))
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

    def set_compressor_threshold(self, channel, threshold):
        self.compressor_threshold[channel] = threshold
        self.calls.append(("set_compressor_threshold", channel, threshold))
        return True

    def set_compressor_gain(self, channel, gain):
        self.calls.append(("set_compressor_gain", channel, gain))
        return True

    def get_compressor_gr(self, channel):
        return self.compressor_gr.get(channel)

    def set_gate(self, channel, threshold, range_db, attack, hold, release, ratio=None):
        self.gate[channel] = (threshold, range_db, attack, hold, release, ratio)
        self.calls.append(("set_gate", channel, threshold, range_db, attack, hold, release, ratio))
        return True

    def set_gate_on(self, channel, on):
        self.calls.append(("set_gate_on", channel, on))
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


def test_engine_preserves_existing_processing_by_default():
    mixer = ResetTrackingMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.mixer_client = mixer

    engine._scan_channels()
    engine._reset_channels()

    assert mixer.reset_calls == []
    assert engine.channels[1].auto_corrections_enabled is False
    assert engine.channels[2].auto_corrections_enabled is True


def test_destructive_reset_requires_explicit_opt_in():
    mixer = ResetTrackingMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.preserve_existing_processing = False
    engine.allow_destructive_reset = True

    engine._scan_channels()
    engine._reset_channels()

    assert mixer.reset_calls == [2]
    assert engine.channels[1].auto_corrections_enabled is False
    assert engine.channels[2].auto_corrections_enabled is True


def test_master_reference_channels_are_not_source_corrected():
    engine = AutoSoundcheckEngine(
        num_channels=24,
        auto_discover=False,
    )
    engine.master_reference_channels = {23, 24}
    info = ChannelInfo(channel=23, name="Master Ref")

    engine._apply_classification_to_info(
        info,
        {
            "preset": "playback",
            "source_role": "playback",
            "stem_roles": ["PLAYBACK"],
            "allowed_controls": ["gain", "hpf", "eq", "compressor", "fader"],
            "confidence": 0.99,
            "match_type": "test",
            "recognized": True,
        },
    )

    assert info.source_role == "mix_bus_reference"
    assert info.stem_roles == ["MASTER"]
    assert info.allowed_controls == []
    assert info.auto_corrections_enabled is False


def test_channel_state_read_captures_group_bus_and_dca_routes():
    mixer = DetectorMixer()
    base_get_channel_settings = mixer.get_channel_settings

    def routed_channel_settings(channel):
        settings = base_get_channel_settings(channel)
        settings["active_bus_sends"] = [
            {
                "bus": 5,
                "on": 1,
                "level_db": -12.0,
                "mode": "POST",
                "active": True,
            }
        ]
        settings["dca_assignments"] = [2]
        settings["tags"] = "#D2"
        return settings

    mixer.get_channel_settings = routed_channel_settings
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            preset="leadVocal",
            auto_corrections_enabled=True,
        )
    }

    engine._read_channel_state()
    status = engine.get_status()

    assert engine.channel_group_routes[1] == {"bus_ids": [5], "dca_ids": [2]}
    assert engine.bus_snapshots[5].name == "VOC BUS"
    assert engine.bus_snapshots[5].source_channels == [1]
    assert engine.dca_snapshots[2].source_channels == [1]
    assert engine.dca_snapshots[6].source_buses == [5]
    assert status["channels"][1]["active_bus_sends"] == [5]
    assert status["group_buses"][5]["dca_assignments"] == [6]
    assert status["dca_groups"][6]["source_buses"] == [5]


def test_group_bus_correction_skips_monitor_buses():
    mixer = DetectorMixer()
    mixer.bus_settings[12] = {
        "bus": 12,
        "name": "VOC GROUP",
        "fader_db": 2.0,
        "muted": False,
        "dca_assignments": [],
    }
    mixer.bus_settings[13] = {
        "bus": 13,
        "name": "DIMA MON",
        "fader_db": 2.0,
        "muted": False,
        "dca_assignments": [],
    }
    base_get_channel_settings = mixer.get_channel_settings

    def routed_channel_settings(channel):
        settings = base_get_channel_settings(channel)
        settings["active_bus_sends"] = [
            {"bus": 12, "on": 1, "level_db": -6.0, "active": True},
            {"bus": 13, "on": 1, "level_db": -6.0, "active": True},
        ]
        return settings

    mixer.get_channel_settings = routed_channel_settings
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    engine.monitor_bus_ids = {13, 14, 15, 16}
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            preset="leadVocal",
            has_signal=True,
            auto_corrections_enabled=True,
            lufs=-18.0,
        )
    }

    engine._read_channel_state()
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    engine._apply_group_bus_corrections()

    bus_blocks = _blocked_history(engine, "set_bus_fader")
    assert bus_blocks
    assert bus_blocks[0].payload["blocked_calls"][0]["bus"] == 12
    assert bus_blocks[0].payload["blocked_calls"][0]["value_db"] == 0.0
    assert not any(call[0] == "set_bus_fader" for call in mixer.calls)
    assert not any(call[:2] == ("set_bus_fader", 13) for call in mixer.calls)


def test_drum_phase_delay_corrects_processed_channels_through_safety(monkeypatch):
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_apply=True,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    engine.audio_capture = DetectorAudioCapture(
        {
            1: np.ones(48000, dtype=np.float32) * 0.1,
            2: np.ones(48000, dtype=np.float32) * 0.1,
        }
    )
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="KICK IN",
            preset="kick",
            has_signal=True,
            original_snapshot=ChannelSnapshot(
                channel=1,
                had_processing=True,
                raw_settings={"delay_enabled": 0, "delay_ms": 0.0},
            ),
        ),
        2: ChannelInfo(
            channel=2,
            name="KICK OUT",
            preset="kick",
            has_signal=True,
            original_snapshot=ChannelSnapshot(
                channel=2,
                had_processing=True,
                raw_settings={"delay_enabled": 0, "delay_ms": 0.0},
            ),
        ),
    }

    monkeypatch.setattr(
        "auto_soundcheck_engine.compare_channels",
        lambda *_args, **_kwargs: InterChannelMetrics(
            channel_a=1,
            channel_b=2,
            cross_correlation=0.8,
            delay_samples=24,
            delay_ms=0.5,
            coherence=0.9,
            spectral_similarity=0.9,
            phase_inverted=False,
        ),
    )

    engine._detect_and_fix_phase()

    delay_blocks = _blocked_history(engine, "set_delay")
    assert delay_blocks
    assert delay_blocks[0].payload["blocked_calls"][0]["channel"] == 2
    # саундчек-деривация (SOURCE_LEARNING): абсолютная цель выравнивания,
    # без пошагового live-лимита
    assert delay_blocks[0].payload["blocked_calls"][0]["delay_ms"] == 0.5
    assert not any(call[0] == "set_delay" for call in mixer.calls)


def test_phase_pairing_does_not_align_unrelated_lead_vocals():
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_discover=False,
    )
    engine.channels = {
        1: ChannelInfo(channel=1, name="KATYA", preset="leadVocal", has_signal=True),
        2: ChannelInfo(channel=2, name="SERGEY", preset="leadVocal", has_signal=True),
    }

    assert engine._find_correlated_pairs() == []


def test_live_shared_mix_uses_master_reference_without_source_correcting_it():
    def sine(freq_hz, amplitude):
        t = np.arange(48000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=24,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    engine.master_reference_channels = {23, 24}
    engine.shared_chat_mix_config.analysis_window_sec = 1.0
    engine.audio_capture = DetectorAudioCapture(
        {
            1: sine(1000.0, 0.2),
            23: sine(1000.0, 0.9),
            24: sine(1000.0, 0.9),
        }
    )
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            preset="leadVocal",
            source_role="lead_vocal",
            stem_roles=["LEAD"],
            allowed_controls=[],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=False,
            fader_db=-6.0,
        ),
        23: ChannelInfo(
            channel=23,
            name="Master L",
            source_role="mix_bus_reference",
            stem_roles=["MASTER"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=False,
        ),
        24: ChannelInfo(
            channel=24,
            name="Master R",
            source_role="mix_bus_reference",
            stem_roles=["MASTER"],
            has_signal=True,
            recognized=True,
            auto_corrections_enabled=False,
        ),
    }

    decisions = engine._apply_live_shared_mix_pass()

    assert any(decision.blocked for decision in decisions)
    assert not any(call[0] == "set_main_fader" for call in mixer.calls)
    assert not any(call[0] == "set_eq_band" and call[1] in {23, 24} for call in mixer.calls)


def test_input_gain_correction_does_not_boost_trim_by_default():
    mixer = GainTrackingMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Kick",
        preset="kick",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
        peak_db=-16.0,
        rms_db=-30.0,
        original_snapshot=ChannelSnapshot(
            channel=1,
            gain_db=4.0,
            raw_settings={"gain_db": 4.0},
            had_processing=True,
        ),
    )
    engine.channels = {1: info}

    engine._apply_input_gain(1, info, "kick")

    assert mixer.gain_calls == []


def test_input_gain_correction_still_reduces_hot_trim():
    mixer = GainTrackingMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Kick",
        preset="kick",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
        peak_db=-3.0,
        rms_db=-18.0,
        original_snapshot=ChannelSnapshot(
            channel=1,
            gain_db=8.0,
            raw_settings={"gain_db": 8.0},
            had_processing=True,
        ),
    )
    engine.channels = {1: info}

    engine._apply_input_gain(1, info, "kick")

    assert mixer.gain_calls[0][0] == 1
    assert mixer.gain_calls[0][1] < 8.0


def test_eq_correction_enables_eq_and_retunes_mismatched_existing_band():
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    # сценарий теста — БЕЗ операторского промоута EQ: включение eq/on
    # остаётся supervised-предложением (blocked evidence)
    engine.correction_executor_config.promoted_action_types["ChannelEQMove"] = False
    info = ChannelInfo(
        channel=1,
        name="Snare",
        preset="snare",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
        allowed_controls=["eq"],
        original_snapshot=ChannelSnapshot(
            channel=1,
            raw_settings={
                "eq_on": False,
                "eq_bands": [
                    (120.0, 1.5, 1.5),
                    (5359.0, -6.0, 10.0),
                    (2500.0, 3.0, 2.5),
                    (5359.0, -6.0, 10.0),
                ],
            },
            had_processing=True,
        ),
    )
    engine.channels = {1: info}

    engine._apply_eq(1, "snare")

    eq_on_blocks = _blocked_console_calls(engine, "set_eq_on")
    assert eq_on_blocks == [
        {
            "cmd": "set_eq_on",
            "channel": 1,
            "on": 1,
            "semantic": "enable_eq_before_auto_soundcheck_eq",
        }
    ]
    assert engine.blocked_console_write_history[-1]["write_policy"] == "proposal_only_quarantined"
    assert engine.blocked_console_write_history[-1]["mutates_mixer"] is False
    assert ("set_eq_on", 1, 1) not in mixer.calls
    band2_calls = [call for call in mixer.calls if call[0] == "set_eq_band" and call[2] == 2]
    assert not band2_calls
    eq_blocks = _blocked_history(engine, "set_eq_band")
    blocked_band2 = [
        call
        for decision in eq_blocks
        for call in decision.payload["blocked_calls"]
        if call.get("band") == 2
    ]
    assert blocked_band2
    assert blocked_band2[0]["freq"] == pytest.approx(400.0)


def test_gate_is_enabled_for_close_drum_mics():
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Rack Tom",
        preset="tom",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
    )
    engine.channels = {1: info}

    engine._apply_gate(1, info, "tom")

    assert _blocked_history(engine, "set_gate")
    assert _blocked_history(engine, "set_gate_on")
    assert not any(call[0] == "set_gate" for call in mixer.calls)
    assert ("set_gate_on", 1, 1) not in mixer.calls


def test_automatic_fx_sends_are_disabled_by_default():
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Lead Vox",
        preset="leadVocal",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
    )
    engine.channels = {1: info}

    engine._apply_fx_sends(1, info, "leadVocal")

    assert not any(call[0] == "set_send_level" for call in mixer.calls)


def test_character_compressor_model_is_preserved_when_gr_is_available():
    mixer = DetectorMixer()
    mixer.compressor_gr[1] = 12.0
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Kick",
        preset="kick",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
        original_snapshot=ChannelSnapshot(
            channel=1,
            raw_settings={
                "compressor_enabled": True,
                "compressor_model": "B160",
                "compressor_threshold_db": -18.0,
            },
            had_processing=True,
        ),
    )
    engine.channels = {1: info}

    engine._apply_compressor(1, info, "kick")

    threshold_blocks = _blocked_console_calls(engine, "set_compressor_threshold")
    assert threshold_blocks
    assert threshold_blocks[-1] == {
        "cmd": "set_compressor_threshold",
        "channel": 1,
        "threshold_db": -12.0,
        "current_threshold_db": -18.0,
        "compressor_model": "B160",
        "gain_reduction_db": 12.0,
        "semantic": "preserve_character_compressor_threshold_relief",
    }
    assert not any(call[0] == "set_compressor_threshold" for call in mixer.calls)
    makeup_blocks = _blocked_calls(engine, "set_compressor_gain")
    assert makeup_blocks
    assert makeup_blocks[-1]["channel"] == 1
    # полная компенсация GR (ratio 1.0; оператор 2026-07-06)
    assert makeup_blocks[-1]["makeup_db"] == 4.0
    assert not any(call[0] == "set_compressor_gain" for call in mixer.calls)
    assert not any(call[0] == "set_compressor" for call in mixer.calls)


def test_existing_generic_compressor_makeup_compensates_gain_reduction():
    mixer = DetectorMixer()
    mixer.compressor_gr[1] = 6.0
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Vocal",
        preset="leadVocal",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
        original_snapshot=ChannelSnapshot(
            channel=1,
            raw_settings={
                "compressor_enabled": True,
                "compressor_model": "COMP",
                "compressor_threshold_db": -22.0,
                "compressor_ratio": 3.0,
                "compressor_attack_ms": 8.0,
                "compressor_release_ms": 120.0,
                "compressor_makeup_db": 0.0,
            },
            had_processing=True,
        ),
    )
    engine.channels = {1: info}

    engine._apply_compressor(1, info, "leadVocal")

    comp_blocks = _blocked_calls(engine, "set_compressor")
    assert comp_blocks
    assert comp_blocks[-1]["channel"] == 1
    assert comp_blocks[-1]["threshold_db"] == -19.0
    assert comp_blocks[-1]["ratio"] == 3.0
    assert comp_blocks[-1]["attack_ms"] == 5.0
    assert comp_blocks[-1]["release_ms"] == 80.0
    # полная компенсация GR (ratio 1.0; оператор 2026-07-06)
    assert comp_blocks[-1]["makeup_db"] == 4.0
    assert comp_blocks[-1]["enabled"] is True
    assert not any(call[0] == "set_compressor" for call in mixer.calls)


def test_b160_missing_gr_meter_gets_conservative_makeup_from_signal_headroom():
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SOURCE_LEARNING
    info = ChannelInfo(
        channel=1,
        name="Kick Out",
        preset="kick",
        has_signal=True,
        recognized=True,
        auto_corrections_enabled=True,
        metrics=ChannelMetrics(
            channel=1,
            level=LevelMetrics(
                peak_db=-6.0,
                true_peak_dbtp=-6.0,
                rms_db=-26.0,
                crest_factor_db=20.0,
            ),
        ),
        original_snapshot=ChannelSnapshot(
            channel=1,
            raw_settings={
                "compressor_enabled": True,
                "compressor_model": "B160",
                "compressor_threshold_db": 0.01,
                "compressor_mix_pct": 100.0,
                "compressor_makeup_db": 0.0,
            },
            had_processing=True,
        ),
    )
    engine.channels = {1: info}

    engine._apply_compressor(1, info, "kick")

    makeup_blocks = _blocked_calls(engine, "set_compressor_gain")
    assert makeup_blocks
    assert makeup_blocks[-1]["channel"] == 1
    # headroom-ограниченный makeup при ratio 1.0
    assert makeup_blocks[-1]["makeup_db"] == 1.5
    assert not any(call[0] == "set_compressor_gain" for call in mixer.calls)
    assert not any(call[0] == "set_compressor" for call in mixer.calls)


def test_feedback_events_are_cooled_down_and_use_least_busy_eq_band():
    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_discover=False,
    )
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.feedback_event_min_interval_sec = 30.0
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            allowed_controls=["feedback_notch"],
            original_snapshot=ChannelSnapshot(
                channel=1,
                raw_settings={
                    "eq_on": True,
                    "eq_bands": [
                        (120.0, 2.0, 1.0),
                        (350.0, -1.5, 1.2),
                        (2500.0, 0.0, 1.5),
                        (8000.0, 3.0, 1.0),
                    ],
                },
                had_processing=True,
            ),
        )
    }
    event = FeedbackEvent(
        channel=1,
        frequency_hz=3150.0,
        magnitude_db=-10.0,
        action="notch",
        confidence=0.9,
    )

    engine._handle_feedback_event(1, event)
    engine._handle_feedback_event(1, event)

    eq_calls = [call for call in mixer.calls if call[0] == "set_eq_band"]
    assert not eq_calls
    eq_blocks = _blocked_calls(engine, "set_eq_band")
    assert len(eq_blocks) == 1
    assert eq_blocks[0]["band"] == 3


def test_monitor_analysis_routes_lead_masking_action_through_safety_layer():
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(
        num_channels=2,
        auto_apply=True,
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
    assert decisions[0].sent is False
    assert decisions[0].blocked is True
    eq_blocks = _blocked_calls(engine, "set_eq_band")
    assert eq_blocks
    assert eq_blocks[-1]["channel"] == 2
    assert not any(call[0] == "set_eq_band" for call in mixer.calls)


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

    assert decision is not None
    assert decision.sent is False
    assert decision.blocked is True

    outcomes = engine._evaluate_pending_actions(force=True)

    assert outcomes == []
    eq_blocks = _blocked_calls(engine, "set_eq_band")
    assert eq_blocks
    assert eq_blocks[-1]["channel"] == 1
    assert eq_blocks[-1]["band"] == 2
    assert eq_blocks[-1]["freq"] == 350.0
    assert eq_blocks[-1]["gain"] == -1.0
    assert not any(call[0] == "set_eq_band" for call in mixer.calls)


def test_restore_original_settings_blocks_fader_and_mute_restore():
    class RestoreMixer:
        def __init__(self):
            self.calls = []

        def set_fader(self, channel, value_db):
            self.calls.append(("set_fader", channel, value_db))
            return True

        def set_mute(self, channel, muted):
            self.calls.append(("set_mute", channel, muted))
            return True

    mixer = RestoreMixer()
    engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
    engine.mixer_client = mixer
    engine.channels = {
        1: ChannelInfo(
            channel=1,
            name="Lead Vox",
            original_snapshot=ChannelSnapshot(
                channel=1,
                fader_db=-12.0,
                muted=True,
                had_processing=True,
            ),
        )
    }

    engine.restore_original_settings()

    assert mixer.calls == []
    fader_blocks = _blocked_console_calls(engine, "set_fader")
    assert fader_blocks == [
        {
            "cmd": "set_fader",
            "channel": 1,
            "value_db": -12.0,
            "manual_gate": "apply_operator_proposal",
            "semantic": "restore_original_fader_state",
        }
    ]
    assert ("set_mute", 1, True) not in mixer.calls
    mute_blocks = _blocked_console_calls(engine, "set_mute")
    assert mute_blocks == [
        {
            "cmd": "set_mute",
            "channel": 1,
            "muted": True,
            "manual_gate": "apply_operator_proposal",
            "semantic": "restore_original_mute_state",
        }
    ]
    assert engine.blocked_console_write_history[-1]["write_policy"] == "proposal_only_quarantined"
    assert engine.blocked_console_write_history[-1]["can_apply_live"] is False


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


def test_phase_target_guard_allows_mirror_eq_inside_green_corridor():
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
    engine.audio_capture = DetectorAudioCapture({1: sine(2800.0, 0.4)})
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
    features = {1: extract_analysis_features(sine(2800.0, 0.4))}
    stems = aggregate_stem_features(features, {1: ["GUITARS", "MUSIC"]})
    profile = build_soundcheck_profile(
        channel_features=features,
        channel_metadata={
            1: {
                "name": "Gtr 1",
                "source_role": "guitar",
                "stem_roles": ["GUITARS", "MUSIC"],
                "allowed_controls": ["eq", "fader"],
                "priority": 0.6,
            },
        },
        stem_features=stems,
        stem_contributions={},
        phase_snapshots={
            "SNAPSHOT_LOCK": build_phase_learning_snapshot(
                phase_name="SNAPSHOT_LOCK",
                runtime_state="SNAPSHOT_LOCK",
                channel_features=features,
                stem_features=stems,
            ),
        },
    )
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_loaded_target_corridor = True
    engine.soundcheck_profile_use_phase_target_action_guards = True

    decision = engine._execute_action(
        ChannelEQMove(
            channel_id=1,
            band=3,
            freq_hz=3000.0,
            gain_db=-1.0,
            q=4.0,
            reason="Mirror EQ cut masker: cross-adaptive overlap at 3000Hz",
        ),
        runtime_state=RuntimeState.SNAPSHOT_LOCK,
        phase_guard_context=engine._build_phase_target_guard_context(
            runtime_state=RuntimeState.SNAPSHOT_LOCK
        ),
    )

    assert decision is not None
    assert decision.sent is False
    assert decision.blocked is True
    eq_blocks = _blocked_calls(engine, "set_eq_band")
    assert eq_blocks
    assert eq_blocks[-1]["channel"] == 1
    assert eq_blocks[-1]["band"] == 3
    assert not any(call[0] == "set_eq_band" for call in mixer.calls)


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


def _burst_signal(sr, hits, amp, seed=0):
    """4 c сигнала: удары amp в моменты hits (сек) + слабый пол."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.001, 4 * sr).astype(np.float32)
    for t in hits:
        i = int(t * sr)
        x[i:i + int(0.04 * sr)] += amp * rng.normal(0, 1, int(0.04 * sr)).astype(np.float32)
    return x


def test_bleed_dominated_tom_frozen_and_snare_with_own_hits_not():
    """Замечание оператора 2026-07-06: TOM регулировался по блидингу.

    Том, в котором каждый онсет громче звучит в снейр-канале (bleed),
    замораживается; снейр с собственными ударами — нет."""
    sr = 48000
    hits = [0.5, 1.2, 1.9, 2.6, 3.3]
    snare = _burst_signal(sr, hits, amp=0.5, seed=1)
    tom_bleed = _burst_signal(sr, hits, amp=0.05, seed=2)  # те же моменты, тише

    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.audio_capture = DetectorAudioCapture({1: snare, 2: tom_bleed})
    engine.channels = {
        1: ChannelInfo(channel=1, name="SNARE", preset="snare", has_signal=True),
        2: ChannelInfo(channel=2, name="TOM", preset="tom", has_signal=True),
    }

    assert engine._close_mic_bleed_dominated(2, engine.channels[2], "tom") is True
    assert engine._close_mic_bleed_dominated(1, engine.channels[1], "snare") is False
    # не-барабанные пресеты не замораживаются этим детектором
    assert engine._close_mic_bleed_dominated(1, engine.channels[1], "leadVocal") is False


def test_snare_pair_mics_do_not_freeze_each_other():
    """Нижний снейр горячее верхнего на том же ударе — это не bleed."""
    sr = 48000
    hits = [0.5, 1.2, 1.9, 2.6, 3.3]
    snare_top = _burst_signal(sr, hits, amp=0.2, seed=3)
    snare_bot = _burst_signal(sr, hits, amp=0.6, seed=4)   # горячее

    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.audio_capture = DetectorAudioCapture({1: snare_top, 2: snare_bot})
    engine.channels = {
        1: ChannelInfo(channel=1, name="SNARE T", preset="snare", has_signal=True),
        2: ChannelInfo(channel=2, name="SNARE B", preset="snare", has_signal=True),
    }
    assert engine._close_mic_bleed_dominated(1, engine.channels[1], "snare") is False
    assert engine._close_mic_bleed_dominated(2, engine.channels[2], "snare") is False


def test_phase_target_guard_allows_mirror_eq_when_stem_below_baseline():
    """Live 2026-07-06: стем-ветка стража душила mirror EQ (planned=8 sent=2).
    Миррор-сепарация — не движение уровня стема; страж её пропускает."""
    def sine(freq_hz, amplitude):
        t = np.arange(12000, dtype=np.float32) / 48000.0
        return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    mixer = DetectorMixer()
    engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    # профиль выучен на ГРОМКОМ сигнале, текущий — тихий: стем ниже baseline
    loud = {1: extract_analysis_features(sine(2800.0, 0.4))}
    loud_stems = aggregate_stem_features(loud, {1: ["GUITARS", "MUSIC"]})
    engine.audio_capture = DetectorAudioCapture({1: sine(2800.0, 0.02)})
    engine.channels = {
        1: ChannelInfo(
            channel=1, name="Gtr 1", source_role="guitar",
            stem_roles=["GUITARS", "MUSIC"], allowed_controls=["eq", "fader"],
            has_signal=True, recognized=True, auto_corrections_enabled=True,
            fader_db=-8.0, priority=0.6,
        ),
    }
    profile = build_soundcheck_profile(
        channel_features=loud,
        channel_metadata={
            1: {"name": "Gtr 1", "source_role": "guitar",
                "stem_roles": ["GUITARS", "MUSIC"],
                "allowed_controls": ["eq", "fader"], "priority": 0.6},
        },
        stem_features=loud_stems,
        stem_contributions={},
        phase_snapshots={
            "SNAPSHOT_LOCK": build_phase_learning_snapshot(
                phase_name="SNAPSHOT_LOCK", runtime_state="SNAPSHOT_LOCK",
                channel_features=loud, stem_features=loud_stems,
            ),
        },
    )
    engine.loaded_soundcheck_profile = profile
    engine.soundcheck_profile_use_loaded_target_corridor = True
    engine.soundcheck_profile_use_phase_target_action_guards = True

    decision = engine._execute_action(
        ChannelEQMove(
            channel_id=1, band=3, freq_hz=3000.0, gain_db=-1.0, q=4.0,
            reason="Mirror EQ cut masker: cross-adaptive overlap at 3000Hz",
        ),
        runtime_state=RuntimeState.SNAPSHOT_LOCK,
        phase_guard_context=engine._build_phase_target_guard_context(
            runtime_state=RuntimeState.SNAPSHOT_LOCK
        ),
    )
    assert decision is not None
    # страж пропустил (нет stem-блока); дальше честный proposal-only карантин
    assert "stem" not in (decision.message or "")
    assert decision.blocked is True
    assert decision.write_policy == "proposal_only_quarantined"


def test_channel_trim_db_reads_console_state():
    """USB-tap pre-trim: пороги из захвата сдвигаются на трим канала."""
    mixer = DetectorMixer()
    mixer.state = {"/ch/1/in/set/trim": -4.0}
    engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
    engine.mixer_client = mixer
    engine.channels = {
        1: ChannelInfo(channel=1, name="KICK", preset="kick", has_signal=True),
    }
    assert engine._channel_trim_db(1) == pytest.approx(-4.0)
    assert engine._channel_trim_db(2) == pytest.approx(0.0)  # нет данных -> 0


def test_balance_solution_measured_in_console_domain():
    """Оператор 2026-07-06: гитары с тримом −8 получали ещё и фейдер −16.
    Решатель обязан видеть capture+trim, а не сырой pre-trim захват."""
    mixer = DetectorMixer()
    mixer.state = {"/ch/1/in/set/trim": 0.0, "/ch/2/in/set/trim": -8.0}
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.mixer_client = mixer

    class _Lvl:
        def __init__(self, lufs): self.lufs_integrated = lufs
    class _M:
        def __init__(self, lufs): self.level = _Lvl(lufs)

    engine.channels = {
        1: ChannelInfo(channel=1, name="VOCAL", preset="leadVocal",
                       source_role="lead_vocal", has_signal=True,
                       recognized=True, auto_corrections_enabled=True,
                       fader_db=0.0),
        2: ChannelInfo(channel=2, name="GTR", preset="electricGuitar",
                       source_role="guitar", has_signal=True,
                       recognized=True, auto_corrections_enabled=True,
                       fader_db=0.0),
    }
    engine.channels[1].metrics = _M(-22.0)
    engine.channels[2].metrics = _M(-18.0)   # pre-trim громче вокала

    solution = engine._compute_balance_v3_solution()
    assert solution is not None
    # в консольном домене гитара уже -26 (trim −8) — ТИШЕ вокала;
    # решатель не должен резать её глубоко
    assert engine._balance_trim_at_solve[2] == pytest.approx(-8.0)
    fader_gtr = solution.faders_db.get(2)
    fader_voc = solution.faders_db.get(1)
    assert fader_gtr is not None and fader_voc is not None
    # без учёта трима гитара получила бы срез глубже вокального минимум
    # на разницу LUFS (4 дБ); с учётом — её срез МЕНЬШЕ вокального
    assert fader_gtr >= fader_voc - 1.0


def test_balance_measured_is_post_fader_no_inversion_on_repeat_runs():
    """Оператор 2026-07-06: «мало вокала и playback, громко гитары».
    Pre-fader measured инвертировал баланс на повторных прогонах: вокал с
    тримом +12 выглядел «уже громким», гитара с −12 — «тихой». Решатель
    обязан получать ПОСТ-фейдерный уровень и восстанавливать иерархию."""
    mixer = DetectorMixer()
    mixer.state = {"/ch/1/in/set/trim": 12.0, "/ch/2/in/set/trim": -12.0}
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.mixer_client = mixer

    class _Lvl:
        def __init__(self, lufs): self.lufs_integrated = lufs
    class _M:
        def __init__(self, lufs): self.level = _Lvl(lufs)

    engine.channels = {
        1: ChannelInfo(channel=1, name="VOCAL", preset="leadVocal",
                       source_role="lead_vocal", has_signal=True,
                       recognized=True, auto_corrections_enabled=True,
                       fader_db=-30.0),
        2: ChannelInfo(channel=2, name="GTR", preset="electricGuitar",
                       source_role="guitar", has_signal=True,
                       recognized=True, auto_corrections_enabled=True,
                       fader_db=-10.0),
    }
    engine.channels[1].metrics = _M(-35.0)   # capture pre-trim одинаковый
    engine.channels[2].metrics = _M(-35.0)

    solution = engine._compute_balance_v3_solution()
    assert solution is not None
    # пост-фейдерные уровни решения: вокал должен оказаться ВЫШЕ гитары
    post = {
        ch: -35.0 + engine._channel_trim_db(ch) + solution.faders_db[ch]
        for ch in (1, 2)
    }
    assert post[1] > post[2]  # вокал над гитарой (план: −22 vs −25)


def test_fx_sends_skipped_when_bus_has_no_insert():
    """Live 2026-07-06: после отката FX-рэка посылы записались на пустые
    шины — сухие параллельные копии каналов в мейне («каша»). Посыл только
    на шину с реальным FX-инсертом."""
    mixer = DetectorMixer()
    mixer.state = {"/bus/13/preins/ins": "NONE", "/bus/15/preins/ins": "NONE"}
    engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(mixer)
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.auto_fx_sends_enabled = True
    info = ChannelInfo(channel=1, name="VOCAL", preset="leadVocal",
                       source_role="lead_vocal", has_signal=True,
                       recognized=True, auto_corrections_enabled=True,
                       allowed_controls=["fx_send"])
    engine.channels = {1: info}
    engine._apply_fx_sends(1, info, "leadVocal")
    assert not any(c[0] == "set_channel_send" for c in mixer.calls)

    # с реальным инсертом посыл разрешён (доходит до safety-слоя)
    mixer.state["/bus/13/preins/ins"] = "FX1"
    mixer.state["/bus/15/preins/ins"] = "FX3"
    engine._apply_fx_sends(1, info, "leadVocal")
    # запись может быть заблокирована карантином (нет промоута в этом
    # контроллере), но до фильтра шин она дошла — иначе бы не было blocked
    blocked = [d for d in engine.safety_controller.history
               if getattr(d.action, "action_type", "") == "SendLevelAdjust"]
    assert blocked


def test_comp_threshold_follows_trim_shift():
    """Live 2026-07-06: после сброса тримов пороги компрессоров остались в
    старом трим-домене (вокал давился GR~12). Сдвиг трима обязан сдвигать
    порог включённого компрессора на ту же дельту."""
    mixer = DetectorMixer()
    mixer.state = {"/ch/1/dyn/on": 1, "/ch/1/dyn/thr": -43.5}
    engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
    engine.mixer_client = mixer
    engine._shift_comp_threshold_with_trim(1, +12.0)
    thr_calls = [c for c in mixer.calls if c[0] == "set_compressor_threshold"]
    assert thr_calls and thr_calls[-1][2] == pytest.approx(-31.5)

    # выключенный комп не трогается
    mixer.state["/ch/1/dyn/on"] = 0
    n = len(mixer.calls)
    engine._shift_comp_threshold_with_trim(1, -6.0)
    assert len(mixer.calls) == n


def test_multi_mic_pair_balanced_as_one_instrument():
    """Оператор 2026-07-06: «вылезает snare из сложенных T+B — это один
    инструмент двумя микрофонами». Вторичный микрофон получает цель на
    6 LU ниже ведущего, а не собственную полную плоскость."""
    engine = AutoSoundcheckEngine(num_channels=4, auto_discover=False)
    engine.channels = {
        2: ChannelInfo(channel=2, name="SNARE T", preset="snare",
                       source_role="snare_top", has_signal=True),
        3: ChannelInfo(channel=3, name="SNARE B", preset="snare",
                       source_role="snare_bottom", has_signal=True),
        1: ChannelInfo(channel=1, name="KICK", preset="kick",
                       source_role="kick", has_signal=True),
        16: ChannelInfo(channel=16, name="VOCAL", preset="leadVocal",
                        source_role="lead_vocal", has_signal=True),
    }
    targets = {2: -25.0, 3: -25.0, 1: -25.0, 16: -22.0}
    adjusted = engine._apply_multi_mic_group_targets(
        targets, {2: "snare", 3: "snare", 1: "kick", 16: "lead_vocal"})
    assert adjusted[2] == pytest.approx(-25.0)   # ведущий на плоскости
    assert adjusted[3] == pytest.approx(-31.0)   # нижний -6 LU приправой
    assert adjusted[1] == pytest.approx(-25.0)   # одиночный kick не тронут
    assert adjusted[16] == pytest.approx(-22.0)


def test_stereo_pair_balanced_as_one_stereo_instrument():
    """Оператор 2026-07-06: каналы с подписью L/R — один стерео-инструмент:
    сумма пары на плоскости, каждый канал −3 LU, уровни равные."""
    engine = AutoSoundcheckEngine(num_channels=4, auto_discover=False)
    engine.channels = {
        9: ChannelInfo(channel=9, name="GUITAR L", preset="electricGuitar",
                       source_role="guitar", has_signal=True),
        10: ChannelInfo(channel=10, name="GUITAR R", preset="electricGuitar",
                        source_role="guitar", has_signal=True),
        16: ChannelInfo(channel=16, name="VOCAL", preset="leadVocal",
                        source_role="lead_vocal", has_signal=True),
    }
    targets = {9: -25.0, 10: -25.0, 16: -22.0}
    adjusted = engine._apply_multi_mic_group_targets(
        targets, {9: "guitar", 10: "guitar", 16: "lead_vocal"})
    assert adjusted[9] == pytest.approx(-28.0)   # плоскость −3 (сумма пары = −25)
    assert adjusted[10] == pytest.approx(-28.0)  # равные уровни
    assert adjusted[16] == pytest.approx(-22.0)


def test_dca_group_balance_trims_stem_to_plane_cuts_only():
    """Директива 2026-07-06: финальный баланс стемов DCA-группами.
    Сумма группы над плоскостью -> DCA вниз; буст выше 0 запрещён."""
    mixer = DetectorMixer()
    mixer.state = {}
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            promoted_direct_write_action_types=("DCAFaderMove",)),
    )
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.config_manager.get_section = lambda name: (
        {"dca_balance": {"enabled": True, "groups": {
            1: {"name": "DRUMS", "channels": [1, 2], "plane_lufs": -25.0}}}}
        if name == "autofoh" else {})

    class _Lvl:
        def __init__(self, lufs): self.lufs_integrated = lufs
    class _M:
        def __init__(self, lufs): self.level = _Lvl(lufs)

    engine.channels = {
        1: ChannelInfo(channel=1, name="KICK", preset="kick", has_signal=True,
                       fader_db=0.0),
        2: ChannelInfo(channel=2, name="SNARE", preset="snare", has_signal=True,
                       fader_db=0.0),
    }
    engine.channels[1].metrics = _M(-22.0)
    engine.channels[2].metrics = _M(-22.0)   # сумма ~ -19 при плоскости -25

    engine._apply_dca_group_balance()
    dca_calls = [c for c in mixer.calls if c[0] == "set_dca_fader"]
    assert dca_calls, "DCA-движение не отправлено"
    assert dca_calls[-1][1] == 1
    assert dca_calls[-1][2] == pytest.approx(-6.0, abs=0.3)   # -25 - (-19)


def test_dca_group_balance_prefers_stem_bus_tap_over_member_sum():
    """Тапы 33-44: реальный замер стем-шины важнее мощностной суммы
    участников (та игнорирует фазу/корреляцию микрофонов)."""
    mixer = DetectorMixer()
    mixer.state = {}
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.mixer_client = mixer
    engine.safety_controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            promoted_direct_write_action_types=("DCAFaderMove",)),
    )
    engine.runtime_state = RuntimeState.SNAPSHOT_LOCK
    engine.config_manager.get_section = lambda name: (
        {"dca_balance": {"enabled": True, "groups": {
            1: {"name": "DRUMS", "channels": [1, 2], "plane_lufs": -25.0}}},
         "measurement_taps": {"stem_buses": {
             1: {"name": "DRUMS", "usb": [33, 34]}}}}
        if name == "autofoh" else {})

    class _Lvl:
        def __init__(self, lufs): self.lufs_integrated = lufs
    class _M:
        def __init__(self, lufs): self.level = _Lvl(lufs)

    engine.channels = {
        1: ChannelInfo(channel=1, name="KICK", preset="kick", has_signal=True,
                       fader_db=0.0),
        2: ChannelInfo(channel=2, name="SNARE", preset="snare", has_signal=True,
                       fader_db=0.0),
    }
    # участники говорят -40 (сумма ~ -37), но тап шины слышит -20/-20
    # (power sum ~ -17): решает тап
    engine.channels[1].metrics = _M(-40.0)
    engine.channels[2].metrics = _M(-40.0)
    engine._tap_lufs = lambda ch, seconds=4.0: (
        -20.0 if ch in (33, 34) else None)

    engine._apply_dca_group_balance()
    dca_calls = [c for c in mixer.calls if c[0] == "set_dca_fader"]
    assert dca_calls, "DCA-движение не отправлено"
    assert dca_calls[-1][2] == pytest.approx(-8.0, abs=0.3)  # -25 - (-17)


def test_postfader_tap_channel_mapping_follows_config():
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.config_manager.get_section = lambda name: (
        {"measurement_taps": {"postfader_channels": {
            "first_usb": 17, "channels": 16}}}
        if name == "autofoh" else {})
    assert engine._postfader_tap_channel(1) == 17
    assert engine._postfader_tap_channel(16) == 32
    assert engine._postfader_tap_channel(17) is None
    assert engine._postfader_tap_channel(0) is None


def test_same_source_pair_rejects_implausible_alignment_delay():
    """SNARE T/B — микрофоны одного барабана в паре сантиметров: замер
    9.9 мс (период резонанса ~100 Гц) — неоднозначность GCC-PHAT, не
    настоящая задержка. Live 2026-07-06: ch2 получил dly 10 мс."""
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    assert engine._same_source_delay_plausible("snare", 0.58) is True
    assert engine._same_source_delay_plausible("snare", 9.90) is False
    assert engine._same_source_delay_plausible("kick", -3.5) is False
    # room-микрофоны стоят в метрах от источника — большие задержки легитимны
    assert engine._same_source_delay_plausible("room", 9.90) is True


def test_master_mix_eval_falls_back_to_prefader_tap_when_master_muted():
    """Мастер замьючен внешним актором: post-fader тап 47/48 молчит,
    оценка микса берётся с PRE-fader тапа 45/46 и помечается source."""
    engine = AutoSoundcheckEngine(num_channels=2, auto_discover=False)
    engine.master_mix_enabled = True
    engine.master_mix_channels = [47, 48]
    engine.master_mix_seconds = 2.0
    engine.master_mix_target_curve = "pink"
    engine.master_mix_target_lufs = -20.0
    engine.config_manager.get_section = lambda name: (
        {"measurement_taps": {"main_prefader": [45, 46]}}
        if name == "autofoh" else {})

    sr = engine.sample_rate
    t = np.arange(int(sr * 2.0)) / sr
    rng = np.random.default_rng(7)
    music = (0.1 * np.sin(2 * np.pi * 220 * t)
             + 0.05 * rng.standard_normal(t.size))

    class _Cap:
        def get_buffer(self, ch, n):
            if ch in (45, 46):
                return music[:n]
            return np.zeros(n)
    engine.audio_capture = _Cap()

    payload = engine._evaluate_master_mix()
    assert payload is not None
    assert payload["source"] == "main_prefader_muted_master"
    assert payload["status"] != "NO_SIGNAL"
