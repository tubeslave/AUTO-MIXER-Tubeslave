"""Tests for arrangement-aware level automation."""

import struct

import numpy as np

from arrangement_automation.activity_detector import ActivityDetector
from arrangement_automation.arrangement_density import ArrangementDensityAnalyzer
from arrangement_automation.channel_role_classifier import ChannelRoleClassifier
from arrangement_automation.level_automation_planner import LevelAutomationPlanner
from arrangement_automation.live_input_trim_controller import LiveInputTrimController
from arrangement_automation.masking_analyzer import MaskingAnalyzer
from arrangement_automation.mix_priority_engine import MixPriorityEngine
from arrangement_automation.safety_limiter import SafetyLimiter
from arrangement_automation.section_detector import SectionDetector
from arrangement_automation.controller import ArrangementAutomationController
from arrangement_automation.models import ActivityState, PlannedOffset
from arrangement_automation.wing_meter_reader import WingMeterReader, WingMeterReading


def _active(channel_id, rms=-24.0, confidence=0.9):
    return ActivityState(
        channel_id=channel_id,
        active=True,
        activity_confidence=confidence,
        rms_db=rms,
        peak_db=rms + 8.0,
        activity_duration_sec=3.0,
    )


def _inactive(channel_id):
    return ActivityState(
        channel_id=channel_id,
        active=False,
        activity_confidence=0.2,
        rms_db=-80.0,
        peak_db=-70.0,
        activity_duration_sec=0.0,
    )


class _TrimMixer:
    is_connected = True

    def __init__(self):
        self.trims = {}
        self.sent = []

    def get_channel_gain(self, channel):
        return self.trims.get(channel, 0.0)

    def set_channel_gain(self, channel, value):
        self.trims[channel] = float(value)
        self.sent.append((channel, float(value)))
        return True

    def confirm_manual_write(self, operation, channel, desired_value, timeout_ms=300):
        return {
            "verification_id": f"verify-{channel}",
            "readback_status": "confirmed",
            "confirmed_value": float(desired_value),
            "failure_reason": None,
            "timeout_ms": int(timeout_ms),
        }


class _LiveTrimMixer(_TrimMixer):
    def __init__(self):
        super().__init__()
        self.cached_trims = {}

    def get_channel_gain(self, channel):
        return self.cached_trims.get(channel, 0.0)

    def get_channel_gain_live(self, channel, timeout=0.35):
        return self.trims.get(channel, 0.0)


class _FakeMeterReader:
    def __init__(self, readings):
        self.readings = readings
        self.started = False

    def start(self, channels):
        self.started = True
        return self.get_status()

    def stop(self):
        self.started = False

    def get_metrics(self, channels, now=None):
        out = {}
        for channel in channels:
            reading = self.readings.get(channel)
            if isinstance(reading, WingMeterReading):
                out[channel] = reading
            elif reading:
                out[channel] = WingMeterReading(channel=channel, **reading)
            else:
                out[channel] = WingMeterReading(
                    channel=channel,
                    trusted=False,
                    status="unavailable",
                    age_sec=None,
                )
        return out

    def get_status(self):
        trusted = any(
            bool(reading.trusted if isinstance(reading, WingMeterReading)
                 else reading.get("trusted", False))
            for reading in self.readings.values()
        )
        return {"enabled": True, "active": self.started, "trusted": trusted}


def _tone(freq=440.0, dbfs=-20.0, sample_rate=48_000, seconds=0.5):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    amp = 10.0 ** (dbfs / 20.0)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _burst(freq=80.0, peak_dbfs=-4.5, sample_rate=48_000, seconds=0.5):
    out = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    size = int(sample_rate * 0.035)
    start = int(sample_rate * 0.2)
    t = np.arange(size, dtype=np.float32) / sample_rate
    amp = 10.0 ** (peak_dbfs / 20.0)
    out[start:start + size] = (
        amp
        * np.sin(2.0 * np.pi * freq * t)
        * np.hanning(size).astype(np.float32)
    )
    return out


def _live_trim_config(extra=None):
    cfg = {
        "automation": {
            "live_input_trim": {
                "run_background_loop": False,
                "analysis_only_mode": False,
                "live_apply_enabled": True,
                "confirm_live_apply": True,
                "activity_attack_hold_sec": 0.0,
                "min_main_signal_confidence": 0.1,
                "analysis_window_sec": 0.5,
                "apply_cooldown_sec": 0.0,
                "max_step_db_per_tick": 0.25,
                "max_boost_step_db": 0.15,
                "deadband_db": 0.1,
                "role_targets": {
                    "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 0.5, "windows": 2},
                    "snare": {"peak_dbfs": -8.0, "main_sec": 0.5, "windows": 2},
                },
            }
        }
    }
    if extra:
        cfg["automation"]["live_input_trim"].update(extra)
    return cfg


def test_channel_role_classifier_english_and_russian_names():
    classifier = ChannelRoleClassifier()

    assert classifier.classify(1, "Lead Vox").role == "lead_vocal"
    assert classifier.classify(2, "Бэк вокал").role == "backing_vocal"
    assert classifier.classify(3, "Бочка").role == "kick"
    assert classifier.classify(4, "SD").role == "snare"
    assert classifier.classify(5, "Bass DI").role == "bass"
    assert classifier.classify(6, "Solo Gtr").role == "lead_guitar"
    assert classifier.classify(7, "Keys L").role == "keys"
    assert classifier.classify(8, "Accordion").role == "keys"
    assert classifier.classify(9, "Playback R").role == "pad"


def test_activity_detector_hysteresis_rejects_short_noise():
    detector = ActivityDetector(
        min_active_db=-50.0,
        noise_margin_db=10.0,
        attack_hold_sec=1.0,
        release_hold_sec=1.0,
    )

    assert not detector.update(1, -35.0, -20.0, noise_floor_db=-70.0, timestamp=0.0).active
    assert not detector.update(1, -35.0, -20.0, noise_floor_db=-70.0, timestamp=0.5).active
    assert detector.update(1, -35.0, -20.0, noise_floor_db=-70.0, timestamp=1.1).active
    assert detector.update(1, -75.0, -70.0, noise_floor_db=-80.0, timestamp=1.5).active
    assert not detector.update(1, -75.0, -70.0, noise_floor_db=-80.0, timestamp=2.6).active


def test_arrangement_density_chorus_is_dense_enough():
    classifier = ChannelRoleClassifier()
    roles = {
        1: classifier.classify(1, "Lead Vox"),
        2: classifier.classify(2, "BV"),
        3: classifier.classify(3, "Kick"),
        4: classifier.classify(4, "Snare"),
        5: classifier.classify(5, "Bass"),
        6: classifier.classify(6, "Gtr"),
        7: classifier.classify(7, "Keys"),
        8: classifier.classify(8, "OH"),
    }
    activities = {ch: _active(ch, -18.0) for ch in roles}
    bands = {
        ch: {
            "sub": -28.0,
            "bass": -24.0,
            "low_mid": -22.0,
            "mid": -18.0,
            "high_mid": -17.0,
            "umf": -16.0,
        }
        for ch in roles
    }

    density = ArrangementDensityAnalyzer(max_channels=8).analyze(activities, roles, bands)

    assert density.active_channel_count == 8
    assert density.density_index >= 0.70
    assert density.density_label in {"dense", "very_dense"}


def test_section_detector_verse_chorus_solo_transitions():
    classifier = ChannelRoleClassifier()
    roles = {
        1: classifier.classify(1, "Lead Vox"),
        2: classifier.classify(2, "BV"),
        3: classifier.classify(3, "Kick"),
        4: classifier.classify(4, "Snare"),
        5: classifier.classify(5, "Bass"),
        6: classifier.classify(6, "Lead Gtr"),
    }
    detector = SectionDetector(hold_time_sec=0.0)

    verse_density = ArrangementDensityAnalyzer(max_channels=6).analyze({
        1: _active(1), 3: _active(3), 5: _active(5), 6: _active(6),
        2: _inactive(2), 4: _inactive(4),
    }, roles)
    verse = detector.update({
        1: _active(1), 3: _active(3), 5: _active(5), 6: _active(6),
        2: _inactive(2), 4: _inactive(4),
    }, roles, verse_density, timestamp=0.0)
    assert verse.section == "verse"

    chorus_activities = {ch: _active(ch, -17.0) for ch in roles}
    chorus_density = ArrangementDensityAnalyzer(max_channels=6).analyze(chorus_activities, roles, {
        ch: {"mid": -16.0, "high_mid": -15.0, "umf": -15.0, "bass": -20.0, "sub": -24.0}
        for ch in roles
    })
    chorus = detector.update(chorus_activities, roles, chorus_density, timestamp=1.0)
    assert chorus.section == "chorus"

    solo_activities = {
        1: _inactive(1), 2: _inactive(2), 3: _active(3), 4: _active(4),
        5: _active(5), 6: _active(6),
    }
    solo_density = ArrangementDensityAnalyzer(max_channels=6).analyze(solo_activities, roles)
    solo = detector.update(solo_activities, roles, solo_density, timestamp=2.0)
    assert solo.section == "solo"


def test_masking_analyzer_finds_guitar_masking_lead_vocal():
    classifier = ChannelRoleClassifier()
    roles = {
        1: classifier.classify(1, "Lead Vox"),
        2: classifier.classify(2, "Rhythm Gtr"),
        3: classifier.classify(3, "Keys"),
    }
    activities = {1: _active(1), 2: _active(2), 3: _active(3)}
    bands = {
        1: {"mid": -22.0, "high_mid": -20.0, "umf": -20.0},
        2: {"mid": -17.0, "high_mid": -15.0, "umf": -15.0, "low_mid": -18.0},
        3: {"mid": -45.0, "high_mid": -45.0, "umf": -45.0},
    }

    masking = MaskingAnalyzer().analyze(roles, activities, bands, primary_channel_id=1)

    assert masking
    assert masking[0].channel_id == 2
    assert masking[0].suggested_action == "level_duck"
    assert masking[0].suggested_gain_reduction_db < 0.0


def test_planner_chorus_protects_vocal_and_ducks_masking_guitar():
    classifier = ChannelRoleClassifier()
    roles = {
        1: classifier.classify(1, "Lead Vox"),
        2: classifier.classify(2, "BV"),
        3: classifier.classify(3, "Rhythm Gtr"),
    }
    activities = {1: _active(1), 2: _active(2), 3: _active(3)}
    density = ArrangementDensityAnalyzer(max_channels=3).analyze(activities, roles, {
        1: {"mid": -18.0, "high_mid": -18.0},
        2: {"mid": -20.0, "high_mid": -20.0},
        3: {"mid": -15.0, "high_mid": -14.0, "umf": -14.0},
    })
    section = SectionDetector(hold_time_sec=0.0).update(activities, roles, density, timestamp=0.0)
    priority = MixPriorityEngine().analyze(roles, activities, section)
    masking = MaskingAnalyzer().analyze(roles, activities, {
        1: {"mid": -20.0, "high_mid": -20.0, "umf": -20.0},
        2: {"mid": -25.0, "high_mid": -25.0, "umf": -25.0},
        3: {"mid": -15.0, "high_mid": -14.0, "umf": -14.0},
    }, primary_channel_id=1)

    planned = LevelAutomationPlanner().plan(roles, activities, density, section, masking, priority)

    assert planned[1].requested_offset_db > 0.0
    assert planned[2].requested_offset_db < planned[1].requested_offset_db
    assert planned[3].requested_offset_db < 0.0


def test_safety_limiter_blocks_low_confidence_live_apply_and_limits_unknown():
    role = ChannelRoleClassifier().classify(9, "Mystery")
    planned = {
        9: PlannedOffset(
            channel_id=9,
            channel_name="Mystery",
            role="unknown",
            requested_offset_db=10.0,
            confidence=0.4,
        )
    }
    limiter = SafetyLimiter({
        "max_step_db_per_tick": 0.25,
        "deadband_db": 0.1,
        "min_confidence_for_live_apply": 0.65,
    })

    result = limiter.limit(
        planned,
        {9: role},
        base_fader_positions={9: -3.0},
        live_apply_enabled=True,
    )

    assert 9 in result.blocked_offsets
    assert result.safe_offsets[9] == 0.0

    analysis_result = limiter.limit(
        planned,
        {9: role},
        base_fader_positions={9: -3.0},
        live_apply_enabled=False,
    )
    assert abs(analysis_result.safe_offsets[9]) <= 0.25


def test_controller_resets_smoothing_when_live_apply_is_enabled():
    class Mixer:
        is_connected = True

        def __init__(self):
            self.sent = []

        def get_channel_fader(self, channel):
            return -5.0

        def set_channel_fader(self, channel, value):
            self.sent.append((channel, value))
            return True

        def confirm_manual_write(self, operation, channel, desired_value, timeout_ms=300):
            return {
                "verification_id": f"verify-{channel}",
                "readback_status": "confirmed",
                "confirmed_value": float(desired_value),
                "failure_reason": None,
                "timeout_ms": int(timeout_ms),
            }

    mixer = Mixer()
    controller = ArrangementAutomationController(
        mixer_client=mixer,
        config={
            "automation": {
                "arrangement_automation": {
                    "run_background_loop": False,
                    "section_hold_time_sec": 0.0,
                    "activity_attack_hold_sec": 0.0,
                    "analysis_only_mode": True,
                    "min_confidence_for_live_apply": 0.4,
                }
            }
        },
    )
    controller.start([1], channel_names={1: "Lead Vox"})
    metrics = {1: {"rms_db": -20.0, "peak_db": -8.0, "lufs": -21.0}}

    controller.process_once(metrics_by_channel=metrics, timestamp=0.0)
    assert mixer.sent == []

    controller.set_live_apply(enabled=True, analysis_only_mode=False)
    controller.process_once(metrics_by_channel=metrics, timestamp=1.0)
    controller.stop()

    assert mixer.sent


def test_live_input_trim_waits_for_main_signal_before_positive_trim():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config(),
    )
    controller.start([1], channel_names={1: "Lead Vox"})
    audio = {1: _tone(700.0, -24.0)}

    controller.process_once(audio_by_channel=audio, timestamp=0.0)
    assert mixer.sent == []

    controller.process_once(audio_by_channel=audio, timestamp=0.6)
    controller.stop()

    assert mixer.sent
    assert mixer.sent[-1][0] == 1
    assert mixer.sent[-1][1] > 0.0


def test_live_input_trim_blocks_real_apply_without_confirmation():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({"confirm_live_apply": False}),
    )
    controller.start([1], channel_names={1: "Lead Vox"})
    audio = {1: _tone(700.0, -24.0)}

    controller.process_once(audio_by_channel=audio, timestamp=0.0)
    state = controller.process_once(audio_by_channel=audio, timestamp=0.6)
    controller.stop()

    assert mixer.sent == []
    assert state["blocked"][1]["blocked_reason"] == "confirm_live_apply_required"
    assert state["blocked"][1]["readback_status"] == "not_requested"


def test_live_input_trim_blocks_positive_trim_on_bleed_only_channel():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({
            "min_main_signal_confidence": 0.65,
            "role_targets": {
                "kick": {"peak_dbfs": -8.0, "main_sec": 0.1, "windows": 1},
                "snare": {"peak_dbfs": -8.0, "main_sec": 0.1, "windows": 1},
            },
        }),
    )
    controller.start([1, 2], channel_names={1: "Kick", 2: "Snare"})
    kick = _tone(70.0, -8.0)
    snare_bleed = kick * 0.08

    state = controller.process_once(
        audio_by_channel={1: kick, 2: snare_bleed},
        timestamp=0.0,
    )
    controller.stop()

    assert all(ch != 2 for ch, _ in mixer.sent)
    assert state["channels"][2]["state"] == "waiting_for_signal"
    assert state["channels"][2]["target_trim_db"] == 0.0


def test_live_input_trim_regular_cut_waits_for_confirmed_main_signal():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({
            "min_main_signal_confidence": 0.65,
            "role_targets": {
                "rhythm_guitar": {"peak_dbfs": -12.0, "main_sec": 2.0, "windows": 5},
            },
        }),
    )
    controller.start([1], channel_names={1: "Guitar L"})

    state = controller.process_once(audio_by_channel={1: _tone(700.0, -5.0)}, timestamp=0.0)
    controller.stop()

    assert mixer.sent == []
    assert state["blocked"][1]["blocked_reason"] == "waiting_for_main_signal_before_cut"


def test_live_input_trim_percussive_peak_hold_does_not_overcut_kick():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({
            "role_targets": {
                "kick": {"peak_dbfs": -8.0, "main_sec": 0.1, "windows": 1},
            },
        }),
    )
    controller.start([1], channel_names={1: "Kick"})

    state = controller.process_once(audio_by_channel={1: _burst(80.0, -4.5)}, timestamp=0.0)
    controller.stop()

    assert mixer.sent == []
    assert state["blocked"][1]["reason"] == "percussive_headroom_ok"
    assert state["channels"][1]["target_trim_db"] == 0.0


def test_live_input_trim_guitar_can_cut_below_old_minus_10_limit():
    mixer = _TrimMixer()
    mixer.trims[1] = -10.0
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({
            "role_targets": {
                "rhythm_guitar": {"peak_dbfs": -12.0, "main_sec": 0.0, "windows": 1},
            },
        }),
    )
    controller.start([1], channel_names={1: "Guitar L"})

    controller.process_once(audio_by_channel={1: _tone(700.0, -4.0)}, timestamp=0.0)
    controller.stop()

    assert mixer.sent
    assert mixer.sent[-1][1] < -10.0


def test_live_input_trim_hot_vocal_cuts_before_learning_completes():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({
            "role_targets": {
                "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 2.0, "windows": 6},
            }
        }),
    )
    controller.start([1], channel_names={1: "Vocal"})

    controller.process_once(audio_by_channel={1: _tone(800.0, -1.0)}, timestamp=0.0)
    controller.stop()

    assert mixer.sent
    assert mixer.sent[-1][1] < 0.0


def test_live_input_trim_cooldown_prevents_repeated_trim_moves():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config({
            "apply_cooldown_sec": 2.0,
            "role_targets": {
                "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 0.0, "windows": 1},
            },
        }),
    )
    controller.start([1], channel_names={1: "Vocal"})
    hot = {1: _tone(800.0, -1.0)}

    controller.process_once(audio_by_channel=hot, timestamp=0.0)
    controller.process_once(audio_by_channel=hot, timestamp=0.5)
    controller.stop()

    assert len(mixer.sent) == 1


def test_live_input_trim_reads_live_trim_baseline_before_cached_gain():
    mixer = _LiveTrimMixer()
    mixer.trims[1] = -12.0
    mixer.cached_trims[1] = 0.0
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config(),
    )

    state = controller.start([1], channel_names={1: "Kick"})
    controller.stop()

    assert state["channels"][1]["current_trim_db"] == -12.0


def test_live_input_trim_emits_replayable_write_plan():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config(),
    )
    controller.start([1], channel_names={1: "Lead Vox"})
    audio = {1: _tone(700.0, -24.0)}

    controller.process_once(audio_by_channel=audio, timestamp=0.0)
    state = controller.process_once(audio_by_channel=audio, timestamp=0.6)
    controller.stop()

    assert state["applied"]
    decision = state["applied"][0]
    assert decision["write_plan"] is not None
    assert decision["write_plan"]["operation"] == "set_gain"
    assert decision["write_plan"]["channel"] == 1
    assert decision["write_plan"]["source"] == "auto_gain_live_input_trim"


def test_live_input_trim_replays_recorded_decision_through_live_apply_gate():
    source_mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=source_mixer,
        config=_live_trim_config(),
    )
    controller.start([1], channel_names={1: "Lead Vox"})
    audio = {1: _tone(700.0, -24.0)}

    controller.process_once(audio_by_channel=audio, timestamp=0.0)
    state = controller.process_once(audio_by_channel=audio, timestamp=0.6)
    recorded = dict(state["applied"][0])
    controller.stop()

    replay_mixer = _TrimMixer()
    replay_controller = LiveInputTrimController(config=_live_trim_config())
    replayed = replay_controller.replay_decision(
        recorded,
        mixer_client=replay_mixer,
        dry_run=False,
    )

    assert replay_mixer.sent == [(1, recorded["write_plan"]["value"])]
    assert replayed["sent"] is True
    assert replayed["readback_status"] == "confirmed"
    assert replayed["current_trim_db"] == replayed["target_trim_db"]


def test_live_input_trim_uses_wing_meter_for_level_decision():
    mixer = _TrimMixer()
    meter = _FakeMeterReader({
        1: {
            "rms_db": -16.0,
            "peak_dbfs": -4.0,
            "true_peak_dbtp": -4.0,
            "trusted": True,
            "confidence": 1.0,
            "age_sec": 0.05,
            "status": "ok",
        }
    })
    controller = LiveInputTrimController(
        mixer_client=mixer,
        meter_reader=meter,
        config=_live_trim_config({
            "level_source": "wing_meter",
            "require_meter_for_analysis": True,
            "role_targets": {
                "rhythm_guitar": {"peak_dbfs": -12.0, "main_sec": 0.0, "windows": 1},
            },
        }),
    )
    controller.start([1], channel_names={1: "Guitar L"})

    state = controller.process_once(audio_by_channel={1: _tone(700.0, -28.0)}, timestamp=0.0)
    controller.stop()

    assert mixer.sent
    assert mixer.sent[-1][1] < 0.0
    assert state["channels"][1]["level_source"] == "wing_meter"
    assert state["channels"][1]["meter_trusted"] is True


def test_live_input_trim_uses_meter_activity_when_usb_capture_is_silent():
    mixer = _TrimMixer()
    meter = _FakeMeterReader({
        1: {
            "rms_db": -22.0,
            "peak_dbfs": -10.0,
            "true_peak_dbtp": -10.0,
            "trusted": True,
            "confidence": 1.0,
            "age_sec": 0.05,
            "status": "ok",
        }
    })
    controller = LiveInputTrimController(
        mixer_client=mixer,
        meter_reader=meter,
        config=_live_trim_config({
            "level_source": "wing_meter",
            "require_meter_for_analysis": True,
            "analysis_only_mode": True,
            "live_apply_enabled": False,
            "min_main_signal_confidence": 0.65,
            "role_targets": {
                "lead_vocal": {"peak_dbfs": -12.0, "main_sec": 0.0, "windows": 1},
            },
        }),
    )
    controller.start([1], channel_names={1: "Vocal"})

    state = controller.process_once(
        audio_by_channel={1: np.zeros(48_000, dtype=np.float32)},
        timestamp=0.0,
    )
    controller.stop()

    assert state["channels"][1]["meter_activity_fallback"] is True
    assert state["channels"][1]["main_signal_confidence"] >= 0.65
    assert state["channels"][1]["state"] == "ready_to_adjust"


def test_live_input_trim_blocks_when_required_wing_meter_is_unavailable():
    mixer = _TrimMixer()
    meter = _FakeMeterReader({})
    controller = LiveInputTrimController(
        mixer_client=mixer,
        meter_reader=meter,
        config=_live_trim_config({
            "level_source": "wing_meter",
            "require_meter_for_analysis": True,
            "require_meter_for_live_apply": True,
            "role_targets": {
                "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 0.0, "windows": 1},
            },
        }),
    )
    controller.start([1], channel_names={1: "Vocal"})

    state = controller.process_once(audio_by_channel={1: _tone(800.0, -1.0)}, timestamp=0.0)
    controller.stop()

    assert mixer.sent == []
    assert state["blocked"][1]["blocked_reason"] == "meter_unavailable_for_level_decision"
    assert state["channels"][1]["target_trim_db"] == 0.0


def test_wing_meter_reader_accepts_push_meter_frames():
    class Mixer:
        is_connected = True

        def __init__(self):
            self.callbacks = []
            self.sent = []

        def subscribe(self, address_pattern, callback):
            self.callbacks.append(callback)

        def send(self, address, *values):
            self.sent.append((address, values))
            return True

        def emit(self, address, *values):
            for callback in self.callbacks:
                callback(address, *values)

    mixer = Mixer()
    reader = WingMeterReader(
        mixer,
        config={"live_input_trim": {"meter_stale_timeout_sec": 1.0}},
    )
    reader.start([1])
    mixer.emit("/$meters/ch/1/in", 0.5)

    readings = reader.get_metrics([1])

    assert readings[1].trusted is True
    assert -6.1 < readings[1].peak_dbfs < -5.9


def test_wing_meter_reader_decodes_native_meter_packets():
    report_id = 0x414D4931
    reader = WingMeterReader(
        None,
        config={
            "live_input_trim": {
                "native_meter_report_id": report_id,
                "native_values_per_channel": 8,
            }
        },
    )
    reader._active = True
    reader._native_channel_order = [1, 2]
    values = [
        -18.0, -20.0, -17.0, -17.0, -30.0, 0.0, -60.0, 0.0,
        -9.0, -11.0, -8.0, -8.0, -30.0, 0.0, -60.0, 0.0,
    ]
    packet = struct.pack(">I", report_id) + b"".join(
        struct.pack(">h", int(value * 256.0)) for value in values
    )

    count = reader._handle_native_packet(packet)
    readings = reader.get_metrics([1, 2])

    assert count == 2
    assert readings[1].trusted is True
    assert readings[1].peak_dbfs == -18.0
    assert readings[2].peak_dbfs == -9.0
    assert readings[2].source == "wing_meter_native"
