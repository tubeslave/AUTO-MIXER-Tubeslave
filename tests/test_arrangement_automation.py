"""Tests for arrangement-aware level automation."""

from arrangement_automation.activity_detector import ActivityDetector
from arrangement_automation.arrangement_density import ArrangementDensityAnalyzer
from arrangement_automation.channel_role_classifier import ChannelRoleClassifier
from arrangement_automation.level_automation_planner import LevelAutomationPlanner
from arrangement_automation.masking_analyzer import MaskingAnalyzer
from arrangement_automation.mix_priority_engine import MixPriorityEngine
from arrangement_automation.safety_limiter import SafetyLimiter
from arrangement_automation.section_detector import SectionDetector
from arrangement_automation.controller import ArrangementAutomationController
from arrangement_automation.models import ActivityState, PlannedOffset


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
