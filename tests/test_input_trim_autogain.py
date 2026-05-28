import pytest

from lufs_gain_staging import LocalInputGainProfile, SignalStats
from backend.operator_recommendation_bridge import build_safe_gain_operator_proposals


def _stats_with_signal(peak_db=-18.0, lufs=-28.0, samples=20):
    stats = SignalStats(channel_id=1)
    for _ in range(samples):
        stats.update_sample(true_peak_db=peak_db, lufs=lufs, sample_rms_db=lufs)
    return stats


def _stats_with_sparse_signal(active_peak_db=-24.0, active_lufs=-34.0, active_samples=3, silent_samples=17):
    stats = SignalStats(channel_id=1)
    for _ in range(active_samples):
        stats.update_sample(true_peak_db=active_peak_db, lufs=active_lufs, sample_rms_db=active_lufs)
    for _ in range(silent_samples):
        stats.update_sample(true_peak_db=-90.0, lufs=-100.0, sample_rms_db=-100.0)
    return stats


def _stats_with_isolated_peak(base_peak_db=-14.0, spike_peak_db=-0.5, lufs=-24.0, base_samples=19):
    stats = SignalStats(channel_id=1)
    for _ in range(base_samples):
        stats.update_sample(true_peak_db=base_peak_db, lufs=lufs, sample_rms_db=lufs)
    stats.update_sample(true_peak_db=spike_peak_db, lufs=lufs, sample_rms_db=lufs)
    return stats


def _stats_with_unstable_loudness(peak_db=-20.0, loud_lufs=-24.0, quiet_lufs=-42.0, samples_each=10):
    stats = SignalStats(channel_id=1)
    for _ in range(samples_each):
        stats.update_sample(true_peak_db=peak_db, lufs=loud_lufs, sample_rms_db=loud_lufs)
    for _ in range(samples_each):
        stats.update_sample(true_peak_db=peak_db, lufs=quiet_lufs, sample_rms_db=quiet_lufs)
    return stats


def _stats_with_noisy_floor(active_peak_db=-22.0, active_lufs=-35.0, noise_lufs=-44.0):
    stats = SignalStats(channel_id=1)
    for _ in range(10):
        stats.update_sample(true_peak_db=active_peak_db, lufs=active_lufs, sample_rms_db=active_lufs)
    for _ in range(10):
        stats.update_sample(true_peak_db=noise_lufs, lufs=noise_lufs, sample_rms_db=noise_lufs)
    return stats


def test_soundcheck_boost_is_capped_by_clean_local_preamp_range():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-32.0)
    profile = LocalInputGainProfile.from_config({
        "character": "clean",
        "clean_gain_max_db": 6.0,
        "soundcheck_max_boost_db": 6.0,
    })

    stats.calculate_input_trim_recommendation(
        current_trim_db=5.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "input_gain_character": {
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 1.0,
                    "evidence_windows": 1,
                }
            }
        },
    )

    report = stats.get_report()
    assert report["method_id"] == "input_trim_preamp_character"
    assert report["suggested_gain_db"] == pytest.approx(1.0)
    assert report["target_trim_db"] == pytest.approx(6.0)
    assert report["limited_by"] == "preamp_clean_range"
    assert report["input_gain_character"]["target_zone"] == "clean"
    assert report["auto_apply_allowed"] is False


def test_soundcheck_blocks_boost_when_local_preamp_is_already_colored():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-32.0)
    profile = LocalInputGainProfile.from_config({
        "character": "clean",
        "clean_gain_max_db": 6.0,
    })

    stats.calculate_input_trim_recommendation(
        current_trim_db=8.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["target_trim_db"] == pytest.approx(8.0)
    assert report["limited_by"] == "preamp_character_limit"
    assert report["operator_action"] == "manual_gain_or_source_check"


def test_live_mode_is_trim_down_only_and_uses_small_cut_steps():
    stats = _stats_with_signal(peak_db=-0.5, lufs=-18.0)
    profile = LocalInputGainProfile.from_config({"live_max_cut_db": 1.0})

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="live",
        mixer_channel=1,
        preset="custom",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-1.0)
    assert report["target_trim_db"] == pytest.approx(-1.0)
    assert report["safety_policy"]["live_policy"] == "trim_down_only"
    assert report["auto_apply_allowed"] is True


def test_live_mode_blocks_quiet_channel_gain_boost():
    stats = _stats_with_signal(peak_db=-22.0, lufs=-36.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="live",
        mixer_channel=1,
        preset="custom",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "live_down_only"
    assert report["operator_action"] == "operator_review"


def test_soundcheck_waits_for_repeated_evidence_before_gain_up():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "gain_up_stability_wait"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["active"] is True
    assert report["input_gain_character"]["stability"]["candidate_direction"] == "up"
    assert report["input_gain_character"]["stability"]["evidence_windows"] == 1


def test_soundcheck_allows_stable_soft_vocal_clean_boost():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "input_gain_character": {
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                }
            }
        },
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["target_trim_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert report["operator_action"] == "apply_trim_proposal"
    assert report["input_gain_character"]["source_behavior"] == "stable_dynamic"
    assert report["input_gain_character"]["boost_guard"]["active"] is False
    assert report["auto_apply_allowed"] is False


def test_soundcheck_blocks_transient_dominant_auto_boost():
    stats = _stats_with_signal(peak_db=-22.0, lufs=-45.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="snare",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["target_trim_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "transient_dominant_boost_block"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "transient_dominant"
    assert report["input_gain_character"]["boost_guard"]["active"] is True
    assert report["input_gain_character"]["boost_guard"]["reason"] == "transient_dominant_boost_block"


def test_soundcheck_blocks_sparse_presence_auto_boost():
    stats = _stats_with_sparse_signal()

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["signal_presence"] == pytest.approx(15.0)
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "unstable_presence_boost_block"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "sparse_source"
    assert report["input_gain_character"]["boost_guard"]["reason"] == "unstable_presence_boost_block"


def test_soundcheck_isolated_peak_cuts_to_clip_margin_not_target_peak():
    stats = _stats_with_isolated_peak()

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-3.5)
    assert report["target_trim_db"] == pytest.approx(-3.5)
    assert report["limited_by"] == "isolated_peak_headroom"
    assert report["input_gain_character"]["source_behavior"] == "isolated_peak_source"
    assert report["input_gain_character"]["isolated_peak_spread_db"] > 6.0
    assert report["input_gain_character"]["peak_p95_db"] < -10.0


def test_soundcheck_persistent_hot_signal_still_cuts_toward_target_peak():
    stats = _stats_with_signal(peak_db=-0.5, lufs=-18.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-6.0)
    assert report["target_trim_db"] == pytest.approx(-6.0)
    assert report["limited_by"] == "peak_headroom"
    assert report["input_gain_character"]["isolated_peak_spread_db"] == pytest.approx(0.0)


def test_soundcheck_blocks_unstable_vocal_loudness_auto_boost():
    stats = _stats_with_unstable_loudness()

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "unstable_dynamics_boost_block"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "unstable_dynamic"
    assert report["input_gain_character"]["lufs_std_db"] > 7.0
    assert report["input_gain_character"]["boost_guard"]["reason"] == "unstable_dynamics_boost_block"


def test_soundcheck_blocks_spiky_quiet_source_auto_boost():
    stats = _stats_with_isolated_peak(base_peak_db=-30.0, spike_peak_db=-18.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="custom",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "isolated_peak_boost_block"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "isolated_peak_source"
    assert report["input_gain_character"]["isolated_peak_spread_db"] > 6.0
    assert report["input_gain_character"]["boost_guard"]["reason"] == "isolated_peak_boost_block"


def test_soundcheck_blocks_boost_when_source_is_close_to_noise_floor():
    stats = _stats_with_noisy_floor()

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "noise_floor_margin_boost_block"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "low_noise_margin"
    assert report["input_gain_character"]["noise_floor_db"] == pytest.approx(-44.0)
    assert report["input_gain_character"]["noise_floor_margin_db"] < 12.0
    assert report["input_gain_character"]["boost_guard"]["reason"] == "noise_floor_margin_boost_block"


def test_soundcheck_keeps_drum_boost_operator_review_even_when_stable():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-32.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="kick",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_personality_boost_review"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "operator_review_source"
    assert report["input_gain_character"]["channel_personality"]["family"] == "transient_percussion"
    assert report["input_gain_character"]["boost_guard"]["personality"]["boost_policy"] == "operator_review"


def test_soundcheck_hysteresis_blocks_small_opposite_direction_boost():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)
    profile = LocalInputGainProfile.from_config({
        "soundcheck_max_boost_db": 1.0,
        "opposite_direction_hold_db": 1.5,
    })

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={"suggested_gain_db": -1.0, "target_trim_db": -1.0},
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["target_trim_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "anti_oscillation_hold"
    assert report["operator_action"] == "hold"
    assert report["input_gain_character"]["hysteresis"]["active"] is True
    assert report["input_gain_character"]["hysteresis"]["previous_direction"] == "down"
    assert report["input_gain_character"]["hysteresis"]["current_direction"] == "up"


def test_live_hysteresis_does_not_block_urgent_peak_cut():
    stats = _stats_with_signal(peak_db=-0.5, lufs=-18.0)
    profile = LocalInputGainProfile.from_config({
        "live_max_cut_db": 1.0,
        "opposite_direction_hold_db": 1.5,
    })

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="live",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={"suggested_gain_db": 1.0, "target_trim_db": 1.0},
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-1.0)
    assert report["limited_by"] == "peak_headroom"
    assert report["operator_action"] == "apply_trim_proposal"
    assert report["auto_apply_allowed"] is True
    assert report["input_gain_character"]["hysteresis"]["active"] is False


def test_digital_clip_forces_headroom_cut_classification():
    stats = _stats_with_isolated_peak(base_peak_db=-20.0, spike_peak_db=0.2, lufs=-24.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-4.2)
    assert report["limited_by"] == "digital_clip_headroom"
    assert report["input_gain_character"]["overload"]["classification"] == "digital_clip"
    assert report["input_gain_character"]["overload"]["digital_clip_detected"] is True


def test_preamp_saturation_zone_forces_trim_down():
    stats = _stats_with_signal(peak_db=-2.0, lufs=-18.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=13.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-6.0)
    assert report["limited_by"] == "preamp_saturation_trim_down"
    assert report["input_gain_character"]["overload"]["classification"] == "preamp_saturation_risk"
    assert report["input_gain_character"]["overload"]["preamp_saturation_detected"] is True


def test_screaming_vocal_is_classified_without_blocking_trim_down():
    stats = _stats_with_signal(peak_db=-2.0, lufs=-14.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-6.0)
    assert report["limited_by"] == "peak_headroom"
    assert report["input_gain_character"]["overload"]["classification"] == "vocal_scream_or_shout"
    assert report["input_gain_character"]["overload"]["vocal_scream_detected"] is True


def test_safe_gain_proposal_carries_preamp_character_evidence():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -0.5,
        "target_peak_db": -10.0,
        "suggested_gain_db": -1.0,
        "target_trim_db": -1.0,
        "confidence": 0.88,
        "operator_note": "Trim down protects input headroom.",
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
        },
    }

    class Mixer:
        def get_channel_gain(self, channel):
            return 0.0

    proposals = build_safe_gain_operator_proposals(
        {1: suggestion},
        channel_mapping={1: 1},
        mixer_client=Mixer(),
    )

    change = proposals[0]["requested_change"]
    assert change["readback_required"] is True
    assert change["rollback_value"] == 0.0
    assert any(item["label"] == "Preamp zone" for item in change["safety_evidence"])
    assert any(item["label"] == "Role" and item["value"] == "input_headroom_not_mix_balance" for item in change["safety_evidence"])
