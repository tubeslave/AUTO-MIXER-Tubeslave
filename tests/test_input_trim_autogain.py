import pytest

from lufs_gain_staging import LocalInputGainProfile, SafeGainCalibrator, SignalStats
from backend.operator_recommendation_bridge import build_safe_gain_operator_proposals


def _spectral_signature(
    bands_db,
    active_sample_count=20,
    band_sample_count=None,
    band_std_db=0.2,
    band_span_db=0.8,
):
    reference_db = sum(bands_db.values()) / max(len(bands_db), 1)
    sample_count = active_sample_count if band_sample_count is None else band_sample_count
    return {
        "available": bool(bands_db),
        "band_count": len(bands_db),
        "bands_db": dict(bands_db),
        "relative_bands_db": {
            band_name: band_db - reference_db
            for band_name, band_db in bands_db.items()
        },
        "band_sample_counts": {
            band_name: sample_count
            for band_name in bands_db
        },
        "band_coverage_ratios": {
            band_name: sample_count / max(active_sample_count, 1)
            for band_name in bands_db
        },
        "band_std_db": {
            band_name: band_std_db
            for band_name in bands_db
        },
        "band_span_db": {
            band_name: band_span_db
            for band_name in bands_db
        },
        "active_sample_count": active_sample_count,
        "signal_lufs_std_db": 0.0,
        "signal_peak_std_db": 0.0,
        "signal_variability_db": 0.0,
        "reference_db": reference_db if bands_db else None,
    }


def _stats_with_signal(peak_db=-18.0, lufs=-28.0, samples=20, spectral_bands_db=None):
    stats = SignalStats(channel_id=1)
    for _ in range(samples):
        stats.update_sample(
            true_peak_db=peak_db,
            lufs=lufs,
            sample_rms_db=lufs,
            spectral_bands_db=spectral_bands_db,
        )
    return stats


def _stats_with_sparse_spectral_signal(peak_db=-18.0, lufs=-28.0, samples=20, spectral_bands_db=None):
    stats = SignalStats(channel_id=1)
    for index in range(samples):
        stats.update_sample(
            true_peak_db=peak_db,
            lufs=lufs,
            sample_rms_db=lufs,
            spectral_bands_db=spectral_bands_db if index == 0 else None,
        )
    return stats


def _stats_with_spectral_sequence(
    peak_db=-18.0,
    lufs=-28.0,
    spectral_sequence=None,
    lufs_sequence=None,
    peak_sequence=None,
):
    stats = SignalStats(channel_id=1)
    spectral_sequence = spectral_sequence or []
    sample_count = len(spectral_sequence)
    for index, spectral_bands in enumerate(spectral_sequence):
        sample_lufs = lufs_sequence[index] if lufs_sequence is not None else lufs
        sample_peak = peak_sequence[index] if peak_sequence is not None else peak_db
        stats.update_sample(
            true_peak_db=sample_peak,
            lufs=sample_lufs,
            sample_rms_db=sample_lufs,
            spectral_bands_db=spectral_bands,
        )
    assert stats.total_samples == sample_count
    return stats


def _stats_with_sparse_signal(active_peak_db=-24.0, active_lufs=-34.0, active_samples=3, silent_samples=17):
    stats = SignalStats(channel_id=1)
    for _ in range(active_samples):
        stats.update_sample(true_peak_db=active_peak_db, lufs=active_lufs, sample_rms_db=active_lufs)
    for _ in range(silent_samples):
        stats.update_sample(true_peak_db=-90.0, lufs=-100.0, sample_rms_db=-100.0)
    return stats


def _stats_with_timed_signal(active_start, active_end, total_samples=20, peak_db=-18.0, lufs=-28.0):
    stats = SignalStats(channel_id=1)
    for sample_index in range(1, total_samples + 1):
        if active_start <= sample_index <= active_end:
            stats.update_sample(true_peak_db=peak_db, lufs=lufs, sample_rms_db=lufs)
        else:
            stats.update_sample(true_peak_db=-90.0, lufs=-100.0, sample_rms_db=-100.0)
    return stats


def test_spectral_fingerprint_normalizes_numeric_analyzer_bands():
    stats = _stats_with_signal(
        peak_db=-18.0,
        lufs=-28.0,
        spectral_bands_db={
            "63 Hz": -30.0,
            "125Hz": -32.0,
            "250 Hz": -35.0,
            "4 kHz": -40.0,
            "12k": -45.0,
        },
    )

    fingerprint = stats.calculate_spectral_fingerprint()

    assert fingerprint["available"] is True
    assert fingerprint["bands_db"]["sub"] == pytest.approx(-30.0)
    assert fingerprint["bands_db"]["low"] == pytest.approx(-32.0)
    assert fingerprint["bands_db"]["low_mid"] == pytest.approx(-35.0)
    assert fingerprint["bands_db"]["presence"] == pytest.approx(-40.0)
    assert fingerprint["bands_db"]["air"] == pytest.approx(-45.0)


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


def _stats_with_peak_sequence(peak_values, lufs=-18.0):
    stats = SignalStats(channel_id=1)
    for peak_db in peak_values:
        stats.update_sample(true_peak_db=peak_db, lufs=lufs, sample_rms_db=lufs)
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
    foh_summary = report["input_gain_character"]["foh_summary"]
    assert report["suggested_gain_db"] == pytest.approx(-1.0)
    assert report["target_trim_db"] == pytest.approx(-1.0)
    assert report["safety_policy"]["live_policy"] == "trim_down_only"
    assert report["auto_apply_allowed"] is True
    assert foh_summary["safe_to_auto_apply"] is True
    assert foh_summary["readiness"] == "auto_apply_trim_down"
    assert foh_summary["next_action"] == "apply_trim_down"
    assert foh_summary["gain_structure"]["target_trim_db"] == pytest.approx(-1.0)


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
    foh_summary = report["input_gain_character"]["foh_summary"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["target_trim_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert report["operator_action"] == "apply_trim_proposal"
    assert report["input_gain_character"]["source_behavior"] == "stable_dynamic"
    assert report["input_gain_character"]["boost_guard"]["active"] is False
    assert report["auto_apply_allowed"] is False
    assert foh_summary["safe_to_auto_apply"] is False
    assert foh_summary["readiness"] == "operator_review_gain_up"
    assert foh_summary["next_action"] == "review_soundcheck_gain_up"
    assert foh_summary["gain_structure"]["target_trim_db"] == pytest.approx(3.0)
    assert foh_summary["measurement_quality"]["readiness"] == "level_only"
    assert "missing_spectral_fingerprint" in foh_summary["measurement_quality"]["warnings"]
    assert "missing_noise_floor_reference" in foh_summary["measurement_quality"]["warnings"]
    assert foh_summary == report["input_gain_character"]["decision_confidence"]["foh_summary"]


def test_soundcheck_large_vocal_gap_converges_faster_but_flags_source_setup():
    stats = _stats_with_signal(peak_db=-31.0, lufs=-41.0)

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
                    "candidate_delta_db": 6.0,
                    "evidence_windows": 1,
                }
            }
        },
    )

    report = stats.get_report()
    character = report["input_gain_character"]
    convergence = character["soundcheck_convergence"]
    foh_summary = character["foh_summary"]

    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["target_trim_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "trim_smoothing_gain_up_slew_limited"
    assert convergence["active"] is True
    assert convergence["requested_delta_db"] == pytest.approx(21.0)
    assert convergence["effective_max_boost_db"] == pytest.approx(9.0)
    assert convergence["setup_check_required"] is True
    assert convergence["setup_check_reason"] == "requested_gain_exceeds_setup_check_threshold"
    assert character["trim_smoothing"]["gain_up_step_db"] == pytest.approx(3.0)
    assert character["trim_smoothing"]["reference_source"] == "soundcheck_convergence_start"
    assert foh_summary["next_action"] == "check_source_setup_and_review_gain_up"
    assert "large_soundcheck_gain_gap" in foh_summary["risk_flags"]
    assert "source_setup_check_required" in foh_summary["risk_flags"]


def test_soundcheck_large_vocal_gap_can_step_beyond_three_db_when_profile_allows_clean_range():
    stats = _stats_with_signal(peak_db=-28.0, lufs=-38.0)
    profile = LocalInputGainProfile.from_config({
        "clean_gain_max_db": 12.0,
        "coloration_start_db": 15.0,
        "saturation_start_db": 18.0,
        "soundcheck_convergence_vocal_max_boost_db": 9.0,
        "soundcheck_convergence_gain_up_step_db": 3.0,
    })

    stats.calculate_input_trim_recommendation(
        current_trim_db=3.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "status": "applied",
            "input_gain_character": {
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 9.0,
                    "evidence_windows": 1,
                },
                "trim_smoothing": {
                    "smoothed_delta_db": 3.0,
                    "applied": True,
                },
            },
        },
    )

    report = stats.get_report()
    character = report["input_gain_character"]
    convergence = character["soundcheck_convergence"]

    assert report["suggested_gain_db"] == pytest.approx(6.0)
    assert report["target_trim_db"] == pytest.approx(9.0)
    assert report["limited_by"] == "trim_smoothing_gain_up_slew_limited"
    assert convergence["active"] is True
    assert convergence["requested_delta_db"] == pytest.approx(18.0)
    assert convergence["effective_max_boost_db"] == pytest.approx(9.0)
    assert convergence["post_smoothing_delta_db"] == pytest.approx(6.0)
    assert character["trim_smoothing"]["reference_delta_db"] == pytest.approx(3.0)
    assert character["trim_smoothing"]["gain_up_step_db"] == pytest.approx(3.0)


def test_soundcheck_ensemble_context_allows_bolder_vocal_gain_up_than_solo():
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
                    "candidate_delta_db": 6.0,
                    "evidence_windows": 1,
                }
            }
        },
        ensemble_context={
            "other_active_sources": 4,
            "other_loud_sources": 3,
            "time_coherence_evaluable": True,
            "time_coherent_loud_sources": 3,
            "required_time_coherent_sources": 2,
            "max_context_lag_samples": 2,
            "target_first_active_sample_index": 1,
            "target_last_active_sample_index": 20,
            "reference_latest_active_sample_index": 20,
            "reference_max_lag_samples": 0,
            "reference_peak_db": -9.0,
            "reference_lufs": -18.0,
        },
    )

    report = stats.get_report()
    character = report["input_gain_character"]
    ensemble = character["soundcheck_ensemble_context"]
    convergence = character["soundcheck_convergence"]
    foh_summary = character["foh_summary"]
    confidence_factors = {
        factor["name"]: factor["delta"]
        for factor in character["decision_confidence"]["factors"]
    }

    assert report["suggested_gain_db"] == pytest.approx(6.0)
    assert report["target_trim_db"] == pytest.approx(6.0)
    assert report["limited_by"] == "quiet_source"
    assert ensemble["active"] is True
    assert ensemble["reason"] == "ensemble_context_vocal_understaged"
    assert ensemble["time_coherent"] is True
    assert ensemble["time_coherent_loud_sources"] == 3
    assert ensemble["vocal_peak_gap_db"] == pytest.approx(11.0)
    assert ensemble["vocal_lufs_gap_db"] == pytest.approx(12.0)
    assert convergence["active"] is True
    assert convergence["ensemble_assisted"] is True
    assert convergence["effective_max_boost_db"] == pytest.approx(6.0)
    assert character["trim_smoothing"]["gain_up_step_db"] == pytest.approx(6.0)
    assert confidence_factors["ensemble_context_vocal_understaged"] == pytest.approx(0.08)
    assert "ensemble_context_vocal_understaged" in foh_summary["risk_flags"]


def test_soundcheck_ensemble_context_ignored_when_other_sources_are_not_loud():
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
        ensemble_context={
            "other_active_sources": 4,
            "other_loud_sources": 1,
            "reference_peak_db": -24.0,
            "reference_lufs": -34.0,
        },
    )

    report = stats.get_report()
    ensemble = report["input_gain_character"]["soundcheck_ensemble_context"]
    convergence = report["input_gain_character"]["soundcheck_convergence"]

    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["target_trim_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert ensemble["active"] is False
    assert ensemble["reason"] == "insufficient_loud_ensemble_sources"
    assert convergence["active"] is False


def test_soundcheck_ensemble_context_blocks_bolder_gain_up_when_sources_are_stale():
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
        ensemble_context={
            "other_active_sources": 4,
            "other_loud_sources": 3,
            "time_coherence_evaluable": True,
            "time_coherent_loud_sources": 1,
            "required_time_coherent_sources": 2,
            "max_context_lag_samples": 2,
            "target_first_active_sample_index": 11,
            "target_last_active_sample_index": 20,
            "reference_latest_active_sample_index": 10,
            "reference_max_lag_samples": 10,
            "reference_peak_db": -9.0,
            "reference_lufs": -18.0,
        },
    )

    report = stats.get_report()
    character = report["input_gain_character"]
    ensemble = character["soundcheck_ensemble_context"]
    confidence_factors = {
        factor["name"]: factor["delta"]
        for factor in character["decision_confidence"]["factors"]
    }

    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["target_trim_db"] == pytest.approx(3.0)
    assert ensemble["active"] is False
    assert ensemble["reason"] == "ensemble_time_context_stale"
    assert ensemble["time_coherent"] is False
    assert ensemble["time_coherent_loud_sources"] == 1
    assert "ensemble_context_vocal_understaged" not in confidence_factors


def test_soundcheck_ensemble_context_reports_source_and_lag_margins():
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
        ensemble_context={
            "other_active_sources": 4,
            "other_loud_sources": 2,
            "time_coherence_evaluable": True,
            "time_coherent_loud_sources": 1,
            "required_time_coherent_sources": 2,
            "max_context_lag_samples": 2,
            "reference_latest_active_sample_index": 15,
            "reference_max_lag_samples": 5,
            "reference_peak_db": -9.0,
            "reference_lufs": -18.0,
        },
    )

    ensemble = stats.get_report()["input_gain_character"]["soundcheck_ensemble_context"]
    assert ensemble["active"] is False
    assert ensemble["reason"] == "ensemble_time_context_stale"
    assert ensemble["other_loud_source_margin"] == 0
    assert ensemble["time_coherent_source_margin"] == -1
    assert ensemble["reference_lag_margin_samples"] == -3


def test_safe_gain_calibrator_builds_ensemble_context_from_other_loud_channels():
    calibrator = SafeGainCalibrator()
    calibrator.add_channel(1)
    calibrator.add_channel(2)
    calibrator.add_channel(3)
    calibrator.channel_settings = {
        1: {"preset": "leadVocal"},
        2: {"preset": "bass"},
        3: {"preset": "electricGuitar"},
    }
    calibrator.channels[1] = _stats_with_signal(peak_db=-20.0, lufs=-30.0)
    calibrator.channels[2] = _stats_with_signal(peak_db=-9.0, lufs=-18.0)
    calibrator.channels[3] = _stats_with_signal(peak_db=-11.0, lufs=-19.0)
    calibrator.suggestions[1] = {
        "input_gain_character": {
            "stability": {
                "candidate_direction": "up",
                "candidate_delta_db": 6.0,
                "evidence_windows": 1,
            }
        }
    }

    calibrator._finalize_analysis()

    vocal = calibrator.suggestions[1]
    ensemble = vocal["input_gain_character"]["soundcheck_ensemble_context"]
    assert vocal["suggested_gain_db"] == pytest.approx(6.0)
    assert ensemble["active"] is True
    assert ensemble["other_loud_sources"] == 2
    assert ensemble["other_loud_source_margin"] == 0
    assert ensemble["time_coherent_source_margin"] == 0
    assert ensemble["reference_lag_margin_samples"] == 2
    assert ensemble["reference_peak_db"] == pytest.approx(-9.5)
    assert ensemble["reference_lufs"] == pytest.approx(-18.25)


def test_safe_gain_calibrator_ignores_stale_ensemble_context_for_bolder_gain_up():
    calibrator = SafeGainCalibrator()
    calibrator.add_channel(1)
    calibrator.add_channel(2)
    calibrator.add_channel(3)
    calibrator.channel_settings = {
        1: {"preset": "leadVocal"},
        2: {"preset": "bass"},
        3: {"preset": "electricGuitar"},
    }
    calibrator.channels[1] = _stats_with_timed_signal(
        active_start=11,
        active_end=20,
        peak_db=-18.0,
        lufs=-28.0,
    )
    calibrator.channels[2] = _stats_with_timed_signal(
        active_start=1,
        active_end=10,
        peak_db=-9.0,
        lufs=-18.0,
    )
    calibrator.channels[3] = _stats_with_timed_signal(
        active_start=1,
        active_end=10,
        peak_db=-11.0,
        lufs=-19.0,
    )
    calibrator.suggestions[1] = {
        "input_gain_character": {
            "stability": {
                "candidate_direction": "up",
                "candidate_delta_db": 3.0,
                "evidence_windows": 1,
            }
        }
    }

    calibrator._finalize_analysis()

    vocal = calibrator.suggestions[1]
    ensemble = vocal["input_gain_character"]["soundcheck_ensemble_context"]
    assert vocal["suggested_gain_db"] == pytest.approx(3.0)
    assert ensemble["active"] is False
    assert ensemble["reason"] == "ensemble_time_context_stale"
    assert ensemble["time_coherent_loud_sources"] == 0
    assert ensemble["reference_max_lag_samples"] == 10
    assert ensemble["other_loud_source_margin"] == 0
    assert ensemble["time_coherent_source_margin"] == -2
    assert ensemble["reference_lag_margin_samples"] == -8


def test_safe_gain_calibrator_reports_disabled_ensemble_context_before_source_counts():
    calibrator = SafeGainCalibrator(config={
        "automation": {
            "safe_gain_calibration": {
                "local_input_preamp": {
                    "soundcheck_ensemble_context_enabled": False,
                },
            },
        },
    })
    calibrator.add_channel(1)
    calibrator.add_channel(2)
    calibrator.channel_settings = {
        1: {"preset": "leadVocal"},
        2: {"preset": "bass"},
    }
    calibrator.channels[1] = _stats_with_signal(peak_db=-20.0, lufs=-30.0)
    calibrator.channels[2] = _stats_with_signal(peak_db=-9.0, lufs=-18.0)
    calibrator.suggestions[1] = {
        "input_gain_character": {
            "stability": {
                "candidate_direction": "up",
                "candidate_delta_db": 3.0,
                "evidence_windows": 1,
            }
        }
    }

    calibrator._finalize_analysis()

    vocal = calibrator.suggestions[1]
    ensemble = vocal["input_gain_character"]["soundcheck_ensemble_context"]
    convergence = vocal["input_gain_character"]["soundcheck_convergence"]
    confidence_factors = {
        factor["name"]: factor["delta"]
        for factor in vocal["input_gain_character"]["decision_confidence"]["factors"]
    }

    assert ensemble["enabled"] is False
    assert ensemble["active"] is False
    assert ensemble["reason"] == "disabled"
    assert convergence["ensemble_assisted"] is False
    assert "ensemble_context_vocal_understaged" not in confidence_factors


def test_foh_summary_marks_full_measurement_context_when_spectral_and_noise_floor_exist():
    stats = SignalStats(channel_id=1)
    for _ in range(12):
        stats.update_sample(
            true_peak_db=-20.0,
            lufs=-30.0,
            sample_rms_db=-30.0,
            spectral_bands_db={
                "125 Hz": -38.0,
                "250 Hz": -36.0,
                "4 kHz": -42.0,
                "12 kHz": -48.0,
            },
        )
    for _ in range(8):
        stats.update_sample(true_peak_db=-90.0, lufs=-100.0, sample_rms_db=-68.0)

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
    measurement = report["input_gain_character"]["foh_summary"]["measurement_quality"]
    assert measurement["readiness"] == "full_context"
    assert measurement["warnings"] == []
    assert measurement["spectral_available"] is True
    assert measurement["spectral_band_count"] == 4
    assert measurement["noise_floor_available"] is True
    assert measurement["noise_floor_db"] == pytest.approx(-68.0)


def test_soundcheck_blocks_vocal_gain_up_when_presence_band_is_unstable():
    spectral_sequence = []
    for index in range(20):
        unstable_presence = -30.0 if index % 2 else -45.0
        spectral_sequence.append({
            "low": -42.0,
            "low_mid": -36.0,
            "presence": unstable_presence,
            "air": -44.0,
        })
    stats = _stats_with_spectral_sequence(
        peak_db=-20.0,
        lufs=-30.0,
        spectral_sequence=spectral_sequence,
    )

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
    guard = report["input_gain_character"]["vocal_gain_up_guard"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "vocal_presence_stability_wait"
    assert report["operator_action"] == "operator_review"
    assert guard["active"] is True
    assert guard["reason"] == "unstable_vocal_presence_band"
    assert guard["presence_available"] is True
    assert guard["presence_unstable"] is True
    assert guard["presence_bands"]["presence"]["stable"] is False
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_stable_vocal_presence_band"


def test_soundcheck_blocks_vocal_gain_up_during_post_scream_cooldown():
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
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "vocal_scream_cooldown_windows": 2,
                },
            }
        },
    )

    report = stats.get_report()
    guard = report["input_gain_character"]["vocal_gain_up_guard"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "vocal_scream_history_wait"
    assert report["operator_action"] == "operator_review"
    assert guard["active"] is True
    assert guard["reason"] == "recent_vocal_scream_or_shout"
    assert guard["previous_scream_cooldown_windows"] == 2
    assert guard["scream_cooldown_windows"] == 1
    assert trend["vocal_scream_cooldown_windows"] == 1
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_after_recent_vocal_scream_or_shout"


def test_trim_smoothing_slew_limits_repeated_gain_up_proposal():
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
            "suggested_gain_db": 0.5,
            "input_gain_character": {
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                }
            },
        },
    )

    report = stats.get_report()
    smoothing = report["input_gain_character"]["trim_smoothing"]
    assert report["suggested_gain_db"] == pytest.approx(1.5)
    assert report["target_trim_db"] == pytest.approx(1.5)
    assert report["limited_by"] == "trim_smoothing_gain_up_slew_limited"
    assert smoothing["active"] is True
    assert smoothing["reason"] == "gain_up_slew_limited"
    assert smoothing["previous_smoothed_delta_db"] == pytest.approx(0.5)
    assert smoothing["smoothed_delta_db"] == pytest.approx(1.5)
    assert smoothing["gain_up_step_db"] == pytest.approx(1.0)


def test_trim_smoothing_resets_reference_for_dismissed_previous_proposal():
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
            "status": "dismissed",
            "suggested_gain_db": 2.5,
            "input_gain_character": {
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trim_smoothing": {
                    "smoothed_delta_db": 2.5,
                },
            },
        },
    )

    report = stats.get_report()
    smoothing = report["input_gain_character"]["trim_smoothing"]
    assert report["suggested_gain_db"] == pytest.approx(1.0)
    assert report["limited_by"] == "trim_smoothing_gain_up_slew_limited"
    assert smoothing["previous_smoothed_delta_db"] == pytest.approx(2.5)
    assert smoothing["previous_apply_status"] == "dismissed"
    assert smoothing["previous_explicitly_unapplied"] is True
    assert smoothing["reference_source"] == "current_trim_due_to_unapplied_previous"
    assert smoothing["reference_delta_db"] == pytest.approx(0.0)
    assert smoothing["smoothed_delta_db"] == pytest.approx(1.0)


def test_trim_smoothing_does_not_limit_urgent_headroom_cut():
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
        previous_recommendation={
            "suggested_gain_db": 0.0,
            "input_gain_character": {
                "trim_smoothing": {
                    "smoothed_delta_db": 0.0,
                }
            },
        },
    )

    report = stats.get_report()
    smoothing = report["input_gain_character"]["trim_smoothing"]
    foh_summary = report["input_gain_character"]["foh_summary"]
    assert report["suggested_gain_db"] == pytest.approx(-6.0)
    assert report["limited_by"] == "peak_headroom"
    assert smoothing["urgent_headroom_bypass"] is True
    assert smoothing["active"] is False
    assert smoothing["smoothed_delta_db"] == pytest.approx(-6.0)
    assert foh_summary["safe_to_auto_apply"] is True
    assert foh_summary["readiness"] == "auto_apply_trim_down"
    assert "peak_over_limit" in foh_summary["risk_flags"]
    assert foh_summary["headroom"]["max_true_peak_db"] == pytest.approx(-0.5)


def test_soundcheck_gain_up_waits_when_source_behavior_flips():
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
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "low_noise_margin",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["low_noise_margin"],
                    "predictive_clip_windows": 0,
                },
            },
        },
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "behavior_change_stability_wait"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_consistent_source_behavior"
    assert report["input_gain_character"]["trend_memory"]["behavior_flip"] is True
    assert report["input_gain_character"]["trend_memory"]["behavior_history"] == [
        "low_noise_margin",
        "stable_dynamic",
    ]


def test_soundcheck_gain_up_waits_when_source_signature_changes_without_behavior_flip():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_bucket_db": 3.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_bucket_db": 3.0,
                        "noise_margin_bucket_db": None,
                    },
                },
            },
        },
    )

    report = stats.get_report()
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_identity_change_wait"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_source_identity_confirmation"
    assert trend["behavior_flip"] is False
    assert trend["source_identity_changed"] is True
    assert set(trend["source_signature_changed_fields"]) == {"crest_bucket_db", "peak_lufs_gap_bucket_db"}


def test_decision_confidence_holds_gain_up_when_source_continuity_is_weak():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"source_signature_change_hold_enabled": False}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_bucket_db": 3.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_bucket_db": 3.0,
                        "noise_margin_bucket_db": None,
                    },
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    foh_summary = confidence["foh_summary"]
    factor_names = {factor["name"] for factor in confidence["factors"]}
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_gain_up_hold"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_higher_gain_up_confidence"
    assert confidence["candidate_direction"] == "up"
    assert confidence["gated"] is True
    assert confidence["threshold_met"] is False
    assert confidence["score"] < confidence["required_score"]
    assert "source_identity_change" in factor_names
    assert foh_summary["policy"] == "display_only_no_apply_authority"
    assert foh_summary["candidate_direction"] == "up"
    assert foh_summary["final_direction"] == "hold"
    assert foh_summary["limited_by"] == "decision_confidence_gain_up_hold"
    assert foh_summary["gate_reason"] == "gain_up_confidence_below_threshold"
    assert foh_summary["headline"] == "Hold for operator review: decision_confidence_gain_up_hold"
    assert foh_summary["evidence_window"]["stability_evidence_windows"] == 2
    assert foh_summary["evidence_window"]["confidence_windows"] == 0
    assert foh_summary["evidence_window"]["threshold_met"] is False
    assert "source_identity_change" in {
        factor["name"]
        for factor in foh_summary["dominant_blockers"]
    }


def test_decision_confidence_hysteresis_holds_first_borderline_gain_up():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_stable_gain_up_confidence"
    assert confidence["threshold_met"] is True
    assert confidence["borderline"] is True
    assert confidence["hysteresis_stable"] is False
    assert confidence["confidence_windows"] == 1


def test_decision_confidence_hysteresis_allows_repeated_borderline_gain_up():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({
            "decision_confidence_gain_up_min_score": 0.84,
            "vocal_gain_up_guard_enabled": False,
        }),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 104.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    foh_summary = confidence["foh_summary"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert report["operator_action"] == "apply_trim_proposal"
    assert confidence["threshold_met"] is True
    assert confidence["borderline"] is True
    assert confidence["hysteresis_stable"] is True
    assert confidence["confidence_windows"] == 2
    assert confidence["previous_confidence_expired"] is False
    assert confidence["elapsed_since_previous_sec"] == pytest.approx(6.0)
    assert foh_summary["policy"] == "display_only_no_apply_authority"
    assert foh_summary["candidate_direction"] == "up"
    assert foh_summary["final_direction"] == "up"
    assert foh_summary["suggested_delta_db"] == pytest.approx(3.0)
    assert foh_summary["headline"] == "Trim up 3.0 dB: quiet_source"
    assert foh_summary["evidence_window"]["confidence_windows"] == 2
    assert foh_summary["evidence_window"]["hysteresis_stable"] is True


def test_decision_confidence_time_memory_expires_stale_borderline_window():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 70.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert report["operator_action"] == "operator_review"
    assert confidence["threshold_met"] is True
    assert confidence["borderline"] is True
    assert confidence["previous_confidence_expired"] is True
    assert confidence["confidence_windows"] == 1
    assert confidence["elapsed_since_previous_sec"] == pytest.approx(40.0)
    assert confidence["max_gap_sec"] == pytest.approx(20.0)


def test_decision_confidence_decay_rejects_old_vocal_memory_before_hard_expiry():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 90.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert confidence["previous_confidence_expired"] is False
    assert confidence["memory_decay_family"] == "vocal"
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 1
    assert confidence["elapsed_since_previous_sec"] == pytest.approx(20.0)


def test_decision_confidence_decay_keeps_playback_memory_for_same_gap():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="playback",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.89,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 90.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert confidence["memory_decay_family"] == "playback"
    assert confidence["memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 2
    assert confidence["hysteresis_stable"] is True


def test_source_signature_drift_penalty_rejects_vocal_borderline_memory():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.80}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_bucket_db": -24.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_bucket_db": None,
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 104.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert trend["source_identity_changed"] is False
    assert trend["source_signature_changed_fields"] == ["integrated_lufs_bucket_db"]
    assert confidence["time_memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["source_signature_drift_penalty"] == pytest.approx(0.3)
    assert confidence["confidence_windows"] == 1


def test_source_signature_drift_weights_vocal_noise_margin_more_heavily():
    stats = _stats_with_noisy_floor(active_peak_db=-20.0, active_lufs=-30.0, noise_lufs=-60.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_bucket_db": 24.0,
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.90,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert trend["source_identity_changed"] is False
    assert trend["source_signature_changed_fields"] == ["noise_margin_bucket_db"]
    assert confidence["source_signature_drift_field_weights"]["noise_margin_bucket_db"] == pytest.approx(1.7)
    assert confidence["source_signature_drift_penalty"] == pytest.approx(0.51)
    assert confidence["time_memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["memory_strength"] < confidence["min_memory_strength"]


def test_continuous_source_signature_drift_keeps_small_vocal_noise_move():
    stats = _stats_with_noisy_floor(active_peak_db=-20.0, active_lufs=-30.0, noise_lufs=-51.1)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": 22.0,
                        "noise_margin_bucket_db": 24.0,
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert trend["source_signature_changed_fields"] == []
    assert confidence["source_signature_drift_penalty"] == pytest.approx(0.0)
    assert confidence["source_signature_continuous_drift_penalty"] < 0.10
    assert confidence["memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 2


def test_continuous_source_signature_drift_rejects_large_in_bucket_vocal_noise_move():
    stats = _stats_with_noisy_floor(active_peak_db=-20.0, active_lufs=-30.0, noise_lufs=-51.1)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": 26.9,
                        "noise_margin_bucket_db": 24.0,
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    continuous = confidence["source_signature_continuous_distances"]["noise_margin_db"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert trend["source_signature_changed_fields"] == []
    assert continuous["distance_db"] == pytest.approx(5.8)
    assert continuous["weight"] == pytest.approx(1.7)
    assert confidence["source_signature_drift_penalty"] == pytest.approx(0.0)
    assert confidence["source_signature_continuous_drift_penalty"] == pytest.approx(0.5751666667)
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 1


def test_spectral_source_signature_keeps_borderline_memory_for_same_vocal_tone_shape():
    current_bands = {
        "low": -39.0,
        "low_mid": -33.5,
        "presence": -37.0,
        "air": -42.5,
    }
    previous_bands = {
        band_name: band_db - 2.0
        for band_name, band_db in current_bands.items()
    }
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0, spectral_bands_db=current_bands)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(previous_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert trend["source_identity_changed"] is False
    assert trend["source_signature"]["spectral_fingerprint"]["available"] is True
    assert confidence["source_signature_spectral_common_band_count"] == 4
    assert confidence["source_signature_spectral_drift_penalty"] == pytest.approx(0.0)
    assert confidence["memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 2


def test_spectral_source_signature_drift_rejects_same_level_borderline_vocal_memory():
    current_bands = {
        "low": -46.0,
        "low_mid": -34.0,
        "presence": -36.0,
        "air": -42.0,
    }
    previous_bands = {
        "low": -34.0,
        "low_mid": -46.0,
        "presence": -44.0,
        "air": -36.0,
    }
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0, spectral_bands_db=current_bands)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(previous_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert trend["source_identity_changed"] is False
    assert trend["source_signature_changed_fields"] == []
    assert confidence["source_signature_drift_penalty"] == pytest.approx(0.0)
    assert confidence["source_signature_continuous_drift_penalty"] < 1e-6
    assert confidence["source_signature_spectral_common_band_count"] == 4
    assert confidence["source_signature_spectral_distance_db"] > 8.0
    assert confidence["source_signature_spectral_drift_penalty"] == pytest.approx(0.65)
    assert confidence["source_signature_spectral_band_weights"]["presence"] == pytest.approx(1.35)
    assert confidence["source_signature_spectral_band_weights"]["air"] == pytest.approx(1.35)
    assert confidence["time_memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 1


def test_spectral_source_signature_playback_tolerates_broad_tone_shift_more_than_vocal():
    current_bands = {
        "low": -46.0,
        "low_mid": -34.0,
        "presence": -36.0,
        "air": -42.0,
    }
    previous_bands = {
        "low": -34.0,
        "low_mid": -46.0,
        "presence": -44.0,
        "air": -36.0,
    }
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0, spectral_bands_db=current_bands)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="playback",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "playback",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(previous_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.89,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert all(
        weight == pytest.approx(0.55)
        for weight in confidence["source_signature_spectral_band_weights"].values()
    )
    assert confidence["source_signature_spectral_drift_penalty"] < 0.45
    assert confidence["memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 2


def test_sparse_spectral_source_signature_is_not_used_for_borderline_gain_up_memory():
    current_bands = {
        "low": -46.0,
        "low_mid": -34.0,
        "presence": -36.0,
        "air": -42.0,
    }
    previous_bands = {
        "low": -34.0,
        "low_mid": -46.0,
        "presence": -44.0,
        "air": -36.0,
    }
    stats = _stats_with_sparse_spectral_signal(
        peak_db=-20.0,
        lufs=-30.0,
        spectral_bands_db=current_bands,
    )

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(previous_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    rejected = confidence["source_signature_spectral_rejected_bands"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert confidence["source_signature_spectral_quality_gate_enabled"] is True
    assert confidence["source_signature_spectral_all_common_bands"] == ["air", "low", "low_mid", "presence"]
    assert confidence["source_signature_spectral_common_band_count"] == 0
    assert set(rejected) == {"air", "low", "low_mid", "presence"}
    assert rejected["presence"]["current"]["reason"] == "too_few_band_samples"
    assert confidence["source_signature_spectral_quality_rejected_count"] == 4
    assert confidence["source_signature_spectral_quality_penalty"] == pytest.approx(0.55)
    assert confidence["source_signature_spectral_drift_penalty"] == pytest.approx(0.0)
    assert confidence["time_memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 1


def test_unstable_spectral_bands_lower_borderline_gain_up_memory():
    median_bands = {
        "low": -42.0,
        "low_mid": -36.0,
        "presence": -38.0,
        "air": -44.0,
    }
    spectral_sequence = []
    for index in range(20):
        swing = 8.0 if index % 2 else -8.0
        spectral_sequence.append({
            "low": median_bands["low"] + swing,
            "low_mid": median_bands["low_mid"] - swing,
            "presence": median_bands["presence"] + swing,
            "air": median_bands["air"] - swing,
        })
    stats = _stats_with_spectral_sequence(
        peak_db=-20.0,
        lufs=-30.0,
        spectral_sequence=spectral_sequence,
    )

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({
            "decision_confidence_gain_up_min_score": 0.84,
            "vocal_gain_up_guard_enabled": False,
        }),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(median_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    rejected = confidence["source_signature_spectral_rejected_bands"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert confidence["source_signature_spectral_common_band_count"] == 0
    assert set(rejected) == {"air", "low", "low_mid", "presence"}
    assert rejected["presence"]["current"]["reason"] == "high_band_variance"
    assert rejected["presence"]["current"]["std_db"] > 6.0
    assert confidence["source_signature_spectral_quality_penalty"] == pytest.approx(0.55)
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 1


def test_playback_spectral_quality_tolerates_mastered_band_motion_more_than_vocal():
    median_bands = {
        "low": -42.0,
        "low_mid": -36.0,
        "presence": -38.0,
        "air": -44.0,
    }
    spectral_sequence = []
    for index in range(20):
        swing = 8.0 if index % 2 else -8.0
        spectral_sequence.append({
            "low": median_bands["low"] + swing,
            "low_mid": median_bands["low_mid"] - swing,
            "presence": median_bands["presence"] + swing,
            "air": median_bands["air"] - swing,
        })
    stats = _stats_with_spectral_sequence(
        peak_db=-20.0,
        lufs=-30.0,
        spectral_sequence=spectral_sequence,
    )

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="playback",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "playback",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(median_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.89,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    distances = confidence["source_signature_spectral_distances"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert confidence["source_signature_spectral_quality_penalty"] == pytest.approx(0.0)
    assert confidence["source_signature_spectral_common_band_count"] == 4
    assert distances["presence"]["weight"] == pytest.approx(0.55)
    assert confidence["memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 2


def test_stuck_spectral_output_lowers_memory_when_signal_is_changing():
    current_bands = {
        "low": -39.0,
        "low_mid": -33.5,
        "presence": -37.0,
        "air": -42.5,
    }
    lufs_sequence = [-32.0, -28.0, -31.0, -27.0] * 5
    stats = _stats_with_spectral_sequence(
        peak_db=-20.0,
        spectral_sequence=[current_bands] * len(lufs_sequence),
        lufs_sequence=lufs_sequence,
    )

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({
            "decision_confidence_gain_up_min_score": 0.84,
            "soundcheck_boost_max_lufs_std_db": 5.0,
        }),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "lead_vocal",
                        "behavior": "stable_dynamic",
                        "crest_db": 10.0,
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_db": -30.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 10.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(current_bands),
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.86,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 109.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    rejected = confidence["source_signature_spectral_rejected_bands"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "decision_confidence_stability_wait"
    assert set(rejected) == {"air", "low", "low_mid", "presence"}
    assert rejected["presence"]["current"]["reason"] == "stuck_band_output"
    assert rejected["presence"]["current"]["signal_variability_db"] >= 1.0
    assert confidence["source_signature_spectral_quality_penalty"] == pytest.approx(0.55)
    assert confidence["memory_strength"] < confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 1


def test_source_signature_drift_penalty_keeps_playback_memory_for_same_drift():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"decision_confidence_gain_up_min_score": 0.84}),
        mode="soundcheck",
        mixer_channel=1,
        preset="playback",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "trend_memory": {
                    "behavior_history": ["stable_dynamic"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "playback",
                        "behavior": "stable_dynamic",
                        "crest_bucket_db": 9.0,
                        "integrated_lufs_bucket_db": -24.0,
                        "peak_lufs_gap_bucket_db": 9.0,
                        "noise_margin_bucket_db": None,
                    },
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.89,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 104.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    trend = report["input_gain_character"]["trend_memory"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert trend["source_identity_changed"] is False
    assert trend["source_signature_changed_fields"] == ["integrated_lufs_bucket_db"]
    assert confidence["source_signature_drift_field_weights"]["integrated_lufs_bucket_db"] == pytest.approx(1.4)
    assert confidence["source_signature_drift_penalty"] == pytest.approx(0.42)
    assert confidence["memory_strength"] >= confidence["min_memory_strength"]
    assert confidence["confidence_windows"] == 2
    assert confidence["hysteresis_stable"] is True


def test_soundcheck_gain_up_waits_when_channel_profile_is_unstable():
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
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 5.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 20.0,
                    "unstable_behavior_ratio": 0.75,
                },
            },
        },
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "channel_profile_stability_wait"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_channel_profile_stability"
    assert report["input_gain_character"]["channel_profile"]["profile_ready"] is True
    assert report["input_gain_character"]["channel_profile"]["profile_blocks_gain_up"] is True
    assert report["input_gain_character"]["channel_profile"]["unstable_behavior_ratio"] >= 0.5


def test_channel_profile_recovery_waits_for_multiple_stable_windows():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 2.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.75,
                    "recovery_windows": 1,
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "channel_profile_stability_wait"
    assert profile["recovery_candidate"] is True
    assert profile["recovery_windows"] == 2
    assert profile["recovery_required_windows"] == 4
    assert profile["recovery_policy_family"] == "lead_vocal"
    assert profile["profile_blocks_gain_up_by_history_raw"] is True
    assert profile["profile_recovered_from_history"] is False


def test_vocal_channel_profile_recovery_allows_gain_up_after_stricter_stable_lane():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 2.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.75,
                    "recovery_windows": 3,
                },
                "decision_confidence": {
                    "candidate_direction": "up",
                    "score": 0.78,
                    "confidence_windows": 1,
                    "analysis_timestamp_sec": 104.0,
                    "analysis_window_sec": 5.0,
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert profile["recovery_candidate"] is True
    assert profile["recovery_windows"] == 4
    assert profile["recovery_required_windows"] == 4
    assert profile["recovery_policy_family"] == "lead_vocal"
    assert profile["profile_blocks_gain_up_by_history_raw"] is True
    assert profile["profile_recovered_from_history"] is True
    assert profile["profile_blocks_gain_up"] is False
    assert report["input_gain_character"]["decision_confidence"]["hysteresis_stable"] is True


def test_playback_channel_profile_recovers_faster_than_vocal_when_stable():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="playback",
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 2.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.75,
                    "recovery_windows": 1,
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert profile["recovery_policy_family"] == "playback"
    assert profile["recovery_windows"] == 2
    assert profile["recovery_required_windows"] == 2
    assert profile["profile_recovered_from_history"] is True


def test_soundcheck_channel_profile_allows_gain_up_after_stable_history():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 3,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
            },
        },
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert report["operator_action"] == "apply_trim_proposal"
    assert report["input_gain_character"]["channel_profile"]["profile_ready"] is True
    assert report["input_gain_character"]["channel_profile"]["profile_blocks_gain_up"] is False
    assert report["input_gain_character"]["channel_profile"]["windows"] == 4
    assert report["input_gain_character"]["channel_profile"]["learning_mode"] == "soundcheck_learning"
    assert report["input_gain_character"]["channel_profile"]["gain_up_profile_frozen"] is False
    assert report["input_gain_character"]["decision_confidence"]["threshold_met"] is True
    assert report["input_gain_character"]["decision_confidence"]["score"] >= 0.72


def test_live_mode_freezes_gain_up_profile_and_updates_risk_only():
    stats = _stats_with_signal(peak_db=-2.0, lufs=-18.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"live_max_cut_db": 1.0}),
        mode="live",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "input_gain_character": {
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.1,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                    "live_risk_windows": 2,
                    "live_overload_events": 0,
                    "live_peak_density_max": 0.2,
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    assert report["suggested_gain_db"] == pytest.approx(-1.0)
    assert report["auto_apply_allowed"] is True
    assert profile["learning_mode"] == "live_risk_only"
    assert profile["gain_up_profile_frozen"] is True
    assert profile["windows"] == 4
    assert profile["unstable_behavior_ratio"] == pytest.approx(0.0)
    assert profile["noise_margin_min_db"] == pytest.approx(24.0)
    assert profile["live_risk_windows"] == 3
    assert profile["live_current_risk"] is True


def test_channel_profile_reset_ignores_stale_unstable_history():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 5,
                    "crest_avg_db": 18.0,
                    "lufs_std_avg_db": 8.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 5.0,
                    "unstable_behavior_ratio": 0.8,
                },
                "profile_lifecycle": {
                    "action": "reset",
                    "reason": "new soundcheck take",
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    lifecycle = report["input_gain_character"]["profile_lifecycle"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert profile["windows"] == 1
    assert profile["profile_blocks_gain_up"] is False
    assert lifecycle["applied_action"] == "reset"
    assert lifecycle["reason"] == "new soundcheck take"
    assert lifecycle["previous_windows"] == 5
    assert lifecycle["retained_windows"] == 0


def test_channel_profile_lock_blocks_soundcheck_gain_up_for_engineer_review():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "profile_lifecycle": {
                    "action": "lock",
                    "reason": "engineer wants manual vocal gain",
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    lifecycle = report["input_gain_character"]["profile_lifecycle"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "channel_profile_locked_review"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["stability"]["reason"] == "waiting_for_profile_unlock_or_engineer_review"
    assert profile["learning_mode"] == "profile_locked"
    assert profile["gain_up_profile_frozen"] is True
    assert profile["profile_locked"] is True
    assert profile["profile_blocks_gain_up"] is True
    assert lifecycle["applied_action"] == "lock"
    assert lifecycle["profile_locked"] is True


def test_live_profile_lock_still_allows_headroom_trim_down():
    stats = _stats_with_signal(peak_db=-0.5, lufs=-18.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"live_max_cut_db": 1.0}),
        mode="live",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "input_gain_character": {
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "profile_lifecycle": {
                    "action": "lock",
                    "reason": "show lock",
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    lifecycle = report["input_gain_character"]["profile_lifecycle"]
    assert report["suggested_gain_db"] == pytest.approx(-1.0)
    assert report["auto_apply_allowed"] is True
    assert profile["learning_mode"] == "profile_locked"
    assert profile["profile_locked"] is True
    assert profile["live_current_risk"] is True
    assert lifecycle["applied_action"] == "lock"


def test_disabled_channel_profile_lifecycle_ignores_previous_lock():
    stats = _stats_with_signal(peak_db=-20.0, lufs=-30.0)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile.from_config({"channel_profile_lifecycle_enabled": False}),
        mode="soundcheck",
        mixer_channel=1,
        preset="leadVocal",
        previous_recommendation={
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 1.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.0,
                },
                "profile_lifecycle": {
                    "profile_locked": True,
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    lifecycle = report["input_gain_character"]["profile_lifecycle"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert profile["learning_mode"] == "soundcheck_learning"
    assert profile["profile_locked"] is False
    assert lifecycle["enabled"] is False
    assert lifecycle["profile_locked"] is False


def test_channel_profile_decay_reduces_weight_of_old_instability():
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
            "suggested_gain_db": 3.0,
            "input_gain_character": {
                "source_behavior": "stable_dynamic",
                "stability": {
                    "candidate_direction": "up",
                    "candidate_delta_db": 3.0,
                    "evidence_windows": 1,
                },
                "channel_profile": {
                    "windows": 4,
                    "crest_avg_db": 10.0,
                    "lufs_std_avg_db": 2.0,
                    "near_clip_density_avg": 0.0,
                    "noise_margin_min_db": 24.0,
                    "unstable_behavior_ratio": 0.6,
                },
                "profile_lifecycle": {
                    "action": "decay",
                    "decay_factor": 0.25,
                    "reason": "new singer warmed up",
                },
            },
        },
    )

    report = stats.get_report()
    profile = report["input_gain_character"]["channel_profile"]
    lifecycle = report["input_gain_character"]["profile_lifecycle"]
    assert report["suggested_gain_db"] == pytest.approx(3.0)
    assert report["limited_by"] == "quiet_source"
    assert profile["windows"] == 2
    assert profile["unstable_behavior_ratio"] == pytest.approx(0.3)
    assert profile["profile_blocks_gain_up"] is False
    assert lifecycle["applied_action"] == "decay"
    assert lifecycle["previous_windows"] == 4
    assert lifecycle["retained_windows"] == 1
    assert lifecycle["decay_factor"] == pytest.approx(0.25)


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


def test_transient_percussion_high_band_drift_does_not_create_auto_gain_up_identity():
    previous_bands = {
        "low": -34.0,
        "low_mid": -38.0,
        "presence": -45.0,
        "air": -50.0,
    }
    current_bands = {
        "low": -34.0,
        "low_mid": -38.0,
        "presence": -34.0,
        "air": -36.0,
    }
    stats = _stats_with_signal(peak_db=-20.0, lufs=-32.0, spectral_bands_db=current_bands)

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="kick",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 0.0,
            "input_gain_character": {
                "source_behavior": "operator_review_source",
                "trend_memory": {
                    "behavior_history": ["operator_review_source"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "transient_percussion",
                        "behavior": "operator_review_source",
                        "crest_db": 12.0,
                        "crest_bucket_db": 12.0,
                        "integrated_lufs_db": -32.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 12.0,
                        "peak_lufs_gap_bucket_db": 12.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(previous_bands),
                    },
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    distances = confidence["source_signature_spectral_distances"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_personality_boost_review"
    assert report["operator_action"] == "operator_review"
    assert report["input_gain_character"]["source_behavior"] == "operator_review_source"
    assert confidence["source_signature_spectral_common_band_count"] == 4
    assert distances["presence"]["weight"] == pytest.approx(0.25)
    assert distances["air"]["weight"] == pytest.approx(0.25)
    assert confidence["source_signature_spectral_band_weights"]["presence"] == pytest.approx(0.25)
    assert report["input_gain_character"]["boost_guard"]["personality"]["boost_policy"] == "operator_review"


def test_transient_window_profile_downweights_attack_brightness_identity():
    body_bands = {
        "low": -34.0,
        "low_mid": -38.0,
        "presence": -50.0,
        "air": -55.0,
    }
    attack_bands = {
        "low": -33.0,
        "low_mid": -37.0,
        "presence": -31.0,
        "air": -32.0,
    }
    spectral_sequence = [body_bands] * 16 + [attack_bands] * 4
    peak_sequence = [-38.0] * 16 + [-20.0] * 4
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="kick",
        analysis_timestamp_sec=110.0,
        analysis_window_sec=5.0,
        previous_recommendation={
            "suggested_gain_db": 0.0,
            "input_gain_character": {
                "source_behavior": "operator_review_source",
                "trend_memory": {
                    "behavior_history": ["operator_review_source"],
                    "predictive_clip_windows": 0,
                    "source_signature": {
                        "family": "transient_percussion",
                        "behavior": "operator_review_source",
                        "crest_db": 12.0,
                        "crest_bucket_db": 12.0,
                        "integrated_lufs_db": -32.0,
                        "integrated_lufs_bucket_db": -30.0,
                        "peak_lufs_gap_db": 12.0,
                        "peak_lufs_gap_bucket_db": 12.0,
                        "noise_margin_db": None,
                        "noise_margin_bucket_db": None,
                        "spectral_fingerprint": _spectral_signature(body_bands),
                    },
                },
            },
        },
    )

    report = stats.get_report()
    confidence = report["input_gain_character"]["decision_confidence"]
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    distances = confidence["source_signature_spectral_distances"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["active"] is True
    assert transient_profile["attack_dominant"] is True
    assert transient_profile["high_band_transient_dominant"] is True
    assert transient_profile["attack_body_gap_db"] >= 8.0
    assert transient_profile["high_band_lift_db"] >= 6.0
    assert distances["presence"]["weight"] == pytest.approx(0.10)
    assert distances["air"]["weight"] == pytest.approx(0.10)
    assert report["operator_action"] == "operator_review"


def test_transient_window_profile_marks_stable_repeated_percussion_hits():
    body_bands = {
        "low": -34.0,
        "low_mid": -38.0,
        "presence": -50.0,
        "air": -55.0,
    }
    attack_bands = {
        "low": -33.0,
        "low_mid": -37.0,
        "presence": -31.0,
        "air": -32.0,
    }
    spectral_sequence = []
    peak_sequence = []
    for _ in range(3):
        spectral_sequence.extend([body_bands] * 4)
        peak_sequence.extend([-38.0] * 4)
        spectral_sequence.append(attack_bands)
        peak_sequence.append(-20.0)
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

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
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    factors = report["input_gain_character"]["decision_confidence"]["factors"]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["repeated_hit_evaluable"] is True
    assert transient_profile["hit_count"] == 3
    assert transient_profile["repeated_hit_body_stable"] is True
    assert transient_profile["repeated_hit_gap_stable"] is True
    assert transient_profile["body_band_reference_available"] is True
    assert transient_profile["repeated_hit_body_band_stable"] is True
    assert transient_profile["repeated_hit_body_band_unstable"] is False
    assert transient_profile["body_band_profile"] == "kick_low_fundamental"
    assert transient_profile["hit_profiles"][0]["body_band_source"] == "primary"
    assert transient_profile["hit_profiles"][0]["body_band_primary_match_count"] == 4
    assert transient_profile["body_band_reference"]["available"] is True
    assert transient_profile["body_band_reference"]["confident"] is True
    assert transient_profile["body_band_reference"]["profile"] == "kick_low_fundamental"
    assert transient_profile["body_band_reference"]["body_band_median_db"] == pytest.approx(-34.0)
    assert transient_profile["repeated_hit_unstable"] is False
    assert transient_profile["body_reference"]["available"] is True
    assert transient_profile["body_reference"]["confident"] is True
    assert transient_profile["body_reference"]["body_band_reference"]["confident"] is True
    assert transient_profile["body_reference"]["body_peak_median_db"] == pytest.approx(-38.0)
    assert transient_profile["body_reference"]["attack_headroom_cap_db"] == pytest.approx(16.0)
    assert transient_profile["body_reference"]["operator_review_delta_db"] == pytest.approx(3.0)
    assert transient_profile["body_reference"]["policy"] == "operator_review_only"
    assert all(factor["name"] != "transient_hit_instability" for factor in factors)


def test_transient_window_profile_uses_numeric_analyzer_bands_for_kick_body_reference():
    body_bands = {
        "63 Hz": -34.0,
        "250 Hz": -38.0,
        "4 kHz": -50.0,
        "12 kHz": -55.0,
    }
    attack_bands = {
        "63 Hz": -33.0,
        "250 Hz": -37.0,
        "4 kHz": -31.0,
        "12 kHz": -32.0,
    }
    spectral_sequence = []
    peak_sequence = []
    for _ in range(3):
        spectral_sequence.extend([body_bands] * 4)
        peak_sequence.extend([-38.0] * 4)
        spectral_sequence.append(attack_bands)
        peak_sequence.append(-20.0)
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

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
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["high_band_available"] is True
    assert transient_profile["high_band_transient_dominant"] is True
    assert transient_profile["body_band_profile"] == "kick_low_fundamental"
    assert transient_profile["body_band_reference_available"] is True
    assert transient_profile["hit_profiles"][0]["body_band_source"] == "primary"
    assert transient_profile["hit_profiles"][0]["body_band_primary_match_count"] == 4
    assert transient_profile["hit_profiles"][0]["body_band_secondary_match_count"] == 4
    assert transient_profile["body_band_reference"]["body_band_median_db"] == pytest.approx(-34.0)
    assert transient_profile["body_reference"]["confident"] is True


def test_transient_window_profile_penalizes_unstable_repeated_percussion_hits():
    body_bands = {
        "low": -34.0,
        "low_mid": -38.0,
        "presence": -50.0,
        "air": -55.0,
    }
    attack_bands = {
        "low": -33.0,
        "low_mid": -37.0,
        "presence": -31.0,
        "air": -32.0,
    }
    spectral_sequence = []
    peak_sequence = []
    for body_peak in (-39.0, -34.0, -27.0):
        spectral_sequence.extend([body_bands] * 4)
        peak_sequence.extend([body_peak] * 4)
        spectral_sequence.append(attack_bands)
        peak_sequence.append(-20.0)
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

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
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    factors = report["input_gain_character"]["decision_confidence"]["factors"]
    instability_factor = [
        factor
        for factor in factors
        if factor["name"] == "transient_hit_instability"
    ]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["repeated_hit_evaluable"] is True
    assert transient_profile["hit_count"] == 3
    assert transient_profile["hit_body_std_db"] > transient_profile["hit_body_std_max_db"]
    assert transient_profile["hit_gap_std_db"] > transient_profile["hit_gap_std_max_db"]
    assert transient_profile["repeated_hit_unstable"] is True
    assert transient_profile["body_reference"]["available"] is True
    assert transient_profile["body_reference"]["confident"] is False
    assert transient_profile["body_reference"]["operator_review_delta_db"] == pytest.approx(3.0)
    assert instability_factor
    assert instability_factor[0]["delta"] == pytest.approx(-0.12)
    assert report["operator_action"] == "operator_review"


def test_transient_window_profile_penalizes_unstable_body_band_reference():
    body_band_sets = [
        {
            "low": -44.0,
            "low_mid": -43.0,
            "presence": -50.0,
            "air": -55.0,
        },
        {
            "low": -34.0,
            "low_mid": -38.0,
            "presence": -50.0,
            "air": -55.0,
        },
        {
            "low": -25.0,
            "low_mid": -30.0,
            "presence": -50.0,
            "air": -55.0,
        },
    ]
    attack_bands = {
        "low": -33.0,
        "low_mid": -37.0,
        "presence": -31.0,
        "air": -32.0,
    }
    spectral_sequence = []
    peak_sequence = []
    for body_bands in body_band_sets:
        spectral_sequence.extend([body_bands] * 4)
        peak_sequence.extend([-38.0] * 4)
        spectral_sequence.append(attack_bands)
        peak_sequence.append(-20.0)
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

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
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    factors = report["input_gain_character"]["decision_confidence"]["factors"]
    body_band_factor = [
        factor
        for factor in factors
        if factor["name"] == "transient_body_band_instability"
    ]
    assert report["suggested_gain_db"] == pytest.approx(0.0)
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["repeated_hit_body_stable"] is True
    assert transient_profile["repeated_hit_gap_stable"] is True
    assert transient_profile["body_band_reference_available"] is True
    assert transient_profile["hit_body_band_std_db"] > transient_profile["hit_body_band_std_max_db"]
    assert transient_profile["repeated_hit_body_band_unstable"] is True
    assert transient_profile["repeated_hit_unstable"] is True
    assert transient_profile["body_band_reference"]["available"] is True
    assert transient_profile["body_band_reference"]["confident"] is False
    assert transient_profile["body_reference"]["available"] is True
    assert transient_profile["body_reference"]["confident"] is False
    assert transient_profile["body_reference"]["body_band_reference"]["confident"] is False
    assert body_band_factor
    assert body_band_factor[0]["delta"] == pytest.approx(-0.10)
    assert report["operator_action"] == "operator_review"


def test_snare_body_band_profile_prefers_low_mid_over_bleeding_low_band():
    body_band_sets = [
        {
            "low": -50.0,
            "low_mid": -35.0,
            "body": -36.0,
            "presence": -47.0,
            "air": -52.0,
        },
        {
            "low": -34.0,
            "low_mid": -35.0,
            "body": -36.0,
            "presence": -47.0,
            "air": -52.0,
        },
        {
            "low": -20.0,
            "low_mid": -35.0,
            "body": -36.0,
            "presence": -47.0,
            "air": -52.0,
        },
    ]
    attack_bands = {
        "low": -33.0,
        "low_mid": -34.0,
        "body": -35.0,
        "presence": -28.0,
        "air": -31.0,
    }
    spectral_sequence = []
    peak_sequence = []
    for body_bands in body_band_sets:
        spectral_sequence.extend([body_bands] * 4)
        peak_sequence.extend([-38.0] * 4)
        spectral_sequence.append(attack_bands)
        peak_sequence.append(-20.0)
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

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
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    factor_names = {
        factor["name"]
        for factor in report["input_gain_character"]["decision_confidence"]["factors"]
    }
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["body_band_profile"] == "snare_low_mid_body"
    assert transient_profile["body_band_primary_tokens"] == ["low_mid", "lowmid", "body", "snare_body"]
    assert transient_profile["hit_profiles"][0]["body_band_source"] == "primary"
    assert transient_profile["hit_profiles"][0]["body_band_primary_match_count"] == 8
    assert transient_profile["body_band_reference"]["body_band_median_db"] == pytest.approx(-35.5)
    assert transient_profile["hit_body_band_std_db"] <= transient_profile["hit_body_band_std_max_db"]
    assert transient_profile["repeated_hit_body_band_stable"] is True
    assert transient_profile["repeated_hit_body_band_unstable"] is False
    assert transient_profile["body_reference"]["confident"] is True
    assert "transient_body_band_instability" not in factor_names


def test_overhead_profile_disables_per_hit_body_reference():
    body_bands = {
        "low": -30.0,
        "low_mid": -34.0,
        "body": -36.0,
        "presence": -32.0,
        "air": -33.0,
    }
    attack_bands = {
        "low": -30.0,
        "low_mid": -34.0,
        "body": -36.0,
        "presence": -22.0,
        "air": -23.0,
    }
    spectral_sequence = []
    peak_sequence = []
    for _ in range(3):
        spectral_sequence.extend([body_bands] * 4)
        peak_sequence.extend([-38.0] * 4)
        spectral_sequence.append(attack_bands)
        peak_sequence.append(-20.0)
    stats = _stats_with_spectral_sequence(
        lufs=-32.0,
        spectral_sequence=spectral_sequence,
        peak_sequence=peak_sequence,
    )

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=LocalInputGainProfile(),
        mode="soundcheck",
        mixer_channel=1,
        preset="overheads",
    )

    report = stats.get_report()
    transient_profile = report["input_gain_character"]["transient_window_profile"]
    assert report["limited_by"] == "source_personality_boost_review"
    assert transient_profile["body_band_profile"] == "drum_image_no_body_reference"
    assert transient_profile["body_band_reference_applicable"] is False
    assert transient_profile["body_band_reference_available"] is False
    assert transient_profile["body_band_reference"]["applicable"] is False
    assert transient_profile["body_band_reference"]["available"] is False
    assert transient_profile["body_reference"]["applicable"] is False
    assert transient_profile["body_reference"]["available"] is False
    assert transient_profile["body_reference"]["confident"] is False
    assert transient_profile["body_reference"]["policy"] == "image_or_cymbal_review_only"


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


def test_predictive_clip_risk_trims_before_actual_clip():
    stats = _stats_with_peak_sequence([-7.0, -6.0, -5.0, -4.6, -4.0, -3.4])

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
    foh_summary = report["input_gain_character"]["decision_confidence"]["foh_summary"]
    assert report["suggested_gain_db"] == pytest.approx(-1.5)
    assert report["limited_by"] == "predictive_clip_risk"
    assert report["input_gain_character"]["overload"]["classification"] == "predictive_clip_risk"
    assert report["input_gain_character"]["predictive_clip"]["active"] is True
    assert report["input_gain_character"]["predictive_clip"]["near_clip_peak_density"] > 0.55
    assert report["input_gain_character"]["predictive_clip"]["peak_trend_db"] > 1.0
    assert foh_summary["final_direction"] == "down"
    assert foh_summary["headline"] == "Trim down 1.5 dB: predictive_clip_risk"
    assert foh_summary["evidence_window"]["samples_analyzed"] == 6
    assert foh_summary["evidence_window"]["active_samples"] == 6


def test_live_predictive_clip_cut_is_capped_to_small_step():
    stats = _stats_with_peak_sequence([-7.0, -6.0, -5.0, -4.6, -4.0, -3.4])
    profile = LocalInputGainProfile.from_config({"live_max_cut_db": 1.0})

    stats.calculate_input_trim_recommendation(
        current_trim_db=0.0,
        target_peak_db=-10.0,
        max_peak_limit=-3.0,
        target_lufs=-18.0,
        preamp_profile=profile,
        mode="live",
        mixer_channel=1,
        preset="leadVocal",
    )

    report = stats.get_report()
    assert report["suggested_gain_db"] == pytest.approx(-1.0)
    assert report["limited_by"] == "predictive_clip_risk"
    assert report["auto_apply_allowed"] is True
    assert report["input_gain_character"]["predictive_clip"]["active"] is True


def test_isolated_near_limit_spike_holds_instead_of_predictive_cut():
    stats = _stats_with_peak_sequence([-10.0] * 19 + [-3.4], lufs=-24.0)

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
    assert report["limited_by"] == "isolated_spike_hold"
    assert report["operator_action"] == "hold"
    assert report["input_gain_character"]["overload"]["classification"] == "isolated_spike"
    assert report["input_gain_character"]["predictive_clip"]["active"] is False


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


def test_safe_gain_proposal_carries_ensemble_time_context_evidence():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -20.0,
        "target_peak_db": -10.0,
        "suggested_gain_db": 3.0,
        "target_trim_db": 3.0,
        "confidence": 0.74,
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
            "soundcheck_ensemble_context": {
                "provided": True,
                "active": False,
                "reason": "ensemble_time_context_stale",
                "time_coherence_enabled": True,
                "time_coherence_evaluable": True,
                "time_coherent": False,
                "time_coherent_loud_sources": 1,
                "required_time_coherent_sources": 2,
                "reference_max_lag_samples": 10,
                "max_context_lag_samples": 2,
                "operator_review_only": True,
            },
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

    evidence = {
        item["label"]: item
        for item in proposals[0]["requested_change"]["safety_evidence"]
    }
    assert evidence["Ensemble context"]["value"] == "ensemble_time_context_stale"
    assert evidence["Ensemble time coherence"]["value"] == "stale"
    assert evidence["Ensemble time coherence"]["reason"] == "ensemble_time_context_stale"
    assert evidence["Ensemble time sources"]["value"] == 1
    assert evidence["Ensemble time sources"]["target"] == 2
    assert evidence["Ensemble lag samples"]["value"] == 10
    assert evidence["Ensemble lag samples"]["target"] == 2
    assert evidence["Ensemble apply policy"]["value"] == "operator_review_only"


def test_safe_gain_proposal_carries_active_ensemble_context_reason():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -21.0,
        "target_peak_db": -10.0,
        "suggested_gain_db": 6.0,
        "target_trim_db": 6.0,
        "confidence": 0.82,
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
            "soundcheck_ensemble_context": {
                "provided": True,
                "active": True,
                "reason": "ensemble_context_vocal_understaged",
                "time_coherence_enabled": True,
                "time_coherence_evaluable": True,
                "time_coherent": True,
                "time_coherent_loud_sources": 3,
                "required_time_coherent_sources": 2,
                "reference_max_lag_samples": 1,
                "max_context_lag_samples": 2,
                "operator_review_only": True,
            },
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

    evidence = {
        item["label"]: item
        for item in proposals[0]["requested_change"]["safety_evidence"]
    }
    assert evidence["Ensemble context"]["value"] == "active"
    assert evidence["Ensemble context"]["reason"] == "ensemble_context_vocal_understaged"
    assert evidence["Ensemble time coherence"]["value"] == "coherent"
    assert evidence["Ensemble time coherence"]["reason"] == "ensemble_context_vocal_understaged"
    assert evidence["Ensemble time sources"]["value"] == 3
    assert evidence["Ensemble time sources"]["target"] == 2
    assert evidence["Ensemble lag samples"]["value"] == 1
    assert evidence["Ensemble lag samples"]["target"] == 2
    assert evidence["Ensemble apply policy"]["value"] == "operator_review_only"


def test_safe_gain_proposal_reason_summarizes_ensemble_context_for_cockpit():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -20.0,
        "target_peak_db": -10.0,
        "suggested_gain_db": 3.0,
        "target_trim_db": 3.0,
        "confidence": 0.74,
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
            "soundcheck_ensemble_context": {
                "provided": True,
                "active": False,
                "reason": "ensemble_time_context_stale",
                "time_coherence_enabled": True,
                "time_coherence_evaluable": True,
                "time_coherent": False,
                "time_coherent_loud_sources": 1,
                "required_time_coherent_sources": 2,
                "reference_max_lag_samples": 10,
                "max_context_lag_samples": 2,
                "operator_review_only": True,
            },
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

    reason = proposals[0]["reason"]
    assert "Ensemble context inactive (ensemble_time_context_stale)" in reason
    assert "time stale" in reason
    assert "sources 1/2" in reason
    assert "lag 10/2 samples" in reason
    assert "operator review only" in reason


def test_safe_gain_proposal_marks_missing_ensemble_time_context_as_missing():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -20.0,
        "target_peak_db": -10.0,
        "suggested_gain_db": 3.0,
        "target_trim_db": 3.0,
        "confidence": 0.7,
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
            "soundcheck_ensemble_context": {
                "provided": True,
                "active": False,
                "reason": "ensemble_time_context_missing",
                "time_coherence_enabled": True,
                "time_coherence_evaluable": False,
                "operator_review_only": True,
            },
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

    evidence = {
        item["label"]: item
        for item in proposals[0]["requested_change"]["safety_evidence"]
    }
    assert evidence["Ensemble context"]["value"] == "ensemble_time_context_missing"
    assert evidence["Ensemble time coherence"]["value"] == "missing"
    assert evidence["Ensemble time coherence"]["reason"] == "ensemble_time_context_missing"
    assert evidence["Ensemble apply policy"]["value"] == "operator_review_only"
    assert "Ensemble time sources" not in evidence
    assert "Ensemble lag samples" not in evidence


def test_safe_gain_proposal_summarizes_disabled_ensemble_context_without_time_noise():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -20.0,
        "target_peak_db": -10.0,
        "suggested_gain_db": 3.0,
        "target_trim_db": 3.0,
        "confidence": 0.7,
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
            "soundcheck_ensemble_context": {
                "enabled": False,
                "provided": True,
                "active": False,
                "reason": "disabled",
                "time_coherence_enabled": True,
                "time_coherence_evaluable": False,
                "time_coherent": False,
                "time_coherent_loud_sources": 0,
                "required_time_coherent_sources": 2,
                "max_context_lag_samples": 2,
                "operator_review_only": True,
            },
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

    evidence = {
        item["label"]: item
        for item in proposals[0]["requested_change"]["safety_evidence"]
    }
    reason = proposals[0]["reason"]

    assert evidence["Ensemble context"]["value"] == "disabled"
    assert evidence["Ensemble context"]["reason"] == "disabled"
    assert evidence["Ensemble apply policy"]["value"] == "operator_review_only"
    assert "Ensemble time coherence" not in evidence
    assert "Ensemble time sources" not in evidence
    assert "Ensemble lag samples" not in evidence
    assert "Ensemble context inactive (disabled)" in reason
    assert "operator review only" in reason
    assert "time missing" not in reason
    assert "sources 0/2" not in reason
