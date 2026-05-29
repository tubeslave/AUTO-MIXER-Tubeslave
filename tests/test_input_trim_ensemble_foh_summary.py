import pytest

from lufs_gain_staging import LocalInputGainProfile, SignalStats


def _stats_with_signal(peak_db=-20.0, lufs=-30.0, samples=20):
    stats = SignalStats(channel_id=1)
    for _ in range(samples):
        stats.update_sample(
            true_peak_db=peak_db,
            lufs=lufs,
            sample_rms_db=lufs,
        )
    return stats


@pytest.mark.parametrize(
    ("ensemble_context", "expected_flag"),
    [
        (
            {
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
            "ensemble_time_context_stale",
        ),
        (
            {
                "other_active_sources": 4,
                "other_loud_sources": 3,
                "time_coherence_evaluable": False,
                "required_time_coherent_sources": 2,
                "max_context_lag_samples": 2,
                "reference_peak_db": -9.0,
                "reference_lufs": -18.0,
            },
            "ensemble_time_context_missing",
        ),
    ],
)
def test_foh_summary_flags_unusable_ensemble_time_context_for_gain_up(
    ensemble_context,
    expected_flag,
):
    stats = _stats_with_signal()

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
        ensemble_context=ensemble_context,
    )

    character = stats.get_report()["input_gain_character"]
    summary = character["foh_summary"]

    assert character["soundcheck_ensemble_context"]["reason"] == expected_flag
    assert summary["candidate_direction"] == "up"
    assert summary["safe_to_auto_apply"] is False
    assert expected_flag in summary["risk_flags"]
