import pytest

from audio_workbench.mastering.export_validation import (
    MasterExportValidationPolicy,
    validate_master_exports,
)


def _metrics(wav_tp=-1.62, mp3_tp=-1.58, lufs=-15.13):
    return {
        "wav": {
            "true_peak_dbtp": wav_tp,
            "integrated_lufs": lufs,
            "integrated_lufs_method": "pyloudnorm",
        },
        "mp3_decoded": {
            "true_peak_dbtp": mp3_tp,
            "integrated_lufs": lufs,
            "integrated_lufs_method": "pyloudnorm",
        },
    }


def _validate(metrics=None, *, ceiling=-1.2, lufs=-15.13, frames=9_128_700):
    return validate_master_exports(
        metrics or _metrics(),
        target_lufs=lufs,
        requested_ceiling_dbtp=ceiling,
        expected_frames=frames,
        export_frames={"wav": frames, "mp3_decoded": frames},
        expected_sample_rate=44_100,
        export_sample_rates={"wav": 44_100, "mp3_decoded": 44_100},
    )


def test_known_belye_stai_v3_export_metrics_pass_requested_minus_1_2_ceiling():
    result = _validate()
    assert result["status"] == "machine_safe_technical"
    assert result["accept_technical_delivery"] is True
    assert result["failures"] == []
    assert result["formats"]["wav"]["true_peak_limit_dbtp"] == pytest.approx(-1.15)
    assert result["formats"]["mp3_decoded"]["true_peak_limit_dbtp"] == pytest.approx(-0.85)
    assert result["baseline_eligible"] is False
    assert result["requires_human_listening"] is True
    assert result["human_acceptance_inferred"] is False


def test_requested_ceiling_not_hardcoded_to_minus_one_dbtp():
    result = _validate(ceiling=-3.0)
    assert result["status"] == "rejected_export_validation"
    assert "wav_requested_ceiling_exceeded" in result["failures"]
    assert "mp3_decoded_requested_ceiling_exceeded" in result["failures"]
    assert result["formats"]["wav"]["requested_ceiling_dbtp"] == -3.0
    assert result["formats"]["mp3_decoded"]["requested_ceiling_dbtp"] == -3.0


def test_lossy_codec_gets_bounded_tolerance_but_not_unlimited_overshoot():
    metrics = _metrics(wav_tp=-3.02, mp3_tp=-2.72)
    result = _validate(metrics, ceiling=-3.0)
    assert result["accept_technical_delivery"] is True

    too_hot = _validate(_metrics(wav_tp=-3.02, mp3_tp=-2.60), ceiling=-3.0)
    assert too_hot["accept_technical_delivery"] is False
    assert "mp3_decoded_requested_ceiling_exceeded" in too_hot["failures"]


def test_loudness_must_be_standards_measured_and_within_tolerance():
    metrics = _metrics(lufs=-16.0)
    result = _validate(metrics)
    assert "wav_loudness_target_missed" in result["failures"]
    assert "mp3_decoded_loudness_target_missed" in result["failures"]

    metrics = _metrics()
    metrics["mp3_decoded"]["integrated_lufs_method"] = "rms_proxy"
    result = _validate(metrics)
    assert "mp3_decoded_loudness_unmeasured" in result["failures"]


def test_missing_format_length_and_sample_rate_fail_closed():
    metrics = {"wav": _metrics()["wav"]}
    result = validate_master_exports(
        metrics,
        target_lufs=-15.13,
        requested_ceiling_dbtp=-1.2,
        expected_frames=1000,
        export_frames={"wav": 999},
        expected_sample_rate=44_100,
        export_sample_rates={"wav": 48_000},
    )
    assert "wav_export_length_mismatch" in result["failures"]
    assert "wav_sample_rate_mismatch" in result["failures"]
    assert "mp3_decoded_measurements_missing" in result["failures"]


def test_policy_and_target_inputs_validate():
    with pytest.raises(ValueError, match="target_lufs"):
        validate_master_exports(
            _metrics(), target_lufs=float("nan"), requested_ceiling_dbtp=-1.2,
            expected_frames=1000, export_frames={"wav": 1000, "mp3_decoded": 1000},
        )
    with pytest.raises(ValueError, match="requested_ceiling"):
        validate_master_exports(
            _metrics(), target_lufs=-15.0, requested_ceiling_dbtp=-0.5,
            expected_frames=1000, export_frames={"wav": 1000, "mp3_decoded": 1000},
        )
    with pytest.raises(ValueError, match="lossy_true_peak_tolerance"):
        validate_master_exports(
            _metrics(), target_lufs=-15.0, requested_ceiling_dbtp=-1.2,
            expected_frames=1000, export_frames={"wav": 1000, "mp3_decoded": 1000},
            policy=MasterExportValidationPolicy(lossy_true_peak_tolerance_db=1.5),
        )
