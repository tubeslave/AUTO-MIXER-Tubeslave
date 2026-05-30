import math
from pathlib import Path

import numpy as np

from autoeq_audit import (
    AutoEQAuditConfig,
    BiquadCoefficients,
    ParametricEQBand,
    PhaseContract,
    audit_biquads,
    audit_frequency_response,
    audit_peq_bands,
    biquad_frequency_response_db,
    check_biquad_stability,
    db_to_lin,
    enforce_peq_limits,
    lin_to_db,
    normalize_to_true_peak_headroom,
    true_peak_dbfs,
    validate_dataset_splits,
    validate_export_reproducibility,
    validate_phase_contract,
)


def test_units_db_linear_round_trip_reference_values():
    for gain_db in [0, 6, -6, 12, -12, 20, -20]:
        round_trip = float(lin_to_db(db_to_lin(gain_db)))
        assert round_trip == pytest_approx(gain_db, abs=1e-9)


def test_phase_contract_requires_matching_phase_assumptions():
    result = validate_phase_contract(
        PhaseContract(
            target_phase="min",
            filter_phase="linear",
            target_source="measurement_magnitude_only",
            filter_design="fir_export",
        )
    )

    assert result.passed is False
    assert any(issue.code == "phase_contract_mismatch" for issue in result.issues)


def test_phase_contract_accepts_explicit_minimum_phase_reconstruction():
    result = validate_phase_contract(
        PhaseContract(
            target_phase="minimum",
            filter_phase="minimum",
            target_source="held_out_measurement",
            filter_design="iir_fit",
            minimum_phase_reconstruction=True,
        )
    )

    assert result.passed is True
    assert result.issues == ()


def test_iir_stability_detects_unstable_biquad_poles():
    stable = BiquadCoefficients(1.0, 0.0, 0.0, 1.0, -0.5, 0.2, label="stable")
    unstable = BiquadCoefficients(1.0, 0.0, 0.0, 1.0, -2.4, 1.44, label="unstable")

    assert check_biquad_stability(stable) is True
    assert check_biquad_stability(unstable) is False

    result = audit_biquads([stable, unstable])
    assert result.passed is False
    assert any(issue.code == "unstable_biquad" for issue in result.issues)


def test_peq_limits_clip_gain_and_q_and_report_errors_before_clipping():
    config = AutoEQAuditConfig(max_abs_gain_db=12.0, max_q=10.0)
    bands = (
        ParametricEQBand(frequency_hz=120.0, gain_db=18.0, q=14.0, label="bad_boost"),
        ParametricEQBand(frequency_hz=2500.0, gain_db=-15.0, q=0.7, label="bad_cut"),
    )

    result = audit_peq_bands(bands, config=config)
    assert result.passed is False
    assert {issue.code for issue in result.issues} >= {"eq_gain_limit_exceeded", "eq_q_limit_exceeded"}

    limited = enforce_peq_limits(bands, config=config)
    assert limited[0].gain_db == 12.0
    assert limited[0].q == 10.0
    assert limited[1].gain_db == -12.0


def test_positive_gain_budget_is_warning_not_live_write_permission():
    result = audit_peq_bands(
        (
            ParametricEQBand(frequency_hz=80.0, gain_db=5.0, q=1.0),
            ParametricEQBand(frequency_hz=500.0, gain_db=5.0, q=1.0),
        ),
        config=AutoEQAuditConfig(max_total_positive_gain_db=6.0),
    )

    assert result.passed is True
    assert any(issue.code == "positive_gain_budget_exceeded" for issue in result.issues)


def test_frequency_response_flags_narrow_high_gain_needles():
    response = np.array([0.0, 0.5, 15.0, 0.2, 0.0])
    result = audit_frequency_response(response)

    assert result.passed is False
    assert result.issues[0].code == "narrow_response_peaks"


def test_dataset_splits_detect_train_test_leakage_by_normalized_id():
    result = validate_dataset_splits(
        train_ids=["measurements/Seat_A_Take_01.wav", "serial-123.json"],
        test_ids=["heldout/seat_a_take_01.csv", "serial-999.json"],
    )

    assert result.passed is False
    assert result.issues[0].code == "dataset_split_leakage"
    assert "seat_a_take_01" in result.issues[0].details["overlap"]


def test_true_peak_headroom_normalizes_reference_signal():
    sample_rate = 48000
    signal = np.array([0.0, 1.0, -1.0, 0.0])

    before = true_peak_dbfs(signal, sample_rate)
    normalized, report = normalize_to_true_peak_headroom(signal, sample_rate, margin_db=3.0)

    assert before == pytest_approx(0.0, abs=1e-9)
    assert report.applied_gain_db == pytest_approx(-3.0, abs=1e-6)
    assert report.true_peak_after_dbfs <= -3.0 + 1e-6
    assert np.max(np.abs(normalized)) < 1.0


def test_biquad_frequency_response_export_reproducibility_gate():
    sample_rate = 48000
    freqs = np.geomspace(20.0, 20000.0, 64)
    identity = [BiquadCoefficients(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)]
    reference = biquad_frequency_response_db(identity, freqs, sample_rate)

    assert np.max(np.abs(reference)) == pytest_approx(0.0, abs=1e-9)
    assert validate_export_reproducibility(reference, reference + 0.1).passed is True

    failed = validate_export_reproducibility(reference, reference + 0.25)
    assert failed.passed is False
    assert failed.issues[0].code == "export_response_delta_exceeded"


def test_autoeq_audit_module_is_transport_free():
    source = Path("backend/autoeq_audit.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_eq",
        "set_fader",
        "set_gain",
        ".send(",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source


def test_autoeq_audit_registry_entry_stays_shadow_only():
    from backend.processing_registry import get_processing_method

    method = get_processing_method("autoeq_audit_validation")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert "lab_dsp" in method.pipeline_profiles
    assert "shadow_quality" in method.pipeline_profiles


def pytest_approx(value, **kwargs):
    # Keeps this test file free of a hard pytest import while still showing intent.
    import pytest

    return pytest.approx(value, **kwargs)
