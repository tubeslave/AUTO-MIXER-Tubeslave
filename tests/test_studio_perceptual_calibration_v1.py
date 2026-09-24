import numpy as np
import pytest

from audio_workbench.mixing.perceptual_calibration import (
    ForegroundStabilityPolicy,
    classify_human_machine_disagreement,
    foreground_stability_evidence,
)


def _critic(*, machine="rejected", failures=None, protected=None):
    return {
        "machine_decision": machine,
        "target": "vocal_intelligibility",
        "failures": list(failures or []),
        "protected_regressions": list(protected or []),
    }


def _phrased(sr=8000, seconds=8.0, stable=False):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    carrier = np.sin(2 * np.pi * 190 * t) + .35 * np.sin(2 * np.pi * 380 * t)
    phrase = ((np.arange(n) // int(.5 * sr)) % 2).astype(float)
    if stable:
        amp = np.where(phrase > .5, .72, .62)
    else:
        amp = np.where(phrase > .5, 1.0, .38)
    amp *= 1.0 + .025 * np.sin(2 * np.pi * .7 * t)
    return (carrier * amp * .08).astype("float32")


def test_foreground_stability_detects_phrase_level_consistency_without_loudness_cheat():
    sr = 8000
    baseline = _phrased(sr=sr, stable=False)
    candidate = _phrased(sr=sr, stable=True) * 1.8
    r = foreground_stability_evidence(baseline, candidate, sr)
    assert r["active_windows"] >= 6
    assert r["spread_improvement_db"] > 4.0
    assert r["p95_absolute_deviation_improvement_db"] > 1.0
    assert abs(r["median_level_match_error_db"]) < 1e-9
    assert r["candidate_constant_match_gain_db"] < -4.0
    assert r["accept"] is None
    assert r["baseline_promotion_allowed"] is False
    assert r["requires_human_listening"] is True


def test_foreground_stability_reuses_explicit_fixed_activity_reference():
    sr = 8000
    baseline = _phrased(sr=sr, stable=False)
    candidate = _phrased(sr=sr, stable=True)
    reference = baseline.copy()
    r = foreground_stability_evidence(baseline, candidate, sr, activity_reference=reference)
    assert r["activity_reference"] == "explicit_fixed_reference"
    assert r["total_windows"] > r["active_windows"] or r["active_windows"] > 0


def test_foreground_stability_fails_closed_on_shape_or_insufficient_windows():
    with pytest.raises(ValueError, match="shapes differ"):
        foreground_stability_evidence(np.ones(4000), np.ones(3999), 8000)
    with pytest.raises(ValueError, match="not enough analysis windows"):
        foreground_stability_evidence(np.ones(100), np.ones(100), 8000)


def test_belye_stai_pattern_is_recorded_as_proxy_miss_not_retroactive_pass():
    critic = _critic(machine="rejected", failures=["target_not_improved"], protected=[])
    review = {
        "human_review": "accepted",
        "observations": ["foreground_stability", "phrase_consistency", "stable_mix_position"],
    }
    r = classify_human_machine_disagreement(critic, review)
    assert r["classification"] == "target_proxy_miss_candidate"
    assert r["metric_hypotheses"] == ["foreground_stability"]
    assert r["machine_failures_preserved"] == ["target_not_improved"]
    assert r["threshold_update_allowed"] is False
    assert r["protected_gate_override_allowed"] is False
    assert r["baseline_promotion_allowed"] is False


def test_human_preference_never_overrides_protected_regression():
    critic = _critic(machine="rejected", failures=["width_regression"], protected=["width_regression"])
    r = classify_human_machine_disagreement(critic, {"human_review": "accepted"})
    assert r["classification"] == "human_preference_safety_conflict"
    assert r["protected_regressions_preserved"] == ["width_regression"]
    assert r["protected_gate_override_allowed"] is False


def test_machine_safe_human_rejection_is_calibration_evidence_not_acceptance():
    critic = _critic(machine="machine_safe", failures=[], protected=[])
    r = classify_human_machine_disagreement(critic, {"human_review": "rejected", "tags": ["too_flat"]})
    assert r["classification"] == "machine_false_positive_candidate"
    assert r["calibration_action"] == "calibrate_missing_protected_or_target_metric"
    assert r["baseline_promotion_allowed"] is False


def test_calibration_validates_inputs():
    with pytest.raises(ValueError, match="machine_decision"):
        classify_human_machine_disagreement({"target": "punch"}, {"human_review": "accepted"})
    with pytest.raises(ValueError, match="accepted or rejected"):
        classify_human_machine_disagreement(_critic(), {"human_review": "maybe"})
    with pytest.raises(ValueError, match="window_ms"):
        foreground_stability_evidence(np.ones(8000), np.ones(8000), 8000,
                                      policy=ForegroundStabilityPolicy(window_ms=10))
