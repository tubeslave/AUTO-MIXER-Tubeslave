import json

import pytest

from audio_workbench.studio_iteration import (
    add_human_calibration_annotation,
    persist_human_calibration_annotation,
)


def _report(*, protected=None, failures=None, decision="rejected"):
    protected = list(protected or [])
    failures = list(failures or ["target_not_improved"])
    return {
        "schema": "studio-autonomous-iteration-v2",
        "status": "rejected",
        "target": "vocal_intelligibility",
        "baseline": {"id": "baseline-1"},
        "candidate": {"id": "candidate-2"},
        "critic": {
            "target": "vocal_intelligibility",
            "machine_decision": decision,
            "failures": failures,
            "protected_regressions": protected,
            "requires_human_listening": True,
        },
        "transition": {
            "status": "rejected",
            "next_action": "rollback_candidate",
            "baseline_before": "baseline-1",
            "baseline_after": "baseline-1",
            "promote_baseline": False,
            "rollback_candidate": True,
            "protected_regressions": protected,
        },
        "delivery": {
            "role": "baseline_rollback",
            "source_id": "baseline-1",
            "rolled_back_to_baseline": True,
        },
        "baseline_promoted": False,
        "baseline_after": "baseline-1",
        "requires_human_listening": True,
        "human_review": "pending",
    }


def _decision_state(report):
    return {
        "status": report["status"],
        "baseline_promoted": report["baseline_promoted"],
        "baseline_after": report["baseline_after"],
        "transition": report["transition"],
        "delivery": report["delivery"],
    }


def test_target_only_human_acceptance_is_annotation_not_machine_reversal():
    original = _report()
    before = _decision_state(original)
    enriched = add_human_calibration_annotation(
        original,
        {
            "human_review": "accepted",
            "observations": ["foreground stability", "phrase consistency"],
        },
    )
    assert enriched["human_calibration"]["classification"] == "target_proxy_miss_candidate"
    assert enriched["human_calibration"]["role"] == "annotation_only"
    assert enriched["human_calibration"]["metric_hypotheses"] == ["foreground_stability"]
    assert not enriched["human_calibration"]["baseline_promotion_allowed"]
    assert not enriched["human_calibration"]["may_change_transition"]
    assert _decision_state(enriched) == before
    assert _decision_state(original) == before


def test_protected_regression_cannot_be_waived_by_human_preference():
    original = _report(
        protected=["width_regression"],
        failures=["target_not_improved", "width_regression"],
    )
    enriched = add_human_calibration_annotation(
        original, {"human_review": "accepted", "observations": ["stable mix position"]}
    )
    annotation = enriched["human_calibration"]
    assert annotation["classification"] == "human_preference_safety_conflict"
    assert annotation["protected_regressions_preserved"] == ["width_regression"]
    assert not annotation["protected_gate_override_allowed"]
    assert enriched["transition"]["rollback_candidate"] is True
    assert enriched["baseline_after"] == "baseline-1"


def test_missing_human_review_is_exact_semantic_noop():
    original = _report()
    enriched = add_human_calibration_annotation(original, None)
    assert enriched == original
    assert enriched is not original
    assert "human_calibration" not in enriched


def test_invalid_human_review_fails_closed_without_mutating_input():
    original = _report()
    snapshot = json.loads(json.dumps(original))
    with pytest.raises(ValueError, match="accepted or rejected"):
        add_human_calibration_annotation(original, {"human_review": "maybe"})
    assert original == snapshot
    assert "human_calibration" not in original


def test_annotation_is_deterministic_for_same_evidence():
    original = _report()
    review = {"human_review": "accepted", "observations": ["foreground stability"]}
    one = add_human_calibration_annotation(original, review)
    two = add_human_calibration_annotation(original, review)
    assert json.dumps(one, sort_keys=True, allow_nan=False) == json.dumps(two, sort_keys=True, allow_nan=False)


def test_persist_adds_annotation_without_rewriting_decision(tmp_path):
    path = tmp_path / "iteration_report.json"
    original = _report()
    path.write_text(json.dumps(original, indent=2), encoding="utf-8")
    enriched = persist_human_calibration_annotation(
        path, {"human_review": "accepted", "observations": ["stable mix position"]}
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == enriched
    assert _decision_state(on_disk) == _decision_state(original)
    assert on_disk["human_review"] == "pending"
    assert on_disk["human_calibration"]["human_review"] == "accepted"


def test_existing_annotation_is_not_silently_overwritten():
    original = _report()
    once = add_human_calibration_annotation(original, {"human_review": "accepted"})
    with pytest.raises(ValueError, match="already contains"):
        add_human_calibration_annotation(once, {"human_review": "rejected"})
