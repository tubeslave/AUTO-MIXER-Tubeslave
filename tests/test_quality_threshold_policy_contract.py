from pathlib import Path

from backend.quality_threshold_policy_contract import (
    REQUIRED_POLICY_CHECKS,
    REQUIRED_THRESHOLD_AXES,
    evaluate_quality_threshold_policy_contract,
)
from backend.offline_replay_evaluation_contract import REQUIRED_QUALITY_AXES


def _complete_threshold(axis: str, **overrides) -> dict:
    threshold = {
        "axis": axis,
        "metric_name": f"{axis}_score",
        "direction": "min",
        "candidate_value": 0.8,
        "evidence_artifact": f"artifacts/evaluation/{axis}.json",
        "rationale": f"Candidate threshold for {axis}",
        "rollback_policy": f"Revert {axis} threshold if regression appears",
    }
    threshold.update(overrides)
    return threshold


def _complete_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "candidate-quality-thresholds",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "policy_checks": {key: True for key in REQUIRED_POLICY_CHECKS},
        "threshold_policy": {"status": "candidate"},
        "thresholds": [_complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
    }
    manifest.update(overrides)
    return manifest


def test_incomplete_candidate_blocks_review_and_threshold_use():
    result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "incomplete-thresholds",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "policy_checks": {"baseline_dataset_bound": True},
            "threshold_policy": {"status": "PENDING_APPROVAL"},
            "thresholds": [_complete_threshold("objective_audio_metrics")],
        }
    )

    assert result["status"] == "blocked"
    assert result["ready_for_owner_review"] is False
    assert result["ready_for_threshold_policy_use"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False
    assert "missing_policy_check:reference_mix_bound" in result["blockers"]
    assert "missing_threshold_axis:reference_mix_comparison" in result["blockers"]
    assert "owner_approved_quality_threshold_policy_required" in result["blockers"]


def test_complete_candidate_is_review_ready_but_not_approved_for_threshold_use():
    result = evaluate_quality_threshold_policy_contract(_complete_manifest())

    assert result["status"] == "ready_for_owner_review"
    assert result["ready_for_owner_review"] is True
    assert result["ready_for_threshold_policy_use"] is False
    assert result["threshold_policy_approved"] is False
    assert result["review_blockers"] == []
    assert result["blockers"] == ["owner_approved_quality_threshold_policy_required"]
    assert result["summary"]["covered_threshold_axes"] == list(REQUIRED_THRESHOLD_AXES)
    assert result["hardware_evidence"] is False


def test_owner_approved_complete_policy_is_ready_for_threshold_use_only_offline():
    result = evaluate_quality_threshold_policy_contract(
        _complete_manifest(
            policy_id="approved-quality-thresholds",
            threshold_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:quality-threshold-policy",
                "approved_at": "2026-06-19T17:20:04Z",
            },
        )
    )

    assert result["status"] == "ready_for_threshold_policy_use"
    assert result["ready_for_owner_review"] is True
    assert result["ready_for_threshold_policy_use"] is True
    assert result["threshold_policy_approved"] is True
    assert result["blockers"] == []
    assert result["live_write"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_live_hardware_and_authority_claims_are_rejected():
    result = evaluate_quality_threshold_policy_contract(
        _complete_manifest(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
            threshold_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:quality-threshold-policy",
                "approved_at": "2026-06-19T17:20:04Z",
            },
        )
    )

    assert result["ready_for_owner_review"] is False
    assert result["ready_for_threshold_policy_use"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert "offline_only_required" in result["blockers"]
    assert "quality_threshold_policy_forbids_live_write" in result["blockers"]
    assert "quality_threshold_policy_forbids_wing_contact" in result["blockers"]
    assert "quality_threshold_policy_forbids_backend_contact" in result["blockers"]
    assert "threshold_policy_evidence_must_not_be_labeled_hardware" in result["blockers"]
    assert "threshold_policy_evidence_cannot_close_readiness" in result["blockers"]
    assert "threshold_policy_evidence_cannot_grant_live_authority" in result["blockers"]


def test_invalid_threshold_values_and_live_transport_imports_stay_blocked():
    result = evaluate_quality_threshold_policy_contract(
        _complete_manifest(
            thresholds=[
                _complete_threshold(
                    "objective_audio_metrics",
                    direction="target_range",
                    candidate_value=[0.9],
                    evidence_artifact="",
                )
            ],
        )
    )
    source = Path("backend/quality_threshold_policy_contract.py").read_text(
        encoding="utf-8"
    )

    assert "objective_audio_metrics:missing_threshold_field:candidate_value" in result[
        "blockers"
    ]
    assert "objective_audio_metrics:missing_threshold_field:evidence_artifact" in result[
        "blockers"
    ]
    assert "objective_audio_metrics:invalid_candidate_value" in result["blockers"]
    assert "missing_threshold_axis:reference_mix_comparison" in result["blockers"]
    assert REQUIRED_THRESHOLD_AXES == REQUIRED_QUALITY_AXES
    assert "wing_client" not in source
    assert "osc" not in source.lower()
    assert "requests" not in source
