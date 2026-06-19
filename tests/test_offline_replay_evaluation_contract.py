from pathlib import Path

from backend.offline_replay_evaluation_contract import (
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_QUALITY_AXES,
    REQUIRED_SAFETY_CHECKS,
    REQUIRED_SCENARIOS,
    evaluate_offline_replay_evaluation_contract,
)
from backend.quality_threshold_policy_contract import (
    REQUIRED_POLICY_CHECKS as THRESHOLD_REQUIRED_POLICY_CHECKS,
    REQUIRED_THRESHOLD_AXES,
)


def _scenario(name: str) -> dict:
    return {
        "scenario": name,
        "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
        "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
        "quality_axes": {
            key: {"status": "passed", "artifact": f"{name}/{key}.json"}
            for key in REQUIRED_QUALITY_AXES
        },
    }


def _threshold_axis(axis: str) -> dict:
    return {
        "axis": axis,
        "metric_name": f"{axis}_score",
        "direction": "min",
        "candidate_value": 0.8,
        "evidence_artifact": f"thresholds/{axis}.json",
        "rationale": f"Replay promotion threshold for {axis}",
        "rollback_policy": f"Revert {axis} threshold on regression",
    }


def _complete_threshold_policy_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "offline-replay-threshold-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
        "threshold_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:offline-replay-thresholds",
            "approved_at": "2026-06-19T17:32:45Z",
        },
        "thresholds": [_threshold_axis(axis) for axis in REQUIRED_THRESHOLD_AXES],
    }
    manifest.update(overrides)
    return manifest


def _complete_manifest(**overrides) -> dict:
    manifest = {
        "manifest_id": "offline-replay-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "quality_threshold_policy_manifest": _complete_threshold_policy_manifest(),
        "scenarios": [_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    manifest.update(overrides)
    return manifest


def test_offline_replay_contract_blocks_incomplete_or_pending_threshold_evidence():
    manifest = _complete_manifest(
        quality_threshold_policy_manifest=_complete_threshold_policy_manifest(
            threshold_policy={"status": "PENDING_APPROVAL"}
        ),
        scenarios=[_scenario("soundcheck")],
    )

    result = evaluate_offline_replay_evaluation_contract(manifest)

    assert result["status"] == "blocked"
    assert result["ready_for_replay_promotion"] is False
    assert result["authority"]["readiness_closure_allowed"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert "approved_quality_threshold_policy_required" in result["blockers"]
    assert "quality_threshold_policy_contract_not_ready" in result["blockers"]
    assert (
        "quality_threshold_policy:owner_approved_quality_threshold_policy_required"
        in result["blockers"]
    )
    assert "missing_required_scenario:studio" in result["blockers"]
    assert "missing_required_scenario:concert" in result["blockers"]


def test_offline_replay_contract_rejects_legacy_threshold_policy_without_contract_manifest():
    result = evaluate_offline_replay_evaluation_contract(
        _complete_manifest(
            quality_threshold_policy_manifest={},
            threshold_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:legacy-threshold-policy",
            },
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_replay_promotion"] is False
    assert "approved_quality_threshold_policy_required" in result["blockers"]
    assert "quality_threshold_policy_contract_not_ready" in result["blockers"]
    assert "legacy_threshold_policy_without_contract_manifest" in result["blockers"]
    assert "quality_threshold_policy:offline_only_required" in result["blockers"]
    assert "quality_threshold_policy:missing_threshold_axis:objective_audio_metrics" in result[
        "blockers"
    ]
    assert result["summary"]["legacy_threshold_policy_status"] == "approved"
    assert result["summary"]["thresholds_approved"] is False


def test_offline_replay_contract_accepts_complete_bundle_without_live_authority():
    result = evaluate_offline_replay_evaluation_contract(_complete_manifest())

    assert result["status"] == "ready_for_replay_promotion"
    assert result["ready_for_replay_promotion"] is True
    assert result["offline_only"] is True
    assert result["hardware_evidence"] is False
    assert result["live_write"] is False
    assert result["authority"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
    }
    assert result["summary"]["covered_scenarios"] == list(REQUIRED_SCENARIOS)
    assert result["summary"]["thresholds_approved"] is True
    assert result["summary"]["threshold_policy_contract_status"] == (
        "ready_for_threshold_policy_use"
    )
    assert result["threshold_policy_result"]["ready_for_threshold_policy_use"] is True
    assert result["blockers"] == []


def test_offline_replay_contract_rejects_live_or_hardware_claims():
    result = evaluate_offline_replay_evaluation_contract(
        _complete_manifest(
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_replay_promotion"] is False
    assert set(result["blockers"]) >= {
        "offline_replay_contract_forbids_live_write",
        "offline_replay_contract_forbids_wing_contact",
        "offline_replay_contract_forbids_backend_contact",
        "offline_replay_evidence_must_not_be_labeled_hardware",
        "offline_replay_evidence_cannot_close_readiness",
        "offline_replay_evidence_cannot_grant_live_authority",
    }
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_offline_replay_contract_rejects_missing_scenario_quality_axes():
    bad_concert = _scenario("concert")
    bad_concert["quality_axes"]["blind_listening_tests"] = {"status": "pending"}
    bad_concert["safety_checks"]["rollback_plan_checked"] = False

    result = evaluate_offline_replay_evaluation_contract(
        _complete_manifest(
            scenarios=[_scenario("soundcheck"), _scenario("studio"), bad_concert],
        )
    )

    assert result["status"] == "blocked"
    assert "concert:missing_quality_axis:blind_listening_tests" in result["blockers"]
    assert "concert:missing_safety_check:rollback_plan_checked" in result["blockers"]


def test_offline_replay_contract_source_has_no_live_transport_imports():
    source = Path("backend/offline_replay_evaluation_contract.py").read_text()
    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "set_fader(",
        "set_gain(",
        "set_eq(",
        "set_compressor(",
        "route_output(",
        "load_snap(",
        "save_snap(",
    ]

    assert not any(token in source for token in forbidden)
