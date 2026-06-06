from pathlib import Path

from backend.supervised_proof_artifact_evaluator import (
    SupervisedProofArtifactInput,
    build_supervised_proof_artifact_snapshot,
    evaluate_supervised_proof_artifact,
    evaluate_supervised_proof_artifact_dict,
    supervised_proof_artifact_from_dict,
)


def complete_artifact(**overrides):
    payload = {
        "proof_id": "unit-proof",
        "value_type": "gain",
        "channel": 3,
        "old_value": 1.0,
        "proposed_value": 0.75,
        "delta_db": -0.25,
        "proof_mode": "dry_run",
        "pre_read_status": "verified_from_physical_readback",
        "pre_read_source": "fake_mixer",
        "readback_status": "verified_from_physical_readback",
        "readback_source": "fake_mixer",
        "proof_status": "verified",
        "rollback_available": True,
        "rollback_value": 1.0,
        "rollback_readback_status": "verified_from_physical_readback",
        "rollback_readback_source": "fake_mixer",
        "rollback_proof_status": "verified",
        "emergency_stop_proof_status": "verified",
        "blocked_followup_status": "verified",
        "operator_approval_recorded": True,
        "idempotency_key_present": True,
        "reason": "operator approved small proof move",
    }
    payload.update(overrides)
    return SupervisedProofArtifactInput(**payload)


def test_supervised_proof_evaluator_is_read_only_and_transport_free():
    source = Path("backend/supervised_proof_artifact_evaluator.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_complete_dry_run_gain_proof_is_ready_but_not_promotable_when_physical_required():
    result = evaluate_supervised_proof_artifact(complete_artifact(), require_physical=True)

    assert result.status == "dry_run_ready"
    assert result.promotion_allowed is False
    assert result.autonomous_ready is False
    assert result.blockers == ("physical_proof_required_for_promotion",)
    assert result.evidence["pre_read_source"] == "fake_mixer"


def test_complete_physical_fader_proof_is_supervised_candidate_only():
    artifact = complete_artifact(
        proof_id="physical-fader-proof",
        value_type="fader",
        old_value=-12.0,
        proposed_value=-12.25,
        delta_db=-0.25,
        proof_mode="physical",
        pre_read_source="physical_readback",
        readback_source="physical_readback",
        rollback_value=-12.0,
        rollback_readback_source="physical_readback",
    )

    result = evaluate_supervised_proof_artifact(artifact, require_physical=True)

    assert result.status == "supervised_candidate"
    assert result.promotion_allowed is True
    assert result.autonomous_ready is False
    assert result.blockers == ()


def test_missing_pre_read_blocks_fail_closed():
    artifact = complete_artifact(pre_read_status="readback_unavailable", pre_read_source="")

    result = evaluate_supervised_proof_artifact(artifact)

    assert result.status == "blocked"
    assert result.promotion_allowed is False
    assert "pre_read_unavailable" in result.blockers


def test_missing_rollback_and_emergency_stop_proof_block():
    artifact = complete_artifact(
        rollback_available=False,
        rollback_value=None,
        rollback_readback_status="",
        rollback_proof_status="",
        emergency_stop_proof_status="",
        blocked_followup_status="",
    )

    result = evaluate_supervised_proof_artifact(artifact)

    assert result.status == "blocked"
    assert "rollback_unavailable" in result.blockers
    assert "rollback_value_missing" in result.blockers
    assert "rollback_readback_missing" in result.blockers
    assert "rollback_proof_missing" in result.blockers
    assert "emergency_stop_proof_missing" in result.blockers
    assert "blocked_followup_missing" in result.blockers


def test_unsupported_console_family_blocks():
    artifact = complete_artifact(value_type="eq")

    result = evaluate_supervised_proof_artifact(artifact)

    assert result.status == "blocked"
    assert "unsupported_value_type" in result.blockers


def test_gain_up_requires_operator_authored_reason():
    artifact = complete_artifact(
        old_value=0.0,
        proposed_value=0.25,
        delta_db=0.25,
        reason="",
    )

    result = evaluate_supervised_proof_artifact(artifact)

    assert result.status == "blocked"
    assert "gain_up_requires_operator_authored_reason" in result.blockers


def gain_dry_run_summary():
    return {
        "run_id": "20260604T000000Z_gain_proof_dry_run_ch3",
        "status": "passed",
        "channel": 3,
        "initial_gain_db": 1.0,
        "delta_db": -0.25,
        "physical_wing": False,
        "write_policy": "fake_mixer_supervised_gate_only",
        "old_value": 1.0,
        "target_value": 0.75,
        "apply_result": {
            "apply_result": {
                "status": "applied",
                "channel": 3,
                "value_type": "gain",
                "old_value": 1.0,
                "effective_value": 0.75,
                "readback_status": "verified_from_physical_readback",
                "pre_read_status": "verified_from_physical_readback",
                "proof_status": "verified",
                "rollback_available": True,
                "rollback_value": 1.0,
            },
        },
        "rollback_result": {
            "type": "manual_write_rollback_applied",
            "rollback_status": "applied",
            "rollback_readback_status": "verified_from_physical_readback",
            "rollback_proof_status": "verified",
        },
        "cooldown_apply_result": {"apply_result": {"reason": "cooldown"}},
        "post_emergency_apply_result": {"apply_result": {"reason": "emergency_stop_active"}},
    }


def test_dict_parser_maps_gain_dry_run_runner_summary():
    artifact = supervised_proof_artifact_from_dict(gain_dry_run_summary())

    assert artifact.proof_id == "20260604T000000Z_gain_proof_dry_run_ch3"
    assert artifact.value_type == "gain"
    assert artifact.proof_mode == "dry_run"
    assert artifact.old_value == 1.0
    assert artifact.proposed_value == 0.75
    assert artifact.pre_read_status == "verified_from_physical_readback"
    assert artifact.pre_read_source == "fake_mixer"
    assert artifact.readback_source == "fake_mixer"
    assert artifact.blocked_followup_status == "verified"
    assert artifact.emergency_stop_proof_status == "verified"
    assert artifact.operator_approval_recorded is True
    assert artifact.idempotency_key_present is True

    result = evaluate_supervised_proof_artifact_dict(gain_dry_run_summary(), require_physical=True)
    assert result.status == "dry_run_ready"
    assert result.promotion_allowed is False
    assert "physical_proof_required_for_promotion" in result.blockers


def test_dict_parser_honors_explicit_physical_runner_schema():
    payload = gain_dry_run_summary()
    payload.update(
        {
            "run_id": "20260604T000000Z_wing_operator_proposal_ch3_gain",
            "proof_artifact_schema": "wing_operator_proposal_validation.v2",
            "proof_runner": "wing_operator_proposal_validation_runner",
            "proof_mode": "physical",
            "physical_wing": True,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
            "operator_reason": "operator approved small gain-down physical proof",
        }
    )
    payload["apply_result"]["apply_result"]["pre_read_source"] = "physical_readback"
    payload["apply_result"]["apply_result"]["readback_source"] = "physical_readback"

    artifact = supervised_proof_artifact_from_dict(payload)
    result = evaluate_supervised_proof_artifact(artifact, require_physical=True)

    assert artifact.proof_mode == "physical"
    assert artifact.pre_read_source == "physical_readback"
    assert artifact.readback_source == "physical_readback"
    assert artifact.rollback_readback_source == "physical_readback"
    assert result.status == "supervised_candidate"
    assert result.promotion_allowed is True
    assert result.autonomous_ready is False


def test_manual_review_timeout_summary_cannot_become_promotion_evidence_even_with_readback_fields():
    payload = gain_dry_run_summary()
    payload.update(
        {
            "run_id": "20260606T000000Z_wing_operator_proposal_ch3_gain",
            "status": "manual_review_required",
            "reason": "connection_or_timeout_after_apply_request",
            "proof_artifact_schema": "wing_operator_proposal_validation.v2",
            "proof_runner": "wing_operator_proposal_validation_runner",
            "proof_mode": "physical",
            "physical_wing": True,
            "execute": True,
            "fail_closed": False,
            "live_apply_request_sent": True,
            "rollback_request_sent": False,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
            "operator_reason": "operator approved small gain-down physical proof",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )

    result = evaluate_supervised_proof_artifact_dict(payload, require_physical=True)

    assert result.status == "blocked"
    assert result.promotion_allowed is False
    assert "summary_status_not_promotable" in result.blockers
    assert "transport_failure_after_apply_requires_manual_review" in result.blockers
    assert result.evidence["metadata"]["live_apply_request_sent"] is True


def test_fail_closed_timeout_summary_before_apply_is_blocked_as_evidence():
    payload = gain_dry_run_summary()
    payload.update(
        {
            "run_id": "20260606T000000Z_wing_operator_proposal_ch3_gain",
            "status": "blocked",
            "reason": "connection_or_timeout_fail_closed",
            "proof_artifact_schema": "wing_operator_proposal_validation.v2",
            "proof_runner": "wing_operator_proposal_validation_runner",
            "proof_mode": "physical",
            "physical_wing": True,
            "execute": True,
            "fail_closed": True,
            "live_apply_request_sent": False,
            "rollback_request_sent": False,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
            "operator_reason": "operator approved small gain-down physical proof",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )

    result = evaluate_supervised_proof_artifact_dict(payload, require_physical=True)

    assert result.status == "blocked"
    assert result.promotion_allowed is False
    assert "summary_status_not_promotable" in result.blockers
    assert "transport_failure_before_apply_fail_closed" in result.blockers
    assert result.evidence["metadata"]["fail_closed"] is True


def test_legacy_execute_wing_summary_without_schema_blocks_instead_of_dry_run_ready():
    payload = gain_dry_run_summary()
    payload.pop("physical_wing")
    payload.update(
        {
            "run_id": "20260523T104641Z_wing_operator_proposal_ch4_gain",
            "execute": True,
            "write_policy": "apply_operator_proposal_only",
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
            "operator_reason": "operator approved small gain-down physical proof",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )

    artifact = supervised_proof_artifact_from_dict(payload)
    result = evaluate_supervised_proof_artifact(artifact, require_physical=True)

    assert artifact.proof_mode == "legacy_ambiguous"
    assert result.status == "blocked"
    assert result.promotion_allowed is False
    assert result.autonomous_ready is False
    assert "legacy_physical_schema_missing" in result.blockers
    assert "physical_proof_required_for_promotion" in result.blockers
    assert result.evidence["metadata"]["execute"] is True
    assert result.evidence["metadata"]["proof_artifact_schema"] is None


def test_dict_parser_preserves_zero_values_and_does_not_treat_summary_reason_as_operator_reason():
    payload = gain_dry_run_summary()
    payload["old_value"] = 0.0
    payload["target_value"] = 0.25
    payload["delta_db"] = 0.25
    payload["reason"] = "proposal_apply_readback_rollback_complete"
    payload["apply_result"]["apply_result"]["old_value"] = 0.0
    payload["apply_result"]["apply_result"]["effective_value"] = 0.25
    payload["apply_result"]["apply_result"]["rollback_value"] = 0.0

    artifact = supervised_proof_artifact_from_dict(payload)
    result = evaluate_supervised_proof_artifact(artifact)

    assert artifact.old_value == 0.0
    assert artifact.proposed_value == 0.25
    assert artifact.rollback_value == 0.0
    assert artifact.reason == ""
    assert result.status == "blocked"
    assert "gain_up_requires_operator_authored_reason" in result.blockers


def test_snapshot_summarizes_promotion_candidates_without_autonomous_ready():
    physical = complete_artifact(
        proof_id="physical-fader-proof",
        value_type="fader",
        old_value=-10.0,
        proposed_value=-10.25,
        proof_mode="physical",
        pre_read_source="physical_readback",
        readback_source="physical_readback",
        rollback_value=-10.0,
        rollback_readback_source="physical_readback",
    )

    snapshot = build_supervised_proof_artifact_snapshot(
        [gain_dry_run_summary(), physical],
        require_physical=True,
    )

    assert snapshot["type"] == "supervised_proof_artifact_evaluation"
    assert snapshot["safety"]["live_write"] is False
    assert snapshot["summary"]["status_counts"]["dry_run_ready"] == 1
    assert snapshot["summary"]["status_counts"]["supervised_candidate"] == 1
    assert snapshot["summary"]["promotion_allowed_proof_ids"] == ["physical-fader-proof"]
    assert snapshot["summary"]["autonomous_ready_proof_ids"] == []
