from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
    build_operator_proposal_apply_sequence_policy_memory_trust_review,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _memory_text() -> str:
    return "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Supervised sequence policy memory",
            "",
            (
                "- Draft ID: "
                "`operator-sequence-policy-memory:memory-write-applied:"
                "memory-write:memory-write-applied`"
            ),
            "- Proposal ID: `memory-write-applied`",
            "- Learning scope: `supervised_fader_gain_sequence_execution`",
            "- Value type: `fader`",
            "- Channel: `2`",
            "- Effective value: `-8.5`",
            "- Idempotency key: `memory-write:memory-write-applied`",
            "- Readback status: `verified_from_cache`",
            "- Proof status: `verified`",
            "- Runtime authority: no live, queue, model, or policy mutation",
            "",
            "Supervised fader/gain sequence execution was verified.",
            "",
        ]
    )


def _sequence_context():
    return {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
    }


def _rollback_evidence():
    return {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_sequence_policy_state",
        "previous_policy_state": "not_trusted_for_sequence_policy",
        "target_policy_store": "operator_sequence_policy_trust_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T13:00:00Z",
    }


def _memory_trust_acceptance_record():
    return {
        "acceptance_id": "sequence-memory-trust-acceptance-1",
        "review_candidate_id": (
            "operator_sequence_memory_trust_review:"
            "operator_sequence_policy_memory_index:"
            "operator-sequence-policy-memory:memory-write-applied:"
            "memory-write:memory-write-applied"
        ),
        "memory_record_id": (
            "operator_sequence_policy_memory_index:"
            "operator-sequence-policy-memory:memory-write-applied:"
            "memory-write:memory-write-applied"
        ),
        "proposal_id": "memory-write-applied",
        "learning_scope": "supervised_fader_gain_sequence_execution",
        "value_type": "fader",
        "idempotency_key": "memory-write:memory-write-applied",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:30:00Z",
    }


def _promotion_acceptance_record():
    return {
        "promotion_acceptance_id": "sequence-memory-promotion-acceptance-1",
        "promotion_contract_record_id": (
            "operator_sequence_memory_trust_promotion_contract:"
            "operator_sequence_memory_trust_promotion:"
            "sequence-memory-trust-acceptance-1"
        ),
        "promotion_candidate_id": (
            "operator_sequence_memory_trust_promotion:"
            "sequence-memory-trust-acceptance-1"
        ),
        "proposal_id": "memory-write-applied",
        "learning_scope": "supervised_fader_gain_sequence_execution",
        "value_type": "fader",
        "idempotency_key": "memory-write:memory-write-applied",
        "previous_policy_state": "not_trusted_for_sequence_policy",
        "rollback_strategy": "manual_revert_to_previous_sequence_policy_state",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T13:30:00Z",
    }


def _executor_dry_run():
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=index,
        current_sequence_review_freshness=_sequence_context(),
    )
    trust_ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=[_memory_trust_acceptance_record()],
    )
    promotion_plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=trust_ledger,
    )
    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=promotion_plan,
        current_sequence_policy_context=_sequence_context(),
        rollback_evidence=_rollback_evidence(),
    )
    acceptance_ledger = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
            promotion_contract=contract,
            acceptance_records=[_promotion_acceptance_record()],
        )
    )
    execution_plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=acceptance_ledger,
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=execution_plan,
    )


def _applied_executor():
    store = {
        "memory-write-applied/memory-write:memory-write-applied": (
            "not_trusted_for_sequence_policy"
        )
    }
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
        apply=True,
        operator_ack=True,
    )


def _apply_audit():
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
        promotion_executor=_applied_executor(),
    )


def _verification_record(audit_record):
    return {
        "rollback_verification_id": "sequence-rollback-verification-1",
        "promotion_apply_audit_id": audit_record["promotion_apply_audit_id"],
        "promotion_executor_apply_id": audit_record["promotion_executor_apply_id"],
        "proposal_id": audit_record["proposal_id"],
        "learning_scope": audit_record["learning_scope"],
        "value_type": audit_record["value_type"],
        "source_idempotency_key": audit_record["source_idempotency_key"],
        "policy_key": audit_record["policy_key"],
        "target_policy_store": audit_record["target_policy_store"],
        "target_policy_path": audit_record["target_policy_path"],
        "target_policy_state": audit_record["target_policy_state"],
        "rollback_target_policy_state": audit_record["rollback_target_policy_state"],
        "rollback_strategy": audit_record["rollback_strategy"],
        "verification_evidence_status": "reviewed",
        "decision": "verified",
        "verified_by": "operator",
        "verified_at": "2026-06-06T14:00:00Z",
        "rollback_verified": True,
        "read_only": True,
        "operator_review_required": True,
        "separate_sequence_policy_gate_required": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "sequence_policy_trust_store_mutation": False,
        "proposal_state_mutation": False,
        "sequence_execution_allowed_by_verification": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }


def test_sequence_memory_trust_promotion_rollback_verification_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py"
    ).read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_trim",
        "set_input_trim",
        "set_channel_trim",
        "set_preamp_trim",
        "set_preamp_gain",
        "set_eq",
        "set_compressor",
        "route_output",
        "load_snap",
        "save_snap",
        "reset",
        "mute",
        "unmute",
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_sequence_memory_trust_promotion_rollback_verification_not_ready_without_audit():
    verification = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification()
    )

    assert (
        verification["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification"
    )
    assert (
        verification["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
    )
    assert (
        verification["status"]
        == "sequence_memory_trust_promotion_rollback_verification_not_ready"
    )
    assert verification["safety"]["read_only"] is True
    assert verification["safety"]["live_write"] is False
    assert verification["safety"]["queue_mutation"] is False
    assert verification["safety"]["runtime_memory_mutation"] is False
    assert verification["safety"]["sequence_policy_trust_store_mutation"] is False
    assert verification["safety"]["sequence_execution_allowed_by_verification"] is False
    assert verification["summary"]["rollback_verified_record_count"] == 0
    assert verification["verified_records"] == []


def test_sequence_memory_trust_promotion_rollback_verification_waits_for_all_audit_records():
    audit = _apply_audit()

    verification = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=audit,
        )
    )

    assert (
        verification["status"]
        == "sequence_memory_trust_promotion_rollback_verification_not_ready"
    )
    assert verification["summary"]["apply_audit_record_count"] == 1
    assert verification["summary"]["verification_record_count"] == 0
    assert verification["summary"]["rollback_verified_record_count"] == 0
    assert verification["summary"]["unverified_apply_audit_record_count"] == 1
    assert verification["summary"]["unverified_apply_audit_record_ids"] == [
        audit["audit_records"][0]["promotion_apply_audit_id"]
    ]
    assert verification["summary"]["usable_as_sequence_policy_gate_evidence"] is False
    assert verification["summary"]["sequence_execution_allowed_by_verification"] is False
    assert verification["verified_records"] == []


def test_sequence_memory_trust_promotion_rollback_verification_records_verified_evidence():
    audit = _apply_audit()
    record = _verification_record(audit["audit_records"][0])

    verification = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=audit,
            verification_records=[record],
        )
    )

    assert (
        verification["status"]
        == "operator_sequence_memory_trust_promotion_rollback_verified_records_available"
    )
    assert verification["safety"]["read_only"] is True
    assert verification["safety"]["rollback_verified"] is True
    assert verification["safety"]["usable_as_sequence_policy_gate_evidence"] is True
    assert verification["safety"]["separate_sequence_policy_gate_required"] is True
    assert verification["safety"]["runtime_memory_mutation"] is False
    assert verification["safety"]["sequence_policy_trust_store_mutation"] is False
    assert verification["safety"]["proposal_state_mutation"] is False
    assert verification["safety"]["live_authority_granted"] is False
    assert verification["summary"]["rollback_verified_record_count"] == 1
    assert verification["summary"]["unverified_apply_audit_record_count"] == 0
    verified = verification["verified_records"][0]
    assert verified["promotion_apply_audit_id"] == record["promotion_apply_audit_id"]
    assert verified["proposal_id"] == "memory-write-applied"
    assert verified["policy_key"] == "memory-write-applied/memory-write:memory-write-applied"
    assert verified["target_policy_state"] == "trusted_for_sequence_policy"
    assert verified["rollback_target_policy_state"] == "not_trusted_for_sequence_policy"
    assert verified["rollback_verified"] is True
    assert verified["usable_as_sequence_policy_gate_evidence"] is True
    assert verified["separate_sequence_policy_gate_required"] is True
    assert verified["runtime_memory_mutation"] is False
    assert verified["queue_mutation"] is False
    assert verified["model_mutation"] is False
    assert verified["proposal_state_mutation"] is False
    assert verified["sequence_execution_allowed_by_verification"] is False
    assert verified["can_import_to_queue"] is False
    assert verified["can_apply_live"] is False
    assert "requested_change" not in verification


def test_sequence_memory_trust_promotion_rollback_verification_rejects_forged_authority():
    audit = _apply_audit()
    forged_record = {
        **_verification_record(audit["audit_records"][0]),
        "live_write": True,
        "sequence_execution_allowed_by_verification": True,
    }
    forged_audit = {
        **audit,
        "summary": {
            **audit["summary"],
            "sequence_execution_allowed_by_audit": True,
        },
    }

    verification = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=forged_audit,
            verification_records=[forged_record],
        )
    )

    assert (
        verification["status"]
        == "sequence_memory_trust_promotion_rollback_verification_replacement_required"
    )
    assert verification["summary"]["rollback_verified_record_count"] == 0
    assert "sequence-rollback-verification-1" in verification["summary"][
        "invalid_record_ids"
    ]
    assert (
        "sequence_memory_trust_promotion_rollback_verification_audit_summary_sequence_execution_allowed_by_audit_not_false"
        in verification["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_rollback_verification_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
    )
    main = get_pipeline_profile("main_live_assist")
    snapshot = build_processing_registry_snapshot(include_methods=False)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {
        "main_live_assist",
        "shadow_quality",
        "lab_dsp",
    }
    assert main is not None
    assert method.id in main.method_ids
    assert method.id not in main.proposal_method_ids
    assert method.id not in main.executor_method_ids
    verification = snapshot[
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
    ]
    assert verification["safety"]["live_write"] is False
    assert verification["safety"]["runtime_memory_mutation"] is False
    assert verification["summary"]["rollback_verified_record_count"] == 0


def test_sequence_memory_trust_promotion_rollback_verification_tracks_goal_matrix_without_live_widening():
    memory = get_project_goal_capability("long_term_memory_learning")

    assert memory is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
        in memory.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py"
        in memory.support_paths
    )
    assert memory.current_live_write_kinds == ()
    assert memory.autonomy_mode_ceiling == "assisted"
