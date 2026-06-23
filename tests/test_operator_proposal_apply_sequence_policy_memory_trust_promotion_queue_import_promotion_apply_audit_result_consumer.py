from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
    _promotion_executor_applied,
    _promotion_executor_dry_run_consumer,
)


def _promotion_executor_result_consumer(tmp_path):
    dry_run_consumer = _promotion_executor_dry_run_consumer(tmp_path)
    applied = _promotion_executor_applied(dry_run_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
            promotion_executor_dry_run_result_consumer=dry_run_consumer,
            promotion_executor_result=applied,
        )
    )


def _promotion_apply_audit(executor_result_consumer):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
        promotion_executor=executor_result_consumer["promotion_apply_audit_request"][
            "promotion_executor"
        ],
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


def _rollback_verification_record(audit_record):
    return {
        "rollback_verification_id": "queue-import-rollback-verification-1",
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
        "current_sequence_review_freshness": _sequence_context(),
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


def test_promotion_apply_audit_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer.py"
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


def test_promotion_apply_audit_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_rollback_verification_handler_invoked"] is False
    assert consumer["summary"]["promotion_rollback_verification_created"] is False
    assert (
        consumer["promotion_rollback_verification_request"]["handler"]
        == "verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback"
    )
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]


def test_promotion_apply_audit_result_consumer_blocks_without_apply_audit_records(
    tmp_path,
):
    executor_result_consumer = _promotion_executor_result_consumer(tmp_path)
    empty_audit = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit()
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=empty_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_blocked"
    )
    assert consumer["summary"]["expected_apply_audit_record_count"] == 1
    assert consumer["summary"]["apply_audit_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is False
    assert (
        "sequence_memory_trust_promotion_apply_audit_records_required_before_rollback_verification"
        in consumer["summary"]["blocked_by"]
    )
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]


def test_promotion_apply_audit_result_consumer_emits_rollback_verification_request(
    tmp_path,
):
    executor_result_consumer = _promotion_executor_result_consumer(tmp_path)
    apply_audit = _promotion_apply_audit(executor_result_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_request_available"
    )
    assert consumer["summary"]["expected_apply_audit_record_count"] == 1
    assert consumer["summary"]["apply_audit_record_count"] == 1
    assert consumer["summary"]["promotion_acceptance_record_template_count"] == 1
    assert consumer["summary"]["memory_trust_review_request_spec_count"] == 2
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "fresh"
    assert consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] == [
        "proposal-first"
    ]
    assert consumer["summary"]["memory_trust_review_request_spec_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is True
    assert consumer["summary"]["promotion_rollback_verification_handler_invoked"] is False
    assert consumer["summary"]["promotion_rollback_verification_created"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_rollback_verification_request"]
    assert request["handler"] == (
        "verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback"
    )
    assert request["read_only_template"] is True
    assert request["promotion_rollback_verification_handler_invoked"] is False
    assert request["promotion_rollback_verification_created"] is False
    assert request["verification_records_required_from_operator"] is True
    assert request["queue_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["apply_audit"] == apply_audit
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["idempotency_key"] == (
        "seq-policy:proposal-first"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == executor_result_consumer["promotion_apply_audit_request"][
        "promotion_acceptance_record_templates"
    ][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == executor_result_consumer["promotion_apply_audit_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

    waiting_for_operator = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=request["apply_audit"],
        )
    )
    assert waiting_for_operator["status"] == (
        "sequence_memory_trust_promotion_rollback_verification_not_ready"
    )
    assert waiting_for_operator["summary"]["unverified_apply_audit_record_count"] == 1
    assert waiting_for_operator["summary"]["rollback_verified_record_count"] == 0
    assert waiting_for_operator["summary"]["live_authority_granted"] is False

    record = _rollback_verification_record(apply_audit["audit_records"][0])
    verified = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=request["apply_audit"],
            verification_records=[record],
        )
    )
    assert verified["status"] == (
        "operator_sequence_memory_trust_promotion_rollback_verified_records_available"
    )
    assert verified["summary"]["rollback_verified_record_count"] == 1
    assert verified["summary"]["sequence_execution_allowed_by_verification"] is False
    assert verified["summary"]["live_authority_granted"] is False
    assert verified["verified_records"][0]["can_import_to_queue"] is False
    assert verified["verified_records"][0]["can_apply_live"] is False


def test_promotion_apply_audit_result_consumer_rejects_forged_apply_audit(
    tmp_path,
):
    executor_result_consumer = _promotion_executor_result_consumer(tmp_path)
    apply_audit = deepcopy(_promotion_apply_audit(executor_result_consumer))
    apply_audit["safety"]["live_authority_granted"] = True
    apply_audit["summary"]["apply_audit_record_count"] = 2
    apply_audit["audit_records"][0]["proposal_id"] = "forged"
    apply_audit["audit_records"][0]["target_policy_path"] = "forged/path"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_audit_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_apply_audit_record_count_mismatch"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(item.endswith(":target_policy_path_mismatch") for item in invalid)
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]


def test_promotion_apply_audit_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    executor_result_consumer = deepcopy(_promotion_executor_result_consumer(tmp_path))
    apply_audit = _promotion_apply_audit(executor_result_consumer)
    executor_result_consumer["summary"]["executor_apply_result_count"] = 2
    executor_result_consumer["summary"][
        "promotion_acceptance_record_template_count"
    ] = 2
    executor_result_consumer["summary"][
        "promotion_acceptance_record_template_proposal_ids"
    ] = ["proposal-second"]
    executor_result_consumer["summary"]["memory_trust_review_request_spec_count"] = 1
    executor_result_consumer["summary"][
        "memory_trust_review_request_spec_proposal_ids"
    ] = ["proposal-second"]
    executor_result_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_executor_apply_result_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_template_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_memory_trust_review_request_spec_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]


def test_promotion_apply_audit_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    executor_result_consumer = deepcopy(_promotion_executor_result_consumer(tmp_path))
    apply_audit = _promotion_apply_audit(executor_result_consumer)
    request = executor_result_consumer["promotion_apply_audit_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    executor_result_consumer["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_memory_trust_review_request_spec_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first:current_sequence_review_freshness_missing" in invalid
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_rollback_verification_request"]
    )


def test_promotion_apply_audit_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    executor_result_consumer = deepcopy(_promotion_executor_result_consumer(tmp_path))
    apply_audit = _promotion_apply_audit(executor_result_consumer)
    live_freshness = {
        "status": "fresh",
        "read_only": False,
        "field_count": 4,
        "fresh_field_count": 4,
        "live_write": True,
        "apply_attempted": True,
        "queries_console": True,
        "queue_mutation": True,
        "policy_mutation": True,
    }
    request = executor_result_consumer["promotion_apply_audit_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "invalid"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "proposal-first:current_sequence_review_freshness_read_only_not_true"
        in invalid
    )
    assert "proposal-first:current_sequence_review_freshness_live_write_not_false" in invalid
    assert (
        "proposal-first:current_sequence_review_freshness_apply_attempted_not_false"
        in invalid
    )
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_rollback_verification_request"]
    )


def test_promotion_apply_audit_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    executor_result_consumer = deepcopy(_promotion_executor_result_consumer(tmp_path))
    apply_audit = _promotion_apply_audit(executor_result_consumer)
    template = executor_result_consumer["promotion_apply_audit_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["idempotency_key"] = "seq-policy:forged"
    template["consumer_record_id"] = ""
    template["source_memory_trust_review_request_spec_id"] = "forged-review-spec"
    template["current_sequence_review_freshness"] = {
        **template["current_sequence_review_freshness"],
        "status": "stale",
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_rollback_verification_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:idempotency_key_apply_result_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "proposal-first:source_memory_trust_review_request_spec_id_mismatch" in invalid
    assert "proposal-first:current_sequence_review_freshness_request_spec_mismatch" in invalid
    assert "apply_audit" not in consumer["promotion_rollback_verification_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_rollback_verification_request"]
    )


def test_promotion_apply_audit_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
    )
    main = get_pipeline_profile("main_live_assist")
    snapshot = build_processing_registry_snapshot(include_methods=False)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "decision_evidence"
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
    consumer = snapshot[
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert (
        consumer["summary"]["promotion_rollback_verification_request_available"]
        is False
    )


def test_promotion_apply_audit_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
