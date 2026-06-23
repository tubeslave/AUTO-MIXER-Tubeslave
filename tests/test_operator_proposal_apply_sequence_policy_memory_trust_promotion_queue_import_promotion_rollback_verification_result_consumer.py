from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
    _promotion_apply_audit,
    _promotion_executor_result_consumer,
    _rollback_verification_record,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer,
)


def _promotion_apply_audit_result_consumer(tmp_path):
    executor_result_consumer = _promotion_executor_result_consumer(tmp_path)
    apply_audit = _promotion_apply_audit(executor_result_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
            promotion_executor_result_consumer=executor_result_consumer,
            promotion_apply_audit_result=apply_audit,
        )
    )


def _rollback_verification_ready(apply_audit_result_consumer):
    request = apply_audit_result_consumer["promotion_rollback_verification_request"]
    audit = request["apply_audit"]
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
        apply_audit=audit,
        verification_records=[
            _rollback_verification_record(audit["audit_records"][0])
        ],
    )


def test_promotion_rollback_verification_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer.py"
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


def test_promotion_rollback_verification_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_sequence_policy_gate_handler_invoked"] is False
    assert consumer["summary"]["promotion_sequence_policy_gate_created"] is False
    assert (
        consumer["promotion_sequence_policy_gate_request"]["handler"]
        == "gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy"
    )
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]


def test_promotion_rollback_verification_result_consumer_blocks_without_verified_records(
    tmp_path,
):
    apply_audit_result_consumer = _promotion_apply_audit_result_consumer(tmp_path)
    request = apply_audit_result_consumer["promotion_rollback_verification_request"]
    waiting = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
        apply_audit=request["apply_audit"],
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=waiting,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_blocked"
    )
    assert consumer["summary"]["expected_rollback_verified_record_count"] == 1
    assert consumer["summary"]["rollback_verified_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    assert (
        "verified_sequence_memory_trust_promotion_rollback_records_required_before_sequence_policy_gate"
        in consumer["summary"]["blocked_by"]
    )
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]


def test_promotion_rollback_verification_result_consumer_emits_sequence_policy_gate_request(
    tmp_path,
):
    apply_audit_result_consumer = _promotion_apply_audit_result_consumer(tmp_path)
    rollback = _rollback_verification_ready(apply_audit_result_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_request_available"
    )
    assert consumer["summary"]["expected_rollback_verified_record_count"] == 1
    assert consumer["summary"]["rollback_verified_record_count"] == 1
    assert consumer["summary"]["promotion_acceptance_record_template_count"] == 1
    assert consumer["summary"]["memory_trust_review_request_spec_count"] == 2
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "fresh"
    assert consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] == [
        "proposal-first"
    ]
    assert consumer["summary"]["memory_trust_review_request_spec_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is True
    assert consumer["summary"]["promotion_sequence_policy_gate_handler_invoked"] is False
    assert consumer["summary"]["promotion_sequence_policy_gate_created"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_sequence_policy_gate_request"]
    assert request["handler"] == (
        "gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy"
    )
    assert request["read_only_template"] is True
    assert request["promotion_sequence_policy_gate_handler_invoked"] is False
    assert request["promotion_sequence_policy_gate_created"] is False
    assert request["queue_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["rollback_verification"] == rollback
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["idempotency_key"] == (
        "seq-policy:proposal-first"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == apply_audit_result_consumer["promotion_rollback_verification_request"][
        "promotion_acceptance_record_templates"
    ][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == apply_audit_result_consumer["promotion_rollback_verification_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

    gate = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate(
        rollback_verification=request["rollback_verification"],
    )
    assert gate["status"] == "verified_sequence_memory_trust_promotion_policy_records_available"
    assert gate["summary"]["rollback_verified_record_count"] == 1
    assert gate["summary"]["sequence_policy_gate_record_count"] == 1
    assert gate["summary"]["eligible_record_count"] == 1
    assert gate["summary"]["sequence_policy_eligible"] is True
    assert gate["summary"]["sequence_policy_trust_store_mutation"] is False
    assert gate["summary"]["queue_mutation"] is False
    assert gate["summary"]["live_authority_granted"] is False
    assert gate["policy_records"][0]["can_import_to_queue"] is False
    assert gate["policy_records"][0]["can_apply_live"] is False


def test_promotion_rollback_verification_result_consumer_rejects_missing_context(
    tmp_path,
):
    apply_audit_result_consumer = _promotion_apply_audit_result_consumer(tmp_path)
    rollback = deepcopy(_rollback_verification_ready(apply_audit_result_consumer))
    rollback["verified_records"][0].pop("current_sequence_review_freshness")

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    assert any(
        item.endswith(":current_sequence_review_freshness_missing")
        for item in consumer["summary"]["invalid_record_ids"]
    )
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]


def test_promotion_rollback_verification_result_consumer_rejects_live_authority_context(
    tmp_path,
):
    apply_audit_result_consumer = _promotion_apply_audit_result_consumer(tmp_path)
    rollback = deepcopy(_rollback_verification_ready(apply_audit_result_consumer))
    rollback["verified_records"][0]["current_sequence_review_freshness"] = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": False,
        "live_write": True,
        "apply_attempted": True,
        "queries_console": True,
        "queue_mutation": True,
        "policy_mutation": True,
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(":current_sequence_review_freshness_read_only_not_true")
        for item in invalid
    )
    assert any(
        item.endswith(":current_sequence_review_freshness_live_write_not_false")
        for item in invalid
    )
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]


def test_promotion_rollback_verification_result_consumer_rejects_forged_rollback(
    tmp_path,
):
    apply_audit_result_consumer = _promotion_apply_audit_result_consumer(tmp_path)
    forged = deepcopy(_rollback_verification_ready(apply_audit_result_consumer))
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["rollback_verified_record_count"] = 2
    forged["verified_records"][0]["proposal_id"] = "forged"
    forged["verified_records"][0]["target_policy_path"] = "forged/path"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_rollback_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_rollback_verified_record_count_mismatch"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(item.endswith(":target_policy_path_mismatch") for item in invalid)
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]


def test_promotion_rollback_verification_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    apply_audit_result_consumer = deepcopy(
        _promotion_apply_audit_result_consumer(tmp_path)
    )
    rollback = _rollback_verification_ready(apply_audit_result_consumer)
    request = apply_audit_result_consumer["promotion_rollback_verification_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    apply_audit_result_consumer["summary"].pop(
        "fresh_sequence_review_context_status",
        None,
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_memory_trust_review_request_spec_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first:current_sequence_review_freshness_missing" in invalid
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_sequence_policy_gate_request"]
    )


def test_promotion_rollback_verification_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    apply_audit_result_consumer = deepcopy(
        _promotion_apply_audit_result_consumer(tmp_path)
    )
    rollback = _rollback_verification_ready(apply_audit_result_consumer)
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
    request = apply_audit_result_consumer["promotion_rollback_verification_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
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
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_sequence_policy_gate_request"]
    )


def test_promotion_rollback_verification_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    apply_audit_result_consumer = deepcopy(
        _promotion_apply_audit_result_consumer(tmp_path)
    )
    rollback = _rollback_verification_ready(apply_audit_result_consumer)
    template = apply_audit_result_consumer["promotion_rollback_verification_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["idempotency_key"] = "seq-policy:forged"
    template["consumer_record_id"] = ""
    template["source_memory_trust_review_request_spec_id"] = "forged-review-spec"
    template["current_sequence_review_freshness"] = {
        **template["current_sequence_review_freshness"],
        "status": "stale",
    }
    rollback["verified_records"][0]["current_sequence_review_freshness"] = {
        **rollback["verified_records"][0]["current_sequence_review_freshness"],
        "status": "stale",
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:idempotency_key_verified_record_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "proposal-first:source_memory_trust_review_request_spec_id_mismatch" in invalid
    assert "proposal-first:current_sequence_review_freshness_request_spec_mismatch" in invalid
    assert (
        "proposal-first:verified_current_sequence_review_freshness_request_spec_mismatch"
        in invalid
    )
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_sequence_policy_gate_request"]
    )


def test_promotion_rollback_verification_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    apply_audit_result_consumer = deepcopy(_promotion_apply_audit_result_consumer(tmp_path))
    rollback = _rollback_verification_ready(apply_audit_result_consumer)
    apply_audit_result_consumer["summary"]["apply_audit_record_count"] = 2
    apply_audit_result_consumer["summary"][
        "promotion_acceptance_record_template_count"
    ] = 2
    apply_audit_result_consumer["summary"][
        "promotion_acceptance_record_template_proposal_ids"
    ] = ["proposal-second"]
    apply_audit_result_consumer["summary"]["memory_trust_review_request_spec_count"] = 1
    apply_audit_result_consumer["summary"][
        "memory_trust_review_request_spec_proposal_ids"
    ] = ["proposal-second"]
    apply_audit_result_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_apply_audit_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_template_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_memory_trust_review_request_spec_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "rollback_verification" not in consumer["promotion_sequence_policy_gate_request"]


def test_promotion_rollback_verification_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_sequence_policy_gate_request_available"] is False


def test_promotion_rollback_verification_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
