from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer,
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
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
    _promotion_apply_audit_result_consumer,
    _rollback_verification_ready,
)


def _promotion_rollback_verification_result_consumer(tmp_path):
    apply_audit_result_consumer = _promotion_apply_audit_result_consumer(tmp_path)
    rollback = _rollback_verification_ready(apply_audit_result_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
            promotion_apply_audit_result_consumer=apply_audit_result_consumer,
            promotion_rollback_verification_result=rollback,
        )
    )


def _sequence_policy_gate_ready(rollback_verification_result_consumer):
    request = rollback_verification_result_consumer["promotion_sequence_policy_gate_request"]
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate(
        rollback_verification=request["rollback_verification"],
    )


def _proposal_for_policy_record(policy_record):
    return {
        "proposal_id": policy_record["proposal_id"],
        "method_id": "safe_gain",
        "kind": "gain_trim_down",
        "target": "ch/02",
        "severity": "high",
        "confidence": 0.82,
        "evidence": {
            "idempotency_key": policy_record["source_idempotency_key"],
        },
    }


def test_promotion_sequence_policy_gate_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer.py"
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


def test_promotion_sequence_policy_gate_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["proposal_order_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_policy_consumer_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_policy_consumer_handler_invoked"] is False
    assert consumer["summary"]["promotion_policy_consumer_created"] is False
    assert (
        consumer["promotion_policy_consumer_request"]["handler"]
        == "consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy"
    )
    assert "sequence_policy_gate" not in consumer["promotion_policy_consumer_request"]


def test_promotion_sequence_policy_gate_result_consumer_blocks_without_policy_records(
    tmp_path,
):
    rollback_result_consumer = _promotion_rollback_verification_result_consumer(tmp_path)
    waiting_gate = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate()
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
            promotion_rollback_verification_result_consumer=rollback_result_consumer,
            promotion_sequence_policy_gate_result=waiting_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_blocked"
    )
    assert consumer["summary"]["expected_sequence_policy_gate_record_count"] == 1
    assert consumer["summary"]["sequence_policy_gate_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_policy_consumer_request_available"] is False
    assert (
        "verified_sequence_memory_trust_promotion_policy_records_required_before_policy_consumer"
        in consumer["summary"]["blocked_by"]
    )
    assert "sequence_policy_gate" not in consumer["promotion_policy_consumer_request"]


def test_promotion_sequence_policy_gate_result_consumer_emits_policy_consumer_request(
    tmp_path,
):
    rollback_result_consumer = _promotion_rollback_verification_result_consumer(tmp_path)
    gate = _sequence_policy_gate_ready(rollback_result_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
            promotion_rollback_verification_result_consumer=rollback_result_consumer,
            promotion_sequence_policy_gate_result=gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_request_available"
    )
    assert consumer["summary"]["expected_sequence_policy_gate_record_count"] == 1
    assert consumer["summary"]["sequence_policy_gate_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_policy_consumer_request_available"] is True
    assert consumer["summary"]["promotion_policy_consumer_handler_invoked"] is False
    assert consumer["summary"]["promotion_policy_consumer_created"] is False
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_policy_consumer_request"]
    assert request["handler"] == (
        "consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy"
    )
    assert request["read_only_template"] is True
    assert request["promotion_policy_consumer_handler_invoked"] is False
    assert request["promotion_policy_consumer_created"] is False
    assert request["proposals_required_from_operator"] is True
    assert request["proposal_order_mutation"] is False
    assert request["queue_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["sequence_policy_gate"] == gate

    policy_record = gate["policy_records"][0]
    policy_consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[_proposal_for_policy_record(policy_record)],
            sequence_policy_gate=request["sequence_policy_gate"],
        )
    )
    assert policy_consumer["status"] == (
        "sequence_memory_trust_promotion_policy_consumer_records_available"
    )
    assert policy_consumer["summary"]["consumer_record_count"] == 1
    assert policy_consumer["summary"]["proposal_order_mutation"] is False
    assert policy_consumer["summary"]["queue_mutation"] is False
    assert policy_consumer["summary"]["sequence_policy_change_allowed_by_consumer"] is False
    assert policy_consumer["summary"]["sequence_execution_allowed_by_consumer"] is False
    assert policy_consumer["summary"]["live_authority_granted"] is False
    assert policy_consumer["consumer_records"][0]["can_import_to_queue"] is False
    assert policy_consumer["consumer_records"][0]["can_apply_live"] is False


def test_promotion_sequence_policy_gate_result_consumer_rejects_forged_gate(
    tmp_path,
):
    rollback_result_consumer = _promotion_rollback_verification_result_consumer(tmp_path)
    forged = deepcopy(_sequence_policy_gate_ready(rollback_result_consumer))
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["sequence_policy_gate_record_count"] = 2
    forged["policy_records"][0]["can_import_to_queue"] = True
    forged["policy_records"][0]["target_policy_path"] = "forged/path"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
            promotion_rollback_verification_result_consumer=rollback_result_consumer,
            promotion_sequence_policy_gate_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_policy_consumer_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_gate_result_consumer_gate_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_gate_result_consumer_gate_summary_sequence_policy_gate_record_count_mismatch"
        in invalid
    )
    assert any(item.endswith(":can_import_to_queue_not_false") for item in invalid)
    assert any(item.endswith(":target_policy_path_mismatch") for item in invalid)
    assert "sequence_policy_gate" not in consumer["promotion_policy_consumer_request"]


def test_promotion_sequence_policy_gate_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    rollback_result_consumer = deepcopy(
        _promotion_rollback_verification_result_consumer(tmp_path)
    )
    gate = _sequence_policy_gate_ready(rollback_result_consumer)
    rollback_result_consumer["summary"]["rollback_verified_record_count"] = 2
    rollback_result_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
            promotion_rollback_verification_result_consumer=rollback_result_consumer,
            promotion_sequence_policy_gate_result=gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_summary_rollback_verified_record_count_mismatch"
        in invalid
    )
    assert "sequence_policy_gate" not in consumer["promotion_policy_consumer_request"]


def test_promotion_sequence_policy_gate_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_policy_consumer_request_available"] is False


def test_promotion_sequence_policy_gate_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
