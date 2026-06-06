from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
    _promotion_rollback_verification_result_consumer,
    _proposal_for_policy_record,
    _sequence_policy_gate_ready,
)


def _promotion_sequence_policy_gate_result_consumer(tmp_path):
    rollback_result_consumer = _promotion_rollback_verification_result_consumer(tmp_path)
    gate = _sequence_policy_gate_ready(rollback_result_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
            promotion_rollback_verification_result_consumer=rollback_result_consumer,
            promotion_sequence_policy_gate_result=gate,
        )
    )


def _policy_consumer_ready(sequence_policy_gate_result_consumer):
    request = sequence_policy_gate_result_consumer["promotion_policy_consumer_request"]
    policy_record = request["sequence_policy_gate"]["policy_records"][0]
    proposal = _proposal_for_policy_record(policy_record)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[proposal],
            sequence_policy_gate=request["sequence_policy_gate"],
        ),
        [proposal],
    )


def test_promotion_policy_consumer_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer.py"
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


def test_promotion_policy_consumer_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["proposal_order_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_ordering_preview_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_ordering_preview_handler_invoked"] is False
    assert consumer["summary"]["promotion_ordering_preview_created"] is False
    assert (
        consumer["promotion_ordering_preview_request"]["handler"]
        == "preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering"
    )
    assert "policy_consumer" not in consumer["promotion_ordering_preview_request"]


def test_promotion_policy_consumer_result_consumer_blocks_without_consumer_records(
    tmp_path,
):
    sequence_gate_result_consumer = _promotion_sequence_policy_gate_result_consumer(
        tmp_path
    )
    request = sequence_gate_result_consumer["promotion_policy_consumer_request"]
    waiting_policy_consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[],
            sequence_policy_gate=request["sequence_policy_gate"],
        )
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
            promotion_sequence_policy_gate_result_consumer=sequence_gate_result_consumer,
            promotion_policy_consumer_result=waiting_policy_consumer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_blocked"
    )
    assert consumer["summary"]["expected_policy_consumer_record_count"] == 1
    assert consumer["summary"]["policy_consumer_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_ordering_preview_request_available"] is False
    assert (
        "sequence_memory_trust_promotion_policy_consumer_records_required_before_ordering_preview"
        in consumer["summary"]["blocked_by"]
    )
    assert "policy_consumer" not in consumer["promotion_ordering_preview_request"]


def test_promotion_policy_consumer_result_consumer_emits_ordering_preview_request(
    tmp_path,
):
    sequence_gate_result_consumer = _promotion_sequence_policy_gate_result_consumer(
        tmp_path
    )
    policy_consumer, proposals = _policy_consumer_ready(sequence_gate_result_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
            promotion_sequence_policy_gate_result_consumer=sequence_gate_result_consumer,
            promotion_policy_consumer_result=policy_consumer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_request_available"
    )
    assert consumer["summary"]["expected_policy_consumer_record_count"] == 1
    assert consumer["summary"]["policy_consumer_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_ordering_preview_request_available"] is True
    assert consumer["summary"]["promotion_ordering_preview_handler_invoked"] is False
    assert consumer["summary"]["promotion_ordering_preview_created"] is False
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_ordering_preview_request"]
    assert request["handler"] == (
        "preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering"
    )
    assert request["read_only_template"] is True
    assert request["promotion_ordering_preview_handler_invoked"] is False
    assert request["promotion_ordering_preview_created"] is False
    assert request["proposals_required_from_operator"] is True
    assert request["proposal_order_mutation"] is False
    assert request["queue_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["policy_consumer"] == policy_consumer

    preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=request["policy_consumer"],
        )
    )
    assert preview["status"] == (
        "sequence_memory_trust_promotion_ordering_preview_available"
    )
    assert preview["summary"]["ordered_preview_count"] == 1
    assert preview["summary"]["proposal_order_mutation"] is False
    assert preview["summary"]["queue_mutation"] is False
    assert preview["summary"]["sequence_policy_change_allowed_by_ordering_preview"] is False
    assert preview["summary"]["sequence_execution_allowed_by_ordering_preview"] is False
    assert preview["summary"]["live_authority_granted"] is False
    assert preview["ordered_proposals"][0]["can_import_to_queue"] is False
    assert preview["ordered_proposals"][0]["can_apply_live"] is False


def test_promotion_policy_consumer_result_consumer_rejects_forged_policy_consumer(
    tmp_path,
):
    sequence_gate_result_consumer = _promotion_sequence_policy_gate_result_consumer(
        tmp_path
    )
    policy_consumer, _ = _policy_consumer_ready(sequence_gate_result_consumer)
    forged = deepcopy(policy_consumer)
    forged["safety"]["proposal_order_mutation"] = True
    forged["summary"]["sequence_execution_allowed_by_consumer"] = True
    forged["consumer_records"][0]["can_import_to_queue"] = True
    forged["consumer_records"][0]["policy_key"] = "forged-policy-key"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
            promotion_sequence_policy_gate_result_consumer=sequence_gate_result_consumer,
            promotion_policy_consumer_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_ordering_preview_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "policy_consumer_result_consumer_policy_consumer_proposal_order_mutation_not_false"
        in invalid
    )
    assert (
        "policy_consumer_result_consumer_policy_consumer_summary_sequence_execution_allowed_by_consumer_not_false"
        in invalid
    )
    assert any(item.endswith(":can_import_to_queue_not_false") for item in invalid)
    assert any(item.endswith(":policy_key_mismatch") for item in invalid)
    assert "policy_consumer" not in consumer["promotion_ordering_preview_request"]


def test_promotion_policy_consumer_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    sequence_gate_result_consumer = deepcopy(
        _promotion_sequence_policy_gate_result_consumer(tmp_path)
    )
    policy_consumer, _ = _policy_consumer_ready(sequence_gate_result_consumer)
    sequence_gate_result_consumer["safety"]["live_authority_granted"] = True
    sequence_gate_result_consumer["summary"][
        "sequence_policy_gate_record_count"
    ] = 2

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
            promotion_sequence_policy_gate_result_consumer=sequence_gate_result_consumer,
            promotion_policy_consumer_result=policy_consumer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "policy_consumer_result_consumer_gate_result_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "policy_consumer_result_consumer_gate_result_consumer_summary_sequence_policy_gate_record_count_mismatch"
        in invalid
    )
    assert "policy_consumer" not in consumer["promotion_ordering_preview_request"]


def test_promotion_policy_consumer_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_ordering_preview_request_available"] is False


def test_promotion_policy_consumer_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
