from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
    _policy_consumer_ready,
    _promotion_sequence_policy_gate_result_consumer,
)


def _promotion_policy_consumer_result_consumer(tmp_path):
    sequence_gate_result_consumer = _promotion_sequence_policy_gate_result_consumer(
        tmp_path
    )
    policy_consumer, proposals = _policy_consumer_ready(sequence_gate_result_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
            promotion_sequence_policy_gate_result_consumer=sequence_gate_result_consumer,
            promotion_policy_consumer_result=policy_consumer,
        ),
        proposals,
    )


def _ordering_preview_ready(tmp_path):
    policy_consumer_result_consumer, proposals = (
        _promotion_policy_consumer_result_consumer(tmp_path)
    )
    request = policy_consumer_result_consumer["promotion_ordering_preview_request"]
    return (
        policy_consumer_result_consumer,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=request["policy_consumer"],
        ),
    )


def test_promotion_ordering_preview_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer.py"
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


def test_promotion_ordering_preview_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["proposal_order_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_queue_ordering_plan_handler_invoked"] is False
    assert consumer["summary"]["promotion_queue_ordering_plan_created"] is False
    assert (
        consumer["promotion_queue_ordering_plan_request"]["handler"]
        == "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering"
    )
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]


def test_promotion_ordering_preview_result_consumer_blocks_without_preview_rows(
    tmp_path,
):
    policy_consumer_result_consumer, _ = (
        _promotion_policy_consumer_result_consumer(tmp_path)
    )
    waiting_preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=[],
        )
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=policy_consumer_result_consumer,
            promotion_ordering_preview_result=waiting_preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_blocked"
    )
    assert consumer["summary"]["expected_policy_ranked_record_count"] == 1
    assert consumer["summary"]["ordered_preview_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False
    assert (
        "sequence_memory_trust_promotion_ordering_preview_required_before_queue_ordering_plan"
        in consumer["summary"]["blocked_by"]
    )
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]


def test_promotion_ordering_preview_result_consumer_emits_queue_ordering_plan_request(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=policy_consumer_result_consumer,
            promotion_ordering_preview_result=ordering_preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_request_available"
    )
    assert consumer["summary"]["expected_policy_ranked_record_count"] == 1
    assert consumer["summary"]["ordered_preview_record_count"] == 1
    assert consumer["summary"]["policy_ranked_preview_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is True
    assert consumer["summary"]["promotion_queue_ordering_plan_handler_invoked"] is False
    assert consumer["summary"]["promotion_queue_ordering_plan_created"] is False
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_queue_ordering_plan_request"]
    assert request["handler"] == (
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering"
    )
    assert request["read_only_template"] is True
    assert request["promotion_queue_ordering_plan_handler_invoked"] is False
    assert request["promotion_queue_ordering_plan_created"] is False
    assert request["ordering_preview_required"] is True
    assert request["proposal_order_mutation"] is False
    assert request["queue_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["ordering_preview"] == ordering_preview

    plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan(
            ordering_preview=request["ordering_preview"],
        )
    )
    assert plan["status"] == "sequence_memory_trust_promotion_queue_ordering_plan_available"
    assert plan["summary"]["queue_ordering_plan_record_count"] == 1
    assert plan["summary"]["proposal_order_mutation"] is False
    assert plan["summary"]["queue_mutation"] is False
    assert plan["summary"]["queue_import_allowed_by_plan"] is False
    assert plan["summary"]["sequence_execution_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False
    assert plan["queue_ordering_plan_records"][0]["can_import_to_queue"] is False
    assert plan["queue_ordering_plan_records"][0]["can_apply_live"] is False


def test_promotion_ordering_preview_result_consumer_rejects_forged_ordering_preview(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged = deepcopy(ordering_preview)
    forged["safety"]["can_import_to_queue"] = True
    forged["summary"]["queue_mutation"] = True
    forged["ordered_proposals"][0]["live_write"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=policy_consumer_result_consumer,
            promotion_ordering_preview_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "ordering_preview_result_consumer_ordering_preview_can_import_to_queue_not_false"
        in invalid
    )
    assert (
        "ordering_preview_result_consumer_ordering_preview_summary_queue_mutation_not_false"
        in invalid
    )
    assert ordering_preview["ordered_proposals"][0]["ordering_preview_record_id"] in invalid
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]


def test_promotion_ordering_preview_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged = deepcopy(policy_consumer_result_consumer)
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["policy_consumer_record_count"] = 2

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=forged,
            promotion_ordering_preview_result=ordering_preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "ordering_preview_result_consumer_prior_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "ordering_preview_result_consumer_prior_summary_policy_consumer_record_count_mismatch"
        in invalid
    )
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]


def test_promotion_ordering_preview_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False


def test_promotion_ordering_preview_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
