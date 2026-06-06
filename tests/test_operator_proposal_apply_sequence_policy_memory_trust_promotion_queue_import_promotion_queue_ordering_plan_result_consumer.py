from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
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
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
    _promotion_policy_consumer_result_consumer,
)


def _promotion_ordering_preview_result_consumer(tmp_path):
    policy_consumer_result_consumer, proposals = (
        _promotion_policy_consumer_result_consumer(tmp_path)
    )
    request = policy_consumer_result_consumer["promotion_ordering_preview_request"]
    ordering_preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=request["policy_consumer"],
        )
    )
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=policy_consumer_result_consumer,
            promotion_ordering_preview_result=ordering_preview,
        ),
        proposals,
    )


def _queue_ordering_plan_ready(tmp_path):
    ordering_preview_result_consumer, proposals = (
        _promotion_ordering_preview_result_consumer(tmp_path)
    )
    request = ordering_preview_result_consumer["promotion_queue_ordering_plan_request"]
    return (
        ordering_preview_result_consumer,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan(
            ordering_preview=request["ordering_preview"],
        ),
        proposals,
    )


def test_promotion_queue_ordering_plan_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer.py"
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


def test_promotion_queue_ordering_plan_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["proposal_order_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_queue_importer_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_queue_importer_handler_invoked"] is False
    assert consumer["summary"]["promotion_queue_importer_created"] is False
    assert (
        consumer["promotion_queue_importer_request"]["handler"]
        == "import_operator_proposal_apply_sequence_policy_queue_ordering_plan"
    )
    assert consumer["promotion_queue_importer_request"]["apply"] is False
    assert consumer["promotion_queue_importer_request"]["operator_ack"] is False
    assert "queue_ordering_plan" not in consumer["promotion_queue_importer_request"]


def test_promotion_queue_ordering_plan_result_consumer_blocks_without_plan_rows(
    tmp_path,
):
    ordering_preview_result_consumer, _ = _promotion_ordering_preview_result_consumer(
        tmp_path
    )
    waiting_plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan()
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
            promotion_ordering_preview_result_consumer=ordering_preview_result_consumer,
            promotion_queue_ordering_plan_result=waiting_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_blocked"
    )
    assert consumer["summary"]["expected_ordering_preview_record_count"] == 1
    assert consumer["summary"]["queue_ordering_plan_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_queue_importer_request_available"] is False
    assert (
        "sequence_memory_trust_promotion_queue_ordering_plan_required_before_queue_importer"
        in consumer["summary"]["blocked_by"]
    )
    assert "queue_ordering_plan" not in consumer["promotion_queue_importer_request"]


def test_promotion_queue_ordering_plan_result_consumer_emits_queue_importer_request(
    tmp_path,
):
    ordering_preview_result_consumer, queue_ordering_plan, proposals = (
        _queue_ordering_plan_ready(tmp_path)
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
            promotion_ordering_preview_result_consumer=ordering_preview_result_consumer,
            promotion_queue_ordering_plan_result=queue_ordering_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_request_available"
    )
    assert consumer["summary"]["expected_ordering_preview_record_count"] == 1
    assert consumer["summary"]["queue_ordering_plan_record_count"] == 1
    assert consumer["summary"]["ranked_queue_candidate_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_queue_importer_request_available"] is True
    assert consumer["summary"]["promotion_queue_importer_handler_invoked"] is False
    assert consumer["summary"]["promotion_queue_importer_created"] is False
    assert consumer["summary"]["proposal_payloads_required_from_operator"] is True
    assert consumer["summary"]["apply"] is False
    assert consumer["summary"]["operator_ack"] is False
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_queue_importer_request"]
    assert request["handler"] == "import_operator_proposal_apply_sequence_policy_queue_ordering_plan"
    assert request["read_only_template"] is True
    assert request["promotion_queue_importer_handler_invoked"] is False
    assert request["promotion_queue_importer_created"] is False
    assert request["proposal_payloads_required_from_operator"] is True
    assert request["apply"] is False
    assert request["operator_ack"] is False
    assert request["queue_mutation"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    assert request["queue_ordering_plan"] == queue_ordering_plan

    importer = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
        queue_ordering_plan=request["queue_ordering_plan"],
        proposal_payloads=proposals,
        apply=request["apply"],
        operator_ack=request["operator_ack"],
    )
    assert importer["status"] == "sequence_memory_trust_promotion_queue_importer_dry_run_ready"
    assert importer["summary"]["queue_import_spec_count"] == 1
    assert importer["summary"]["queue_import_result_count"] == 0
    assert importer["summary"]["queue_mutation"] is False
    assert importer["summary"]["live_write"] is False
    assert importer["queue_import_specs"][0]["can_import_to_queue"] is False
    assert importer["queue_import_specs"][0]["can_apply_live"] is False


def test_promotion_queue_ordering_plan_result_consumer_rejects_forged_plan(
    tmp_path,
):
    ordering_preview_result_consumer, queue_ordering_plan, _ = (
        _queue_ordering_plan_ready(tmp_path)
    )
    forged = deepcopy(queue_ordering_plan)
    forged["safety"]["queue_mutation"] = True
    forged["summary"]["can_import_to_queue"] = True
    forged["queue_ordering_plan_records"][0]["live_write"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
            promotion_ordering_preview_result_consumer=ordering_preview_result_consumer,
            promotion_queue_ordering_plan_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_queue_importer_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "queue_ordering_plan_result_consumer_plan_queue_mutation_not_false" in invalid
    assert (
        "queue_ordering_plan_result_consumer_plan_summary_can_import_to_queue_not_false"
        in invalid
    )
    assert (
        queue_ordering_plan["queue_ordering_plan_records"][0][
            "queue_ordering_plan_record_id"
        ]
        in invalid
    )
    assert "queue_ordering_plan" not in consumer["promotion_queue_importer_request"]


def test_promotion_queue_ordering_plan_result_consumer_rejects_forged_prior(
    tmp_path,
):
    ordering_preview_result_consumer, queue_ordering_plan, _ = (
        _queue_ordering_plan_ready(tmp_path)
    )
    forged = deepcopy(ordering_preview_result_consumer)
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["ordered_preview_record_count"] = 2

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
            promotion_ordering_preview_result_consumer=forged,
            promotion_queue_ordering_plan_result=queue_ordering_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "queue_ordering_plan_result_consumer_prior_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "queue_ordering_plan_result_consumer_prior_summary_ordered_preview_record_count_mismatch"
        in invalid
    )
    assert "queue_ordering_plan" not in consumer["promotion_queue_importer_request"]


def test_promotion_queue_ordering_plan_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_queue_importer_request_available"] is False


def test_promotion_queue_ordering_plan_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
