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
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["idempotency_key"] == (
        "seq-policy:proposal-first"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == policy_consumer_result_consumer["promotion_ordering_preview_request"][
        "promotion_acceptance_record_templates"
    ][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == policy_consumer_result_consumer["promotion_ordering_preview_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

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


def test_promotion_ordering_preview_result_consumer_rejects_forged_identity(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged = deepcopy(ordering_preview)
    forged["ordered_proposals"][0]["source_idempotency_key"] = (
        "memory-write:unrelated-proposal"
    )

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
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert ordering_preview["ordered_proposals"][0]["ordering_preview_record_id"] in invalid
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]


def test_promotion_ordering_preview_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged_prior = deepcopy(policy_consumer_result_consumer)
    template = forged_prior["promotion_ordering_preview_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["idempotency_key"] = "seq-policy:forged"
    template["consumer_record_id"] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=forged_prior,
            promotion_ordering_preview_result=ordering_preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:idempotency_key_ordering_row_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_queue_ordering_plan_request"]
    )


def test_promotion_ordering_preview_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged_prior = deepcopy(policy_consumer_result_consumer)
    request = forged_prior["promotion_ordering_preview_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    forged_prior["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=forged_prior,
            promotion_ordering_preview_result=ordering_preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "ordering_preview_result_consumer_prior_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "ordering_preview_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first:current_sequence_review_freshness_missing" in invalid
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_queue_ordering_plan_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_queue_ordering_plan_request"]
    )


def test_promotion_ordering_preview_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged_prior = deepcopy(policy_consumer_result_consumer)
    live_freshness = {
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
    request = forged_prior["promotion_ordering_preview_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
            promotion_policy_consumer_result_consumer=forged_prior,
            promotion_ordering_preview_result=ordering_preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_queue_ordering_plan_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "invalid"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:current_sequence_review_freshness_read_only_not_true" in invalid
    assert "proposal-first:current_sequence_review_freshness_live_write_not_false" in invalid
    assert (
        "proposal-first:current_sequence_review_freshness_apply_attempted_not_false"
        in invalid
    )
    assert (
        "proposal-first:current_sequence_review_freshness_ordering_row_mismatch"
        in invalid
    )
    assert "ordering_preview" not in consumer["promotion_queue_ordering_plan_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_queue_ordering_plan_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_queue_ordering_plan_request"]
    )


def test_promotion_ordering_preview_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    policy_consumer_result_consumer, ordering_preview = _ordering_preview_ready(tmp_path)
    forged = deepcopy(policy_consumer_result_consumer)
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["policy_consumer_record_count"] = 2
    forged["summary"]["promotion_acceptance_record_template_count"] = 2
    forged["summary"][
        "promotion_acceptance_record_template_proposal_ids"
    ] = ["proposal-second"]

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
    assert (
        "ordering_preview_result_consumer_prior_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "ordering_preview_result_consumer_prior_summary_template_proposal_ids_mismatch"
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
