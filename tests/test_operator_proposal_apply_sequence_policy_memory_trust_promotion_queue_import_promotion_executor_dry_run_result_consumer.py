from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
    _promotion_acceptance_consumer,
    _promotion_execution_plan,
)


def _promotion_execution_plan_consumer(tmp_path):
    acceptance_consumer = _promotion_acceptance_consumer(tmp_path)
    execution_plan = _promotion_execution_plan(acceptance_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer(
            promotion_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_execution_plan_result=execution_plan,
        )
    )


def _promotion_executor_dry_run(execution_plan_consumer):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=execution_plan_consumer[
            "promotion_executor_dry_run_request"
        ]["promotion_execution_plan"],
    )


def test_promotion_executor_dry_run_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer.py"
    ).read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "apply_operator_proposal",
        "_apply_manual_console_write",
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
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_promotion_executor_dry_run_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_executor_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_executor_handler_invoked"] is False
    assert consumer["summary"]["promotion_executor_created"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_executor_request"]["handler"] == (
        "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert "executor_dry_run" not in consumer["promotion_executor_request"]


def test_promotion_executor_dry_run_result_consumer_emits_executor_request_template(
    tmp_path,
):
    execution_plan_consumer = _promotion_execution_plan_consumer(tmp_path)
    dry_run = _promotion_executor_dry_run(execution_plan_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_request_available"
    )
    assert consumer["summary"]["expected_executor_dry_run_record_count"] == 1
    assert consumer["summary"]["executor_dry_run_record_count"] == 1
    assert consumer["summary"]["promotion_acceptance_record_template_count"] == 1
    assert consumer["summary"]["memory_trust_review_request_spec_count"] == 2
    assert consumer["summary"]["would_change_policy_state_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "fresh"
    assert consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] == [
        "proposal-first"
    ]
    assert consumer["summary"]["memory_trust_review_request_spec_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["promotion_executor_request_available"] is True
    assert consumer["summary"]["promotion_executor_handler_invoked"] is False
    assert consumer["summary"]["promotion_executor_created"] is False
    assert consumer["summary"]["target_policy_store"] == "operator_sequence_policy_trust_gate"
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["model_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_executor_request"]
    assert request["handler"] == (
        "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert request["read_only_template"] is True
    assert request["promotion_executor_handler_invoked"] is False
    assert request["promotion_executor_created"] is False
    assert request["apply"] is False
    assert request["operator_ack"] is False
    assert request["policy_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["executor_dry_run"]["status"] == (
        "sequence_memory_trust_promotion_executor_dry_run_ready"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["idempotency_key"] == (
        "seq-policy:proposal-first"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == execution_plan_consumer["promotion_executor_dry_run_request"][
        "promotion_acceptance_record_templates"
    ][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == execution_plan_consumer["promotion_executor_dry_run_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

    executor_preview = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=request["executor_dry_run"],
        target_store={},
        apply=request["apply"],
        operator_ack=request["operator_ack"],
    )
    assert executor_preview["status"] == "sequence_memory_trust_promotion_executor_ready"
    assert executor_preview["summary"]["apply_requested"] is False
    assert executor_preview["summary"]["operator_ack"] is False
    assert executor_preview["summary"]["sequence_policy_promotion_applied"] is False
    assert executor_preview["summary"]["runtime_memory_mutation"] is False
    assert executor_preview["summary"]["sequence_policy_trust_store_mutation"] is False
    assert executor_preview["summary"]["live_authority_granted"] is False


def test_promotion_executor_dry_run_result_consumer_rejects_forged_dry_run(
    tmp_path,
):
    execution_plan_consumer = _promotion_execution_plan_consumer(tmp_path)
    forged = deepcopy(_promotion_executor_dry_run(execution_plan_consumer))
    forged["safety"]["sequence_policy_promotion_allowed_by_dry_run"] = True
    forged["summary"]["dry_run_record_count"] = 2
    forged["summary"]["sequence_policy_promotion_allowed_by_dry_run"] = True
    forged["dry_run_records"][0]["proposal_id"] = "forged"
    forged["dry_run_records"][0]["policy_mutation_allowed_by_dry_run"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_executor_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_sequence_policy_promotion_allowed_by_dry_run_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_dry_run_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_sequence_policy_promotion_allowed_by_dry_run_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(
        item.endswith(":policy_mutation_allowed_by_dry_run_not_false")
        for item in invalid
    )


def test_promotion_executor_dry_run_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    execution_plan_consumer = deepcopy(_promotion_execution_plan_consumer(tmp_path))
    dry_run = _promotion_executor_dry_run(execution_plan_consumer)
    execution_plan_consumer["summary"]["execution_candidate_count"] = 2
    execution_plan_consumer["summary"][
        "promotion_acceptance_record_template_count"
    ] = 2
    execution_plan_consumer["summary"][
        "promotion_acceptance_record_template_proposal_ids"
    ] = ["proposal-second"]
    execution_plan_consumer["safety"]["live_authority_granted"] = True
    execution_plan_consumer["promotion_executor_dry_run_request"]["apply"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_execution_candidate_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_template_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_request_apply_not_false"
        in invalid
    )


def test_promotion_executor_dry_run_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    execution_plan_consumer = deepcopy(_promotion_execution_plan_consumer(tmp_path))
    dry_run = _promotion_executor_dry_run(execution_plan_consumer)
    template = execution_plan_consumer["promotion_executor_dry_run_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["idempotency_key"] = "seq-policy:forged"
    template["consumer_record_id"] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_executor_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:idempotency_key_candidate_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "executor_dry_run" not in consumer["promotion_executor_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_executor_request"]
    )


def test_promotion_executor_dry_run_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    execution_plan_consumer = deepcopy(_promotion_execution_plan_consumer(tmp_path))
    dry_run = _promotion_executor_dry_run(execution_plan_consumer)
    request = execution_plan_consumer["promotion_executor_dry_run_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    execution_plan_consumer["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_executor_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_dry_run_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first:current_sequence_review_freshness_missing" in invalid
    assert "executor_dry_run" not in consumer["promotion_executor_request"]
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_executor_request"]
    )


def test_promotion_executor_dry_run_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    execution_plan_consumer = deepcopy(_promotion_execution_plan_consumer(tmp_path))
    dry_run = _promotion_executor_dry_run(execution_plan_consumer)
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
    request = execution_plan_consumer["promotion_executor_dry_run_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_executor_request_available"] is False
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
    assert "executor_dry_run" not in consumer["promotion_executor_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_executor_request"]
    )


def test_promotion_executor_dry_run_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["summary"]["promotion_executor_request_available"] is False


def test_promotion_executor_dry_run_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
