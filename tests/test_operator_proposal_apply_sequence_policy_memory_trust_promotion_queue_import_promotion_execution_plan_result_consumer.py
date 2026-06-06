from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
    _promotion_acceptance_ledger,
    _promotion_contract_consumer,
)


def _promotion_acceptance_consumer(tmp_path):
    contract_consumer = _promotion_contract_consumer(tmp_path)
    ledger = _promotion_acceptance_ledger(contract_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )


def _promotion_execution_plan(acceptance_consumer):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=acceptance_consumer[
            "promotion_execution_plan_request"
        ]["promotion_acceptance_ledger"],
    )


def test_promotion_execution_plan_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer.py"
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
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_promotion_execution_plan_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_executor_dry_run_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_executor_dry_run_handler_invoked"] is False
    assert consumer["summary"]["promotion_executor_dry_run_created"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_executor_dry_run_request"]["handler"] == (
        "dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
    )
    assert "promotion_execution_plan" not in consumer[
        "promotion_executor_dry_run_request"
    ]


def test_promotion_execution_plan_result_consumer_emits_executor_dry_run_request(
    tmp_path,
):
    acceptance_consumer = _promotion_acceptance_consumer(tmp_path)
    execution_plan = _promotion_execution_plan(acceptance_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer(
            promotion_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_execution_plan_result=execution_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_request_available"
    )
    assert consumer["summary"]["expected_execution_candidate_count"] == 1
    assert consumer["summary"]["execution_candidate_count"] == 1
    assert consumer["summary"]["accepted_record_count"] == 1
    assert consumer["summary"]["rejected_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_executor_dry_run_request_available"] is True
    assert consumer["summary"]["promotion_executor_dry_run_handler_invoked"] is False
    assert consumer["summary"]["promotion_executor_dry_run_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["model_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_executor_dry_run_request"]
    assert request["handler"] == (
        "dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
    )
    assert request["read_only_template"] is True
    assert request["promotion_executor_dry_run_handler_invoked"] is False
    assert request["promotion_executor_dry_run_created"] is False
    assert request["apply"] is False
    assert request["operator_ack"] is False
    assert request["policy_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["promotion_execution_plan"]["status"] == (
        "manual_sequence_memory_trust_promotion_execution_candidate_available"
    )

    dry_run = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=request["promotion_execution_plan"],
    )
    assert dry_run["status"] == "sequence_memory_trust_promotion_executor_dry_run_ready"
    assert dry_run["summary"]["dry_run_record_count"] == 1
    assert dry_run["summary"]["dry_run"] is True
    assert dry_run["summary"]["execution_allowed_by_dry_run"] is False
    assert dry_run["summary"]["sequence_policy_promotion_applied"] is False
    assert (
        dry_run["summary"]["sequence_policy_promotion_allowed_by_dry_run"]
        is False
    )
    assert dry_run["summary"]["policy_mutation_allowed_by_dry_run"] is False
    assert dry_run["summary"]["runtime_memory_mutation_allowed_by_dry_run"] is False
    assert dry_run["summary"]["live_authority_granted"] is False


def test_promotion_execution_plan_result_consumer_rejects_forged_plan(tmp_path):
    acceptance_consumer = _promotion_acceptance_consumer(tmp_path)
    forged = deepcopy(_promotion_execution_plan(acceptance_consumer))
    forged["safety"]["sequence_policy_promotion_allowed_by_plan"] = True
    forged["summary"]["execution_candidate_count"] = 2
    forged["summary"]["sequence_policy_promotion_allowed_by_plan"] = True
    forged["execution_candidates"][0]["proposal_id"] = "forged"
    forged["execution_candidates"][0]["policy_mutation_allowed_by_plan"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer(
            promotion_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_execution_plan_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_executor_dry_run_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_execution_plan_result_consumer_plan_sequence_policy_promotion_allowed_by_plan_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_execution_candidate_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_sequence_policy_promotion_allowed_by_plan_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(
        item.endswith(":policy_mutation_allowed_by_plan_not_false")
        for item in invalid
    )


def test_promotion_execution_plan_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    acceptance_consumer = deepcopy(_promotion_acceptance_consumer(tmp_path))
    execution_plan = _promotion_execution_plan(acceptance_consumer)
    acceptance_consumer["summary"]["accepted_record_count"] = 2
    acceptance_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer(
            promotion_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_execution_plan_result=execution_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_summary_accepted_record_count_mismatch"
        in invalid
    )


def test_promotion_execution_plan_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_executor_dry_run_request_available"] is False


def test_promotion_execution_plan_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
