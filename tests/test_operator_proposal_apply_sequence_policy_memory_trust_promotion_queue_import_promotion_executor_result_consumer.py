from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
    _promotion_execution_plan_consumer,
    _promotion_executor_dry_run,
)


def _promotion_executor_dry_run_consumer(tmp_path):
    execution_plan_consumer = _promotion_execution_plan_consumer(tmp_path)
    dry_run = _promotion_executor_dry_run(execution_plan_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
            promotion_execution_plan_result_consumer=execution_plan_consumer,
            promotion_executor_dry_run_result=dry_run,
        )
    )


def _promotion_executor_preview(dry_run_consumer):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=dry_run_consumer["promotion_executor_request"][
            "executor_dry_run"
        ],
    )


def _promotion_executor_applied(dry_run_consumer):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=dry_run_consumer["promotion_executor_request"][
            "executor_dry_run"
        ],
        target_store={},
        apply=True,
        operator_ack=True,
    )


def test_promotion_executor_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer.py"
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


def test_promotion_executor_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_apply_audit_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_apply_audit_handler_invoked"] is False
    assert consumer["summary"]["promotion_apply_audit_created"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_apply_audit_request"]["handler"] == (
        "audit_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert "promotion_executor" not in consumer["promotion_apply_audit_request"]


def test_promotion_executor_result_consumer_blocks_preview_executor(tmp_path):
    dry_run_consumer = _promotion_executor_dry_run_consumer(tmp_path)
    preview = _promotion_executor_preview(dry_run_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
            promotion_executor_dry_run_result_consumer=dry_run_consumer,
            promotion_executor_result=preview,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_blocked"
    )
    assert consumer["summary"]["executor_apply_result_count"] == 0
    assert consumer["summary"]["promotion_apply_audit_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        "applied_sequence_memory_trust_promotion_executor_result_required_before_apply_audit"
        in consumer["summary"]["blocked_by"]
    )
    assert "promotion_executor" not in consumer["promotion_apply_audit_request"]


def test_promotion_executor_result_consumer_emits_apply_audit_request(tmp_path):
    dry_run_consumer = _promotion_executor_dry_run_consumer(tmp_path)
    applied = _promotion_executor_applied(dry_run_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
            promotion_executor_dry_run_result_consumer=dry_run_consumer,
            promotion_executor_result=applied,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_request_available"
    )
    assert consumer["summary"]["expected_executor_apply_result_count"] == 1
    assert consumer["summary"]["executor_apply_result_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_apply_audit_request_available"] is True
    assert consumer["summary"]["promotion_apply_audit_handler_invoked"] is False
    assert consumer["summary"]["promotion_apply_audit_created"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_apply_audit_request"]
    assert request["handler"] == (
        "audit_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert request["read_only_template"] is True
    assert request["promotion_apply_audit_handler_invoked"] is False
    assert request["promotion_apply_audit_created"] is False
    assert request["policy_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["promotion_executor"]["status"] == (
        "sequence_memory_trust_promotion_executor_applied"
    )

    audit = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
        promotion_executor=request["promotion_executor"],
    )
    assert audit["status"] == "sequence_memory_trust_promotion_apply_audit_records_available"
    assert audit["summary"]["executor_apply_result_count"] == 1
    assert audit["summary"]["apply_audit_record_count"] == 1
    assert audit["summary"]["runtime_memory_mutation"] is False
    assert audit["summary"]["sequence_policy_trust_store_mutation"] is False
    assert audit["summary"]["live_authority_granted"] is False
    assert audit["summary"]["rollback_verified"] is False


def test_promotion_executor_result_consumer_rejects_forged_executor(tmp_path):
    dry_run_consumer = _promotion_executor_dry_run_consumer(tmp_path)
    forged = deepcopy(_promotion_executor_applied(dry_run_consumer))
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["apply_result_count"] = 2
    forged["summary"]["runtime_memory_mutation"] = False
    forged["apply_results"][0]["proposal_id"] = "forged"
    forged["apply_results"][0]["target_policy_path"] = "forged/path"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
            promotion_executor_dry_run_result_consumer=dry_run_consumer,
            promotion_executor_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_apply_audit_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_executor_result_consumer_executor_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_result_consumer_executor_summary_apply_result_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_result_consumer_executor_summary_runtime_memory_mutation_mismatch"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(item.endswith(":target_policy_path_mismatch") for item in invalid)


def test_promotion_executor_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    dry_run_consumer = deepcopy(_promotion_executor_dry_run_consumer(tmp_path))
    applied = _promotion_executor_applied(dry_run_consumer)
    dry_run_consumer["summary"]["executor_dry_run_record_count"] = 2
    dry_run_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
            promotion_executor_dry_run_result_consumer=dry_run_consumer,
            promotion_executor_result=applied,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_summary_executor_dry_run_record_count_mismatch"
        in invalid
    )


def test_promotion_executor_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_apply_audit_request_available"] is False


def test_promotion_executor_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
