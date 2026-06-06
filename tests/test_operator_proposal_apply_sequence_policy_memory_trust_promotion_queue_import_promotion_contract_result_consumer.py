from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
    _acceptance_ledger_consumer,
    _promotion_plan,
    _rollback_evidence,
    _sequence_context,
)


def _promotion_plan_consumer(tmp_path):
    acceptance_consumer = _acceptance_ledger_consumer(tmp_path)
    plan = _promotion_plan(acceptance_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer(
            memory_trust_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_plan_result=plan,
        )
    )


def _promotion_contract(plan_consumer, **overrides):
    request = plan_consumer["promotion_contract_request"]
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=request["promotion_plan"],
        current_sequence_policy_context=overrides.pop(
            "current_sequence_policy_context",
            _sequence_context(),
        ),
        rollback_evidence=overrides.pop("rollback_evidence", _rollback_evidence()),
    )


def _promotion_acceptance_record(template, *, decision="accepted", **overrides):
    record = {
        "promotion_acceptance_id": "sequence-memory-promotion-acceptance-1",
        "promotion_contract_record_id": template["promotion_contract_record_id"],
        "promotion_candidate_id": template["promotion_candidate_id"],
        "proposal_id": template["proposal_id"],
        "learning_scope": template["learning_scope"],
        "value_type": template["value_type"],
        "idempotency_key": template["idempotency_key"],
        "previous_policy_state": template["previous_policy_state"],
        "rollback_strategy": template["rollback_strategy"],
        "decision": decision,
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T14:30:00Z",
    }
    record.update(overrides)
    return record


def test_promotion_contract_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py"
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


def test_promotion_contract_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_acceptance_ledger_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_acceptance_ledger_handler_invoked"] is False
    assert consumer["summary"]["promotion_acceptance_ledger_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_acceptance_ledger_request"]["handler"] == (
        "record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance"
    )
    assert consumer["promotion_acceptance_ledger_request"]["acceptance_records"] == []
    assert consumer["promotion_acceptance_record_templates"] == []


def test_promotion_contract_result_consumer_emits_acceptance_request(tmp_path):
    plan_consumer = _promotion_plan_consumer(tmp_path)
    contract = _promotion_contract(plan_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
            promotion_plan_result_consumer=plan_consumer,
            promotion_contract_result=contract,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_request_available"
    )
    assert consumer["summary"]["expected_contract_record_count"] == 1
    assert consumer["summary"]["contract_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_acceptance_ledger_request_available"] is True
    assert consumer["summary"]["promotion_acceptance_ledger_handler_invoked"] is False
    assert consumer["summary"]["promotion_acceptance_ledger_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["model_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    request = consumer["promotion_acceptance_ledger_request"]
    assert request["handler"] == (
        "record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance"
    )
    assert request["read_only_template"] is True
    assert request["promotion_acceptance_ledger_handler_invoked"] is False
    assert request["promotion_acceptance_ledger_created"] is False
    assert request["operator_must_supply_promotion_acceptance_records"] is True
    assert request["acceptance_records"] == []
    assert request["promotion_contract"]["status"] == (
        "manual_sequence_memory_trust_promotion_contract_ready"
    )
    template = consumer["promotion_acceptance_record_templates"][0]
    assert template["proposal_id"] == "proposal-first"
    assert template["decision_options"] == ["accepted", "rejected"]
    assert template["operator_must_choose_decision"] is True
    assert template["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert template["execution_allowed_by_acceptance"] is False
    assert template["live_authority_granted"] is False

    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=request["promotion_contract"],
        acceptance_records=[_promotion_acceptance_record(template)],
    )
    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_records_available"
    )
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert ledger["summary"]["execution_allowed_by_acceptance"] is False


def test_promotion_contract_result_consumer_rejects_not_ready_contract(tmp_path):
    plan_consumer = _promotion_plan_consumer(tmp_path)
    contract = _promotion_contract(
        plan_consumer,
        current_sequence_policy_context=_sequence_context(status="stale"),
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
            promotion_plan_result_consumer=plan_consumer,
            promotion_contract_result=contract,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_acceptance_ledger_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "sequence_policy_promotion_contract_result_consumer_contract_status_not_ready" in invalid
    assert (
        "sequence_policy_promotion_contract_result_consumer_contract_summary_context_not_fresh"
        in invalid
    )
    assert (
        "sequence_policy_promotion_contract_result_consumer_record_count_mismatch"
        in invalid
    )


def test_promotion_contract_result_consumer_rejects_forged_contract(tmp_path):
    plan_consumer = _promotion_plan_consumer(tmp_path)
    forged = deepcopy(_promotion_contract(plan_consumer))
    forged["safety"]["policy_mutation_allowed_by_contract"] = True
    forged["summary"]["contract_record_count"] = 2
    forged["summary"]["sequence_policy_promotion_allowed_by_contract"] = True
    forged["contract_records"][0]["proposal_id"] = "forged"
    forged["contract_records"][0]["execution_allowed_by_contract"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
            promotion_plan_result_consumer=plan_consumer,
            promotion_contract_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_acceptance_ledger_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_contract_result_consumer_contract_policy_mutation_allowed_by_contract_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_contract_result_consumer_contract_summary_contract_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_contract_result_consumer_contract_summary_sequence_policy_promotion_allowed_by_contract_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(
        item.endswith(":execution_allowed_by_contract_not_false")
        for item in invalid
    )
    assert "promotion_contract" not in consumer["promotion_acceptance_ledger_request"]


def test_promotion_contract_result_consumer_rejects_forged_prior_consumer(tmp_path):
    plan_consumer = deepcopy(_promotion_plan_consumer(tmp_path))
    contract = _promotion_contract(plan_consumer)
    plan_consumer["summary"]["promotion_candidate_count"] = 2
    plan_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
            promotion_plan_result_consumer=plan_consumer,
            promotion_contract_result=contract,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_contract_result_consumer_plan_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_contract_result_consumer_plan_consumer_summary_promotion_candidate_count_mismatch"
        in invalid
    )


def test_promotion_contract_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_acceptance_ledger_request_available"] is False


def test_promotion_contract_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
