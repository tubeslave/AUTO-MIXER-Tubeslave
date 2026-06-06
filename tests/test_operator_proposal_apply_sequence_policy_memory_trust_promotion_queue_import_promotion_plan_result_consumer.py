from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
    _accepted_rejected_ledger,
    _trust_review_consumer,
)


def _acceptance_ledger_consumer(tmp_path):
    trust_consumer = _trust_review_consumer(tmp_path)
    ledger = _accepted_rejected_ledger(trust_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
            memory_trust_review_result_consumer=trust_consumer,
            memory_trust_acceptance_ledger_result=ledger,
        )
    )


def _promotion_plan(acceptance_consumer):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=acceptance_consumer["promotion_plan_request"][
            "acceptance_ledger"
        ],
    )


def _sequence_context(**overrides):
    payload = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
    }
    payload.update(overrides)
    return payload


def _rollback_evidence(**overrides):
    payload = {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_sequence_policy_state",
        "previous_policy_state": "not_trusted_for_sequence_policy",
        "target_policy_store": "operator_sequence_policy_trust_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T14:00:00Z",
    }
    payload.update(overrides)
    return payload


def test_promotion_plan_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py"
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


def test_promotion_plan_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_plan_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_contract_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_contract_handler_invoked"] is False
    assert consumer["summary"]["promotion_contract_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_contract_request"]["handler"] == (
        "contract_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert "promotion_plan" not in consumer["promotion_contract_request"]


def test_promotion_plan_result_consumer_emits_contract_request(tmp_path):
    acceptance_consumer = _acceptance_ledger_consumer(tmp_path)
    promotion_plan = _promotion_plan(acceptance_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer(
            memory_trust_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_plan_result=promotion_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_contract_request_available"
    )
    assert consumer["summary"]["expected_promotion_candidate_count"] == 1
    assert consumer["summary"]["promotion_candidate_count"] == 1
    assert consumer["summary"]["accepted_record_count"] == 1
    assert consumer["summary"]["rejected_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_contract_request_available"] is True
    assert consumer["summary"]["promotion_contract_handler_invoked"] is False
    assert consumer["summary"]["promotion_contract_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["model_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    request = consumer["promotion_contract_request"]
    assert request["handler"] == (
        "contract_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert request["read_only_template"] is True
    assert request["promotion_contract_handler_invoked"] is False
    assert request["promotion_contract_created"] is False
    assert request["operator_must_supply_current_sequence_policy_context"] is True
    assert request["operator_must_supply_rollback_evidence"] is True
    assert request["promotion_plan"]["status"] == (
        "sequence_memory_trust_promotion_plan_candidate_available"
    )
    assert request["execution_allowed_by_consumer"] is False
    assert request["policy_mutation"] is False
    assert request["live_authority_granted"] is False

    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=request["promotion_plan"],
        current_sequence_policy_context=_sequence_context(),
        rollback_evidence=_rollback_evidence(),
    )
    assert contract["status"] == "manual_sequence_memory_trust_promotion_contract_ready"
    assert contract["summary"]["contract_record_count"] == 1
    assert contract["summary"]["sequence_policy_promotion_allowed_by_contract"] is False
    assert contract["summary"]["execution_allowed_by_contract"] is False


def test_promotion_plan_result_consumer_rejects_not_ready_plan(tmp_path):
    acceptance_consumer = _acceptance_ledger_consumer(tmp_path)
    plan = deepcopy(_promotion_plan(acceptance_consumer))
    plan["status"] = "sequence_memory_trust_promotion_plan_not_ready"
    plan["summary"]["promotion_candidate_count"] = 0
    plan["promotion_candidates"] = []

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer(
            memory_trust_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_plan_result=plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_contract_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "sequence_policy_promotion_plan_result_consumer_plan_status_not_available" in invalid
    assert (
        "sequence_policy_promotion_plan_result_consumer_candidate_count_mismatch"
        in invalid
    )
    assert "sequence-memory-trust-acceptance-1:promotion_candidate_missing" in invalid


def test_promotion_plan_result_consumer_rejects_forged_plan(tmp_path):
    acceptance_consumer = _acceptance_ledger_consumer(tmp_path)
    forged = deepcopy(_promotion_plan(acceptance_consumer))
    forged["safety"]["policy_mutation_allowed_by_plan"] = True
    forged["summary"]["accepted_record_count"] = 2
    forged["summary"]["sequence_policy_promotion_allowed_by_plan"] = True
    forged["promotion_candidates"][0]["proposal_id"] = "forged"
    forged["promotion_candidates"][0]["execution_allowed_by_plan"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer(
            memory_trust_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_plan_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_contract_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_plan_result_consumer_plan_policy_mutation_allowed_by_plan_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_plan_result_consumer_plan_summary_accepted_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_plan_result_consumer_plan_summary_sequence_policy_promotion_allowed_by_plan_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(item.endswith(":execution_allowed_by_plan_not_false") for item in invalid)
    assert "promotion_plan" not in consumer["promotion_contract_request"]


def test_promotion_plan_result_consumer_rejects_forged_prior_consumer(tmp_path):
    acceptance_consumer = deepcopy(_acceptance_ledger_consumer(tmp_path))
    promotion_plan = _promotion_plan(acceptance_consumer)
    acceptance_consumer["summary"]["accepted_record_count"] = 2
    acceptance_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer(
            memory_trust_acceptance_ledger_result_consumer=acceptance_consumer,
            promotion_plan_result=promotion_plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_plan_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_plan_result_consumer_acceptance_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_plan_result_consumer_acceptance_consumer_summary_accepted_record_count_mismatch"
        in invalid
    )


def test_promotion_plan_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_contract_request_available"] is False


def test_promotion_plan_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
