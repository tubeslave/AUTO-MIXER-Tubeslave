from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
    build_operator_proposal_apply_sequence_policy_memory_trust_review,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
    _freshness,
    _memory_index,
    _memory_writer_apply_result_consumer,
)


def _memory_index_result_consumer(tmp_path):
    apply_consumer = _memory_writer_apply_result_consumer()
    index = _memory_index(apply_consumer, root=tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
            memory_writer_apply_result_consumer=apply_consumer,
            memory_index_result=index,
            current_sequence_review_freshness=_freshness(),
        )
    )


def _memory_trust_review(index_consumer):
    request = index_consumer["memory_trust_review_request"]
    return build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=request["memory_index"],
        current_sequence_review_freshness=request[
            "current_sequence_review_freshness"
        ],
    )


def test_memory_trust_review_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer.py"
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


def test_memory_trust_review_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_not_ready"
    )
    assert consumer["summary"]["acceptance_ledger_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["acceptance_ledger_handler_invoked"] is False
    assert consumer["summary"]["acceptance_ledger_created"] is False
    assert consumer["summary"]["operator_acceptance_required"] is True
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["acceptance_ledger_request"]["handler"] == (
        "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance"
    )
    assert consumer["acceptance_ledger_request"]["acceptance_records"] == []
    assert consumer["acceptance_record_templates"] == []


def test_memory_trust_review_result_consumer_emits_acceptance_request(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    trust_review = _memory_trust_review(index_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=index_consumer,
            memory_trust_review_result=trust_review,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_request_available"
    )
    assert consumer["summary"]["expected_trust_review_candidate_count"] == 2
    assert consumer["summary"]["trust_review_candidate_count"] == 2
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["acceptance_ledger_request_available"] is True
    assert consumer["summary"]["acceptance_ledger_handler_invoked"] is False
    assert consumer["summary"]["acceptance_ledger_created"] is False
    assert consumer["summary"]["operator_acceptance_required"] is True
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    request = consumer["acceptance_ledger_request"]
    assert request["handler"] == (
        "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance"
    )
    assert request["read_only_template"] is True
    assert request["acceptance_ledger_handler_invoked"] is False
    assert request["acceptance_ledger_created"] is False
    assert request["operator_must_supply_acceptance_records"] is True
    assert request["acceptance_records"] == []
    assert request["memory_trust_review"]["status"] == (
        "sequence_memory_trust_review_candidate_available"
    )
    assert request["memory_trust_review_request_specs"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ]["status"] == "fresh"
    template = consumer["acceptance_record_templates"][0]
    assert template["proposal_id"] == "proposal-first"
    assert template["source_memory_trust_review_request_spec_id"].endswith(":proposal-first")
    assert template["source_memory_index_request_spec_id"].endswith(":proposal-first")
    assert template["source_memory_writer_apply_request_spec_id"].endswith(":proposal-first")
    assert template["source_memory_writer_request_spec_id"].endswith(":proposal-first")
    assert template["source_memory_draft_request_spec_id"].endswith(":proposal-first")
    assert template["source_audit_policy_gate_request_spec_id"].endswith(":proposal-first")
    assert template["source_audit_request_spec_id"].endswith(":proposal-first")
    assert template["source_executor_request_spec_id"].endswith(":proposal-first")
    assert template["source_execution_gate_request_spec_id"].endswith(":proposal-first")
    assert template["source_dry_run_request_spec_id"].endswith(":proposal-first")
    assert template["source_sequence_plan_request_spec_id"].endswith(":proposal-first")
    assert template["source_preflight_request_spec_id"].endswith(":proposal-first")
    assert template["queue_ordering_plan_record_id"] == "plan:proposal-first"
    assert template["consumer_record_id"] == "consumer:proposal-first"
    assert template["sequence_policy_gate_record_id"] == "gate:proposal-first"
    assert template["policy_key"] == "proposal-first/memory-write:proposal-first"
    assert template["source_idempotency_key"] == "memory-write:proposal-first"
    assert template["current_sequence_review_freshness"] == index_consumer[
        "memory_trust_review_request_specs"
    ][0]["current_sequence_review_freshness"]
    assert template["rank_source"] == "sequence_policy_consumer"
    assert template["decision_options"] == ["accepted", "rejected"]
    assert template["operator_must_choose_decision"] is True
    assert template["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert template["execution_allowed_by_acceptance"] is False
    assert template["live_authority_granted"] is False


def test_memory_trust_review_result_consumer_rejects_not_ready_review(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    request = index_consumer["memory_trust_review_request"]
    not_ready_review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=request["memory_index"],
        current_sequence_review_freshness={**_freshness("stale")},
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=index_consumer,
            memory_trust_review_result=not_ready_review,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
    )
    assert consumer["summary"]["acceptance_ledger_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_trust_review_result_consumer_review_status_not_available"
        in invalid
    )
    assert (
        "sequence_policy_memory_trust_review_result_consumer_review_summary_freshness_not_fresh"
        in invalid
    )
    assert consumer["acceptance_record_templates"] == []


def test_memory_trust_review_result_consumer_rejects_forged_review_request_identity(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    forged_consumer = deepcopy(index_consumer)
    forged_consumer["memory_trust_review_request_specs"][0][
        "source_idempotency_key"
    ] = "forged-memory-write:proposal-first"
    forged_consumer["memory_trust_review_request_specs"][1]["consumer_record_id"] = ""
    trust_review = _memory_trust_review(index_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=forged_consumer,
            memory_trust_review_result=trust_review,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
    )
    assert consumer["summary"]["acceptance_ledger_request_available"] is False
    assert (
        "proposal-first:policy_key_identity_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:consumer_record_id_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["acceptance_record_templates"] == []


def test_memory_trust_review_result_consumer_rejects_missing_sequence_context(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    trust_review = _memory_trust_review(index_consumer)
    for spec in index_consumer["memory_trust_review_request_specs"]:
        spec.pop("current_sequence_review_freshness", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=index_consumer,
            memory_trust_review_result=trust_review,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
    )
    assert consumer["summary"]["acceptance_ledger_request_available"] is False
    assert (
        "proposal-first:current_sequence_review_freshness_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:current_sequence_review_freshness_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["acceptance_record_templates"] == []


def test_memory_trust_review_result_consumer_rejects_live_sequence_context(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    trust_review = _memory_trust_review(index_consumer)
    index_consumer["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = {
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

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=index_consumer,
            memory_trust_review_result=trust_review,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
    )
    assert consumer["summary"]["acceptance_ledger_request_available"] is False
    assert (
        "proposal-first:current_sequence_review_freshness_read_only_not_true"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:current_sequence_review_freshness_live_write_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:current_sequence_review_freshness_apply_attempted_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["acceptance_record_templates"] == []


def test_memory_trust_review_result_consumer_rejects_forged_review(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    forged = deepcopy(_memory_trust_review(index_consumer))
    forged["summary"]["policy_mutation"] = True
    forged["summary"]["trust_review_candidate_count"] = 3
    forged["safety"]["live_authority_granted"] = True
    forged["trust_review_candidates"][0]["proposal_id"] = "forged"
    forged["trust_review_candidates"][1]["execution_allowed_by_review"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=index_consumer,
            memory_trust_review_result=forged,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
    )
    assert consumer["summary"]["acceptance_ledger_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_trust_review_result_consumer_review_safety_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_trust_review_result_consumer_review_summary_trust_review_candidate_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_memory_trust_review_result_consumer_review_summary_policy_mutation_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(item.endswith(":execution_allowed_by_review_not_false") for item in invalid)
    assert consumer["acceptance_record_templates"] == []


def test_memory_trust_review_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["acceptance_ledger_request_available"] is False


def test_memory_trust_review_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
