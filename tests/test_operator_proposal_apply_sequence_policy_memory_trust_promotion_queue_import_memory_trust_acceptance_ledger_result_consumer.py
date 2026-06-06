from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
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
)
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
    _memory_index_result_consumer,
)


def _trust_review_consumer(tmp_path):
    index_consumer = _memory_index_result_consumer(tmp_path)
    request = index_consumer["memory_trust_review_request"]
    trust_review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=request["memory_index"],
        current_sequence_review_freshness=request[
            "current_sequence_review_freshness"
        ],
    )
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
            memory_index_result_consumer=index_consumer,
            memory_trust_review_result=trust_review,
        )
    )


def _acceptance_record(template, *, index=0, decision="accepted", **overrides):
    record = {
        "acceptance_id": f"sequence-memory-trust-acceptance-{index + 1}",
        "review_candidate_id": template["review_candidate_id"],
        "memory_record_id": template["memory_record_id"],
        "proposal_id": template["proposal_id"],
        "learning_scope": template["learning_scope"],
        "value_type": template["value_type"],
        "idempotency_key": template["idempotency_key"],
        "decision": decision,
        "reviewed_by": "operator",
        "reviewed_at": f"2026-06-06T12:3{index}:00Z",
    }
    record.update(overrides)
    return record


def _acceptance_ledger(trust_review_consumer, records):
    request = trust_review_consumer["acceptance_ledger_request"]
    return build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=request["memory_trust_review"],
        acceptance_records=records,
    )


def _accepted_rejected_ledger(trust_review_consumer):
    templates = trust_review_consumer["acceptance_record_templates"]
    return _acceptance_ledger(
        trust_review_consumer,
        [
            _acceptance_record(templates[0], index=0, decision="accepted"),
            _acceptance_record(templates[1], index=1, decision="rejected"),
        ],
    )


def test_memory_trust_acceptance_ledger_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer.py"
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


def test_memory_trust_acceptance_ledger_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_plan_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_plan_handler_invoked"] is False
    assert consumer["summary"]["promotion_plan_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_plan_request"]["handler"] == (
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert "acceptance_ledger" not in consumer["promotion_plan_request"]


def test_memory_trust_acceptance_ledger_result_consumer_emits_promotion_plan_request(tmp_path):
    trust_review_consumer = _trust_review_consumer(tmp_path)
    ledger = _accepted_rejected_ledger(trust_review_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
            memory_trust_review_result_consumer=trust_review_consumer,
            memory_trust_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available"
    )
    assert consumer["summary"]["expected_trust_review_candidate_count"] == 2
    assert consumer["summary"]["acceptance_record_count"] == 2
    assert consumer["summary"]["accepted_record_count"] == 1
    assert consumer["summary"]["rejected_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["unreviewed_candidate_count"] == 0
    assert consumer["summary"]["promotion_plan_request_available"] is True
    assert consumer["summary"]["promotion_plan_handler_invoked"] is False
    assert consumer["summary"]["promotion_plan_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["model_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    request = consumer["promotion_plan_request"]
    assert request["handler"] == (
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    )
    assert request["read_only_template"] is True
    assert request["promotion_plan_handler_invoked"] is False
    assert request["promotion_plan_created"] is False
    assert request["acceptance_ledger"]["status"] == (
        "operator_sequence_memory_trust_acceptance_records_available"
    )
    assert request["execution_allowed_by_consumer"] is False
    assert request["policy_mutation"] is False
    assert request["live_authority_granted"] is False

    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=request["acceptance_ledger"],
    )
    assert plan["status"] == "sequence_memory_trust_promotion_plan_candidate_available"
    assert plan["summary"]["promotion_candidate_count"] == 1
    assert plan["summary"]["accepted_proposal_ids"] == ["proposal-first"]
    assert plan["summary"]["rejected_proposal_ids"] == ["proposal-second"]
    assert plan["summary"]["sequence_policy_promotion_allowed_by_plan"] is False
    assert plan["summary"]["execution_allowed_by_plan"] is False


def test_memory_trust_acceptance_ledger_result_consumer_blocks_unreviewed_ledger(tmp_path):
    trust_review_consumer = _trust_review_consumer(tmp_path)
    ledger = _acceptance_ledger(trust_review_consumer, [])

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
            memory_trust_review_result_consumer=trust_review_consumer,
            memory_trust_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_blocked"
    )
    assert consumer["summary"]["promotion_plan_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["unreviewed_candidate_count"] == 2
    assert (
        "operator_must_supply_accept_or_reject_records_before_promotion_plan"
        in consumer["summary"]["blocked_by"]
    )
    assert "acceptance_ledger" not in consumer["promotion_plan_request"]


def test_memory_trust_acceptance_ledger_result_consumer_blocks_partially_reviewed_ledger(
    tmp_path,
):
    trust_review_consumer = _trust_review_consumer(tmp_path)
    templates = trust_review_consumer["acceptance_record_templates"]
    ledger = _acceptance_ledger(
        trust_review_consumer,
        [_acceptance_record(templates[0], index=0, decision="accepted")],
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
            memory_trust_review_result_consumer=trust_review_consumer,
            memory_trust_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_blocked"
    )
    assert consumer["summary"]["acceptance_record_count"] == 1
    assert consumer["summary"]["accepted_record_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["unreviewed_candidate_count"] == 1
    assert consumer["summary"]["promotion_plan_request_available"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert (
        "operator_must_supply_accept_or_reject_records_before_promotion_plan"
        in consumer["summary"]["blocked_by"]
    )
    assert "acceptance_ledger" not in consumer["promotion_plan_request"]


def test_memory_trust_acceptance_ledger_result_consumer_rejects_forged_ledger(tmp_path):
    trust_review_consumer = _trust_review_consumer(tmp_path)
    forged = deepcopy(_accepted_rejected_ledger(trust_review_consumer))
    forged["safety"]["policy_mutation"] = True
    forged["summary"]["accepted_record_count"] = 2
    forged["summary"]["sequence_policy_promotion_allowed_by_acceptance"] = True
    forged["records"][0]["proposal_id"] = "forged"
    forged["records"][0]["execution_allowed_by_acceptance"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
            memory_trust_review_result_consumer=trust_review_consumer,
            memory_trust_acceptance_ledger_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_policy_mutation_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_accepted_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_sequence_policy_promotion_allowed_by_acceptance_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(
        item.endswith(":execution_allowed_by_acceptance_not_false")
        for item in invalid
    )
    assert "acceptance_ledger" not in consumer["promotion_plan_request"]


def test_memory_trust_acceptance_ledger_result_consumer_rejects_stale_consumer(tmp_path):
    trust_review_consumer = deepcopy(_trust_review_consumer(tmp_path))
    ledger = _accepted_rejected_ledger(trust_review_consumer)
    trust_review_consumer["summary"]["trust_review_candidate_proposal_ids"] = [
        "proposal-first"
    ]
    trust_review_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
            memory_trust_review_result_consumer=trust_review_consumer,
            memory_trust_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_proposal_ids_mismatch"
        in invalid
    )


def test_memory_trust_acceptance_ledger_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_plan_request_available"] is False


def test_memory_trust_acceptance_ledger_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
