from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
    _promotion_acceptance_record,
    _promotion_contract,
    _promotion_plan_consumer,
)


def _promotion_contract_consumer(tmp_path):
    plan_consumer = _promotion_plan_consumer(tmp_path)
    contract = _promotion_contract(plan_consumer)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
            promotion_plan_result_consumer=plan_consumer,
            promotion_contract_result=contract,
        )
    )


def _promotion_acceptance_ledger(contract_consumer, *, decision="accepted", **overrides):
    request = contract_consumer["promotion_acceptance_ledger_request"]
    template = contract_consumer["promotion_acceptance_record_templates"][0]
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=request["promotion_contract"],
        acceptance_records=[
            _promotion_acceptance_record(template, decision=decision, **overrides)
        ],
    )


def test_promotion_acceptance_ledger_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py"
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


def test_promotion_acceptance_ledger_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_not_ready"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_execution_plan_handler_invoked"] is False
    assert consumer["summary"]["promotion_execution_plan_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["promotion_execution_plan_request"]["handler"] == (
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution"
    )
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]


def test_promotion_acceptance_ledger_result_consumer_emits_execution_plan_request(
    tmp_path,
):
    contract_consumer = _promotion_contract_consumer(tmp_path)
    ledger = _promotion_acceptance_ledger(contract_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_request_available"
    )
    assert consumer["summary"]["expected_promotion_contract_record_count"] == 1
    assert consumer["summary"]["promotion_acceptance_record_template_count"] == 1
    assert consumer["summary"]["promotion_contract_record_count"] == 1
    assert consumer["summary"]["acceptance_record_count"] == 1
    assert consumer["summary"]["accepted_record_count"] == 1
    assert consumer["summary"]["rejected_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "fresh"
    assert consumer["summary"]["promotion_execution_plan_request_available"] is True
    assert consumer["summary"]["promotion_execution_plan_handler_invoked"] is False
    assert consumer["summary"]["promotion_execution_plan_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["model_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] == [
        "proposal-first"
    ]
    request = consumer["promotion_execution_plan_request"]
    assert request["handler"] == (
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution"
    )
    assert request["read_only_template"] is True
    assert request["promotion_execution_plan_handler_invoked"] is False
    assert request["promotion_execution_plan_created"] is False
    assert request["accepted_promotion_record_required"] is True
    assert request["promotion_acceptance_ledger"]["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_records_available"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["policy_key"] == (
        "proposal-first/memory-write:proposal-first"
    )
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == contract_consumer["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == contract_consumer["promotion_acceptance_ledger_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

    execution_plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
            promotion_acceptance_ledger=request["promotion_acceptance_ledger"],
        )
    )
    assert execution_plan["status"] == (
        "manual_sequence_memory_trust_promotion_execution_candidate_available"
    )
    assert execution_plan["summary"]["execution_candidate_count"] == 1
    assert execution_plan["summary"]["execution_allowed_by_plan"] is False
    assert (
        execution_plan["summary"]["sequence_policy_promotion_allowed_by_plan"]
        is False
    )
    assert execution_plan["summary"]["live_authority_granted"] is False


def test_promotion_acceptance_ledger_result_consumer_blocks_rejected_only_ledger(
    tmp_path,
):
    contract_consumer = _promotion_contract_consumer(tmp_path)
    ledger = _promotion_acceptance_ledger(contract_consumer, decision="rejected")

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_blocked"
    )
    assert consumer["summary"]["accepted_record_count"] == 0
    assert consumer["summary"]["rejected_record_count"] == 1
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    assert (
        "accepted_sequence_memory_trust_promotion_record_required_before_execution_plan"
        in consumer["summary"]["blocked_by"]
    )
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]


def test_promotion_acceptance_ledger_result_consumer_rejects_forged_ledger(
    tmp_path,
):
    contract_consumer = _promotion_contract_consumer(tmp_path)
    forged = deepcopy(_promotion_acceptance_ledger(contract_consumer))
    forged["safety"]["policy_mutation_allowed_by_acceptance"] = True
    forged["summary"]["accepted_record_count"] = 2
    forged["summary"]["sequence_policy_promotion_allowed_by_acceptance"] = True
    forged["records"][0]["proposal_id"] = "forged"
    forged["records"][0]["execution_allowed_by_acceptance"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_policy_mutation_allowed_by_acceptance_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_accepted_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_sequence_policy_promotion_allowed_by_acceptance_not_false"
        in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(
        item.endswith(":execution_allowed_by_acceptance_not_false")
        for item in invalid
    )


def test_promotion_acceptance_ledger_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    contract_consumer = deepcopy(_promotion_contract_consumer(tmp_path))
    ledger = _promotion_acceptance_ledger(contract_consumer)
    contract_consumer["summary"]["contract_record_count"] = 2
    contract_consumer["safety"]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_summary_contract_record_count_mismatch"
        in invalid
    )


def test_promotion_acceptance_ledger_result_consumer_rejects_forged_prior_template_summary(
    tmp_path,
):
    contract_consumer = deepcopy(_promotion_contract_consumer(tmp_path))
    ledger = _promotion_acceptance_ledger(contract_consumer)
    contract_consumer["summary"]["promotion_acceptance_record_template_count"] = 2
    contract_consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] = [
        "proposal-second"
    ]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_template_proposal_ids_mismatch"
        in invalid
    )
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]


def test_promotion_acceptance_ledger_result_consumer_rejects_forged_template_identity(
    tmp_path,
):
    contract_consumer = _promotion_contract_consumer(tmp_path)
    ledger = _promotion_acceptance_ledger(contract_consumer)
    forged_consumer = deepcopy(contract_consumer)
    forged_consumer["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] = "forged-memory-write:proposal-first"
    forged_consumer["promotion_acceptance_record_templates"][0][
        "consumer_record_id"
    ] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=forged_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:policy_key_identity_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_execution_plan_request"]
    )


def test_promotion_acceptance_ledger_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    contract_consumer = deepcopy(_promotion_contract_consumer(tmp_path))
    ledger = _promotion_acceptance_ledger(contract_consumer)
    for template in contract_consumer["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    for template in contract_consumer["promotion_acceptance_ledger_request"][
        "promotion_acceptance_record_templates"
    ]:
        template.pop("current_sequence_review_freshness", None)
    contract_consumer["promotion_acceptance_ledger_request"].pop(
        "memory_trust_review_request_specs"
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:current_sequence_review_freshness_missing" in invalid
    assert (
        "sequence_policy_promotion_acceptance_ledger_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_execution_plan_request"]
    )


def test_promotion_acceptance_ledger_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    contract_consumer = deepcopy(_promotion_contract_consumer(tmp_path))
    ledger = _promotion_acceptance_ledger(contract_consumer)
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
    contract_consumer["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    contract_consumer["promotion_acceptance_ledger_request"][
        "promotion_acceptance_record_templates"
    ][0]["current_sequence_review_freshness"] = live_freshness
    contract_consumer["promotion_acceptance_ledger_request"][
        "memory_trust_review_request_specs"
    ][0]["current_sequence_review_freshness"] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=contract_consumer,
            promotion_acceptance_ledger_result=ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
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
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_execution_plan_request"]
    )


def test_promotion_acceptance_ledger_result_consumer_rejects_self_consistent_forged_template(
    tmp_path,
):
    contract_consumer = _promotion_contract_consumer(tmp_path)
    ledger = _promotion_acceptance_ledger(contract_consumer)
    forged_consumer = deepcopy(contract_consumer)
    for templates in (
        forged_consumer["promotion_acceptance_record_templates"],
        forged_consumer["promotion_acceptance_ledger_request"][
            "promotion_acceptance_record_templates"
        ],
    ):
        template = templates[0]
        template["promotion_candidate_id"] = "promotion-candidate-second"
        template["proposal_id"] = "proposal-second"
        template["source_idempotency_key"] = "memory-write:proposal-second"
        template["policy_key"] = "proposal-second/memory-write:proposal-second"
        template["idempotency_key"] = "seq-policy:proposal-second"
    forged_consumer["summary"]["proposal_ids"] = ["proposal-second"]
    forged_consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] = [
        "proposal-second"
    ]
    forged_ledger = deepcopy(ledger)
    record = forged_ledger["records"][0]
    record["promotion_candidate_id"] = "promotion-candidate-second"
    record["proposal_id"] = "proposal-second"
    record["idempotency_key"] = "seq-policy:proposal-second"
    forged_ledger["summary"]["accepted_proposal_ids"] = ["proposal-second"]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
            promotion_contract_result_consumer=forged_consumer,
            promotion_acceptance_ledger_result=forged_ledger,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-second:promotion_candidate_id_contract_record_mismatch" in invalid
    assert "proposal-second:proposal_id_contract_record_mismatch" in invalid
    assert "proposal-second:idempotency_key_contract_record_mismatch" in invalid
    assert "promotion_acceptance_ledger" not in consumer["promotion_execution_plan_request"]
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_execution_plan_request"]
    )


def test_promotion_acceptance_ledger_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_execution_plan_request_available"] is False


def test_promotion_acceptance_ledger_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
