from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer,
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
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
    _freshness,
    _memory_index,
    _promotion_writer_apply_result_consumer_ready,
)


def _promotion_memory_index_result_consumer_ready(tmp_path):
    apply_consumer = _promotion_writer_apply_result_consumer_ready(tmp_path)
    index = _memory_index(apply_consumer, root=tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer(
            promotion_memory_writer_apply_result_consumer=apply_consumer,
            promotion_memory_index_result=index,
            current_sequence_review_freshness=_freshness(),
        )
    )


def _memory_trust_review(index_consumer):
    request = index_consumer["promotion_memory_trust_review_request"]
    return build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=request["memory_index"],
        current_sequence_review_freshness=request[
            "current_sequence_review_freshness"
        ],
    )


def test_promotion_memory_trust_review_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer.py"
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
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_promotion_memory_trust_review_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["acceptance_ledger_handler_invoked"] is False
    assert consumer["safety"]["acceptance_ledger_created"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["safety"]["live_write"] is False
    assert (
        consumer["summary"]["promotion_acceptance_ledger_request_available"]
        is False
    )
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_acceptance_ledger_request"]["handler"] == (
        "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance"
    )
    assert consumer["promotion_acceptance_ledger_request"]["acceptance_records"] == []
    assert consumer["promotion_acceptance_record_templates"] == []


def test_promotion_memory_trust_review_result_consumer_emits_acceptance_request(
    tmp_path,
):
    index_consumer = _promotion_memory_index_result_consumer_ready(tmp_path)
    trust_review = _memory_trust_review(index_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer(
            promotion_memory_index_result_consumer=index_consumer,
            promotion_memory_trust_review_result=trust_review,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_request_available"
    )
    assert consumer["summary"]["expected_trust_review_candidate_count"] == 1
    assert consumer["summary"]["trust_review_candidate_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        consumer["summary"]["promotion_acceptance_ledger_request_available"]
        is True
    )
    assert consumer["summary"]["acceptance_ledger_handler_invoked"] is False
    assert consumer["summary"]["acceptance_ledger_created"] is False
    assert consumer["summary"]["operator_acceptance_required"] is True
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["generic_memory_trust_review_result_consumer"]["summary"][
        "acceptance_ledger_request_available"
    ] is True

    request = consumer["promotion_acceptance_ledger_request"]
    assert request["handler"] == (
        "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance"
    )
    assert request["read_only_template"] is True
    assert request["promotion_memory_trust_review_result_consumer_only"] is True
    assert request["promotion_acceptance_ledger_request_template_only"] is True
    assert request["acceptance_ledger_handler_invoked"] is False
    assert request["acceptance_ledger_created"] is False
    assert request["operator_must_supply_acceptance_records"] is True
    assert request["acceptance_records"] == []
    assert request["memory_trust_review"]["status"] == (
        "sequence_memory_trust_review_candidate_available"
    )
    assert request["memory_file_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["promotion_memory_trust_review_result_consumer"][
        "trust_review_candidate_proposal_ids"
    ] == ["proposal-first"]

    template = consumer["promotion_acceptance_record_templates"][0]
    assert template["proposal_id"] == "proposal-first"
    assert template["decision_options"] == ["accepted", "rejected"]
    assert template["operator_must_choose_decision"] is True
    assert template["promotion_memory_trust_review_result_consumer_only"] is True
    assert template["promotion_acceptance_ledger_request_template_only"] is True
    assert template["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert template["execution_allowed_by_acceptance"] is False
    assert template["sequence_policy_trust_store_mutation"] is False
    assert template["proposal_state_mutation"] is False
    assert template["can_import_to_queue"] is False
    assert template["can_apply_live"] is False
    assert template["live_authority_granted"] is False


def test_promotion_memory_trust_review_result_consumer_rejects_forged_review(
    tmp_path,
):
    index_consumer = _promotion_memory_index_result_consumer_ready(tmp_path)
    forged = deepcopy(_memory_trust_review(index_consumer))
    forged["summary"]["policy_mutation"] = True
    forged["summary"]["trust_review_candidate_count"] = 2
    forged["safety"]["live_authority_granted"] = True
    forged["trust_review_candidates"][0]["proposal_id"] = "forged"
    forged["trust_review_candidates"][0]["execution_allowed_by_review"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer(
            promotion_memory_index_result_consumer=index_consumer,
            promotion_memory_trust_review_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_acceptance_ledger_request_available"]
        is False
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            "sequence_policy_memory_trust_review_result_consumer_review_safety_live_authority_granted_not_false"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            "sequence_policy_memory_trust_review_result_consumer_review_summary_policy_mutation_not_false"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            "sequence_policy_memory_trust_review_result_consumer_review_summary_trust_review_candidate_count_mismatch"
        )
        for item in invalid
    )
    assert any(item.endswith(":proposal_id_mismatch") for item in invalid)
    assert any(item.endswith(":execution_allowed_by_review_not_false") for item in invalid)
    assert consumer["promotion_acceptance_record_templates"] == []


def test_promotion_memory_trust_review_result_consumer_rejects_forged_prior(
    tmp_path,
):
    index_consumer = _promotion_memory_index_result_consumer_ready(tmp_path)
    forged_prior = deepcopy(index_consumer)
    forged_prior["summary"]["live_authority_granted"] = True
    forged_prior["promotion_memory_trust_review_request"]["policy_mutation"] = True
    forged_prior["promotion_memory_trust_review_request_specs"][0][
        "sequence_policy_trust_store_mutation"
    ] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer(
            promotion_memory_index_result_consumer=forged_prior,
            promotion_memory_trust_review_result=_memory_trust_review(index_consumer),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_acceptance_ledger_request_available"]
        is False
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_memory_trust_review_result_consumer_prior_summary_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "promotion_memory_trust_review_result_consumer_prior_request_policy_mutation_not_false"
        in invalid
    )
    assert any(
        item.endswith(":sequence_policy_trust_store_mutation_not_false")
        for item in invalid
    )


def test_promotion_memory_trust_review_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert (
        consumer["summary"]["promotion_acceptance_ledger_request_available"]
        is False
    )


def test_promotion_memory_trust_review_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
