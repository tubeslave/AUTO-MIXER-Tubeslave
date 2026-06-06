from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
    _queue_importer_applied,
    _queue_importer_dry_run,
)


def _preflight_plan_ready(tmp_path):
    queue_ordering_plan_consumer, importer, queue = _queue_importer_applied(tmp_path)
    prior_consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=queue_ordering_plan_consumer,
            promotion_queue_importer_result=importer,
        )
    )
    request = prior_consumer["promotion_queue_import_preflight_plan_request"]
    plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
            pending_actions=queue.pending_actions(),
            idempotency_key_prefix=request["idempotency_key_prefix"],
            limit=request["limit"],
        )
    )
    return prior_consumer, plan


def test_promotion_preflight_plan_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer.py"
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
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_promotion_preflight_plan_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["idempotency_reserved"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert consumer["summary"]["promotion_batch_preflight_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_batch_preflight_request"]["proposal_ids"] == []


def test_promotion_preflight_plan_result_consumer_blocks_without_applied_importer(
    tmp_path,
):
    queue_ordering_plan_consumer, importer, _ = _queue_importer_dry_run(tmp_path)
    blocked_prior = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=queue_ordering_plan_consumer,
            promotion_queue_importer_result=importer,
        )
    )

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer(
            promotion_queue_importer_result_consumer=blocked_prior,
            promotion_queue_import_preflight_plan_result=(
                build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan()
            ),
        )
    )

    assert consumer["status"] in {
        "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_blocked",
        "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_replacement_required",
    }
    assert consumer["summary"]["promotion_batch_preflight_request_available"] is False


def test_promotion_preflight_plan_result_consumer_emits_batch_request(tmp_path):
    prior_consumer, plan = _preflight_plan_ready(tmp_path)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer(
            promotion_queue_importer_result_consumer=prior_consumer,
            promotion_queue_import_preflight_plan_result=plan,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_request_available"
    )
    assert consumer["summary"]["expected_imported_proposal_count"] == 1
    assert consumer["summary"]["preflight_candidate_count"] == 1
    assert consumer["summary"]["preflight_request_spec_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_batch_preflight_request_available"] is True
    assert consumer["summary"]["promotion_batch_preflight_handler_invoked"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_batch_preflight_request"]
    assert request["handler"] == "preflight_operator_proposal_apply_batch"
    assert request["read_only_template"] is True
    assert request["promotion_preflight_plan_result_consumer_only"] is True
    assert request["promotion_batch_preflight_request_template_only"] is True
    assert request["promotion_batch_preflight_handler_invoked"] is False
    assert request["operator_must_set_approved_true"] is True
    assert request["approved"] is False
    assert request["proposal_ids"] == plan["summary"]["preflight_candidate_proposal_ids"]
    assert request["idempotency_key_prefix"] == plan["summary"]["idempotency_key_prefix"]
    assert request["queue_mutation"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    assert request["queue_import_preflight_plan"] == plan


def test_promotion_preflight_plan_result_consumer_rejects_forged_plan(tmp_path):
    prior_consumer, plan = _preflight_plan_ready(tmp_path)
    forged = deepcopy(plan)
    forged["summary"]["batch_preflight_request_available"] = False
    forged["batch_preflight_request"]["approved"] = True
    forged["preflight_request_specs"][0]["live_write"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer(
            promotion_queue_importer_result_consumer=prior_consumer,
            promotion_queue_import_preflight_plan_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_batch_preflight_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_preflight_plan_result_consumer_plan_summary_request_available_mismatch"
        in invalid
    )
    assert (
        "promotion_preflight_plan_result_consumer_batch_request_approved_not_false"
        in invalid
    )
    assert any(item.endswith(":live_write_not_false") for item in invalid)


def test_promotion_preflight_plan_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_batch_preflight_request_available"] is False


def test_promotion_preflight_plan_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
