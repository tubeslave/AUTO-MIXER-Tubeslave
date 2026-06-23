from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
    _preflight_plan_ready,
)


def _promotion_preflight_plan_consumer_ready(tmp_path):
    prior_consumer, plan = _preflight_plan_ready(tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer(
            promotion_queue_importer_result_consumer=prior_consumer,
            promotion_queue_import_preflight_plan_result=plan,
        )
    )


def _preflight_result(proposal_id, *, prefix, ready=True):
    return {
        "type": "operator_proposal_apply_preflight",
        "status": "ready" if ready else "blocked",
        "success": ready,
        "ready_to_apply": ready,
        "reason": None if ready else "approval_required",
        "proposal": {
            "id": proposal_id,
            "title": f"Ready {proposal_id}",
            "status": "pending",
        },
        "idempotency": {
            "required": True,
            "provided": True,
            "idempotency_key": f"{prefix}:{proposal_id}",
            "reserved": False,
            "would_reserve_on_apply": ready,
            "duplicate": False,
        },
        "normalized_change": {
            "value_type": "fader",
            "channel": 2,
            "address": "/ch/2/fdr",
            "effective_value": -8.5,
            "old_value": -8.0,
            "pre_read_value": -8.0,
            "delta_db": -0.5,
        },
        "normalized_changes": [
            {
                "value_type": "fader",
                "channel": 2,
                "address": "/ch/2/fdr",
                "effective_value": -8.5,
                "old_value": -8.0,
                "pre_read_value": -8.0,
                "delta_db": -0.5,
            }
        ],
        "safety": {
            "preflight_only": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "live_apply_entrypoint": "apply_operator_proposal",
        },
    }


def _batch_result(prior_consumer, *, ready=True):
    request = prior_consumer["promotion_batch_preflight_request"]
    prefix = request["idempotency_key_prefix"]
    ids = list(request["proposal_ids"])
    preflight_results = [
        _preflight_result(proposal_id, prefix=prefix, ready=ready)
        for proposal_id in ids
    ]
    ready_ids = [
        item["proposal"]["id"]
        for item in preflight_results
        if item.get("ready_to_apply") is True
    ]
    blocked_ids = [
        item["proposal"]["id"]
        for item in preflight_results
        if item.get("ready_to_apply") is not True
    ]
    return {
        "type": "operator_proposal_apply_preflight_batch",
        "status": "ok",
        "success": True,
        "generated_at": 123.0,
        "summary": {
            "selected_count": len(preflight_results),
            "preflight_result_count": len(preflight_results),
            "ready_count": len(ready_ids),
            "blocked_count": len(blocked_ids),
            "skipped_analysis_only_count": 0,
            "not_found_count": 0,
            "ready_proposal_ids": ready_ids,
            "blocked_proposal_ids": blocked_ids,
            "skipped_analysis_only_ids": [],
            "not_found_ids": [],
            "approved": True,
            "idempotency_key_prefix": prefix,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "idempotency_reserved": False,
        },
        "safety": {
            "batch_preflight_only": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_apply_entrypoint": "apply_operator_proposal",
        },
        "preflight_results": preflight_results,
    }


def test_promotion_batch_preflight_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer.py"
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


def test_promotion_batch_preflight_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["idempotency_reserved"] is False
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_sequence_plan_request"]["preflight_results"] == []


def test_promotion_batch_preflight_result_consumer_emits_sequence_plan_request(
    tmp_path,
):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    batch = _batch_result(prior_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=prior_consumer,
            promotion_batch_preflight_result=batch,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_sequence_plan_request_available"
    )
    assert consumer["summary"]["planned_preflight_candidate_count"] == 1
    assert consumer["summary"]["promotion_acceptance_record_template_count"] == 1
    assert consumer["summary"]["memory_trust_review_request_spec_count"] == 2
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "fresh"
    assert consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] == [
        "proposal-first"
    ]
    assert consumer["summary"]["memory_trust_review_request_spec_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["batch_preflight_result_count"] == 1
    assert consumer["summary"]["ready_preflight_result_count"] == 1
    assert consumer["summary"]["blocked_preflight_result_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is True
    assert consumer["summary"]["sequence_plan_handler_invoked"] is False
    assert consumer["summary"]["sequence_plan_created"] is False
    assert consumer["summary"]["sequence_execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["generic_preflight_result_consumer"]["summary"][
        "sequence_plan_request_available"
    ] is True

    request = consumer["promotion_sequence_plan_request"]
    assert request["handler"] == "plan_operator_proposal_apply_sequence"
    assert request["read_only_template"] is True
    assert request["promotion_batch_preflight_result_consumer_only"] is True
    assert request["promotion_sequence_plan_request_template_only"] is True
    assert len(request["preflight_results"]) == 1
    assert request["sequence_plan_created"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["idempotency_key"] == (
        "seq-policy:proposal-first"
    )
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == prior_consumer["promotion_batch_preflight_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]
    spec = consumer["promotion_sequence_plan_request_specs"][0]
    assert spec["proposal_id"] == batch["summary"]["ready_proposal_ids"][0]
    assert spec["sequence_plan_created"] is False
    assert spec["live_authority_granted"] is False


def test_promotion_batch_preflight_result_consumer_blocks_blocked_rows(tmp_path):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    batch = _batch_result(prior_consumer, ready=False)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=prior_consumer,
            promotion_batch_preflight_result=batch,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_blocked"
    )
    assert consumer["summary"]["ready_preflight_result_count"] == 0
    assert consumer["summary"]["blocked_preflight_result_count"] == 1
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    assert consumer["promotion_sequence_plan_request"]["preflight_results"] == []


def test_promotion_batch_preflight_result_consumer_rejects_forged_batch(tmp_path):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    forged = deepcopy(_batch_result(prior_consumer))
    forged["summary"]["approved"] = False
    forged["safety"]["live_write"] = True
    forged["preflight_results"][0]["safety"]["live_write"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=prior_consumer,
            promotion_batch_preflight_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith("batch_summary_approved_not_true")
        for item in invalid
    )
    assert any(item.endswith("batch_live_write_not_false") for item in invalid)
    assert any(item.endswith(":preflight_result_live_write_not_false") for item in invalid)


def test_promotion_batch_preflight_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    batch = _batch_result(prior_consumer)
    forged_prior = deepcopy(prior_consumer)
    template = forged_prior["promotion_batch_preflight_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["idempotency_key"] = "seq-policy:forged"
    template["consumer_record_id"] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=forged_prior,
            promotion_batch_preflight_result=batch,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:idempotency_key_sequence_plan_spec_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert consumer["promotion_sequence_plan_request"]["preflight_results"] == []
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_sequence_plan_request"]
    )
    assert consumer["promotion_sequence_plan_request_specs"] == []


def test_promotion_batch_preflight_result_consumer_rejects_missing_prior_sequence_review_freshness(
    tmp_path,
):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    batch = _batch_result(prior_consumer)
    forged_prior = deepcopy(prior_consumer)
    request = forged_prior["promotion_batch_preflight_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    forged_prior["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=forged_prior,
            promotion_batch_preflight_result=batch,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_batch_preflight_result_consumer_prior_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "promotion_batch_preflight_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first_current_sequence_review_freshness_missing" in invalid
    assert consumer["promotion_sequence_plan_request"]["preflight_results"] == []
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_sequence_plan_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_sequence_plan_request"]
    )
    assert consumer["promotion_sequence_plan_request_specs"] == []


def test_promotion_batch_preflight_result_consumer_rejects_live_prior_sequence_review_freshness(
    tmp_path,
):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    batch = _batch_result(prior_consumer)
    forged_prior = deepcopy(prior_consumer)
    live_freshness = {
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
    request = forged_prior["promotion_batch_preflight_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=forged_prior,
            promotion_batch_preflight_result=batch,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "invalid"
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first_current_sequence_review_freshness_read_only_not_true" in invalid
    assert "proposal-first_current_sequence_review_freshness_live_write_not_false" in invalid
    assert (
        "proposal-first_current_sequence_review_freshness_apply_attempted_not_false"
        in invalid
    )
    assert (
        "proposal-first:current_sequence_review_freshness_sequence_plan_spec_mismatch"
        in invalid
    )
    assert consumer["promotion_sequence_plan_request"]["preflight_results"] == []
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_sequence_plan_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_sequence_plan_request"]
    )
    assert consumer["promotion_sequence_plan_request_specs"] == []


def test_promotion_batch_preflight_result_consumer_rejects_forged_prior_template_summary(
    tmp_path,
):
    prior_consumer = _promotion_preflight_plan_consumer_ready(tmp_path)
    batch = _batch_result(prior_consumer)
    forged_prior = deepcopy(prior_consumer)
    forged_prior["summary"]["promotion_acceptance_record_template_count"] = 2
    forged_prior["summary"]["promotion_acceptance_record_template_proposal_ids"] = [
        "proposal-second"
    ]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
            promotion_preflight_plan_result_consumer=forged_prior,
            promotion_batch_preflight_result=batch,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_batch_preflight_result_consumer_prior_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "promotion_batch_preflight_result_consumer_prior_summary_template_proposal_ids_mismatch"
        in invalid
    )
    assert consumer["promotion_sequence_plan_request"]["preflight_results"] == []
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_sequence_plan_request"]
    )
    assert consumer["promotion_sequence_plan_request_specs"] == []


def test_promotion_batch_preflight_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_sequence_plan_request_available"] is False


def test_promotion_batch_preflight_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
