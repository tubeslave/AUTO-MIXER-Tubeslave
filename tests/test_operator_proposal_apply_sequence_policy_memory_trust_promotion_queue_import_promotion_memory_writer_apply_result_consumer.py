from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer import (
    _promotion_memory_draft_result_consumer_ready,
    _writer_dry_run_payload,
)


def _promotion_memory_writer_dry_run_result_consumer_ready(tmp_path):
    prior_consumer = _promotion_memory_draft_result_consumer_ready(tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer(
            promotion_memory_draft_result_consumer=prior_consumer,
            promotion_memory_writer_dry_run_result=_writer_dry_run_payload(record_count=1),
        )
    )


def _writer_apply_payload(dry_run_consumer):
    records = dry_run_consumer["promotion_memory_writer_apply_request"][
        "memory_draft_payload"
    ]["memory_draft_records"]
    write_results = [
        {
            "draft_id": record["draft_id"],
            "target_memory_path": ".ai/memory/project.md",
            "applied": True,
            "bytes_appended": 192 + index,
        }
        for index, record in enumerate(records)
    ]
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_writer",
        "status": "append_only_memory_write_applied",
        "success": True,
        "reason": None,
        "generated_at": 123.0,
        "source_memory_draft_status": "ready",
        "summary": {
            "memory_draft_record_count": len(records),
            "invalid_record_count": 0,
            "write_result_count": len(write_results),
            "append_only_memory_write_applied": True,
            "operator_ack_required": True,
            "operator_ack": True,
            "apply_requested": True,
            "target_memory_path": ".ai/memory/project.md",
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_file_mutation": True,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [],
        },
        "safety": {
            "operator_required": True,
            "operator_ack_required": True,
            "apply_requested": True,
            "operator_ack": True,
            "append_only_memory_file_write_applied": True,
            "target_memory_path": ".ai/memory/project.md",
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_file_mutation": True,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "write_results": write_results,
        "blocked_results": [],
    }


def test_promotion_memory_writer_apply_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer.py"
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


def test_promotion_memory_writer_apply_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["source_memory_file_write_applied"] is False
    assert consumer["safety"]["memory_index_handler_invoked"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["safety"]["live_write"] is False
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_memory_index_request"]["handler"] == (
        "index_operator_proposal_apply_sequence_policy_memory"
    )
    assert consumer["promotion_memory_index_request_specs"] == []


def test_promotion_memory_writer_apply_result_consumer_emits_index_request(tmp_path):
    dry_run_consumer = _promotion_memory_writer_dry_run_result_consumer_ready(tmp_path)
    writer = _writer_apply_payload(dry_run_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=dry_run_consumer,
            promotion_memory_writer_apply_result=writer,
            root_path=str(tmp_path),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_index_request_available"
    )
    assert consumer["summary"]["expected_memory_write_result_count"] == 1
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
    assert consumer["summary"]["memory_write_result_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_memory_index_request_available"] is True
    assert consumer["summary"]["source_memory_file_write_applied"] is True
    assert consumer["summary"]["memory_index_handler_invoked"] is False
    assert consumer["summary"]["memory_index_created"] is False
    assert consumer["summary"]["memory_file_write_applied_by_consumer"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["generic_memory_writer_apply_result_consumer"]["summary"][
        "memory_index_request_available"
    ] is True

    request = consumer["promotion_memory_index_request"]
    assert request["handler"] == "index_operator_proposal_apply_sequence_policy_memory"
    assert request["root_path"] == str(tmp_path)
    assert request["target_memory_path"] == ".ai/memory/project.md"
    assert request["read_only_template"] is True
    assert request["promotion_memory_writer_apply_result_consumer_only"] is True
    assert request["promotion_memory_index_request_template_only"] is True
    assert request["memory_index_handler_invoked"] is False
    assert request["memory_index_created"] is False
    assert request["source_memory_file_write_applied"] is True
    assert request["memory_file_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["promotion_memory_writer_apply_result_consumer"][
        "memory_write_result_proposal_ids"
    ] == ["proposal-first"]
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == dry_run_consumer["promotion_memory_writer_apply_request"][
        "promotion_acceptance_record_templates"
    ][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == dry_run_consumer["promotion_memory_writer_apply_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

    spec = consumer["promotion_memory_index_request_specs"][0]
    assert spec["proposal_id"] == "proposal-first"
    assert spec["draft_id"] == (
        dry_run_consumer["promotion_memory_writer_apply_request"][
            "memory_draft_payload"
        ]["memory_draft_records"][0]["draft_id"]
    )
    assert spec["promotion_memory_writer_apply_result_consumer_only"] is True
    assert spec["promotion_memory_index_request_template_only"] is True
    assert spec["source_memory_file_write_applied"] is True
    assert spec["memory_file_write_applied_by_consumer"] is False
    assert spec["memory_file_mutation"] is False
    assert spec["sequence_policy_trust_store_mutation"] is False
    assert spec["live_authority_granted"] is False


def test_promotion_memory_writer_apply_result_consumer_rejects_dry_run_writer_result(
    tmp_path,
):
    dry_run_consumer = _promotion_memory_writer_dry_run_result_consumer_ready(tmp_path)
    writer = _writer_dry_run_payload(record_count=1)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=dry_run_consumer,
            promotion_memory_writer_apply_result=writer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            "sequence_policy_memory_writer_apply_consumer_writer_status_not_applied"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            "sequence_policy_memory_writer_apply_consumer_writer_success_not_true"
        )
        for item in invalid
    )
    assert consumer["promotion_memory_index_request_specs"] == []


def test_promotion_memory_writer_apply_result_consumer_rejects_forged_apply_evidence(
    tmp_path,
):
    dry_run_consumer = _promotion_memory_writer_dry_run_result_consumer_ready(tmp_path)
    writer = _writer_apply_payload(dry_run_consumer)
    forged_prior = deepcopy(dry_run_consumer)
    forged_prior["summary"]["live_authority_granted"] = True
    forged_prior["promotion_memory_writer_apply_request"]["memory_file_mutation"] = True
    forged_writer = deepcopy(writer)
    forged_writer["summary"]["live_authority_granted"] = True
    forged_writer["write_results"][0]["draft_id"] = "forged"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=forged_prior,
            promotion_memory_writer_apply_result=forged_writer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_summary_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_request_memory_file_mutation_not_false"
        in invalid
    )
    assert any(
        item.endswith(
            "sequence_policy_memory_writer_apply_consumer_writer_summary_live_authority_granted_not_false"
        )
        for item in invalid
    )
    assert any(item.endswith(":write_result_draft_id_mismatch") for item in invalid)
    assert consumer["promotion_memory_index_request_specs"] == []


def test_promotion_memory_writer_apply_result_consumer_rejects_forged_prior_summary_ids(
    tmp_path,
):
    dry_run_consumer = _promotion_memory_writer_dry_run_result_consumer_ready(tmp_path)
    writer = _writer_apply_payload(dry_run_consumer)
    forged_prior = deepcopy(dry_run_consumer)
    forged_prior["summary"]["memory_writer_dry_run_proposal_ids"] = [
        "proposal-second"
    ]
    forged_prior["summary"]["idempotency_key_prefix"] = "forged-prefix"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=forged_prior,
            promotion_memory_writer_apply_result=writer,
            root_path=str(tmp_path),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_summary_prefix_mismatch"
        in invalid
    )
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_summary_proposal_ids_mismatch"
        in invalid
    )
    assert consumer["promotion_memory_index_request_specs"] == []
    assert (
        "promotion_memory_writer_apply_result_consumer"
        not in consumer["promotion_memory_index_request"]
    )


def test_promotion_memory_writer_apply_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    dry_run_consumer = deepcopy(_promotion_memory_writer_dry_run_result_consumer_ready(tmp_path))
    writer = _writer_apply_payload(dry_run_consumer)
    request = dry_run_consumer["promotion_memory_writer_apply_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    dry_run_consumer["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=dry_run_consumer,
            promotion_memory_writer_apply_result=writer,
            root_path=str(tmp_path),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_summary_review_request_spec_count_mismatch"
        in invalid
    )
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "promotion_memory_writer_apply_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first_current_sequence_review_freshness_missing" in invalid
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_memory_index_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_memory_index_request"]
    )


def test_promotion_memory_writer_apply_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    dry_run_consumer = deepcopy(_promotion_memory_writer_dry_run_result_consumer_ready(tmp_path))
    writer = _writer_apply_payload(dry_run_consumer)
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
    request = dry_run_consumer["promotion_memory_writer_apply_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=dry_run_consumer,
            promotion_memory_writer_apply_result=writer,
            root_path=str(tmp_path),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "invalid"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first_current_sequence_review_freshness_read_only_not_true" in invalid
    assert "proposal-first_current_sequence_review_freshness_live_write_not_false" in invalid
    assert (
        "proposal-first_current_sequence_review_freshness_apply_attempted_not_false"
        in invalid
    )
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_memory_index_request"]
    )


def test_promotion_memory_writer_apply_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    dry_run_consumer = deepcopy(_promotion_memory_writer_dry_run_result_consumer_ready(tmp_path))
    writer = _writer_apply_payload(dry_run_consumer)
    template = dry_run_consumer["promotion_memory_writer_apply_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["consumer_record_id"] = ""
    template["source_memory_trust_review_request_spec_id"] = "forged-review-spec"
    template["current_sequence_review_freshness"] = {
        **template["current_sequence_review_freshness"],
        "status": "stale",
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
            promotion_memory_writer_dry_run_result_consumer=dry_run_consumer,
            promotion_memory_writer_apply_result=writer,
            root_path=str(tmp_path),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_index_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "proposal-first:source_memory_trust_review_request_spec_id_mismatch" in invalid
    assert "proposal-first:current_sequence_review_freshness_request_spec_mismatch" in invalid
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_memory_index_request"]
    )


def test_promotion_memory_writer_apply_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_memory_index_request_available"] is False


def test_promotion_memory_writer_apply_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
