from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer import (
    _audit_payload,
    _promotion_execution_result_consumer_ready,
)


def _promotion_execution_audit_result_consumer_ready(tmp_path):
    prior_consumer = _promotion_execution_result_consumer_ready(tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer(
            promotion_execution_result_consumer=prior_consumer,
            promotion_execution_audit_result=_audit_payload(prior_consumer),
        )
    )


def _policy_row(proposal_id, *, index, prefix, status="policy_evidence_ready"):
    return {
        "proposal_id": proposal_id,
        "audit_row_index": index,
        "status": status,
        "eligible_for_supervised_sequence_policy": True,
        "execution_allowed_by_policy_gate": False,
        "policy_mutation_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "value_type": "fader",
        "channel": 2,
        "effective_value": -8.5,
        "idempotency_key": f"{prefix}:{proposal_id}",
        "readback_status": "verified_from_cache",
        "proof_status": "verified",
        "rollback_available": True,
        "live_apply_entrypoint": "apply_operator_proposal",
    }


def _policy_gate_payload(prior_consumer, *, rows=None, blocked=None, status="ready"):
    prefix = prior_consumer["summary"]["idempotency_key_prefix"]
    ids = list(prior_consumer["summary"]["execution_audit_row_proposal_ids"])
    policy_evidence = list(
        rows
        if rows is not None
        else [
            _policy_row(
                proposal_id,
                index=index,
                prefix=prefix,
            )
            for index, proposal_id in enumerate(ids)
        ]
    )
    blocked_results = list(blocked if blocked is not None else [])
    ready_rows = [
        row for row in policy_evidence if row.get("status") == "policy_evidence_ready"
    ]
    blocked_rows = [row for row in policy_evidence if row.get("status") == "blocked"]
    return {
        "type": "operator_proposal_apply_sequence_audit_policy_gate",
        "status": status,
        "success": status == "ready",
        "reason": None if status == "ready" else "sequence_audit_policy_gate_blocked",
        "generated_at": 123.0,
        "source_audit_status": "passed",
        "summary": {
            "input_audit_row_count": len(ids),
            "policy_evidence_ready_count": len(ready_rows),
            "blocked_count": len(blocked_rows) + len(blocked_results),
            "policy_evidence_ready_proposal_ids": [
                row["proposal_id"] for row in ready_rows
            ],
            "blocked_proposal_ids": [
                row["proposal_id"] for row in [*blocked_rows, *blocked_results]
            ],
            "policy_gate_only": True,
            "audit_payload_required": True,
            "execution_allowed_by_policy_gate": False,
            "execution_allowed_by_audit": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "safety": {
            "policy_gate_only": True,
            "audit_payload_required": True,
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "execution_allowed_by_policy_gate": False,
            "execution_allowed_by_audit": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "policy_evidence": policy_evidence,
        "blocked_results": blocked_results,
    }


def test_promotion_audit_policy_gate_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer.py"
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


def test_promotion_audit_policy_gate_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_draft_created"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_memory_draft_request"]["policy_gate_payload"] == {}


def test_promotion_audit_policy_gate_result_consumer_emits_memory_draft_request(tmp_path):
    prior_consumer = _promotion_execution_audit_result_consumer_ready(tmp_path)
    policy_gate = _policy_gate_payload(prior_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=prior_consumer,
            promotion_audit_policy_gate_result=policy_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_memory_draft_request_available"
    )
    assert consumer["summary"]["expected_policy_evidence_count"] == 1
    assert consumer["summary"]["policy_evidence_count"] == 1
    assert consumer["summary"]["ready_policy_evidence_count"] == 1
    assert consumer["summary"]["blocked_policy_evidence_count"] == 0
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
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_memory_draft_request_available"] is True
    assert consumer["summary"]["memory_draft_handler_invoked"] is False
    assert consumer["summary"]["memory_draft_created"] is False
    assert consumer["summary"]["memory_write_allowed_by_consumer"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["generic_audit_policy_gate_result_consumer"]["summary"][
        "memory_draft_request_available"
    ] is True

    request = consumer["promotion_memory_draft_request"]
    assert request["handler"] == "draft_operator_proposal_apply_sequence_policy_memory"
    assert request["read_only_template"] is True
    assert request["promotion_audit_policy_gate_result_consumer_only"] is True
    assert request["promotion_memory_draft_request_template_only"] is True
    assert len(request["policy_gate_payload"]["policy_evidence"]) == 1
    assert request["memory_draft_handler_invoked"] is False
    assert request["memory_draft_created"] is False
    assert request["memory_file_mutation"] is False
    assert request["live_authority_granted"] is False
    assert request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] == prior_consumer["promotion_audit_policy_gate_request"][
        "promotion_acceptance_record_templates"
    ][0][
        "current_sequence_review_freshness"
    ]
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == prior_consumer["promotion_audit_policy_gate_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]
    spec = consumer["promotion_memory_draft_request_specs"][0]
    assert spec["proposal_id"] == "proposal-first"
    assert (
        spec["idempotency_key"]
        == f"{prior_consumer['summary']['idempotency_key_prefix']}:proposal-first"
    )
    assert spec["memory_draft_handler_invoked"] is False
    assert spec["memory_write_allowed_by_consumer"] is False
    assert spec["live_authority_granted"] is False


def test_promotion_audit_policy_gate_result_consumer_rejects_forged_policy_gate(tmp_path):
    prior_consumer = _promotion_execution_audit_result_consumer_ready(tmp_path)
    policy_gate = _policy_gate_payload(prior_consumer)
    policy_gate["summary"]["live_authority_granted"] = True
    policy_gate["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    policy_gate["policy_evidence"][0]["idempotency_key"] = "forged"
    policy_gate["policy_evidence"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=prior_consumer,
            promotion_audit_policy_gate_result=policy_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            "sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_live_authority_granted_not_false"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            "sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_allowed_kinds_mismatch"
        )
        for item in invalid
    )
    assert any(item.endswith("proposal-first:idempotency_key_mismatch") for item in invalid)
    assert any(item.endswith("proposal-first:fader_ceiling_exceeded") for item in invalid)
    assert consumer["promotion_memory_draft_request"]["policy_gate_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_promotion_audit_policy_gate_result_consumer_rejects_blocked_policy_evidence(tmp_path):
    prior_consumer = _promotion_execution_audit_result_consumer_ready(tmp_path)
    prefix = prior_consumer["summary"]["idempotency_key_prefix"]
    rows = [
        _policy_row(
            "proposal-first",
            index=0,
            prefix=prefix,
            status="blocked",
        )
    ]
    blocked = [
        {
            "proposal_id": "proposal-first",
            "audit_row_index": 0,
            "status": "blocked",
            "reason": "policy_evidence_readback_not_verified",
        }
    ]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=prior_consumer,
            promotion_audit_policy_gate_result=_policy_gate_payload(
                prior_consumer,
                rows=rows,
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
    assert consumer["promotion_memory_draft_request"]["policy_gate_payload"] == {}
    assert any(
        item.endswith(
            "sequence_policy_audit_policy_gate_result_consumer_policy_gate_has_blocked_results"
        )
        for item in consumer["summary"]["invalid_record_ids"]
    )


def test_promotion_audit_policy_gate_result_consumer_rejects_forged_prior_summary_ids(
    tmp_path,
):
    prior_consumer = _promotion_execution_audit_result_consumer_ready(tmp_path)
    policy_gate = _policy_gate_payload(prior_consumer)
    forged_prior = deepcopy(prior_consumer)
    forged_prior["summary"]["expected_execution_audit_proposal_ids"] = [
        "proposal-second"
    ]
    forged_prior["summary"]["execution_audit_row_proposal_ids"] = [
        "proposal-second"
    ]
    forged_prior["summary"]["passed_execution_audit_proposal_ids"] = [
        "proposal-second"
    ]
    forged_prior["summary"]["idempotency_key_prefix"] = "forged-prefix"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=forged_prior,
            promotion_audit_policy_gate_result=policy_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "promotion_audit_policy_gate_result_consumer_prior_summary_prefix_mismatch" in invalid
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_summary_expected_ids_mismatch"
        in invalid
    )
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_summary_row_ids_mismatch"
        in invalid
    )
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_summary_passed_ids_mismatch"
        in invalid
    )
    assert consumer["promotion_memory_draft_request"]["policy_gate_payload"] == {}
    assert (
        "promotion_audit_policy_gate_result_consumer"
        not in consumer["promotion_memory_draft_request"]
    )
    assert consumer["promotion_memory_draft_request_specs"] == []


def test_promotion_audit_policy_gate_result_consumer_rejects_missing_sequence_review_freshness(
    tmp_path,
):
    prior_consumer = deepcopy(_promotion_execution_audit_result_consumer_ready(tmp_path))
    policy_gate = _policy_gate_payload(prior_consumer)
    request = prior_consumer["promotion_audit_policy_gate_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    prior_consumer["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=prior_consumer,
            promotion_audit_policy_gate_result=policy_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_summary_review_request_spec_count_mismatch"
        in invalid
    )
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "promotion_audit_policy_gate_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first_current_sequence_review_freshness_missing" in invalid
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_memory_draft_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_memory_draft_request"]
    )


def test_promotion_audit_policy_gate_result_consumer_rejects_live_sequence_review_freshness(
    tmp_path,
):
    prior_consumer = deepcopy(_promotion_execution_audit_result_consumer_ready(tmp_path))
    policy_gate = _policy_gate_payload(prior_consumer)
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
    request = prior_consumer["promotion_audit_policy_gate_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=prior_consumer,
            promotion_audit_policy_gate_result=policy_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
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
        not in consumer["promotion_memory_draft_request"]
    )


def test_promotion_audit_policy_gate_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    prior_consumer = deepcopy(_promotion_execution_audit_result_consumer_ready(tmp_path))
    policy_gate = _policy_gate_payload(prior_consumer)
    template = prior_consumer["promotion_audit_policy_gate_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["consumer_record_id"] = ""
    template["source_memory_trust_review_request_spec_id"] = "forged-review-spec"
    template["current_sequence_review_freshness"] = {
        **template["current_sequence_review_freshness"],
        "status": "stale",
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
            promotion_execution_audit_result_consumer=prior_consumer,
            promotion_audit_policy_gate_result=policy_gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert "proposal-first:source_memory_trust_review_request_spec_id_mismatch" in invalid
    assert "proposal-first:current_sequence_review_freshness_request_spec_mismatch" in invalid
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_memory_draft_request"]
    )


def test_promotion_audit_policy_gate_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_draft_created"] is False
    assert consumer["summary"]["promotion_memory_draft_request_available"] is False


def test_promotion_audit_policy_gate_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
