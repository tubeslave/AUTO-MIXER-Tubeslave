from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
)
from backend.operator_proposal_queue import OperatorProposalQueue
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
    _queue_ordering_plan_ready,
)


def _queue_ordering_plan_result_consumer_ready(tmp_path):
    ordering_preview_result_consumer, queue_ordering_plan, proposals = (
        _queue_ordering_plan_ready(tmp_path)
    )
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
            promotion_ordering_preview_result_consumer=ordering_preview_result_consumer,
            promotion_queue_ordering_plan_result=queue_ordering_plan,
        )
    )
    return consumer, proposals


def _applyable_proposals(proposals):
    upgraded = deepcopy(proposals)
    for proposal in upgraded:
        if proposal.get("proposal_id") == "proposal-first":
            proposal["kind"] = "fader"
            proposal["requested_change"] = {
                "value_type": "fader",
                "channel": 2,
                "value": -1.0,
                "current_value": 0.0,
                "rollback_value": 0.0,
            }
    return upgraded


def _queue_importer_dry_run(tmp_path):
    consumer, proposals = _queue_ordering_plan_result_consumer_ready(tmp_path)
    request = consumer["promotion_queue_importer_request"]
    importer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
            queue_ordering_plan=request["queue_ordering_plan"],
            proposal_payloads=proposals,
            apply=False,
            operator_ack=False,
        )
    )
    return consumer, importer, proposals


def _queue_importer_applied(tmp_path):
    consumer, proposals = _queue_ordering_plan_result_consumer_ready(tmp_path)
    request = consumer["promotion_queue_importer_request"]
    queue = OperatorProposalQueue()
    importer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
            queue_ordering_plan=request["queue_ordering_plan"],
            proposal_payloads=_applyable_proposals(proposals),
            target_queue=queue,
            apply=True,
            operator_ack=True,
        )
    )
    return consumer, importer, queue


def _partial_queue_importer_result(tmp_path):
    class PartiallyBlockingQueue:
        def __init__(self):
            self.queue = OperatorProposalQueue()
            self.create_count = 0

        def create(self, payload, operator_mode):
            self.create_count += 1
            if self.create_count == 1:
                return self.queue.create(payload, operator_mode)
            return {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "unit_test_partial_queue_block",
                "operator_mode": operator_mode,
                "proposal": None,
            }

        def find(self, proposal_id=None, index=None):
            return self.queue.find(proposal_id=proposal_id, index=index)

    consumer, proposals = _queue_ordering_plan_result_consumer_ready(tmp_path)
    consumer = deepcopy(consumer)
    request = consumer["promotion_queue_importer_request"]
    plan = request["queue_ordering_plan"]
    first_record = plan["queue_ordering_plan_records"][0]
    second_record = deepcopy(first_record)
    second_record.update(
        {
            "consumer_record_id": (
                "operator_sequence_policy_memory_trust_promotion_policy_consumer:"
                "proposal-second"
            ),
            "ordering_preview_record_id": (
                "operator_proposal_apply_sequence_policy_memory_trust_promotion_"
                "ordering_preview:proposal-second"
            ),
            "queue_ordering_plan_record_id": (
                "operator_proposal_apply_sequence_policy_memory_trust_promotion_"
                "queue_ordering_plan:proposal-second"
            ),
            "proposal_id": "proposal-second",
            "proposal_target": "ch/03",
            "planned_queue_index": 1,
            "original_index": 1,
            "recommended_index": 1,
            "policy_key": "proposal-second/seq-policy:proposal-second",
            "source_idempotency_key": "seq-policy:proposal-second",
        }
    )
    plan["queue_ordering_plan_records"] = [first_record, second_record]
    plan["summary"]["input_ordering_preview_count"] = 2
    plan["summary"]["queue_ordering_plan_record_count"] = 2
    plan["summary"]["ranked_queue_candidate_count"] = 2
    plan["summary"]["planned_queue_proposal_ids"] = [
        "proposal-first",
        "proposal-second",
    ]
    plan["summary"]["ranked_queue_candidate_proposal_ids"] = [
        "proposal-first",
        "proposal-second",
    ]
    consumer["summary"]["queue_ordering_plan_record_count"] = 2
    consumer["summary"]["ranked_queue_candidate_count"] = 2
    consumer["summary"]["planned_queue_proposal_ids"] = [
        "proposal-first",
        "proposal-second",
    ]
    consumer["summary"]["ordering_preview_proposal_ids"] = [
        "proposal-first",
        "proposal-second",
    ]
    second_template = {
        key: (
            value.replace("proposal-first", "proposal-second")
            if isinstance(value, str)
            else value
        )
        for key, value in consumer["promotion_queue_importer_request"][
            "promotion_acceptance_record_templates"
        ][0].items()
    }
    second_template["planned_queue_index"] = 1
    second_template["idempotency_key"] = "seq-policy:proposal-second"
    second_template["source_idempotency_key"] = "memory-write:proposal-second"
    second_template["policy_key"] = "proposal-second/memory-write:proposal-second"
    consumer["promotion_queue_importer_request"][
        "promotion_acceptance_record_templates"
    ].append(second_template)
    consumer["summary"]["promotion_acceptance_record_template_count"] = 2
    consumer["summary"]["promotion_acceptance_record_template_proposal_ids"] = [
        "proposal-first",
        "proposal-second",
    ]

    second_proposal = deepcopy(proposals[0])
    second_proposal["proposal_id"] = "proposal-second"
    second_proposal["target"] = "ch/03"
    second_proposal["evidence"]["idempotency_key"] = "seq-policy:proposal-second"
    queue = PartiallyBlockingQueue()
    importer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
            queue_ordering_plan=plan,
            proposal_payloads=_applyable_proposals([proposals[0], second_proposal]),
            target_queue=queue,
            apply=True,
            operator_ack=True,
        )
    )
    return consumer, importer


def test_promotion_queue_importer_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py"
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
        "requested_change",
        "requested_changes",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_promotion_queue_importer_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["runtime_memory_mutation"] is False
    assert consumer["safety"]["live_authority_granted"] is False
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        consumer["promotion_queue_import_preflight_plan_request"]["handler"]
        == "plan_operator_proposal_apply_sequence_policy_queue_import_preflight"
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_blocks_dry_run_before_preflight(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_dry_run(tmp_path)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked"
    )
    assert consumer["summary"]["queue_ordering_plan_record_count"] == 1
    assert consumer["summary"]["queue_import_spec_count"] == 1
    assert consumer["summary"]["queue_import_result_count"] == 0
    assert consumer["summary"]["importer_dry_run_seen"] is True
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert (
        "applied_sequence_policy_queue_importer_result_required_before_preflight_plan"
        in consumer["summary"]["blocked_by"]
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_emits_preflight_plan_request(
    tmp_path,
):
    prior_consumer, importer, queue = _queue_importer_applied(tmp_path)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
    )
    assert consumer["summary"]["queue_ordering_plan_record_count"] == 1
    assert consumer["summary"]["queue_import_spec_count"] == 1
    assert consumer["summary"]["queue_import_result_count"] == 1
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
    assert consumer["summary"]["imported_proposal_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is True
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_handler_invoked"]
        is False
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_status_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False

    request = consumer["promotion_queue_import_preflight_plan_request"]
    assert (
        request["handler"]
        == "plan_operator_proposal_apply_sequence_policy_queue_import_preflight"
    )
    assert request["read_only_template"] is True
    assert request["promotion_queue_importer_result_consumer_only"] is True
    assert request["promotion_queue_import_preflight_plan_request_template_only"] is True
    assert request["promotion_queue_import_preflight_plan_handler_invoked"] is False
    assert request["pending_actions_required_from_operator_queue"] is True
    assert request["queue_mutation"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    assert request["imported_proposal_ids"] == ["proposal-first"]
    assert request["promotion_queue_importer_result"] == importer
    request_importer = request["promotion_queue_importer_result"]
    spec_context = request_importer["queue_import_specs"][0][
        "current_sequence_review_freshness"
    ]
    payload_context = request_importer["queue_import_specs"][0][
        "operator_proposal_payload"
    ]["sequence_policy_queue_import"]["current_sequence_review_freshness"]
    result_context = request_importer["queue_import_results"][0]["operator_proposal"][
        "raw"
    ]["sequence_policy_queue_import"]["current_sequence_review_freshness"]
    assert spec_context["status"] == "fresh"
    assert spec_context["read_only"] is True
    assert payload_context == spec_context
    assert result_context == spec_context
    assert request["promotion_acceptance_record_templates"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert request["promotion_acceptance_record_templates"][0]["idempotency_key"] == (
        "seq-policy:proposal-first"
    )
    assert request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] == prior_consumer["promotion_queue_importer_request"][
        "memory_trust_review_request_specs"
    ][0][
        "current_sequence_review_freshness"
    ]

    preflight_plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
            pending_actions=queue.pending_actions(),
            idempotency_key_prefix=request["idempotency_key_prefix"],
            limit=request["limit"],
        )
    )
    assert preflight_plan["status"] == (
        "sequence_memory_trust_promotion_queue_import_preflight_plan_available"
    )
    assert preflight_plan["summary"]["preflight_candidate_count"] == 1
    assert preflight_plan["summary"]["batch_preflight_request_available"] is True
    assert preflight_plan["summary"]["queue_mutation"] is False
    assert preflight_plan["summary"]["live_authority_granted"] is False


def test_promotion_queue_importer_result_consumer_accepts_idempotent_import_noop(
    tmp_path,
):
    prior_consumer, proposals = _queue_ordering_plan_result_consumer_ready(tmp_path)
    request = prior_consumer["promotion_queue_importer_request"]
    queue = OperatorProposalQueue()
    proposal_payloads = _applyable_proposals(proposals)
    first_importer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
            queue_ordering_plan=request["queue_ordering_plan"],
            proposal_payloads=proposal_payloads,
            target_queue=queue,
            apply=True,
            operator_ack=True,
        )
    )
    second_importer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
            queue_ordering_plan=request["queue_ordering_plan"],
            proposal_payloads=proposal_payloads,
            target_queue=queue,
            apply=True,
            operator_ack=True,
        )
    )

    assert first_importer["summary"]["queue_mutation"] is True
    assert second_importer["status"] == "sequence_memory_trust_promotion_queue_importer_applied"
    assert second_importer["summary"]["queue_mutation"] is False
    assert second_importer["queue_import_results"][0]["status"] == "exists"
    assert second_importer["queue_import_results"][0]["queue_mutation"] is False

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=second_importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
    )
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is True
    )
    assert (
        consumer["promotion_queue_import_preflight_plan_request"][
            "promotion_queue_importer_result"
        ]
        == second_importer
    )


def test_promotion_queue_importer_result_consumer_blocks_partial_import_without_replacement(
    tmp_path,
):
    prior_consumer, importer = _partial_queue_importer_result(tmp_path)

    assert (
        importer["status"]
        == "sequence_memory_trust_promotion_queue_importer_apply_blocked"
    )
    assert importer["summary"]["queue_mutation"] is True
    assert importer["summary"]["queue_mutation_count"] == 1
    assert importer["summary"]["blocked_import_count"] == 1

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked"
    )
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["queue_ordering_plan_record_count"] == 2
    assert consumer["summary"]["queue_import_result_count"] == 2
    assert consumer["summary"]["imported_proposal_count"] == 1
    assert consumer["summary"]["imported_proposal_ids"] == ["proposal-first"]
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert (
        "applied_sequence_policy_queue_importer_result_required_before_preflight_plan"
        in consumer["summary"]["blocked_by"]
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False


def test_promotion_queue_importer_result_consumer_rejects_missing_sequence_context(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged = deepcopy(importer)
    forged["queue_import_specs"][0].pop("current_sequence_review_freshness", None)
    forged["queue_import_specs"][0]["operator_proposal_payload"][
        "sequence_policy_queue_import"
    ].pop("current_sequence_review_freshness", None)
    forged["queue_import_results"][0]["operator_proposal"]["raw"][
        "sequence_policy_queue_import"
    ].pop("current_sequence_review_freshness", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_status_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_spec_current_sequence_review_freshness_missing"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_spec_payload_current_sequence_review_freshness_missing"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_result_current_sequence_review_freshness_missing"
        )
        for item in invalid
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_live_sequence_context(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged = deepcopy(importer)
    live_context = {
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
    forged["queue_import_specs"][0][
        "current_sequence_review_freshness"
    ] = live_context
    forged["queue_import_specs"][0]["operator_proposal_payload"][
        "sequence_policy_queue_import"
    ]["current_sequence_review_freshness"] = live_context
    forged["queue_import_results"][0]["operator_proposal"]["raw"][
        "sequence_policy_queue_import"
    ]["current_sequence_review_freshness"] = live_context

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_status_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_spec_current_sequence_review_freshness_read_only_not_true"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_spec_payload_current_sequence_review_freshness_live_write_not_false"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_result_current_sequence_review_freshness_apply_attempted_not_false"
        )
        for item in invalid
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_missing_prior_sequence_review_freshness(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged_prior = deepcopy(prior_consumer)
    request = forged_prior["promotion_queue_importer_request"]
    for template in request["promotion_acceptance_record_templates"]:
        template.pop("current_sequence_review_freshness", None)
    request.pop("memory_trust_review_request_specs", None)
    forged_prior["summary"].pop("fresh_sequence_review_context_status", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=forged_prior,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "queue_importer_result_consumer_prior_summary_freshness_not_fresh"
        in invalid
    )
    assert (
        "queue_importer_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        in invalid
    )
    assert "proposal-first_current_sequence_review_freshness_missing" in invalid
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_live_prior_sequence_review_freshness(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
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
    request = forged_prior["promotion_queue_importer_request"]
    request["promotion_acceptance_record_templates"][0][
        "current_sequence_review_freshness"
    ] = live_freshness
    request["memory_trust_review_request_specs"][0][
        "current_sequence_review_freshness"
    ] = live_freshness

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=forged_prior,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "invalid"
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first_current_sequence_review_freshness_read_only_not_true" in invalid
    assert "proposal-first_current_sequence_review_freshness_live_write_not_false" in invalid
    assert (
        "proposal-first_current_sequence_review_freshness_apply_attempted_not_false"
        in invalid
    )
    assert (
        "proposal-first:current_sequence_review_freshness_queue_import_spec_mismatch"
        in invalid
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )
    assert (
        "memory_trust_review_request_specs"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_forged_importer_authority(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged = deepcopy(importer)
    forged["safety"]["live_authority_granted"] = True
    forged["summary"]["can_apply_live"] = True
    forged["queue_import_results"][0]["operator_proposal"]["raw"][
        "sequence_policy_queue_import"
    ]["can_apply_live"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "queue_importer_result_consumer_importer_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "queue_importer_result_consumer_importer_summary_can_apply_live_not_false"
        in invalid
    )
    assert any(item.endswith(":metadata_can_apply_live_not_false") for item in invalid)
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_forged_spec_identity(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged = deepcopy(importer)
    forged["queue_import_specs"][0]["source_idempotency_key"] = (
        "memory-write:unrelated-proposal"
    )
    forged["queue_import_specs"][0]["operator_proposal_payload"][
        "sequence_policy_queue_import"
    ]["source_idempotency_key"] = "memory-write:unrelated-proposal"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_status_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_spec_source_idempotency_key_mismatch"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_spec_payload_source_idempotency_key_mismatch"
        )
        for item in invalid
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_forged_acceptance_template_identity(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged_prior = deepcopy(prior_consumer)
    template = forged_prior["promotion_queue_importer_request"][
        "promotion_acceptance_record_templates"
    ][0]
    template["idempotency_key"] = "seq-policy:forged"
    template["consumer_record_id"] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=forged_prior,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_status_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert "proposal-first:idempotency_key_queue_import_spec_mismatch" in invalid
    assert "proposal-first:consumer_record_id_missing" in invalid
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )
    assert (
        "promotion_acceptance_record_templates"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_forged_result_identity(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged = deepcopy(importer)
    forged["queue_import_results"][0]["operator_proposal"]["raw"][
        "sequence_policy_queue_import"
    ]["source_idempotency_key"] = "memory-write:unrelated-proposal"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=prior_consumer,
            promotion_queue_importer_result=forged,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["proposal_status_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            ":queue_importer_result_consumer_result_import_source_idempotency_key_mismatch"
        )
        for item in invalid
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_rejects_forged_prior_consumer(
    tmp_path,
):
    prior_consumer, importer, _ = _queue_importer_applied(tmp_path)
    forged_prior = deepcopy(prior_consumer)
    forged_prior["summary"]["promotion_queue_importer_created"] = True
    forged_prior["summary"]["promotion_acceptance_record_template_count"] = 2
    forged_prior["summary"][
        "promotion_acceptance_record_template_proposal_ids"
    ] = ["proposal-second"]
    forged_prior["promotion_queue_importer_request"]["operator_ack"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
            promotion_queue_ordering_plan_result_consumer=forged_prior,
            promotion_queue_importer_result=importer,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
    )
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "queue_importer_result_consumer_prior_summary_promotion_queue_importer_created_not_false"
        in invalid
    )
    assert (
        "queue_importer_result_consumer_prior_summary_promotion_acceptance_record_template_count_mismatch"
        in invalid
    )
    assert (
        "queue_importer_result_consumer_prior_summary_template_proposal_ids_mismatch"
        in invalid
    )
    assert (
        "queue_importer_result_consumer_prior_request_operator_ack_not_false"
        in invalid
    )
    assert (
        "promotion_queue_importer_result"
        not in consumer["promotion_queue_import_preflight_plan_request"]
    )


def test_promotion_queue_importer_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert (
        consumer["summary"]["promotion_queue_import_preflight_plan_request_available"]
        is False
    )


def test_promotion_queue_importer_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
