from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _imported_action(
    proposal_id,
    *,
    index,
    can_apply=True,
    rank_source="sequence_policy_consumer",
    status="pending",
):
    return {
        "id": proposal_id,
        "title": f"Imported {proposal_id}",
        "source": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
        ),
        "status": status,
        "can_apply": can_apply,
        "raw": {
            "sequence_policy_queue_import": {
                "queue_ordering_plan_record_id": f"plan:{proposal_id}",
                "ordering_preview_record_id": f"preview:{proposal_id}",
                "consumer_record_id": f"consumer:{proposal_id}",
                "sequence_policy_gate_record_id": f"gate:{proposal_id}",
                "policy_key": f"{proposal_id}/memory-write:{proposal_id}",
                "planned_queue_index": index,
                "rank_source": rank_source,
                "recommended_sequence_policy_bucket": "trusted_sequence_policy",
                "operator_ack_required_before_queue_import": True,
                "operator_ack_required_before_sequence_execution": True,
                "sequence_consumer_required_before_execution": True,
                "live_write": False,
                "can_apply_live": False,
            }
        },
    }


def test_sequence_policy_queue_import_preflight_plan_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py"
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


def test_sequence_policy_queue_import_preflight_plan_not_ready_without_imports():
    plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan()
    )

    assert (
        plan["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan"
    )
    assert (
        plan["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
    )
    assert (
        plan["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_plan_not_ready"
    )
    assert plan["summary"]["sequence_policy_imported_pending_count"] == 0
    assert plan["summary"]["preflight_candidate_count"] == 0
    assert plan["summary"]["batch_preflight_request_available"] is False
    assert plan["summary"]["batch_preflight_invoked"] is False
    assert plan["summary"]["live_write"] is False
    assert plan["summary"]["queue_mutation"] is False
    assert plan["summary"]["idempotency_reserved"] is False
    assert plan["batch_preflight_request"]["proposal_ids"] == []
    assert plan["preflight_request_specs"] == []


def test_sequence_policy_queue_import_preflight_plan_orders_applyable_imports():
    actions = [
        _imported_action("proposal-late", index=1),
        {
            "id": "ordinary-proposal",
            "status": "pending",
            "can_apply": True,
            "source": "operator_proposal_queue",
        },
        _imported_action(
            "proposal-analysis",
            index=2,
            can_apply=False,
            rank_source="original_order",
        ),
        _imported_action("proposal-first", index=0),
    ]

    plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
            pending_actions=actions,
            idempotency_key_prefix="seq-import",
        )
    )

    assert (
        plan["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_plan_available"
    )
    assert plan["summary"]["input_pending_action_count"] == 4
    assert plan["summary"]["sequence_policy_imported_pending_count"] == 3
    assert plan["summary"]["preflight_candidate_count"] == 2
    assert plan["summary"]["analysis_only_count"] == 1
    assert plan["summary"]["imported_proposal_ids"] == [
        "proposal-first",
        "proposal-late",
        "proposal-analysis",
    ]
    assert plan["summary"]["preflight_candidate_proposal_ids"] == [
        "proposal-first",
        "proposal-late",
    ]
    assert plan["summary"]["analysis_only_proposal_ids"] == ["proposal-analysis"]
    assert plan["summary"]["batch_preflight_request_available"] is True
    assert plan["summary"]["sequence_plan_created"] is False
    assert plan["summary"]["sequence_execution_allowed_by_preflight_plan"] is False
    assert plan["summary"]["idempotency_reserved"] is False
    assert plan["batch_preflight_request"] == {
        "handler": "preflight_operator_proposal_apply_batch",
        "proposal_ids": ["proposal-first", "proposal-late"],
        "idempotency_key_prefix": "seq-import",
        "limit": 20,
        "approved": False,
        "operator_must_set_approved_true": True,
        "read_only_template": True,
        "live_write": False,
        "apply_attempted": False,
        "queue_mutation": False,
        "idempotency_reserved": False,
    }
    first = plan["preflight_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_preflight_index"] == 0
    assert first["planned_queue_index"] == 0
    assert first["idempotency_key"] == "seq-import:proposal-first"
    assert first["operator_approval_required_for_preflight_ready"] is True
    assert first["sequence_consumer_required_before_execution"] is True
    assert first["live_write"] is False
    assert first["queue_mutation"] is False
    assert first["idempotency_reserved"] is False


def test_sequence_policy_queue_import_preflight_plan_rejects_forged_metadata():
    action = _imported_action("forged", index=0)
    metadata = action["raw"]["sequence_policy_queue_import"]
    metadata["live_write"] = True
    metadata["sequence_consumer_required_before_execution"] = False

    plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
            pending_actions=[action],
        )
    )

    assert (
        plan["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_plan_replacement_required"
    )
    assert plan["summary"]["preflight_candidate_count"] == 0
    assert plan["summary"]["batch_preflight_request_available"] is False
    assert "forged:live_write_not_false" in plan["summary"]["invalid_record_ids"]
    assert (
        "forged:sequence_consumer_required_not_true"
        in plan["summary"]["invalid_record_ids"]
    )
    assert plan["batch_preflight_request"]["proposal_ids"] == []


def test_sequence_policy_queue_import_preflight_plan_rejects_ambiguous_import_order():
    plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
            pending_actions=[
                _imported_action("duplicate-proposal", index=0),
                _imported_action("duplicate-proposal", index=0),
            ],
        )
    )

    assert (
        plan["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_plan_replacement_required"
    )
    assert plan["summary"]["preflight_candidate_count"] == 0
    assert plan["summary"]["batch_preflight_request_available"] is False
    assert (
        "duplicate_proposal_id:duplicate-proposal"
        in plan["summary"]["invalid_record_ids"]
    )
    assert "duplicate_planned_queue_index:0" in plan["summary"]["invalid_record_ids"]
    assert plan["batch_preflight_request"]["proposal_ids"] == []
    assert plan["batch_preflight_request"]["approved"] is False
    assert plan["summary"]["live_authority_granted"] is False


def test_sequence_policy_queue_import_preflight_plan_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
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
    preflight_plan = snapshot[
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
    ]
    assert preflight_plan["safety"]["live_write"] is False
    assert preflight_plan["safety"]["queue_mutation"] is False
    assert preflight_plan["summary"]["preflight_candidate_count"] == 0


def test_sequence_policy_queue_import_preflight_plan_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
