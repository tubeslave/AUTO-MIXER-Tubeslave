from unittest.mock import AsyncMock, patch

import pytest

from server import AutoMixerServer


@pytest.fixture
def server():
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch(
        "server.BleedService"
    ):
        yield AutoMixerServer()


def _activate_wing_boundary(server):
    server.operator_mode = "supervised"
    server.connection_mode = "wing"
    server._is_wing_deployment_write_boundary_active = lambda: True


def _policy_gate_payload(*, value_type="pan", allowed_live_write_kinds=None):
    allowed_live_write_kinds = allowed_live_write_kinds or ["fader", "gain", "pan"]
    return {
        "type": "operator_proposal_apply_sequence_audit_policy_gate",
        "status": "ready",
        "summary": {
            "policy_gate_only": True,
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
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": allowed_live_write_kinds,
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
        "policy_evidence": [
            {
                "proposal_id": "proposal-sequence-pan",
                "status": "policy_evidence_ready",
                "eligible_for_supervised_sequence_policy": True,
                "execution_allowed_by_policy_gate": False,
                "policy_mutation_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "value_type": value_type,
                "channel": 2,
                "effective_value": 0.25,
                "idempotency_key": "memory-write:proposal-sequence-pan",
                "readback_status": "verified_from_cache",
                "proof_status": "verified",
                "rollback_available": True,
                "live_apply_entrypoint": "apply_operator_proposal",
            }
        ],
        "blocked_results": [],
    }


def _memory_draft_payload(*, value_type="pan", allowed_live_write_kinds=None):
    allowed_live_write_kinds = allowed_live_write_kinds or ["fader", "gain", "pan"]
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_draft",
        "status": "ready",
        "summary": {
            "memory_draft_only": True,
            "append_only_memory_write_allowed_by_draft": False,
            "execution_allowed_by_draft": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "safety": {
            "memory_draft_only": True,
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": allowed_live_write_kinds,
            "append_only_memory_write_allowed_by_draft": False,
            "execution_allowed_by_draft": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "memory_draft_records": [
            {
                "draft_id": "operator-sequence-policy-memory:proposal-sequence-pan",
                "proposal_id": "proposal-sequence-pan",
                "status": "memory_draft_ready",
                "learning_scope": "supervised_fader_gain_sequence_execution",
                "target_memory_path": ".ai/memory/project.md",
                "append_only": True,
                "memory_write_allowed_by_draft": False,
                "append_only_memory_write_allowed_by_draft": False,
                "memory_mutation": False,
                "runtime_memory_mutation": False,
                "policy_mutation": False,
                "model_mutation": False,
                "queue_mutation": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "evidence": {
                    "value_type": value_type,
                    "channel": 2,
                    "effective_value": 0.25,
                    "idempotency_key": "memory-write:proposal-sequence-pan",
                    "readback_status": "verified_from_cache",
                    "proof_status": "verified",
                    "rollback_available": True,
                    "live_apply_entrypoint": "apply_operator_proposal",
                },
                "draft_text": "Pan evidence must not enter WING sequence memory.",
            }
        ],
        "blocked_results": [],
    }


@pytest.mark.asyncio
async def test_sequence_policy_memory_draft_blocks_pan_on_wing_boundary(server):
    _activate_wing_boundary(server)
    server.send_to_client = AsyncMock()

    await server._dispatch["draft_operator_proposal_apply_sequence_policy_memory"](
        object(),
        {"policy_gate_payload": _policy_gate_payload()},
    )

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_pipeline_migration_required"
    assert payload["status"] == "migration_required"
    assert payload["reason"] == "server_owned_sequence_state_required"
    assert payload["blocked_handler"] == (
        "draft_operator_proposal_apply_sequence_policy_memory"
    )
    assert payload["no_additional_console_send_proven"] is True
    assert payload["soundcheck_completion_blocked"] is True


@pytest.mark.asyncio
async def test_sequence_policy_memory_writer_blocks_pan_on_wing_boundary(server):
    _activate_wing_boundary(server)
    server.send_to_client = AsyncMock()

    await server._dispatch["write_operator_proposal_apply_sequence_policy_memory"](
        object(),
        {"memory_draft_payload": _memory_draft_payload()},
    )

    payload = server.send_to_client.await_args.args[1]
    reasons = {item["reason"] for item in payload["blocked_results"]}
    assert payload["status"] == "memory_write_replacement_required"
    assert payload["summary"]["write_result_count"] == 0
    assert payload["safety"]["allowed_live_write_kinds"] == ["fader", "gain"]
    assert "safety_allowed_live_write_kinds_widened" in reasons
    assert "memory_draft_record_value_type_not_allowed" in reasons
