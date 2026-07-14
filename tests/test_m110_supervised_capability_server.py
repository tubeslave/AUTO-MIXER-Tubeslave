from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autofoh_safety import ChannelFaderMove
from decision_queue_wire_contract import (
    canonical_json_bytes,
    project_outbound_message,
)
from master_capture_route_proof import MasterCaptureRouteAuthority
from m110_m006_restore import (
    M110_M006_RESTORE_ADAPTER_REVISION,
    M110_M006_RESTORE_PROTOCOL_REVISION,
    build_m110_m006_restore_plan,
)
from m110_manual_write_wal import M110_WAL_CONTRACT
from operator_proposal_queue import OperatorProposalQueue
from operator_proposal_result_store import (
    ResultEvidenceReservationBinding,
    ResultEvidenceStore,
)
from server import AutoMixerServer
from soundcheck_transaction_plan import build_soundcheck_transaction_plans
from supervised_console_transaction import (
    AudioRecheckResult,
    ConsoleTransactionOperation,
)
from supervised_proof_capability import (
    M110_M006_RESTORE_BUNDLE_KIND,
    M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
    M110CapabilityBinding,
    M110RestoredProofReceipt,
    SupervisedCapabilityStore,
    new_m110_attended_control_authority,
    new_m110_restored_proof_receipt,
)


@pytest.fixture
def m110_server(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(tmp_path / "console-runtime"),
    )
    monkeypatch.setenv(
        "AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR",
        str(tmp_path / "manual-events"),
    )
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch(
        "server.BleedService"
    ):
        instance = AutoMixerServer(ws_host="localhost", ws_port=8765)
    instance.operator_proposal_queue = OperatorProposalQueue(
        result_store=ResultEvidenceStore(
            tmp_path / "proposal-results",
            min_free_bytes=0,
        ),
        result_evidence_lifecycle_ready=True,
    )
    # This module exercises the post-creation capability lifecycle.  Proposal
    # provenance itself is covered separately; keep these fixtures inside the
    # private server-owned lane introduced by M110.
    instance.is_m110_server_owned_proposal = MagicMock(return_value=True)
    return instance


def _binding(*, websocket, operation_digest=None):
    bundle = b'{"bundle":"m110-unit"}'
    digest = operation_digest or (
        "sha256:" + hashlib.sha256(bundle).hexdigest()
    )
    return M110CapabilityBinding(
        capability_kind="restore_only_proof",
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "3" * 64,
        source_head="4" * 40,
        source_tree="5" * 40,
        connection_generation=6,
        transport_generation=7,
        console_serial="0100Q1V0604AAE214",
        firmware_version="3.0.5",
        route_proof_id="m110-route-1",
        route_artifact_sha256="sha256:" + "6" * 64,
        capture_hardware_uid="usb:5015:1291:214",
        emergency_generation=8,
        owner_connection_id=id(websocket),
        proposal_id="m110-proposal-1",
        review_contract_version=2,
        review_content_digest="sha256:" + "7" * 64,
        operation_digest=digest,
        canonical_bundle_bytes=bundle,
        bundle_kind="single_soundcheck_operation",
        approval_id="m110-approval-1",
        idempotency_key="m110-proof:unit",
        scope_addresses=("/ch/1/eq/on",),
        target_fingerprints=(("/ch/1/eq/on", "1"),),
        _owner_ref=websocket,
        _route_ref=object(),
        _transport_ref=object(),
        _capture_ref=object(),
    )


def _apply_request(binding):
    return {
        "type": "apply_operator_proposal",
        "proposal_id": binding.proposal_id,
        "approved": True,
        "approval_id": binding.approval_id,
        "idempotency_key": binding.idempotency_key,
        "expected_review_contract_version": 2,
        "expected_review_content_digest": binding.review_content_digest,
        "expected_operation_digest": binding.operation_digest,
    }


def _plan():
    return {
        "transaction_id": "m110-transaction-1",
        "operations": [
            {
                "operation_id": "m110-operation-1",
                "family_id": "eq",
                "write_kind": "eq_enable",
                "channel": 1,
                "parameter": "eq_enabled",
                "target_value": True,
            }
        ],
    }


def test_capability_blockers_recognize_server_owned_nested_m110_wal_row(
    m110_server,
):
    server = m110_server
    websocket = object()
    binding = _binding(websocket=websocket)
    payload = {
        "scope": "supervised_transaction",
        "proof_run": True,
        "restore_after_verify": True,
        "capability_kind": "restore_only_proof",
        "proposal_id": binding.proposal_id,
        "review_content_digest": binding.review_content_digest,
        "operation_digest": binding.operation_digest,
    }
    event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": "m110-nested-own-row",
        "phase": "prepared",
        "wal_contract": M110_WAL_CONTRACT,
        "wal_payload": payload,
    }
    server._manual_write_prepared_transactions = {
        "m110-nested-own-row": event,
    }
    server._manual_write_recovery_transactions = {
        "m110-nested-own-row": {
            **event,
            "phase_sequence": ["prepared"],
            "latest_event": event,
        },
    }
    server._m110_route_global_blockers = MagicMock(
        return_value=[
            "manual_write_wal_prepared_transaction_present",
            "manual_write_wal_recovery_required",
            "independent_blocker",
        ]
    )

    assert server._m110_capability_global_blockers(
        websocket,
        active_binding=binding,
    ) == ["independent_blocker"]


def _restored_proof_result(*, proposal_id, review_digest, operation_digest):
    return {
        "transaction_id": "m110-transaction-1",
        "status": "proof_passed",
        "reason": "proof_target_verified_and_restored",
        "success": True,
        "applied_operation_ids": ["m110-operation-1"],
        "rollback_operation_ids": ["m110-operation-1"],
        "events": [],
        "proof_run": True,
        "restore_after_verify": True,
        "capability_kind": "restore_only_proof",
        "proposal_id": proposal_id,
        "review_content_digest": review_digest,
        "operation_digest": operation_digest,
        "accepted": False,
        "owner_review_required": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "no_console_send_proven": False,
        "physical_verified": True,
        "proof_target_verified": True,
        "physical_state_restored": True,
        "terminal_after_console_send": True,
        "rollback_attempted": True,
        "rollback_succeeded": True,
        "rollback_status": "verified",
        "wal_terminal": "proof_restored",
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "fresh_analysis_required": False,
    }


@pytest.mark.asyncio
async def test_capability_is_consumed_before_headroom_precheck_and_never_writes(
    m110_server,
):
    server = m110_server
    websocket = object()
    proposal = object()
    binding = _binding(websocket=websocket)
    store = SupervisedCapabilityStore(token_factory=lambda: "a" * 64)
    issued, _ = store.issue(binding)
    grant, _ = store.consume(issued.token, binding)
    server._m110_capability_store = store
    server._verify_m110_cp05_hardware_gate = MagicMock()
    server._verified_master_capture_route_proof_with_fresh_hardware = AsyncMock(
        return_value={"verified": True}
    )
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._m110_capability_binding_for_apply = MagicMock(
        return_value=(binding, "current")
    )
    server._m110_soundcheck_child_proposal_contexts[binding.proposal_id] = {
        "proposal": proposal,
        "canonical_bundle_bytes": binding.canonical_bundle_bytes,
        "required_capability_kind": binding.capability_kind,
        "source_plan": _plan(),
    }
    server.refresh_soundcheck_child_preflight = AsyncMock(
        return_value={"refreshed": True}
    )
    server.validate_soundcheck_child_for_apply = MagicMock(
        return_value={"allowed": True}
    )
    server._m110_combined_live_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "failed",
            "master_true_peak_headroom_insufficient",
            {
                "master_true_peak_dbtp": -0.2,
                "age_sec": 0.1,
                "max_age_sec": 5.0,
                "ceiling_dbtp": -1.0,
                "route_verified": True,
                "capture_real_input_fresh": True,
                "master_channels": [47, 48],
            },
        )
    )
    server._apply_supervised_soundcheck_transaction_unlocked = AsyncMock()
    reservation = ResultEvidenceReservationBinding(
        token="b" * 32,
        proposal_id=binding.proposal_id,
        review_content_digest=binding.review_content_digest,
    )

    result = await server.apply_m110_restore_only_proof_transaction(
        websocket=websocket,
        request=_apply_request(binding),
        capability_grant=grant,
        proposal=proposal,
        plan_payload=_plan(),
        result_evidence_reservation_binding=reservation,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "master_true_peak_headroom_insufficient"
    assert result["no_console_send_proven"] is True
    server._apply_supervised_soundcheck_transaction_unlocked.assert_not_awaited()
    assert store.active_count() == 0
    assert grant.execution_status() == "terminal"
    assert store.consume(issued.token, binding) == (
        None,
        "m110_capability_token_replayed_or_revoked",
    )


def test_global_drift_blocks_before_preclaim_consume(
    m110_server,
):
    server = m110_server
    websocket = object()
    binding = _binding(websocket=websocket)
    store = SupervisedCapabilityStore(token_factory=lambda: "c" * 64)
    issued, _ = store.issue(binding)
    server._m110_capability_store = store
    server._m110_capability_global_blockers = MagicMock(
        return_value=["m110_live_source_snapshot_drift"]
    )
    server._m110_capability_binding_for_apply = MagicMock(
        return_value=(binding, "current")
    )

    result = server.consume_m110_capability_for_apply(
        websocket=websocket,
        request=_apply_request(binding),
        token=issued.token,
        proposal=object(),
    )

    assert result == {
        "consumed": False,
        "reason": "m110_capability_global_safety_gate_failed",
        "blockers": ["m110_live_source_snapshot_drift"],
    }
    assert store.active_count() == 1


@pytest.mark.asyncio
async def test_result_reservation_is_required_before_writer_or_hardware(
    m110_server,
):
    server = m110_server
    websocket = object()
    binding = _binding(websocket=websocket)
    store = SupervisedCapabilityStore(token_factory=lambda: "e" * 64)
    issued, _ = store.issue(binding)
    grant, _ = store.consume(issued.token, binding)
    server._verified_master_capture_route_proof_with_fresh_hardware = AsyncMock()

    result = await server.apply_m110_restore_only_proof_transaction(
        websocket=websocket,
        request=_apply_request(binding),
        capability_grant=grant,
        proposal=object(),
        plan_payload=_plan(),
        result_evidence_reservation_binding=None,
    )

    assert result["reason"] == "m110_result_evidence_reservation_binding_required"
    server._verified_master_capture_route_proof_with_fresh_hardware.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_uses_capability_lane_and_never_persists_plaintext_token(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    plan = _plan()
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        created = server.operator_proposal_queue.create(
            {
                "id": "m110-proposal-1",
                "title": "M110 restore-only proof",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": plan,
            },
            server.get_operator_mode_status(),
        )
    assert created["status"] == "created"
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-proposal-1"
    )
    operation_digest = "sha256:" + "9" * 64
    apply_result = _restored_proof_result(
        proposal_id=proposal.id,
        review_digest=proposal.review_content_digest,
        operation_digest=operation_digest,
    )
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.is_m110_server_owned_proposal = MagicMock(return_value=True)
    grant = MagicMock()
    grant.binding.bundle_kind = "single_soundcheck_operation"
    grant.binding.capability_kind = "restore_only_proof"
    consume_statuses = []

    def consume_before_claim(**kwargs):
        consume_statuses.append(kwargs["proposal"].status)
        return {
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }

    server.consume_m110_capability_for_apply = MagicMock(
        side_effect=consume_before_claim
    )
    server.apply_m110_capability_transaction = AsyncMock(
        return_value=apply_result
    )
    server.record_m110_restored_proof_receipt = MagicMock(
        return_value={"recorded": True, "reason": "recorded"}
    )
    server.record_automation_decision_trace = MagicMock()
    server.send_to_client = AsyncMock()
    token = "m110cap_v1_" + "f" * 64
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-1",
        "idempotency_key": "m110-proof:handler",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": operation_digest,
        "supervised_capability_token": token,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    terminal = server.operator_proposal_queue.find(proposal_id=proposal.id)
    assert terminal.status == "proof_passed"
    assert consume_statuses == ["pending"]
    assert request.get("supervised_capability_token") is None
    assert server.consume_m110_capability_for_apply.call_args.kwargs["token"] == token
    assert (
        server.apply_m110_capability_transaction.await_args.kwargs[
            "capability_grant"
        ]
        is grant
    )
    server.record_m110_restored_proof_receipt.assert_called_once()
    trace = server.record_automation_decision_trace.call_args.kwargs
    assert trace["status"] == "proof_passed"
    normal_mode = server.get_operator_mode_status()
    assert normal_mode["capabilities"]["can_apply_to_console"] is False
    assert normal_mode["capabilities"]["allowed_live_write_kinds"] == []
    sent = server.send_to_client.await_args.args[1]
    stored = server.operator_proposal_queue.load_result_evidence(
        terminal,
        expected_result_digest=terminal.result_digest,
    )
    assert token not in json.dumps(sent, default=str)
    assert token.encode("ascii") not in stored
    assert token not in repr(trace)
    assert token not in repr(
        server.record_m110_restored_proof_receipt.call_args.kwargs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("already_satisfied", [False, True])
@pytest.mark.parametrize("commit_recorded", [False, True])
async def test_handler_projects_trusted_soundcheck_parent_commit_truth(
    m110_server,
    already_satisfied,
    commit_recorded,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    plan = _plan()
    proposal_id = (
        "m110-handler-plan-commit-"
        + ("satisfied" if already_satisfied else "write")
        + ("-recorded" if commit_recorded else "-failed")
    )
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        created = server.operator_proposal_queue.create(
            {
                "id": proposal_id,
                "title": "M110 soundcheck child commit",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": plan,
            },
            server.get_operator_mode_status(),
        )
    assert created["status"] == "created"
    proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
    operation_digest = "sha256:" + "9" * 64
    operation_ids = ["m110-operation-1"]
    apply_result = {
        "transaction_id": plan["transaction_id"],
        "status": "applied",
        "reason": "all_operations_applied",
        "success": True,
        "applied_operation_ids": (
            [] if already_satisfied else operation_ids
        ),
        "rollback_operation_ids": [],
        "already_satisfied_operation_ids": (
            operation_ids if already_satisfied else []
        ),
        "events": [],
        "proof_run": False,
        "restore_after_verify": False,
        "capability_kind": "soundcheck_child_commit_once",
        "proposal_id": proposal.id,
        "review_content_digest": proposal.review_content_digest,
        "operation_digest": operation_digest,
        "bundle_kind": M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
        "proof_receipt_id": "m110proof_v1_" + "8" * 64,
        "accepted": True,
        "owner_review_required": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "no_console_send_proven": already_satisfied,
        "physical_verified": True,
        "proof_target_verified": False,
        "physical_state_restored": False,
        "terminal_after_console_send": not already_satisfied,
        "rollback_attempted": False,
        "rollback_succeeded": False,
        "rollback_status": "not_attempted",
        "wal_terminal": "committed",
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "fresh_analysis_required": False,
        # This mutable value must never outrank the post-finalize recorder.
        "parent_commit_recorded": not commit_recorded,
    }
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.is_m110_server_owned_proposal = MagicMock(return_value=True)
    grant = MagicMock()
    grant.binding = SimpleNamespace(
        bundle_kind=M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
        capability_kind="soundcheck_child_commit_once",
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    server.freeze_m110_plan_commit_context = MagicMock(
        return_value={"schema": "automixer.m110_plan_commit_context.v1"}
    )
    server.apply_m110_capability_transaction = AsyncMock(
        return_value=apply_result
    )
    commit_reason = (
        "m110_plan_commit_recorded"
        if commit_recorded
        else "m110_plan_commit_batch_journal_failed"
    )
    server.record_m110_plan_commit = MagicMock(
        return_value={
            "recorded": commit_recorded,
            "reason": commit_reason,
            "run_id": "m110-handler-run" if commit_recorded else None,
        }
    )
    server.reconcile_soundcheck_readiness = AsyncMock(
        return_value={"run_id": "m110-handler-run", "ready": False}
    )
    server._revoke_m110_route_authority = MagicMock()
    server.record_automation_decision_trace = MagicMock()
    server.send_to_client = AsyncMock()
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-handler-approval",
        "idempotency_key": f"m110-handler:{proposal.id}",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": operation_digest,
        "supervised_capability_token": "m110cap_v1_" + "f" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    raw_sent = server.send_to_client.await_args.args[1]
    assert "parent_commit_recorded" not in raw_sent
    receipt = server.send_to_client.await_args.kwargs["trusted_receipt"]
    projected = json.loads(
        project_outbound_message(raw_sent, trusted_receipt=receipt)
    )
    assert projected["status"] == "applied"
    assert projected["success"] is True
    assert projected["physical_verified"] is True
    assert projected["applied_operation_ids"] == apply_result[
        "applied_operation_ids"
    ]
    assert projected["rollback_operation_ids"] == []
    assert projected["already_satisfied_operation_ids"] == apply_result[
        "already_satisfied_operation_ids"
    ]
    assert projected["parent_commit_recorded"] is commit_recorded
    assert projected["parent_commit_reason"] == commit_reason
    server.record_m110_plan_commit.assert_called_once()
    if commit_recorded:
        server.reconcile_soundcheck_readiness.assert_awaited_once_with(
            "m110-handler-run"
        )
        server._revoke_m110_route_authority.assert_not_called()
    else:
        server.reconcile_soundcheck_readiness.assert_not_awaited()
        server._revoke_m110_route_authority.assert_called_once_with(
            "m110_plan_commit_not_recorded"
        )


@pytest.mark.asyncio
async def test_handler_routes_fixed_reset_without_client_transaction_plan(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    created = server.operator_proposal_queue.create(
        {
            "id": "m110-fixed-reset-handler",
            "title": "M110 server-owned fixed reset proof",
            "kind": "m110_maintenance_dsp_reset_proof",
            "source": "m110_server_owned_maintenance",
            "m110_maintenance_reset": {
                "bundle_kind": "maintenance_dsp_reset_bundle",
                "operation_count": 64,
            },
        },
        server.get_operator_mode_status(),
    )
    assert created["status"] == "created"
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-fixed-reset-handler"
    )
    operation_digest = "sha256:" + "9" * 64
    apply_result = _restored_proof_result(
        proposal_id=proposal.id,
        review_digest=proposal.review_content_digest,
        operation_digest=operation_digest,
    )
    apply_result.update(
        {
            "bundle_kind": "maintenance_dsp_reset_bundle",
            "review_preimage_sha256": "sha256:" + "8" * 64,
        }
    )
    server.is_m110_fixed_reset_proposal = MagicMock(return_value=True)
    server.is_m110_server_owned_proposal = MagicMock(return_value=True)
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    grant = MagicMock()
    grant.binding = SimpleNamespace(
        bundle_kind="maintenance_dsp_reset_bundle",
        capability_kind="restore_only_proof",
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    server.apply_m110_capability_transaction = AsyncMock(
        return_value=apply_result
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock(
        return_value={"recorded": True, "reason": "recorded"}
    )
    server.record_automation_decision_trace = MagicMock()
    server.send_to_client = AsyncMock()
    token = "m110cap_v1_" + "e" * 64
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-fixed-reset-handler-approval",
        "idempotency_key": "m110-proof:fixed-reset-handler",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": operation_digest,
        "supervised_capability_token": token,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    terminal = server.operator_proposal_queue.find(proposal_id=proposal.id)
    assert terminal.status == "proof_passed"
    dispatched = server.apply_m110_capability_transaction.await_args.kwargs
    assert dispatched["capability_grant"] is grant
    assert dispatched["plan_payload"] is None
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()
    receipt_call = server.record_m110_restored_proof_receipt.call_args.kwargs
    assert receipt_call["capability_grant"] is grant
    assert request.get("supervised_capability_token") is None
    assert token not in json.dumps(
        server.send_to_client.await_args.args[1],
        default=str,
    )


@pytest.mark.asyncio
async def test_post_consume_claim_exception_releases_evidence_and_grant(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-claim-exception",
                "title": "M110 claim exception",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-claim-exception"
    )
    operation_digest = "sha256:" + "8" * 64
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    original_claim = server.operator_proposal_queue.claim_verified_for_apply

    def claim_then_raise(*args, **kwargs):
        claimed = original_claim(*args, **kwargs)
        assert claimed["claimed"] is True
        raise RuntimeError("injected_after_claim")

    server.operator_proposal_queue.claim_verified_for_apply = claim_then_raise
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-claim-exception",
        "idempotency_key": "m110-proof:claim-exception",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": operation_digest,
        "supervised_capability_token": "m110cap_v1_" + "7" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_consume_exception_releases_evidence_and_revokes_prepared_authority(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-consume-exception",
                "title": "M110 consume exception",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-consume-exception"
    )
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        side_effect=RuntimeError("injected_consume_failure")
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-consume-exception",
        "idempotency_key": "m110-proof:consume-exception",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "2" * 64,
        "supervised_capability_token": "m110cap_v1_" + "3" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "m110_capability_consume_exception"
    assert sent["result_evidence_reserved"] is False
    assert sent["apply_result"]["result_persistence_failed"] is False
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    assert request.get("supervised_capability_token") is None
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_claim_execution_aborts_claim_and_releases_evidence(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-invalid-claim-execution",
                "title": "M110 invalid claim execution",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-invalid-claim-execution"
    )
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    original_claim = server.operator_proposal_queue.claim_verified_for_apply

    def claim_without_execution(*args, **kwargs):
        result = original_claim(*args, **kwargs)
        assert result["claimed"] is True
        return {**result, "execution": None}

    server.operator_proposal_queue.claim_verified_for_apply = (
        claim_without_execution
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-invalid-claim-execution",
        "idempotency_key": "m110-proof:invalid-claim-execution",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "4" * 64,
        "supervised_capability_token": "m110cap_v1_" + "5" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_claim_execution_invalid"
    assert sent["proposal_status_mutation"] is False
    assert sent["result_evidence_reserved"] is False
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    assert server.operator_proposal_queue.result_persistence_failed is False
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_mapping_claim_result_exactly_aborts_real_claim(m110_server):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-non-mapping-claim",
                "title": "M110 non-mapping claim",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-non-mapping-claim"
    )
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    original_claim = server.operator_proposal_queue.claim_verified_for_apply

    def claim_then_return_none(*args, **kwargs):
        result = original_claim(*args, **kwargs)
        assert result["claimed"] is True
        return None

    server.operator_proposal_queue.claim_verified_for_apply = (
        claim_then_return_none
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-non-mapping-claim",
        "idempotency_key": "m110-proof:non-mapping-claim",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "0" * 64,
        "supervised_capability_token": "m110cap_v1_" + "1" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_claim_exception"
    assert sent["proposal_status_mutation"] is False
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_persistence_failed is False
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_identity_drift_cannot_use_mutated_id_to_abort(m110_server):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-claim-identity-before",
                "title": "M110 claim identity drift",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-claim-identity-before"
    )
    frozen_proposal_id = proposal.id
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    original_claim = server.operator_proposal_queue.claim_verified_for_apply

    def claim_then_mutate_identity(*args, **kwargs):
        result = original_claim(*args, **kwargs)
        assert result["claimed"] is True
        proposal.id = "m110-claim-identity-mutated"
        return {**result, "execution": None}

    server.operator_proposal_queue.claim_verified_for_apply = (
        claim_then_mutate_identity
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[frozen_proposal_id] = {
        "prepared": True
    }
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": frozen_proposal_id,
        "approved": True,
        "approval_id": "m110-approval-claim-identity-drift",
        "idempotency_key": "m110-proof:claim-identity-drift",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "a" * 64,
        "supervised_capability_token": "m110cap_v1_" + "b" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_claim_cleanup_failed"
    assert sent["proposal_status_mutation"] is True
    assert proposal.status == "applying"
    assert server.operator_proposal_queue.result_persistence_failed is True
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert frozen_proposal_id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_identity_drift_during_claim_exception_latches_fence(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-pending-identity-before",
                "title": "M110 pending identity drift",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-pending-identity-before"
    )
    frozen_proposal_id = proposal.id
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )

    def mutate_pending_identity_then_raise(*_args, **_kwargs):
        proposal.id = "m110-pending-identity-mutated"
        raise RuntimeError("injected_pending_identity_drift")

    server.operator_proposal_queue.claim_verified_for_apply = (
        mutate_pending_identity_then_raise
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[frozen_proposal_id] = {
        "prepared": True
    }
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": frozen_proposal_id,
        "approved": True,
        "approval_id": "m110-approval-pending-identity-drift",
        "idempotency_key": "m110-proof:pending-identity-drift",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "e" * 64,
        "supervised_capability_token": "m110cap_v1_" + "f" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_claim_cleanup_failed"
    assert sent["proposal_status_mutation"] is True
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_persistence_failed is True
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert frozen_proposal_id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_post_claim_status_latches_cleanup_fence(m110_server):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-unexpected-claim-status",
                "title": "M110 unexpected claim status",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-unexpected-claim-status"
    )
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )

    def claim_with_unexpected_status(*_args, **_kwargs):
        proposal.status = "apply_error"
        raise RuntimeError("injected_unexpected_claim_status")

    server.operator_proposal_queue.claim_verified_for_apply = (
        claim_with_unexpected_status
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-unexpected-claim-status",
        "idempotency_key": "m110-proof:unexpected-claim-status",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "c" * 64,
        "supervised_capability_token": "m110cap_v1_" + "d" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_claim_cleanup_failed"
    assert sent["proposal_status_mutation"] is True
    assert proposal.status == "apply_error"
    assert server.operator_proposal_queue.result_persistence_failed is True
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_execution_view_exception_finalizes_claim_and_releases_idempotency(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-execution-view-exception",
                "title": "M110 execution view exception",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-execution-view-exception"
    )
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    original_claim = server.operator_proposal_queue.claim_verified_for_apply

    def claim_with_unreadable_execution(*args, **kwargs):
        result = original_claim(*args, **kwargs)
        assert result["claimed"] is True

        def fail_content_copy():
            raise ValueError("injected_execution_view_failure")

        execution = SimpleNamespace(
            proposal_id=proposal.id,
            review_contract_version=proposal.review_contract_version,
            review_content_digest=proposal.review_content_digest,
            content_copy=fail_content_copy,
        )
        return {**result, "execution": execution}

    server.operator_proposal_queue.claim_verified_for_apply = (
        claim_with_unreadable_execution
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-execution-view-exception",
        "idempotency_key": "m110-proof:execution-view-exception",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "6" * 64,
        "supervised_capability_token": "m110cap_v1_" + "7" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["status"] == "apply_blocked"
    assert sent["apply_result"]["reason"] == "proposal_execution_view_invalid"
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    assert request["idempotency_key"] not in server._operator_apply_idempotency_keys
    grant.mark_terminal.assert_called_with(
        "execution_view_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_time_expiry_reports_terminal_mutation_and_no_reservation(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-claim-time-expiry",
                "title": "M110 claim-time expiry",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-claim-time-expiry"
    )
    expiry_checks = 0

    def expires_between_precheck_and_claim():
        nonlocal expiry_checks
        expiry_checks += 1
        return expiry_checks >= 2

    proposal.is_expired = expires_between_precheck_and_claim
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-claim-time-expiry",
        "idempotency_key": "m110-proof:claim-time-expiry",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "8" * 64,
        "supervised_capability_token": "m110cap_v1_" + "9" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "proposal_expired"
    assert sent["proposal_status_mutation"] is True
    assert sent["result_evidence_reserved"] is False
    assert proposal.status == "apply_blocked"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with(
        "claim_failed_after_capability_consume"
    )
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_consume_apply_cancellation_drains_evidence_and_authority(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-apply-cancelled",
                "title": "M110 apply cancellation",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-apply-cancelled"
    )
    operation_digest = "sha256:" + "3" * 64
    grant = MagicMock()
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={
            "consumed": True,
            "reason": "m110_capability_consumed",
            "grant": grant,
        }
    )
    apply_entered = asyncio.Event()
    hold_apply = asyncio.Event()

    async def wait_before_private_transaction(**_kwargs):
        apply_entered.set()
        await hold_apply.wait()

    server.apply_m110_restore_only_proof_transaction = AsyncMock(
        side_effect=wait_before_private_transaction
    )
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-apply-cancelled",
        "idempotency_key": "m110-proof:apply-cancelled",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": operation_digest,
        "supervised_capability_token": "m110cap_v1_" + "4" * 64,
    }

    task = asyncio.create_task(
        server._dispatch["apply_operator_proposal"](websocket, request)
    )
    await asyncio.wait_for(apply_entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No send occurred, so the proposal itself remains safely retryable after
    # a fresh arm.  The consumed grant and prepared bundle must not survive.
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 0
    assert proposal.id not in server._m110_prepared_capability_bundles
    grant.mark_terminal.assert_called_with("cancelled_after_capability_consume")
    server.record_m110_restored_proof_receipt.assert_not_called()
    server.send_to_client.assert_not_awaited()
    assert request.get("supervised_capability_token") is None


@pytest.mark.asyncio
async def test_consume_failure_release_error_latches_result_fence(m110_server):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-consume-release-error",
                "title": "M110 consume release error",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-consume-release-error"
    )
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = MagicMock(
        return_value={"consumed": False, "reason": "token_rejected"}
    )
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server.operator_proposal_queue.release_result_evidence = MagicMock(
        side_effect=RuntimeError("injected_release_failure")
    )
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-release-error",
        "idempotency_key": "m110-proof:release-error",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "6" * 64,
        "supervised_capability_token": "m110cap_v1_" + "5" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    assert server.operator_proposal_queue.result_persistence_failed is True
    assert proposal.status == "pending"
    assert server.operator_proposal_queue.result_store.active_reservation_count == 1
    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_result_persistence_failed"
    assert sent["result_evidence_reserved"] is True
    assert sent["apply_result"]["consume_reason"] == "token_rejected"
    assert sent["apply_result"]["result_persistence_failed"] is True
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_consume_path_release_error_reports_reserved_evidence(
    m110_server,
):
    server = m110_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server.mixer_client = MagicMock(is_connected=True)
    with patch(
        "operator_proposal_queue._apply_blockers_for_transaction_plan",
        return_value=[],
    ):
        server.operator_proposal_queue.create(
            {
                "id": "m110-missing-consume-release-error",
                "title": "M110 missing consume release error",
                "kind": "supervised_soundcheck_transaction",
                "supervised_transaction_plan": _plan(),
            },
            server.get_operator_mode_status(),
        )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-missing-consume-release-error"
    )
    server.preview_m110_capability_apply = MagicMock(
        return_value={"valid": True, "reason": "current"}
    )
    server.consume_m110_capability_for_apply = None
    server.apply_m110_restore_only_proof_transaction = AsyncMock()
    server.record_m110_restored_proof_receipt = MagicMock()
    server.send_to_client = AsyncMock()
    server.operator_proposal_queue.release_result_evidence = MagicMock(
        side_effect=RuntimeError("injected_release_failure")
    )
    request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-approval-missing-consume-release-error",
        "idempotency_key": "m110-proof:missing-consume-release-error",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": "sha256:" + "1" * 64,
        "supervised_capability_token": "m110cap_v1_" + "2" * 64,
    }

    await server._dispatch["apply_operator_proposal"](websocket, request)

    sent = server.send_to_client.await_args.args[1]
    assert sent["reason"] == "operator_proposal_result_persistence_failed"
    assert sent["result_evidence_reserved"] is True
    assert sent["apply_result"]["result_persistence_failed"] is True
    assert server.operator_proposal_queue.result_persistence_failed is True
    assert server.operator_proposal_queue.result_store.active_reservation_count == 1
    assert proposal.status == "pending"
    server.apply_m110_restore_only_proof_transaction.assert_not_awaited()


def _route_authority(*, websocket, transport, capture, stream):
    return MasterCaptureRouteAuthority(
        proof_id="m110-route-1",
        issued_at_epoch=1.0,
        expires_at_epoch=10_000_000_000.0,
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "3" * 64,
        connection_generation=4,
        transport_generation=5,
        transport_instance_id=id(transport),
        owner_connection_id=id(websocket),
        emergency_generation=6,
        console_serial="0100Q1V0604AAE214",
        firmware_version="3.0.5",
        capture_device="WING",
        capture_device_index=1,
        capture_channel_count=48,
        capture_sample_rate_hz=48000,
        capture_instance_id=id(capture),
        capture_hardware_uid="usb:5015:1291:214",
        capture_hardware_binding_sha256="7" * 64,
        capture_usb_serial="214",
        capture_usb_location_id=1,
        master_channels=(47, 48),
        artifact_path="/private/m110-route.json",
        artifact_sha256="sha256:" + "8" * 64,
        evidence_sha256="sha256:" + "9" * 64,
        _transport_ref=transport,
        _capture_ref=capture,
        _capture_stream_ref=stream,
        _owner_ref=websocket,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0.9, 1.1, True, "1"])
async def test_main_mute_checkpoint_rejects_non_exact_numeric_readback(
    m110_server,
    value,
):
    server = m110_server
    query = AsyncMock(
        return_value={
            "status": "received",
            "address": "/main/1/mute",
            "value": value,
            "confirmed_round_trips": 2,
        }
    )
    server.mixer_client = SimpleNamespace(
        is_connected=True,
        query_parameter=query,
        send=MagicMock(),
    )

    result = await server._m110_main_mute_physical_readback(
        expected_muted=True,
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_main_mute_physical_readback_mismatch"
    assert result["console_mutation_write_attempt_count"] == 0
    assert query.await_count == 2
    assert all(
        call.args == ("/main/1/mute", 0.5)
        for call in query.await_args_list
    )
    server.mixer_client.send.assert_not_called()


@pytest.mark.asyncio
async def test_main_mute_checkpoint_accepts_two_exact_physical_round_trips(
    m110_server,
):
    server = m110_server
    query = AsyncMock(
        return_value={
            "status": "received",
            "address": "/main/1/mute",
            "value": 1.0,
            "confirmed_round_trips": 2,
        }
    )
    server.mixer_client = SimpleNamespace(
        is_connected=True,
        query_parameter=query,
        send=MagicMock(),
    )

    result = await server._m110_main_mute_physical_readback(
        expected_muted=True,
    )

    assert result["verified"] is True
    assert [row["normalized_value"] for row in result["passes"]] == [1, 1]
    assert query.await_count == 2
    server.mixer_client.send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("physical_route", "expected_status", "expected_reason"),
    [
        (
            {
                "verified": False,
                "reason": "master_capture_route_physical_readback_mismatch",
            },
            "failed",
            "master_capture_route_physical_readback_mismatch",
        ),
        (
            {"verified": True, "route_proof_id": "route-after-feedback"},
            "passed",
            "fresh_headroom_and_feedback_clear",
        ),
    ],
)
async def test_combined_live_recheck_revalidates_physical_route_after_feedback(
    m110_server,
    physical_route,
    expected_status,
    expected_reason,
):
    server = m110_server
    operation = ConsoleTransactionOperation(
        operation_id="m110-live-route-recheck",
        family_id="eq_hpf",
        write_kind="hpf_enable",
        channel=1,
        parameter="hpf_enabled",
        target_value=True,
    )
    server._supervised_transaction_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "passed",
            "master_true_peak_safe",
            {"master_channels": [47, 48], "route_verified": True},
        )
    )
    server.audio_capture = SimpleNamespace(running=True)
    server._m110_feedback_clear_recheck = MagicMock(
        return_value=AudioRecheckResult(
            "passed",
            "fresh_feedback_clear_state",
            {"feedback_clear": True},
        )
    )
    route_proof = {
        "verified": True,
        "route_proof_id": "route-proof-after-feedback",
        "console_serial": "0100Q1V0604AAE214",
        "firmware_version": "3.0.5",
    }
    server._verified_master_capture_route_proof_with_fresh_hardware = (
        AsyncMock(return_value=route_proof)
    )
    server._fresh_master_usb_route_readback = AsyncMock(
        return_value=physical_route
    )

    result = await server._m110_combined_live_audio_recheck(
        operation,
        {},
        {},
    )

    assert result.status == expected_status
    assert result.reason == expected_reason
    assert result.evidence["route_after_feedback"] == route_proof
    assert result.evidence["physical_route_after_feedback"] == physical_route
    server._verified_master_capture_route_proof_with_fresh_hardware.assert_awaited_once()
    server._fresh_master_usb_route_readback.assert_awaited_once_with(route_proof)


@pytest.mark.asyncio
async def test_reset_containment_accepts_only_verified_post_mute_silence(
    m110_server,
):
    server = m110_server
    operation = ConsoleTransactionOperation(
        operation_id="m110-reset-containment-test",
        family_id="eq_hpf",
        write_kind="hpf_enable",
        channel=1,
        parameter="hpf_enabled",
        target_value=False,
    )
    server._m110_main_mute_physical_readback = AsyncMock(
        return_value={"verified": True, "expected": 1}
    )
    server._supervised_transaction_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "failed",
            "master_audio_signal_missing",
            {
                "route_verified": True,
                "capture_real_input_fresh": True,
                "capture_identity_matches_proof": True,
                "capture_sample_progressed": True,
                "master_channels": [47, 48],
                "master_sample_peaks_dbfs": [-96.0, -95.0],
            },
        )
    )
    server.audio_capture = SimpleNamespace(running=True)
    server._m110_feedback_clear_recheck = MagicMock(
        return_value=AudioRecheckResult("passed", "feedback_clear", {})
    )
    server._verified_master_capture_route_proof_with_fresh_hardware = (
        AsyncMock(return_value={"verified": True})
    )
    server._fresh_master_usb_route_readback = AsyncMock(
        return_value={"verified": True}
    )

    result = await server._m110_reset_contained_audio_recheck(
        operation,
        {},
        {},
    )

    assert result.status == "passed"
    assert result.reason == (
        "m110_reset_contained_silence_and_feedback_clear"
    )
    assert result.evidence["contained_silence_verified"] is True
    assert server._m110_main_mute_physical_readback.await_count == 2


@pytest.mark.asyncio
async def test_reset_containment_rejects_unbounded_near_silence(
    m110_server,
):
    server = m110_server
    operation = ConsoleTransactionOperation(
        operation_id="m110-reset-containment-too-loud",
        family_id="eq_hpf",
        write_kind="hpf_enable",
        channel=1,
        parameter="hpf_enabled",
        target_value=False,
    )
    server._m110_main_mute_physical_readback = AsyncMock(
        return_value={"verified": True, "expected": 1}
    )
    server._supervised_transaction_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "failed",
            "master_audio_signal_missing",
            {
                "route_verified": True,
                "capture_real_input_fresh": True,
                "capture_identity_matches_proof": True,
                "capture_sample_progressed": True,
                "master_channels": [47, 48],
                "master_sample_peaks_dbfs": [-89.9, -96.0],
            },
        )
    )
    server._m110_feedback_clear_recheck = MagicMock()

    result = await server._m110_reset_contained_audio_recheck(
        operation,
        {},
        {},
    )

    assert result.status == "failed"
    assert result.reason == "master_audio_signal_missing"
    assert result.evidence["contained_silence_verified"] is False
    server._m110_feedback_clear_recheck.assert_not_called()


@pytest.mark.asyncio
async def test_reset_containment_rejects_post_mute_signal_even_when_headroom_safe(
    m110_server,
):
    server = m110_server
    operation = ConsoleTransactionOperation(
        operation_id="m110-reset-containment-audible",
        family_id="eq_hpf",
        write_kind="hpf_enable",
        channel=1,
        parameter="hpf_enabled",
        target_value=False,
    )
    server._m110_main_mute_physical_readback = AsyncMock(
        return_value={"verified": True, "expected": 1}
    )
    server._supervised_transaction_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "passed",
            "master_true_peak_safe",
            {
                "route_verified": True,
                "capture_real_input_fresh": True,
                "capture_identity_matches_proof": True,
                "capture_sample_progressed": True,
                "master_channels": [47, 48],
                "master_sample_peaks_dbfs": [-12.0, -11.0],
            },
        )
    )
    server._m110_feedback_clear_recheck = MagicMock()

    result = await server._m110_reset_contained_audio_recheck(
        operation,
        {},
        {},
    )

    assert result.status == "failed"
    assert result.reason == "m110_reset_post_mute_signal_present"
    assert result.evidence["contained_silence_verified"] is False
    server._m110_feedback_clear_recheck.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_version", [1.0, True, "1", None])
async def test_main_output_checkpoint_request_schema_blocks_before_writer(
    m110_server,
    contract_version,
):
    server = m110_server
    server._console_write_scope = MagicMock()

    result = await server.verify_m110_main_output_checkpoint(
        websocket=object(),
        request={
            "type": "verify_m110_main_output_checkpoint",
            "contract_version": contract_version,
            "checkpoint": "reset_muted",
        },
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m110_main_output_checkpoint_request_invalid"
    assert result["console_mutation_write_attempt_count"] == 0
    server._console_write_scope.assert_not_called()


@pytest.mark.asyncio
async def test_main_output_checkpoint_rejects_client_supplied_address(
    m110_server,
):
    server = m110_server
    server._console_write_scope = MagicMock()

    result = await server.verify_m110_main_output_checkpoint(
        websocket=object(),
        request={
            "type": "verify_m110_main_output_checkpoint",
            "contract_version": 1,
            "checkpoint": "reset_muted",
            "osc_address": "/main/1/mute",
        },
    )

    assert result["reason"] == "m110_main_output_checkpoint_request_invalid"
    assert result["console_mutation_write_attempt_count"] == 0
    server._console_write_scope.assert_not_called()


@pytest.mark.asyncio
async def test_main_output_checkpoint_writer_conflict_clears_old_marker(
    m110_server,
):
    server = m110_server
    server._m110_main_output_checkpoint_authority = {"checkpoint": "stale"}

    @asynccontextmanager
    async def conflict_scope(**_kwargs):
        yield False

    server._console_write_scope = conflict_scope
    server._verify_m110_main_output_checkpoint_locked = AsyncMock()

    result = await server.verify_m110_main_output_checkpoint(
        websocket=object(),
        request={
            "type": "verify_m110_main_output_checkpoint",
            "contract_version": 1,
            "checkpoint": "reset_muted",
        },
    )

    assert result["reason"] == "console_writer_conflict_timeout"
    assert result["console_mutation_write_attempt_count"] == 0
    assert server._m110_main_output_checkpoint_authority is None
    server._verify_m110_main_output_checkpoint_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_output_checkpoint_cancellation_clears_old_marker(
    m110_server,
):
    server = m110_server
    server._m110_main_output_checkpoint_authority = {"checkpoint": "stale"}
    entered = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def blocked_scope(**_kwargs):
        entered.set()
        await release.wait()
        yield True

    server._console_write_scope = blocked_scope
    task = asyncio.create_task(
        server.verify_m110_main_output_checkpoint(
            websocket=object(),
            request={
                "type": "verify_m110_main_output_checkpoint",
                "contract_version": 1,
                "checkpoint": "reset_muted",
            },
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    assert server._m110_main_output_checkpoint_authority is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_reset_muted_checkpoint_is_read_only_and_run_bound(m110_server):
    server = m110_server
    websocket = object()
    stream = object()
    capture = SimpleNamespace(running=True, _stream=stream)
    query = AsyncMock()
    transport = SimpleNamespace(
        is_connected=True,
        query_parameter=query,
        send=MagicMock(),
    )
    route = _route_authority(
        websocket=websocket,
        transport=transport,
        capture=capture,
        stream=stream,
    )
    control = object()
    mute_evidence = {
        "verified": True,
        "address": "/main/1/mute",
        "expected": 1,
    }
    server.mixer_client = transport
    server.audio_capture = capture
    server._master_capture_route_authority = route
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._verify_m110_cp05_hardware_gate = MagicMock()
    server._verified_master_capture_route_proof_with_fresh_hardware = (
        AsyncMock(return_value={"verified": True})
    )
    server._m110_committed_reset_is_current = MagicMock(return_value=False)
    server._m110_reset_contained_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "passed",
            "m110_reset_contained_silence_and_feedback_clear",
            {
                "main_mute_before": mute_evidence,
                "main_mute_after": mute_evidence,
            },
        )
    )

    result = await server.verify_m110_main_output_checkpoint(
        websocket=websocket,
        request={
            "type": "verify_m110_main_output_checkpoint",
            "contract_version": 1,
            "checkpoint": "reset_muted",
        },
    )

    assert result["status"] == "verified"
    assert result["verified"] is True
    assert result["console_mutation_write_attempt_count"] == 0
    assert server._m110_main_output_checkpoint_is_current(
        "reset_muted",
        websocket=websocket,
        control=control,
        route=route,
    ) is True
    query.assert_not_awaited()
    transport.send.assert_not_called()


def test_terminal_proposal_revocation_clears_only_pending_authority(m110_server):
    server = m110_server
    websocket = object()
    binding = _binding(websocket=websocket)
    store = SupervisedCapabilityStore(token_factory=lambda: "1" * 64)
    issued, _ = store.issue(binding)
    server._m110_capability_store = store
    server._m110_prepared_capability_bundles = {
        binding.proposal_id: {"prepared": True},
        "other-proposal": {"prepared": True},
    }
    receipt = object()
    server._m110_restored_proof_receipts = {binding.operation_digest: receipt}

    assert server.revoke_m110_capability_for_proposal(
        binding.proposal_id,
        reason="proposal_terminal",
    ) == 1
    assert binding.proposal_id not in server._m110_prepared_capability_bundles
    assert "other-proposal" in server._m110_prepared_capability_bundles
    assert server._m110_restored_proof_receipts[binding.operation_digest] is receipt
    assert store.consume(issued.token, binding) == (
        None,
        "m110_capability_token_replayed_or_revoked",
    )


def test_queue_terminal_callback_revokes_automatic_expiry(m110_server):
    server = m110_server
    server.operator_proposal_queue._terminal_callback = (
        server._on_operator_proposal_terminal
    )
    payload = {
        "id": "m110-auto-expiry",
        "title": "M110 auto expiry",
        "kind": "analysis",
    }
    created = server.operator_proposal_queue.create(
        payload,
        server.get_operator_mode_status(),
    )
    proposal = server.operator_proposal_queue.find(
        proposal_id="m110-auto-expiry"
    )
    assert created["success"] is True and proposal is not None
    binding = _binding(websocket=object())
    binding = __import__("dataclasses").replace(
        binding,
        proposal_id=proposal.id,
    )
    store = SupervisedCapabilityStore(token_factory=lambda: "5" * 64)
    issued, _ = store.issue(binding)
    server._m110_capability_store = store
    server._m110_prepared_capability_bundles[proposal.id] = {"prepared": True}
    proposal.is_expired = lambda: True

    duplicate = server.operator_proposal_queue.create(
        payload,
        server.get_operator_mode_status(),
    )

    assert duplicate["reason"] == "proposal_expired_fresh_analysis_required"
    assert proposal.id not in server._m110_prepared_capability_bundles
    assert store.consume(issued.token, binding) == (
        None,
        "m110_capability_token_replayed_or_revoked",
    )


def test_receipt_mint_requires_exact_wal_and_durable_result_identity(m110_server):
    server = m110_server
    websocket = object()
    transport = object()
    capture = object()
    route = _route_authority(
        websocket=websocket,
        transport=transport,
        capture=capture,
        stream=object(),
    )
    server._runtime_identity = {
        "instance_nonce": "1" * 32,
        "source": {"head": "4" * 40, "tree": "5" * 40},
    }
    control = new_m110_attended_control_authority(
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "3" * 64,
        source_head="4" * 40,
        source_tree="5" * 40,
        route_proof_id=route.proof_id,
        route_artifact_sha256=route.artifact_sha256,
        connection_generation=4,
        transport_generation=5,
        emergency_generation=6,
        owner_connection_id=id(websocket),
        approval_id="approval-1",
        armed_at_epoch=1.0,
        _owner_ref=websocket,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
    )
    terminal = _restored_proof_result(
        proposal_id="m110-proposal-1",
        review_digest="sha256:" + "7" * 64,
        operation_digest="sha256:" + "9" * 64,
    )
    prepared_event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": terminal["transaction_id"],
        "phase": "prepared",
        "scope": "supervised_transaction",
        "proof_run": True,
        "restore_after_verify": True,
        "capability_kind": "restore_only_proof",
        "proposal_id": terminal["proposal_id"],
        "review_content_digest": terminal["review_content_digest"],
        "operation_digest": terminal["operation_digest"],
        "expected_operation_ids": terminal["applied_operation_ids"],
    }
    terminal_event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": terminal["transaction_id"],
        "phase": "proof_restored",
        "result_status": "proof_passed",
        **{
            key: terminal[key]
            for key in (
                "proof_run",
                "restore_after_verify",
                "capability_kind",
                "proposal_id",
                "review_content_digest",
                "operation_digest",
                "applied_operation_ids",
                "rollback_operation_ids",
                "no_console_send_proven",
                "physical_verified",
                "proof_target_verified",
                "audio_recheck_passed",
                "physical_state_restored",
                "terminal_after_console_send",
                "rollback_attempted",
                "rollback_succeeded",
                "rollback_status",
                "write_state_unknown",
                "rollback_state_unknown",
                "manual_review_required",
                "emergency_stop_activated",
                "emergency_stop_required",
                "accepted",
                "owner_review_required",
                "live_authority_granted",
                "readiness_closure_allowed",
            )
        },
    }
    server._manual_write_wal_lifecycles = {
        terminal["transaction_id"]: {
            "transaction_id": terminal["transaction_id"],
            "phases": ["prepared", "proof_restored"],
            "prepared": prepared_event,
            "latest_event": terminal_event,
        }
    }
    stored_bytes = canonical_json_bytes(terminal)
    result_digest = "sha256:" + hashlib.sha256(stored_bytes).hexdigest()
    proposal = SimpleNamespace(
        id=terminal["proposal_id"],
        status="proof_passed",
        review_contract_version=2,
        review_content_digest=terminal["review_content_digest"],
        result_detail_available=True,
        result_digest=result_digest,
    )
    server.operator_proposal_queue = MagicMock()
    server.operator_proposal_queue.load_result_evidence.return_value = stored_bytes
    server._master_capture_route_authority = route
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._m110_capability_global_blockers = MagicMock(return_value=[])

    recorded = server.record_m110_restored_proof_receipt(
        websocket=websocket,
        proposal=proposal,
        terminal_result=terminal,
        result_digest=result_digest,
    )

    assert recorded == {
        "recorded": True,
        "reason": "m110_restored_proof_receipt_recorded",
    }
    receipt = server._m110_restored_proof_receipts[
        terminal["operation_digest"]
    ]
    assert type(receipt) is M110RestoredProofReceipt
    assert receipt.is_authentic() is True

    mismatched = {
        **terminal,
        "operation_digest": "sha256:" + "a" * 64,
    }
    mismatched_bytes = canonical_json_bytes(mismatched)
    mismatched_digest = (
        "sha256:" + hashlib.sha256(mismatched_bytes).hexdigest()
    )
    proposal.result_digest = mismatched_digest
    server.operator_proposal_queue.load_result_evidence.return_value = (
        mismatched_bytes
    )
    rejected = server.record_m110_restored_proof_receipt(
        websocket=websocket,
        proposal=proposal,
        terminal_result=mismatched,
        result_digest=mismatched_digest,
    )
    assert rejected["recorded"] is False
    assert rejected["reason"] == "m110_restored_proof_wal_result_mismatch"


@pytest.mark.parametrize("already_satisfied", [False, True])
def test_m006_commit_marker_advances_only_from_durable_exact_terminal(
    m110_server,
    already_satisfied,
):
    server = m110_server
    websocket = object()
    control = object()
    transport = object()
    capture = object()
    route = _route_authority(
        websocket=websocket,
        transport=transport,
        capture=capture,
        stream=object(),
    )
    server.mixer_client = transport
    server.audio_capture = capture
    server._master_capture_route_authority = route
    server._mixer_connection_generation = route.connection_generation
    server._manual_write_emergency_generation = route.emergency_generation
    server._current_mixer_transport_generation = MagicMock(
        return_value=route.transport_generation
    )
    server._runtime_identity.update(
        {
            "instance_nonce": route.runtime_instance_nonce,
            "manifest_sha256": "2" * 64,
            "source": {"head": "4" * 40, "tree": "5" * 40},
        }
    )
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._m110_committed_reset_is_current = MagicMock(return_value=True)
    server._m110_main_output_checkpoint_is_current = MagicMock(
        return_value=True
    )
    server._manual_write_wal_integrity_ok = True
    server._m110_committed_reset_authority = {
        "operation_digest": "sha256:" + "1" * 64,
        "committed_surface_ledger": {},
        "committed_surface_ledger_sha256": (
            server._m110_committed_surface_ledger_sha256({})
        ),
        "committed_bundle_records": {},
        "m006_restore_stages": (),
        "m006_restore_complete": False,
        "_owner_ref": websocket,
        "_control_ref": control,
        "_route_ref": route,
        "_transport_ref": transport,
        "_capture_ref": capture,
    }

    proposal_id = (
        "m110-m006-restore-foundation_level-commit-" + "a" * 32
    )
    plan = build_m110_m006_restore_plan(
        stage="foundation_level",
        role="commit",
        proposal_nonce="a" * 32,
    )
    built = server._m110_build_canonical_plan_bundle(
        plan_payload=plan,
        route=route,
        bundle_kind=M110_M006_RESTORE_BUNDLE_KIND,
        adapter_revision=M110_M006_RESTORE_ADAPTER_REVISION,
        protocol_revision=M110_M006_RESTORE_PROTOCOL_REVISION,
        fixed_preimage_restore_authorized=True,
    )
    review_digest = "sha256:" + "6" * 64
    proof_receipt = new_m110_restored_proof_receipt(
        proposal_id="m110-m006-foundation-proof",
        review_contract_version=2,
        review_content_digest="sha256:" + "7" * 64,
        review_preimage_sha256="",
        operation_digest=built["operation_digest"],
        bundle_kind=M110_M006_RESTORE_BUNDLE_KIND,
        canonical_bundle_bytes=built["canonical_bundle_bytes"],
        result_digest="sha256:" + "8" * 64,
        transaction_id="m110-m006-foundation-proof-transaction",
        runtime_instance_nonce=route.runtime_instance_nonce,
        source_head="4" * 40,
        source_tree="5" * 40,
        route_proof_id=route.proof_id,
        route_artifact_sha256=route.artifact_sha256,
        owner_connection_id=id(websocket),
        created_at_epoch=100.0,
        created_at_monotonic=100.0,
        _owner_ref=websocket,
        _control_ref=control,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
    )
    binding = M110CapabilityBinding(
        capability_kind="maintenance_dsp_reset_commit_once",
        runtime_instance_nonce=route.runtime_instance_nonce,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "3" * 64,
        source_head="4" * 40,
        source_tree="5" * 40,
        connection_generation=route.connection_generation,
        transport_generation=route.transport_generation,
        console_serial=route.console_serial,
        firmware_version=route.firmware_version,
        route_proof_id=route.proof_id,
        route_artifact_sha256=route.artifact_sha256,
        capture_hardware_uid=route.capture_hardware_uid,
        emergency_generation=route.emergency_generation,
        owner_connection_id=id(websocket),
        proposal_id=proposal_id,
        review_contract_version=2,
        review_content_digest=review_digest,
        operation_digest=built["operation_digest"],
        canonical_bundle_bytes=built["canonical_bundle_bytes"],
        bundle_kind=M110_M006_RESTORE_BUNDLE_KIND,
        review_preimage_sha256="",
        proof_receipt_id=proof_receipt.receipt_id,
        approval_id="m110-m006-foundation-approval",
        idempotency_key="m110-reset-commit:foundation",
        scope_addresses=built["scope_addresses"],
        target_fingerprints=built["target_fingerprints"],
        _owner_ref=websocket,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
        _control_ref=control,
        _proof_receipt_ref=proof_receipt,
    )
    capability_store = SupervisedCapabilityStore(
        epoch_clock=lambda: 101.0,
        monotonic_clock=lambda: 101.0,
        token_factory=lambda: "9" * 64,
    )
    issued, _ = capability_store.issue(binding)
    grant, _ = capability_store.consume(issued.token, binding)
    assert grant.mark_terminal("completed") is True

    operation_ids = [
        operation.operation_id for operation in built["plan"].operations
    ]
    terminal = {
        "transaction_id": built["plan"].transaction_id,
        "status": "applied",
        "reason": "all_operations_applied",
        "success": True,
        "applied_operation_ids": [] if already_satisfied else operation_ids,
        "rollback_operation_ids": [],
        "already_satisfied_operation_ids": (
            operation_ids if already_satisfied else []
        ),
        "events": [],
        "proof_run": False,
        "restore_after_verify": False,
        "capability_kind": binding.capability_kind,
        "proposal_id": proposal_id,
        "review_content_digest": review_digest,
        "operation_digest": built["operation_digest"],
        "bundle_kind": M110_M006_RESTORE_BUNDLE_KIND,
        "review_preimage_sha256": "",
        "proof_receipt_id": proof_receipt.receipt_id,
        "accepted": True,
        "owner_review_required": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "no_console_send_proven": already_satisfied,
        "physical_verified": True,
        "proof_target_verified": False,
        "physical_state_restored": False,
        "terminal_after_console_send": not already_satisfied,
        "rollback_attempted": False,
        "rollback_succeeded": False,
        "rollback_status": "not_attempted",
        "wal_terminal": "committed",
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "fresh_analysis_required": False,
    }
    prepared_event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": terminal["transaction_id"],
        "phase": "prepared",
        "scope": "supervised_transaction",
        "approval_id": binding.approval_id,
        "idempotency_key": binding.idempotency_key,
        "proof_run": False,
        "restore_after_verify": False,
        "capability_kind": binding.capability_kind,
        "proposal_id": proposal_id,
        "review_content_digest": review_digest,
        "operation_digest": built["operation_digest"],
        "bundle_kind": M110_M006_RESTORE_BUNDLE_KIND,
        "proof_receipt_id": proof_receipt.receipt_id,
        "expected_operation_ids": operation_ids,
    }
    terminal_event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": terminal["transaction_id"],
        "phase": "committed",
        "result_status": "applied",
        "result_reason": terminal["reason"],
        "applied_operation_ids": [] if already_satisfied else operation_ids,
        "rollback_operation_ids": [],
        "already_satisfied_operation_ids": (
            operation_ids if already_satisfied else []
        ),
        "proof_run": False,
        "restore_after_verify": False,
        "capability_kind": binding.capability_kind,
        "proposal_id": proposal_id,
        "review_content_digest": review_digest,
        "operation_digest": built["operation_digest"],
        "bundle_kind": M110_M006_RESTORE_BUNDLE_KIND,
        "proof_receipt_id": proof_receipt.receipt_id,
    }
    server._manual_write_wal_lifecycles = {
        terminal["transaction_id"]: {
            "transaction_id": terminal["transaction_id"],
            "phases": ["prepared", "committed"],
            "events": [prepared_event, terminal_event],
            "prepared": prepared_event,
            "latest_event": terminal_event,
        }
    }
    stored_bytes = canonical_json_bytes(terminal)
    result_digest = "sha256:" + hashlib.sha256(stored_bytes).hexdigest()
    proposal = SimpleNamespace(
        id=proposal_id,
        status="applied",
        review_contract_version=2,
        review_content_digest=review_digest,
        result_detail_available=True,
        result_digest=result_digest,
        raw={"supervised_transaction_plan": plan},
    )
    server.operator_proposal_queue = MagicMock()
    server.operator_proposal_queue.load_result_evidence.return_value = (
        stored_bytes
    )

    recorded = server.record_m110_m006_restore_commit(
        websocket=websocket,
        proposal=proposal,
        capability_grant=grant,
        terminal_result=terminal,
        result_digest=result_digest,
    )

    assert recorded == {
        "recorded": True,
        "reason": "m110_m006_restore_commit_recorded",
        "stage": "foundation_level",
        "m006_restore_complete": False,
    }
    assert server._m110_committed_reset_authority[
        "m006_restore_stages"
    ] == ("foundation_level",)
    assert server._m110_committed_reset_authority[
        "m006_restore_complete"
    ] is False


@pytest.mark.parametrize("already_satisfied", [False, True])
@pytest.mark.parametrize("journal_failure", [False, True])
def test_m110_soundcheck_commit_advances_outcome_and_ledger_only_after_fsync(
    m110_server,
    already_satisfied,
    journal_failure,
):
    server = m110_server
    websocket = object()
    control = object()
    transport = object()
    capture = object()
    route = _route_authority(
        websocket=websocket,
        transport=transport,
        capture=capture,
        stream=object(),
    )
    server.mixer_client = transport
    server.audio_capture = capture
    server._master_capture_route_authority = route
    server._mixer_connection_generation = route.connection_generation
    server._manual_write_emergency_generation = route.emergency_generation
    server._current_mixer_transport_generation = MagicMock(
        return_value=route.transport_generation
    )
    server._runtime_identity.update(
        {
            "instance_nonce": route.runtime_instance_nonce,
            "manifest_sha256": "2" * 64,
            "source": {"head": "4" * 40, "tree": "5" * 40},
        }
    )
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._m110_full_reset_is_current = MagicMock(return_value=True)
    server._m110_main_output_checkpoint_is_current = MagicMock(
        return_value=True
    )
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("commit balance", 1, -12.0)],
        run_id="m110-plan-commit-run",
    ).to_dict()
    source_plan = manifest["plans"][0]
    operation_id = source_plan["operations"][0]["operation_id"]
    proposal_id = "m110-soundcheck-child-commit-" + "a" * 32
    execution_plan = copy.deepcopy(source_plan)
    execution_plan["source_transaction_id"] = source_plan[
        "transaction_id"
    ]
    execution_plan["transaction_id"] = (
        "m110:soundcheck:commit:" + "a" * 32
    )
    built = server._m110_build_canonical_plan_bundle(
        plan_payload=execution_plan,
        route=route,
        bundle_kind=M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
        adapter_revision="wing-supervised-console-adapters.v1",
        protocol_revision="wing-osc-fw3.0.5",
    )
    review_digest = "sha256:" + "6" * 64
    proof_receipt = new_m110_restored_proof_receipt(
        proposal_id="m110-soundcheck-child-proof-" + "b" * 32,
        review_contract_version=2,
        review_content_digest="sha256:" + "7" * 64,
        review_preimage_sha256="",
        operation_digest=built["operation_digest"],
        bundle_kind=M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
        canonical_bundle_bytes=built["canonical_bundle_bytes"],
        result_digest="sha256:" + "8" * 64,
        transaction_id="m110:soundcheck:proof:" + "b" * 32,
        runtime_instance_nonce=route.runtime_instance_nonce,
        source_head="4" * 40,
        source_tree="5" * 40,
        route_proof_id=route.proof_id,
        route_artifact_sha256=route.artifact_sha256,
        owner_connection_id=id(websocket),
        created_at_epoch=100.0,
        created_at_monotonic=100.0,
        _owner_ref=websocket,
        _control_ref=control,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
    )
    binding = M110CapabilityBinding(
        capability_kind="soundcheck_child_commit_once",
        runtime_instance_nonce=route.runtime_instance_nonce,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "3" * 64,
        source_head="4" * 40,
        source_tree="5" * 40,
        connection_generation=route.connection_generation,
        transport_generation=route.transport_generation,
        console_serial=route.console_serial,
        firmware_version=route.firmware_version,
        route_proof_id=route.proof_id,
        route_artifact_sha256=route.artifact_sha256,
        capture_hardware_uid=route.capture_hardware_uid,
        emergency_generation=route.emergency_generation,
        owner_connection_id=id(websocket),
        proposal_id=proposal_id,
        review_contract_version=2,
        review_content_digest=review_digest,
        operation_digest=built["operation_digest"],
        canonical_bundle_bytes=built["canonical_bundle_bytes"],
        bundle_kind=M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
        review_preimage_sha256="",
        proof_receipt_id=proof_receipt.receipt_id,
        approval_id="m110-plan-commit-approval",
        idempotency_key="m110-soundcheck-commit:unit",
        scope_addresses=built["scope_addresses"],
        target_fingerprints=built["target_fingerprints"],
        _owner_ref=websocket,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
        _control_ref=control,
        _proof_receipt_ref=proof_receipt,
    )
    capability_store = SupervisedCapabilityStore(
        epoch_clock=lambda: 101.0,
        monotonic_clock=lambda: 101.0,
        token_factory=lambda: "9" * 64,
    )
    issued, _ = capability_store.issue(binding)
    grant, _ = capability_store.consume(issued.token, binding)
    assert grant.mark_terminal("completed") is True
    operation_ids = [
        operation.operation_id for operation in built["plan"].operations
    ]
    terminal = {
        "transaction_id": built["plan"].transaction_id,
        "status": "applied",
        "reason": "all_operations_applied",
        "success": True,
        "applied_operation_ids": [] if already_satisfied else operation_ids,
        "rollback_operation_ids": [],
        "already_satisfied_operation_ids": (
            operation_ids if already_satisfied else []
        ),
        "events": [],
        "proof_run": False,
        "restore_after_verify": False,
        "capability_kind": binding.capability_kind,
        "proposal_id": proposal_id,
        "review_content_digest": review_digest,
        "operation_digest": built["operation_digest"],
        "bundle_kind": binding.bundle_kind,
        "review_preimage_sha256": "",
        "proof_receipt_id": proof_receipt.receipt_id,
        "accepted": True,
        "owner_review_required": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "no_console_send_proven": already_satisfied,
        "physical_verified": True,
        "proof_target_verified": False,
        "physical_state_restored": False,
        "terminal_after_console_send": not already_satisfied,
        "rollback_attempted": False,
        "rollback_succeeded": False,
        "rollback_status": "not_attempted",
        "wal_terminal": "committed",
        "audio_recheck_passed": True,
        "audible_state_verified": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "fresh_analysis_required": False,
    }
    prepared_event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": terminal["transaction_id"],
        "phase": "prepared",
        "scope": "supervised_transaction",
        "proposal_id": proposal_id,
        "review_content_digest": binding.review_content_digest,
        "operation_digest": binding.operation_digest,
        "bundle_kind": binding.bundle_kind,
        "capability_kind": binding.capability_kind,
        "approval_id": binding.approval_id,
        "idempotency_key": binding.idempotency_key,
        "proof_run": False,
        "restore_after_verify": False,
        "proof_receipt_id": binding.proof_receipt_id,
        "expected_operation_ids": operation_ids,
    }
    terminal_event = {
        "schema": "automixer.manual_write_wal.v1",
        "transaction_id": terminal["transaction_id"],
        "phase": "committed",
        "result_status": "applied",
        "result_reason": terminal["reason"],
        "proposal_id": proposal_id,
        "review_content_digest": binding.review_content_digest,
        "operation_digest": binding.operation_digest,
        "bundle_kind": binding.bundle_kind,
        "capability_kind": binding.capability_kind,
        "proof_receipt_id": binding.proof_receipt_id,
        "proof_run": False,
        "restore_after_verify": False,
        "applied_operation_ids": terminal["applied_operation_ids"],
        "rollback_operation_ids": [],
        "already_satisfied_operation_ids": terminal[
            "already_satisfied_operation_ids"
        ],
    }
    server._manual_write_wal_integrity_ok = True
    server._manual_write_wal_lifecycles = {
        terminal["transaction_id"]: {
            "transaction_id": terminal["transaction_id"],
            "phases": ["prepared", "committed"],
            "events": [prepared_event, terminal_event],
            "prepared": prepared_event,
            "latest_event": terminal_event,
        }
    }
    stored_bytes = canonical_json_bytes(terminal)
    result_digest = "sha256:" + hashlib.sha256(stored_bytes).hexdigest()
    proposal = SimpleNamespace(
        id=proposal_id,
        status="applied",
        review_contract_version=2,
        review_content_digest=review_digest,
        result_detail_available=True,
        result_digest=result_digest,
        raw={"supervised_transaction_plan": execution_plan},
    )
    server.operator_proposal_queue = MagicMock()
    server.operator_proposal_queue.load_result_evidence.return_value = (
        stored_bytes
    )
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._soundcheck_outcomes = {manifest["run_id"]: {}}
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_preflight_assessments = {}
    server._active_soundcheck_run_id = manifest["run_id"]
    empty_ledger = {}
    server._m110_committed_reset_authority = {
        "operation_digest": "sha256:" + "1" * 64,
        "result_digest": "sha256:" + "0" * 64,
        "committed_surface_ledger": empty_ledger,
        "committed_surface_ledger_sha256": (
            server._m110_committed_surface_ledger_sha256(empty_ledger)
        ),
        "committed_result_digests": ("sha256:" + "0" * 64,),
        "committed_operation_digests": ("sha256:" + "1" * 64,),
        "committed_bundle_records": {},
        "m006_restore_complete": True,
    }
    commit_context = {
        "schema": "automixer.m110_plan_commit_context.v1",
        "lane": "soundcheck",
        "proposal_id": proposal_id,
        "bundle_kind": binding.bundle_kind,
        "operation_digest": binding.operation_digest,
        "canonical_bundle_bytes": binding.canonical_bundle_bytes,
        "source_plans": [copy.deepcopy(source_plan)],
        "source_operation_ids": (operation_id,),
        "candidate_id": None,
    }
    marker_before = copy.deepcopy(server._m110_committed_reset_authority)
    if journal_failure:
        server._append_soundcheck_parent_journal = MagicMock(
            return_value=None
        )

    recorded = server.record_m110_plan_commit(
        websocket=websocket,
        proposal=proposal,
        capability_grant=grant,
        terminal_result=terminal,
        result_digest=result_digest,
        commit_context=commit_context,
    )

    if journal_failure:
        assert recorded == {
            "recorded": False,
            "reason": "m110_plan_commit_batch_journal_failed",
        }
        assert server._soundcheck_outcomes[manifest["run_id"]] == {}
        assert server._m110_committed_reset_authority == marker_before
        return
    assert recorded["recorded"] is True
    outcome = server._soundcheck_outcomes[manifest["run_id"]][operation_id]
    if already_satisfied:
        assert outcome["disposition"] == "already_satisfied"
        assert outcome["physical_step_count"] == 0
    else:
        assert outcome["status"] == "applied"
        assert outcome["physical_step_count"] == 1
    assert "/ch/1/fdr" in server._m110_committed_reset_authority[
        "committed_surface_ledger"
    ]
    journal_events = [
        json.loads(line)
        for line in server._soundcheck_parent_journal_path.read_text().splitlines()
    ]
    assert journal_events[-1]["event_type"] == "m110_child_commit_batch"
    assert len(journal_events[-1]["batches"]) == 1
    committed_record = server._m110_committed_reset_authority[
        "committed_bundle_records"
    ][result_digest]
    assert journal_events[-1]["target_delta_sha256"] == committed_record[
        "target_delta_sha256"
    ]


def _configure_m110_final_audit(server, websocket):
    control = object()
    transport = object()
    capture = object()
    route = _route_authority(
        websocket=websocket,
        transport=transport,
        capture=capture,
        stream=object(),
    )
    server.mixer_client = transport
    server.audio_capture = capture
    server._master_capture_route_authority = route
    server._mixer_connection_generation = route.connection_generation
    server._manual_write_emergency_generation = route.emergency_generation
    server._current_mixer_transport_generation = MagicMock(
        return_value=route.transport_generation
    )
    server._runtime_identity.update(
        {
            "instance_nonce": route.runtime_instance_nonce,
            "manifest_sha256": "2" * 64,
            "source": {"head": "4" * 40, "tree": "5" * 40},
        }
    )
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._verify_m110_cp05_hardware_gate = MagicMock(return_value={})
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._verified_master_capture_route_proof_with_fresh_hardware = (
        AsyncMock(return_value={"verified": True})
    )
    server._m110_full_reset_is_current = MagicMock(return_value=True)
    server._m110_main_output_checkpoint_is_current = MagicMock(
        return_value=True
    )
    server._m110_soundcheck_apply_in_progress = MagicMock(
        return_value=False
    )
    manifest = {
        "schema_version": "soundcheck_batch_manifest/v2",
        "run_id": "m110-final-audit-run",
        "plan_revision": 3,
        "plan_fingerprint": "fingerprint-final-audit",
        "manifest_content_sha256": "sha256:" + "a" * 64,
    }
    server._m110_current_soundcheck_manifest = MagicMock(
        return_value=manifest
    )
    server.reconcile_soundcheck_readiness = AsyncMock(
        return_value={
            "run_id": manifest["run_id"],
            "mix_readiness": "show_mix_ready",
        }
    )
    audit_readiness_fence = {"verified": True, "fence": "final-audit"}
    server._soundcheck_preflight_assessments[manifest["run_id"]] = {
        "console_change_fence": audit_readiness_fence,
    }
    server._soundcheck_global_blockers[manifest["run_id"]] = []
    server._soundcheck_console_fence_still_current = MagicMock(
        return_value=True
    )
    server.refresh_soundcheck_readiness_truth = MagicMock(
        return_value={
            "run_id": manifest["run_id"],
            "mix_readiness": "show_mix_ready",
        }
    )
    ledger = {
        "/ch/1/fdr": {
            "target_value": -12.0,
            "tolerance": "0.05",
            "source_kind": "soundcheck_child",
            "operation_digest": "sha256:" + "b" * 64,
        },
        "/main/1/mute": {
            "target_value": 0,
            "tolerance": "0",
            "source_kind": "main_output_checkpoint:soundcheck_unmuted",
            "operation_digest": "sha256:" + "c" * 64,
        },
    }
    marker = {
        "operation_digest": "sha256:" + "d" * 64,
        "result_digest": "sha256:" + "e" * 64,
        "committed_surface_ledger": ledger,
        "committed_surface_ledger_sha256": (
            server._m110_committed_surface_ledger_sha256(ledger)
        ),
        "committed_result_digests": ("sha256:" + "e" * 64,),
        "committed_operation_digests": ("sha256:" + "d" * 64,),
        "committed_bundle_records": {},
        "m006_restore_stages": (
            "foundation_level",
            "tonal_dynamics",
            "spatial_fx",
        ),
        "m006_restore_complete": True,
        "main_unmute_checkpoint_digest": "sha256:" + "f" * 64,
    }
    server._m110_committed_reset_authority = marker

    async def query(address):
        return {
            "verified": True,
            "address": address,
            "value": 0 if address == "/main/1/mute" else -12.0,
            "confirmed_round_trips": 2,
        }

    server._m110_query_parameter_exact = AsyncMock(side_effect=query)
    server._m110_combined_live_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult("passed", "audio_safe", {})
    )
    server._m110_main_mute_physical_readback = AsyncMock(
        return_value={"verified": True, "value": 0}
    )
    return manifest, marker


@pytest.mark.asyncio
async def test_m110_final_audit_is_two_pass_read_only_and_state_hash_stable(
    m110_server,
):
    websocket = object()
    manifest, marker = _configure_m110_final_audit(
        m110_server,
        websocket,
    )
    request = {
        "type": "run_m110_soundcheck_final_audit",
        "contract_version": 1,
        "audit_selector": "active_soundcheck_full_reconciliation",
    }

    first = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request=request,
    )
    second = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request=request,
    )

    assert first["verified"] is True
    assert first["mix_readiness"] == "show_mix_ready"
    assert first["audited_address_count"] == 2
    assert first["audit_pass_count"] == 2
    assert first["operator_listening_required"] is True
    assert first["live_mixer_writes"] is False
    assert first["readiness_closure_allowed"] is False
    assert second["verified_state_sha256"] == first["verified_state_sha256"]
    assert m110_server._m110_query_parameter_exact.await_count == 8
    assert m110_server._m110_committed_reset_authority is marker
    assert m110_server._m110_current_soundcheck_manifest() is manifest


@pytest.mark.asyncio
async def test_m110_final_audit_rejects_second_pass_target_drift(m110_server):
    websocket = object()
    _configure_m110_final_audit(m110_server, websocket)
    values = iter([-12.0, 0, -10.0])

    async def drifting_query(address):
        return {
            "verified": True,
            "address": address,
            "value": next(values),
            "confirmed_round_trips": 2,
        }

    m110_server._m110_query_parameter_exact = AsyncMock(
        side_effect=drifting_query
    )

    result = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request={
            "type": "run_m110_soundcheck_final_audit",
            "contract_version": 1,
            "audit_selector": "active_soundcheck_full_reconciliation",
        },
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_final_audit_target_mismatch"
    assert result["audit_pass"] == 2
    assert result["live_mixer_writes"] is False


@pytest.mark.asyncio
async def test_m110_final_audit_stops_immediately_when_emergency_fence_changes(
    m110_server,
):
    websocket = object()
    _configure_m110_final_audit(m110_server, websocket)
    initial_generation = m110_server._manual_write_emergency_generation

    async def query_then_latch(address):
        m110_server._manual_write_emergency_generation = (
            initial_generation + 1
        )
        return {
            "verified": True,
            "address": address,
            "value": 0 if address == "/main/1/mute" else -12.0,
            "confirmed_round_trips": 2,
        }

    m110_server._m110_query_parameter_exact = AsyncMock(
        side_effect=query_then_latch
    )

    result = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request={
            "type": "run_m110_soundcheck_final_audit",
            "contract_version": 1,
            "audit_selector": "active_soundcheck_full_reconciliation",
        },
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_final_audit_state_fence_changed"
    assert result["fence_reason"] == "emergency_generation_changed"
    assert result["audit_pass"] == 1
    assert m110_server._m110_query_parameter_exact.await_count == 1
    m110_server._m110_combined_live_audio_recheck.assert_not_awaited()
    assert result["live_mixer_writes"] is False


@pytest.mark.asyncio
async def test_m110_final_audit_rechecks_ledger_after_audio_await(m110_server):
    websocket = object()
    _configure_m110_final_audit(m110_server, websocket)
    values = {"/ch/1/fdr": -12.0, "/main/1/mute": 0}

    async def query(address):
        return {
            "verified": True,
            "address": address,
            "value": values[address],
            "confirmed_round_trips": 2,
        }

    async def audio_then_external_drift(*_args, **_kwargs):
        values["/ch/1/fdr"] = -10.0
        return AudioRecheckResult("passed", "audio_safe", {})

    m110_server._m110_query_parameter_exact = AsyncMock(side_effect=query)
    m110_server._m110_combined_live_audio_recheck = AsyncMock(
        side_effect=audio_then_external_drift
    )

    result = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request={
            "type": "run_m110_soundcheck_final_audit",
            "contract_version": 1,
            "audit_selector": "active_soundcheck_full_reconciliation",
        },
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_final_audit_target_mismatch"
    assert result["address"] == "/ch/1/fdr"
    assert result["audit_pass"] == 2
    assert result["live_mixer_writes"] is False


@pytest.mark.asyncio
async def test_m110_final_audit_rejects_readiness_invalidation_on_last_read(
    m110_server,
):
    websocket = object()
    manifest, _marker = _configure_m110_final_audit(
        m110_server,
        websocket,
    )
    fence_current = True
    query_count = 0

    def current_fence(_fence):
        return fence_current

    def current_readiness(run_id):
        return {
            "run_id": run_id,
            "mix_readiness": (
                "show_mix_ready"
                if fence_current
                else "soundcheck_incomplete"
            ),
        }

    async def invalidate_on_last_read(address):
        nonlocal fence_current, query_count
        query_count += 1
        if query_count == 4:
            fence_current = False
            m110_server._soundcheck_global_blockers[manifest["run_id"]].append(
                "console_state_changed_after_reconciliation"
            )
        return {
            "verified": True,
            "address": address,
            "value": 0 if address == "/main/1/mute" else -12.0,
            "confirmed_round_trips": 2,
        }

    m110_server._soundcheck_console_fence_still_current = MagicMock(
        side_effect=current_fence
    )
    m110_server.refresh_soundcheck_readiness_truth = MagicMock(
        side_effect=current_readiness
    )
    m110_server._m110_query_parameter_exact = AsyncMock(
        side_effect=invalidate_on_last_read
    )

    result = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request={
            "type": "run_m110_soundcheck_final_audit",
            "contract_version": 1,
            "audit_selector": "active_soundcheck_full_reconciliation",
        },
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_final_audit_state_fence_changed"
    assert result["fence_reason"] == (
        "m110_show_mix_readiness_fence_not_current"
    )
    assert result["audit_pass"] == 2
    assert query_count == 4


@pytest.mark.asyncio
async def test_m110_final_audit_rejects_readiness_expiry_during_audio_await(
    m110_server,
):
    websocket = object()
    manifest, _marker = _configure_m110_final_audit(
        m110_server,
        websocket,
    )
    expired = False

    def current_readiness(run_id):
        return {
            "run_id": run_id,
            "mix_readiness": (
                "soundcheck_incomplete" if expired else "show_mix_ready"
            ),
        }

    async def expire_during_audio(*_args, **_kwargs):
        nonlocal expired
        expired = True
        return AudioRecheckResult("passed", "audio_safe", {})

    m110_server.refresh_soundcheck_readiness_truth = MagicMock(
        side_effect=current_readiness
    )
    m110_server._m110_combined_live_audio_recheck = AsyncMock(
        side_effect=expire_during_audio
    )

    result = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request={
            "type": "run_m110_soundcheck_final_audit",
            "contract_version": 1,
            "audit_selector": "active_soundcheck_full_reconciliation",
        },
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_final_audit_state_fence_changed"
    assert result["fence_reason"] == "m110_show_mix_readiness_not_current"
    assert result["audit_pass"] == 1
    assert m110_server._m110_query_parameter_exact.await_count == 2
    m110_server._m110_main_mute_physical_readback.assert_not_awaited()


@pytest.mark.asyncio
async def test_m110_final_audit_includes_committed_audible_state_guards(
    m110_server,
):
    websocket = object()
    _manifest, marker = _configure_m110_final_audit(
        m110_server,
        websocket,
    )
    operation_digest = "sha256:" + "6" * 64
    result_digest = "sha256:" + "7" * 64
    marker = m110_server._m110_marker_with_committed_bundle_overlay(
        marker,
        canonical_bundle={
            "operations": [
                {
                    "operation_id": "bus-eq-guard",
                    "writes": [
                        {
                            "address": "/bus/1/eq/1/gain",
                            "target_value": "-3",
                            "tolerance": "0.1",
                        }
                    ],
                    "audible_state_reads": [
                        {
                            "address": "/bus/1/eq/on",
                            "expected_value": True,
                            "tolerance": "0",
                        }
                    ],
                }
            ]
        },
        operation_digest=operation_digest,
        result_digest=result_digest,
        source_kind="soundcheck_child",
    )
    m110_server._m110_committed_reset_authority = marker
    assert marker["committed_surface_ledger"]["/bus/1/eq/on"] == {
        "target_value": True,
        "tolerance": "0",
        "source_kind": "soundcheck_child:audible_guard",
        "operation_digest": operation_digest,
    }
    target_delta = m110_server._m110_committed_bundle_target_delta(
        marker,
        result_digest=result_digest,
    )
    assert set(target_delta) == {
        "/bus/1/eq/1/gain",
        "/bus/1/eq/on",
    }
    assert marker["committed_bundle_records"][result_digest][
        "target_delta_sha256"
    ] == m110_server._m110_committed_surface_ledger_sha256(target_delta)

    async def query(address):
        expected = marker["committed_surface_ledger"][address][
            "target_value"
        ]
        return {
            "verified": True,
            "address": address,
            "value": 0 if address == "/bus/1/eq/on" else expected,
            "confirmed_round_trips": 2,
        }

    m110_server._m110_query_parameter_exact = AsyncMock(side_effect=query)

    result = await m110_server.run_m110_soundcheck_final_audit(
        websocket=websocket,
        request={
            "type": "run_m110_soundcheck_final_audit",
            "contract_version": 1,
            "audit_selector": "active_soundcheck_full_reconciliation",
        },
    )

    assert result["verified"] is False
    assert result["reason"] == "m110_final_audit_target_mismatch"
    assert result["address"] == "/bus/1/eq/on"
    assert result["audit_pass"] == 1
    assert result["live_mixer_writes"] is False


@pytest.mark.asyncio
async def test_m110_final_audit_ingress_is_exact_and_has_typed_block(m110_server):
    websocket = object()
    m110_server._runtime_identity.setdefault("launch_binding", {})[
        "mode"
    ] = "m110_attended_runner"
    m110_server._verify_m110_cp05_hardware_gate = MagicMock(return_value={})
    m110_server.send_to_client = AsyncMock()
    valid = {
        "type": "run_m110_soundcheck_final_audit",
        "contract_version": 1,
        "audit_selector": "active_soundcheck_full_reconciliation",
    }

    assert await m110_server._block_m110_protocol_message(
        websocket,
        valid["type"],
        valid,
    ) is False
    malformed = {**valid, "client_target": "/main/1/fdr"}
    assert await m110_server._block_m110_protocol_message(
        websocket,
        malformed["type"],
        malformed,
    ) is True

    response = m110_server.send_to_client.await_args.args[1]
    assert response["type"] == "m110_soundcheck_final_audit_result"
    assert response["verified"] is False
    assert response["operator_listening_required"] is False
    assert response["live_mixer_writes"] is False
    assert response["readiness_closure_allowed"] is False
    assert response["blockers"] == [
        "m110_soundcheck_final_audit_request_invalid"
    ]


@pytest.mark.asyncio
async def test_connection_handler_registers_m110_final_audit():
    from handlers.connection_handlers import register_handlers

    server = SimpleNamespace(
        run_m110_soundcheck_final_audit=AsyncMock(
            return_value={"type": "m110_soundcheck_final_audit_result"}
        ),
        send_to_client=AsyncMock(),
    )
    handlers = register_handlers(server)
    websocket = object()
    request = {
        "type": "run_m110_soundcheck_final_audit",
        "contract_version": 1,
        "audit_selector": "active_soundcheck_full_reconciliation",
    }

    await handlers["run_m110_soundcheck_final_audit"](websocket, request)

    server.run_m110_soundcheck_final_audit.assert_awaited_once_with(
        websocket=websocket,
        request=request,
    )
    server.send_to_client.assert_awaited_once()
