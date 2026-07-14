from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decision_queue_wire_contract import canonical_json_bytes
from master_capture_route_proof import MasterCaptureRouteAuthority
from m110_manual_write_wal import M110_WAL_CONTRACT
from operator_proposal_queue import OperatorProposalQueue
from operator_proposal_result_store import (
    ResultEvidenceReservationBinding,
    ResultEvidenceStore,
)
from server import AutoMixerServer
from supervised_console_transaction import AudioRecheckResult
from supervised_proof_capability import (
    M110CapabilityBinding,
    M110RestoredProofReceipt,
    SupervisedCapabilityStore,
    new_m110_attended_control_authority,
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
    server._supervised_transaction_audio_recheck = AsyncMock(
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
        proposal=object(),
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
    grant = MagicMock()
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
    server.apply_m110_restore_only_proof_transaction = AsyncMock(
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
        server.apply_m110_restore_only_proof_transaction.await_args.kwargs[
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
                "physical_verified",
                "proof_target_verified",
                "audio_recheck_passed",
                "physical_state_restored",
                "terminal_after_console_send",
                "rollback_attempted",
                "rollback_succeeded",
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
