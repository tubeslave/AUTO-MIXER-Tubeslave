from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maintenance_dsp_reset import (
    M110_PRESERVATION_ADDRESSES,
    M110_PRESERVATION_SPECS,
    M110_RESET_ADAPTER_REVISION,
    M110_RESET_ADDRESSES,
    M110_RESET_OPERATION_DEFINITIONS,
    M110_RESET_TARGETS,
    M110_RESET_PROTOCOL_REVISION,
    M110DeliveredMutationObserver,
    M110MaintenanceDspResetExecutor,
    M110MaintenanceMutationGate,
    M110ResetHooks,
    build_fixed_m110_reset_canonical_bundle,
    capture_m110_reset_preimage,
    compare_preservation_manifest_rows,
    fixed_m110_reset_bundle_operations,
    fixed_m110_reset_prepared_operations,
    m110_reset_review_preimage_document,
)
from supervised_family_proof_plan import (
    canonical_bundle_scope,
    canonical_operation_bundle_bytes,
)
from supervised_proof_capability import (
    M110CapabilityBinding,
    SupervisedCapabilityStore,
    new_m110_restored_proof_receipt,
)
from supervised_console_transaction import AudioRecheckResult
from operator_proposal_result_store import ResultEvidenceReservationBinding
from m110_manual_write_wal import (
    M110_LIFECYCLE_RESOLVED,
    classify_m110_lifecycle,
)
from master_capture_route_proof import MasterCaptureRouteAuthority
from server import AutoMixerServer


IDENTITY_VALUES = {
    "/$syscfg/$serial": "0100Q1V0604AAE",
    "/$syscfg/$firmware": "3.0.5",
    "/io/out/USB/47/grp": "MAIN",
    "/io/out/USB/47/in": 0,
    "/io/out/USB/48/grp": "MAIN",
    "/io/out/USB/48/in": 1,
}


def _server_route_authority(*, websocket, transport, capture):
    return MasterCaptureRouteAuthority(
        proof_id="m110-fixed-reset-route",
        issued_at_epoch=time.time(),
        expires_at_epoch=time.time() + 600.0,
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "8" * 64,
        connection_generation=1,
        transport_generation=2,
        transport_instance_id=id(transport),
        owner_connection_id=id(websocket),
        emergency_generation=3,
        console_serial=IDENTITY_VALUES["/$syscfg/$serial"],
        firmware_version=IDENTITY_VALUES["/$syscfg/$firmware"],
        capture_device="WING",
        capture_device_index=1,
        capture_channel_count=48,
        capture_sample_rate_hz=48_000,
        capture_instance_id=id(capture),
        capture_hardware_uid="m110-reset-unit-capture",
        capture_hardware_binding_sha256="9" * 64,
        capture_usb_serial="unit",
        capture_usb_location_id=1,
        master_channels=(47, 48),
        artifact_path="/tmp/m110-fixed-reset-route.json",
        artifact_sha256="sha256:" + "5" * 64,
        evidence_sha256="sha256:" + "6" * 64,
        _transport_ref=transport,
        _capture_ref=capture,
        _capture_stream_ref=object(),
        _owner_ref=websocket,
    )


def _payload_sha256(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _physical_values():
    values = {}
    for address in M110_RESET_ADDRESSES:
        values[address] = 1 if address.endswith(("eq/on", "flt/lc")) else 0
    for spec in M110_PRESERVATION_SPECS:
        if spec.address in IDENTITY_VALUES:
            values[spec.address] = IDENTITY_VALUES[spec.address]
        elif spec.address == "/main/1/mute":
            values[spec.address] = 1
        elif spec.value_kind == "binary":
            values[spec.address] = 0
        elif spec.value_kind == "integer":
            values[spec.address] = 1
        elif spec.value_kind == "decimal":
            values[spec.address] = -12.0
        else:
            values[spec.address] = "LOCAL"
    return values


def _query(values, *, mutate=None):
    def query(address, timeout):
        result = {
            "status": "received",
            "address": address,
            "value": values[address],
            "confirmed_round_trips": 2,
        }
        if mutate is not None:
            mutate(address, result)
        return result

    return query


class _FakeWing:
    def __init__(self, values):
        self.values = dict(values)
        self.callbacks = {}
        self.boundary = None
        self.boundary_token = None
        self.remove_boundary_ok = True
        self.unsubscribe_ok = True
        self.on_unsubscribe = None
        self.on_remove_boundary = None
        self.send_calls = []
        self.query_calls = []
        self.fail_send_at = None
        self.raise_send_at = None
        self.after_value_send = None
        self.after_query = None
        self.block_send_at = None
        self.send_started = threading.Event()
        self.release_send = threading.Event()

    def subscribe(self, pattern, callback):
        self.callbacks.setdefault(pattern, []).append(callback)

    def unsubscribe(self, pattern, callback):
        if self.on_unsubscribe is not None:
            self.on_unsubscribe(self, pattern, callback)
        if not self.unsubscribe_ok:
            return False
        callbacks = self.callbacks.get(pattern, [])
        if callback not in callbacks:
            return False
        callbacks.remove(callback)
        if not callbacks:
            self.callbacks.pop(pattern, None)
        return True

    def _deliver(self, address, value):
        for callback in tuple(self.callbacks.get(address, ())):
            callback(address, value)
        for callback in tuple(self.callbacks.get("*", ())):
            callback(address, value)

    def query_parameter(self, address, timeout):
        self.query_calls.append((address, timeout))
        value = self.values[address]
        self._deliver(address, value)
        if self.after_query is not None:
            self.after_query(self, address)
        return {
            "status": "received",
            "address": address,
            "value": value,
            "confirmed_round_trips": 2,
        }

    def send(self, address, *values):
        if not values:
            return True
        if self.boundary is not None and self.boundary(address, tuple(values)) is not True:
            return False
        self.send_calls.append((address, values[0]))
        call_number = len(self.send_calls)
        if self.block_send_at == call_number:
            self.send_started.set()
            assert self.release_send.wait(timeout=5.0)
        if self.raise_send_at == call_number:
            raise OSError("injected send error")
        if self.fail_send_at == call_number:
            return False
        self.values[address] = values[0]
        self._deliver(address, values[0])
        if self.after_value_send is not None:
            self.after_value_send(self, call_number)
        return True

    def install_mutation_boundary(self, boundary):
        if self.boundary is not None:
            raise RuntimeError("already active")
        self.boundary = boundary
        self.boundary_token = object()
        return self.boundary_token

    def remove_mutation_boundary(self, token):
        if self.on_remove_boundary is not None:
            self.on_remove_boundary(self)
        if (
            not self.remove_boundary_ok
            or self.boundary is None
            or token is not self.boundary_token
        ):
            return False
        self.boundary = None
        self.boundary_token = None
        return True

    def mutation_boundary_matches(self, token):
        return bool(
            self.boundary is not None
            and self.boundary_token is token
        )


class _Hooks:
    def __init__(self):
        self.prepared = []
        self.terminals = []
        self.emergencies = []
        self.authority = True
        self.authority_calls = 0
        self.raise_authority_at = None
        self.prepare_ok = True
        self.boundary_ok = True
        self.terminal_ok = True
        self.audio_status = "passed"
        self.audio_calls = 0
        self.audio_queries = None
        self.on_audio = None
        self.on_terminal = None
        self.emergency_result = {"success": True, "teardown_complete": True}

    def values(self):
        async def audio():
            self.audio_calls += 1
            if self.audio_queries is not None:
                self.audio_queries.query_parameter("/main/1/fdr", 0.1)
            if self.on_audio is not None:
                self.on_audio()
            return AudioRecheckResult(self.audio_status)

        async def emergency(reason):
            self.emergencies.append(reason)
            return self.emergency_result

        def authority():
            self.authority_calls += 1
            if self.raise_authority_at == self.authority_calls:
                raise RuntimeError("injected authority error")
            return self.authority

        def terminal(phase, payload):
            recorded = copy.deepcopy(dict(payload))
            self.terminals.append((phase, recorded))
            if self.on_terminal is not None:
                self.on_terminal(phase)
            return (
                {
                    "path": "wal.jsonl",
                    "payload_sha256": _payload_sha256(recorded),
                }
                if self.terminal_ok
                else None
            )

        return M110ResetHooks(
            authority_current=authority,
            persist_prepared=lambda payload: (
                self.prepared.append(copy.deepcopy(dict(payload)))
                or ("wal.jsonl" if self.prepare_ok else None)
            ),
            mark_send_may_have_occurred=lambda: self.boundary_ok,
            persist_terminal=terminal,
            audio_recheck=audio,
            emergency_stop=emergency,
        )


async def _executor_fixture(*, proof_run=False):
    wing = _FakeWing(_physical_values())
    approved = await capture_m110_reset_preimage(
        wing.query_parameter,
        expected_identity_values=IDENTITY_VALUES,
    )
    wing.query_calls.clear()
    hooks = _Hooks()
    executor = M110MaintenanceDspResetExecutor(wing, hooks.values())
    return wing, hooks, executor, approved, _reset_authority(
        proof_run=proof_run,
        approved_preimage=approved,
        wing_client=wing,
    )


def _reset_authority(*, proof_run, approved_preimage, wing_client):
    owner = object()
    route = object()
    transport = wing_client
    capture = object()
    control = object()
    safety_binding = {
        "adapter_revision": M110_RESET_ADAPTER_REVISION,
        "protocol_revision": M110_RESET_PROTOCOL_REVISION,
        "runtime_instance_nonce": "1" * 32,
        "runtime_manifest_sha256": "sha256:" + "2" * 64,
        "source_head": "3" * 40,
        "source_tree": "4" * 40,
        "console_serial": IDENTITY_VALUES["/$syscfg/$serial"],
        "firmware_version": IDENTITY_VALUES["/$syscfg/$firmware"],
        "route_proof_id": "m110-reset-unit-route",
        "route_artifact_sha256": "sha256:" + "5" * 64,
        "capture_hardware_uid": "m110-reset-unit-capture",
        "connection_generation": 1,
        "transport_generation": 2,
        "emergency_generation": 3,
    }
    bundle = build_fixed_m110_reset_canonical_bundle(safety_binding)
    bundle_bytes = canonical_operation_bundle_bytes(bundle)
    operation_digest = "sha256:" + hashlib.sha256(bundle_bytes).hexdigest()
    review_preimage_sha256 = m110_reset_review_preimage_document(
        approved_preimage
    )["review_preimage_sha256"]
    scope, targets = canonical_bundle_scope(bundle)
    receipt = None
    kind = (
        "restore_only_proof"
        if proof_run
        else "maintenance_dsp_reset_commit_once"
    )
    if not proof_run:
        receipt = new_m110_restored_proof_receipt(
            proposal_id="m110-reset-proof-proposal",
            review_contract_version=2,
            review_content_digest="sha256:" + "6" * 64,
            review_preimage_sha256=review_preimage_sha256,
            operation_digest=operation_digest,
            bundle_kind="maintenance_dsp_reset_bundle",
            canonical_bundle_bytes=bundle_bytes,
            result_digest="sha256:" + "7" * 64,
            transaction_id="m110-reset-proof-transaction",
            runtime_instance_nonce="1" * 32,
            source_head="3" * 40,
            source_tree="4" * 40,
            route_proof_id="m110-reset-unit-route",
            route_artifact_sha256="sha256:" + "5" * 64,
            owner_connection_id=9,
            created_at_epoch=time.time(),
            created_at_monotonic=time.monotonic(),
            _owner_ref=owner,
            _control_ref=control,
            _route_ref=route,
            _transport_ref=transport,
            _capture_ref=capture,
        )
    binding = M110CapabilityBinding(
        capability_kind=kind,
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "8" * 64,
        source_head="3" * 40,
        source_tree="4" * 40,
        connection_generation=1,
        transport_generation=2,
        console_serial=IDENTITY_VALUES["/$syscfg/$serial"],
        firmware_version=IDENTITY_VALUES["/$syscfg/$firmware"],
        route_proof_id="m110-reset-unit-route",
        route_artifact_sha256="sha256:" + "5" * 64,
        capture_hardware_uid="m110-reset-unit-capture",
        emergency_generation=3,
        owner_connection_id=9,
        proposal_id=(
            "m110-reset-proof-proposal"
            if proof_run
            else "m110-reset-commit-proposal"
        ),
        review_contract_version=2,
        review_content_digest="sha256:" + ("6" if proof_run else "9") * 64,
        operation_digest=operation_digest,
        canonical_bundle_bytes=bundle_bytes,
        bundle_kind="maintenance_dsp_reset_bundle",
        review_preimage_sha256=review_preimage_sha256,
        proof_receipt_id=receipt.receipt_id if receipt is not None else "",
        approval_id="m110-reset-approval",
        idempotency_key=(
            "m110-reset-proof-idempotency"
            if proof_run
            else "m110-reset-commit-idempotency"
        ),
        scope_addresses=scope,
        target_fingerprints=targets,
        _owner_ref=owner,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
        _control_ref=control,
        _proof_receipt_ref=receipt,
    )
    store = SupervisedCapabilityStore()
    issued, reason = store.issue(binding)
    assert issued is not None, reason
    grant, reason = store.consume(issued.token, binding)
    assert grant is not None, reason
    return grant, receipt


def test_fixed_reset_is_exact_channel_major_64_address_surface():
    assert len(M110_RESET_ADDRESSES) == 64
    assert len(M110_RESET_TARGETS) == 64
    assert len(M110_RESET_OPERATION_DEFINITIONS) == 64
    assert M110_RESET_ADDRESSES[:8] == (
        "/ch/1/eq/on",
        "/ch/1/flt/lc",
        "/ch/1/gate/on",
        "/ch/1/dyn/on",
        "/ch/2/eq/on",
        "/ch/2/flt/lc",
        "/ch/2/gate/on",
        "/ch/2/dyn/on",
    )
    assert M110_RESET_ADDRESSES[-4:] == (
        "/ch/16/eq/on",
        "/ch/16/flt/lc",
        "/ch/16/gate/on",
        "/ch/16/dyn/on",
    )
    assert set(M110_RESET_TARGETS.values()) == {0}
    assert all(
        row["family_id"] == "maintenance_dsp_reset"
        and row["write_kind"] == "processor_bypass"
        and row["target_value"] == 0
        for row in M110_RESET_OPERATION_DEFINITIONS
    )


def test_fixed_reset_bundle_and_prepared_rows_are_exact_and_singular():
    operations = fixed_m110_reset_bundle_operations()
    prepared = fixed_m110_reset_prepared_operations()
    assert len(operations) == len(prepared) == 64
    assert [row["operation_id"] for row in operations] == [
        row["operation_id"] for row in M110_RESET_OPERATION_DEFINITIONS
    ]
    assert [row["addresses"][0] for row in prepared] == list(
        M110_RESET_ADDRESSES
    )
    assert all(
        row["write_count"] == 1
        and len(row["writes"]) == 1
        and row["writes"][0]["target_value"] == 0
        and row["audible_state_reads"] == ()
        for row in prepared
    )


def test_preservation_manifest_is_exact_sorted_184_address_surface():
    assert len(M110_PRESERVATION_ADDRESSES) == 184
    assert M110_PRESERVATION_ADDRESSES == tuple(
        sorted(M110_PRESERVATION_ADDRESSES)
    )
    assert len(set(M110_PRESERVATION_ADDRESSES)) == 184
    assert not set(M110_PRESERVATION_ADDRESSES).intersection(
        M110_RESET_ADDRESSES
    )
    for channel in range(1, 17):
        expected = {
            f"/ch/{channel}/in/set/trim",
            f"/ch/{channel}/fdr",
            f"/ch/{channel}/mute",
            f"/ch/{channel}/pan",
            f"/ch/{channel}/in/set/inv",
            f"/ch/{channel}/in/set/dly",
            f"/ch/{channel}/in/set/dlyon",
            f"/ch/{channel}/in/conn/grp",
            f"/ch/{channel}/in/conn/in",
            f"/ch/{channel}/in/conn/altgrp",
            f"/ch/{channel}/in/conn/altin",
        }
        assert expected.issubset(M110_PRESERVATION_ADDRESSES)
    assert set(IDENTITY_VALUES).issubset(M110_PRESERVATION_ADDRESSES)
    assert {"/main/1/fdr", "/main/1/mute"}.issubset(
        M110_PRESERVATION_ADDRESSES
    )


@pytest.mark.asyncio
async def test_capture_requires_double_physical_readback_and_identity_match():
    values = _physical_values()
    capture = await capture_m110_reset_preimage(
        _query(values),
        expected_identity_values=IDENTITY_VALUES,
    )

    assert capture["reset_row_count"] == 64
    assert capture["preservation_manifest"]["row_count"] == 184
    assert [row["address"] for row in capture["reset_rows"]] == list(
        M110_RESET_ADDRESSES
    )
    assert [
        row["address"] for row in capture["preservation_manifest"]["rows"]
    ] == list(M110_PRESERVATION_ADDRESSES)
    assert capture["reset_preimage_sha256"].startswith("sha256:")
    assert capture["preservation_manifest_sha256"].startswith("sha256:")
    assert capture["live_authority_granted"] is False
    assert len(capture["reset_query_evidence"]) == 64
    assert len(capture["preservation_query_evidence"]) == 184


@pytest.mark.asyncio
async def test_capture_total_deadline_bounds_both_physical_sweeps():
    query = _query(_physical_values())

    async def slow_query(address, timeout):
        await asyncio.sleep(0.01)
        return query(address, timeout)

    with pytest.raises(
        TimeoutError,
        match="m110_(?:physical_sweep|total_capture)_deadline_exceeded",
    ):
        await capture_m110_reset_preimage(
            slow_query,
            expected_identity_values=IDENTITY_VALUES,
            total_deadline_seconds=0.005,
        )


@pytest.mark.asyncio
async def test_capture_rejects_one_round_trip_wrong_address_and_identity():
    values = _physical_values()

    def one_round_trip(address, result):
        if address == "/ch/1/eq/on":
            result["confirmed_round_trips"] = 1

    with pytest.raises(ValueError, match="m110_physical_query_unverified"):
        await capture_m110_reset_preimage(
            _query(values, mutate=one_round_trip),
            expected_identity_values=IDENTITY_VALUES,
        )

    def wrong_address(address, result):
        if address == "/ch/1/eq/on":
            result["address"] = "/ch/2/eq/on"

    with pytest.raises(ValueError, match="m110_physical_query_unverified"):
        await capture_m110_reset_preimage(
            _query(values, mutate=wrong_address),
            expected_identity_values=IDENTITY_VALUES,
        )

    wrong_identity = dict(IDENTITY_VALUES)
    wrong_identity["/$syscfg/$firmware"] = "3.0.4"
    with pytest.raises(ValueError, match="m110_preservation_identity_mismatch"):
        await capture_m110_reset_preimage(
            _query(values),
            expected_identity_values=wrong_identity,
        )


@pytest.mark.asyncio
async def test_capture_rejects_non_binary_reset_preimage():
    values = _physical_values()
    values["/ch/1/eq/on"] = 2
    with pytest.raises(ValueError, match="m110_physical_value_invalid"):
        await capture_m110_reset_preimage(
            _query(values),
            expected_identity_values=IDENTITY_VALUES,
        )


@pytest.mark.asyncio
async def test_preservation_compare_uses_address_specific_tolerances():
    capture = await capture_m110_reset_preimage(
        _query(_physical_values()),
        expected_identity_values=IDENTITY_VALUES,
    )
    before = capture["preservation_manifest"]["rows"]
    within = copy.deepcopy(before)
    outside = copy.deepcopy(before)
    fader_index = next(
        index
        for index, row in enumerate(before)
        if row["address"] == "/ch/1/fdr"
    )
    within[fader_index]["value"] = "-11.96"
    outside[fader_index]["value"] = "-11.94"

    assert compare_preservation_manifest_rows(before, within) == ()
    mismatch = compare_preservation_manifest_rows(before, outside)
    assert mismatch == (
        {
            "address": "/ch/1/fdr",
            "before": "-12",
            "after": "-11.94",
            "tolerance": "0.05",
        },
    )


@pytest.mark.asyncio
async def test_maintenance_commit_writes_exact_64_and_preserves_manifest():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    hooks.audio_queries = wing

    result = await executor.execute(
        transaction_id="m110-reset-commit",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "applied"
    assert result.wal_terminal == "committed"
    assert result.physical_verified is True
    assert result.audio_recheck_passed is True
    assert result.verification_cut_epoch is not None
    assert result.verification_cut_epoch > 0
    assert result.verification_cut_reset_surface_sha256.startswith("sha256:")
    assert (
        result.verification_cut_preservation_manifest_sha256
        == result.after_manifest_sha256
    )
    assert [address for address, _value in wing.send_calls] == list(
        M110_RESET_ADDRESSES
    )
    assert {value for _address, value in wing.send_calls} == {0}
    assert len(result.applied_operation_ids) == 64
    assert result.preservation_mismatches == ()
    assert result.mutation_audit["no_outside_mutation_attempt_proven"] is True
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "committed",
    ]
    assert hooks.emergencies == []
    # The Main read performed by audio is outside the observer window.
    assert not any(
        audit["unexpected_events"] for audit in result.observer_audits
    )
    assert wing.boundary is None
    assert "*" not in wing.callbacks
    assert wing.send("/ch/1/fdr", -7.0) is True


@pytest.mark.asyncio
async def test_maintenance_proof_restores_all_64_members_in_reverse():
    wing, hooks, executor, approved, authority = await _executor_fixture(proof_run=True)
    original = dict(wing.values)

    result = await executor.execute(
        transaction_id="m110-reset-proof",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "proof_passed"
    assert result.wal_terminal == "proof_restored"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert len(result.applied_operation_ids) == 64
    assert len(result.rollback_operation_ids) == 64
    assert result.rollback_operation_ids == list(
        reversed(result.applied_operation_ids)
    )
    forward = wing.send_calls[:64]
    rollback = wing.send_calls[64:]
    assert [address for address, _value in forward] == list(M110_RESET_ADDRESSES)
    expected_rollback = [
        address
        for address in reversed(M110_RESET_ADDRESSES)
        if original[address] == 1
    ]
    assert [address for address, _value in rollback] == expected_rollback
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "proof_restored",
    ]
    assert wing.boundary is None
    assert "*" not in wing.callbacks


@pytest.mark.asyncio
async def test_server_reset_capture_timeout_returns_strict_blocked_terminal(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(tmp_path / "console-runtime"),
    )
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch(
        "server.BleedService"
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
    websocket = object()
    wing = _FakeWing(_physical_values())
    wing.is_connected = True
    capture = object()
    control = object()
    route = _server_route_authority(
        websocket=websocket,
        transport=wing,
        capture=capture,
    )
    server.mixer_client = wing
    server.audio_capture = capture
    server._master_capture_route_authority = route
    server._runtime_identity = {
        "instance_nonce": "1" * 32,
        "manifest_sha256": "2" * 64,
        "launch_binding": {"prestart_manifest_sha256": "8" * 64},
        "source": {"head": "3" * 40, "tree": "4" * 40},
    }
    server._mixer_connection_generation = 1
    server._manual_write_emergency_generation = 3
    server._current_mixer_transport_generation = MagicMock(return_value=2)
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._verify_m110_cp05_hardware_gate = MagicMock(return_value={})
    server._verified_master_capture_route_proof_with_fresh_hardware = AsyncMock(
        return_value={"verified": True}
    )
    server._m110_main_output_checkpoint_is_current = MagicMock(
        return_value=True
    )

    with patch(
        "server.capture_m110_reset_preimage",
        new=AsyncMock(
            side_effect=TimeoutError("m110_physical_sweep_deadline_exceeded")
        ),
    ):
        result = await server.create_m110_maintenance_reset_proposal(
            websocket=websocket,
            request={
                "type": "create_m110_maintenance_reset_proof_proposal",
                "contract_version": 1,
                "bundle_selector": "fixed_channels_1_16_dsp_bypass",
            },
        )

    assert result["type"] == "m110_maintenance_reset_proposal_result"
    assert result["status"] == "blocked"
    assert result["success"] is False
    assert result["reason"] == (
        "m110_maintenance_reset_capture_invalid:"
        "m110_physical_sweep_deadline_exceeded"
    )
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert wing.send_calls == []


@pytest.mark.asyncio
async def test_server_owns_fixed_reset_proposal_capture_and_private_provenance(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(tmp_path / "console-runtime"),
    )
    monkeypatch.setenv(
        "AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR",
        str(tmp_path / "manual-events"),
    )
    monkeypatch.setenv(
        "AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR",
        str(tmp_path / "proposal-results"),
    )
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch(
        "server.BleedService"
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
    websocket = object()
    wing = _FakeWing(_physical_values())
    wing.is_connected = True
    capture = object()
    control = object()
    route = _server_route_authority(
        websocket=websocket,
        transport=wing,
        capture=capture,
    )
    server.mixer_client = wing
    server.audio_capture = capture
    server._master_capture_route_authority = route
    server._runtime_identity = {
        "instance_nonce": "1" * 32,
        "manifest_sha256": "2" * 64,
        "launch_binding": {
            "prestart_manifest_sha256": "8" * 64,
        },
        "source": {"head": "3" * 40, "tree": "4" * 40},
    }
    server._mixer_connection_generation = 1
    server._manual_write_emergency_generation = 3
    server._current_mixer_transport_generation = MagicMock(return_value=2)
    server._m110_capability_global_blockers = MagicMock(return_value=[])
    server._verified_m110_attended_control = MagicMock(
        return_value=(control, "current")
    )
    server._verify_m110_cp05_hardware_gate = MagicMock(return_value={})
    server._verified_master_capture_route_proof_with_fresh_hardware = AsyncMock(
        return_value={"verified": True}
    )
    server._m110_main_output_checkpoint_is_current = MagicMock(
        return_value=True
    )

    created = await server.create_m110_maintenance_reset_proposal(
        websocket=websocket,
        request={
            "type": "create_m110_maintenance_reset_proof_proposal",
            "contract_version": 1,
            "bundle_selector": "fixed_channels_1_16_dsp_bypass",
        },
    )

    assert created["status"] == "created"
    assert created["required_capability_kind"] == "restore_only_proof"
    assert created["bundle_kind"] == "maintenance_dsp_reset_bundle"
    assert created["operation_count"] == 64
    assert created["scope_addresses"] == list(M110_RESET_ADDRESSES)
    assert "approved_preimage" not in created
    assert "canonical_bundle_bytes" not in created
    assert len(wing.query_calls) == (
        len(M110_RESET_ADDRESSES) + len(M110_PRESERVATION_ADDRESSES)
    )
    assert wing.send_calls == []
    proposal = server.operator_proposal_queue.find(
        proposal_id=created["proposal_id"]
    )
    assert proposal is not None
    assert "supervised_transaction_plan" not in proposal.raw
    context = server._m110_fixed_reset_proposal_contexts[proposal.id]
    assert context["proposal"] is proposal
    assert context["approved_preimage"]["reset_rows"]
    assert context["approved_preimage"]["preservation_manifest"][
        "row_count"
    ] == 184

    prepared = await server.prepare_m110_restore_only_proof(
        websocket=websocket,
        request={
            "type": "prepare_m110_restore_only_proof",
            "contract_version": 1,
            "proposal_id": proposal.id,
            "bundle_selector": "reviewed_supervised_transaction",
        },
    )
    assert prepared["status"] == "prepared"
    assert prepared["operation_count"] == 64
    assert prepared["required_capability_kind"] == "restore_only_proof"
    assert prepared["review_preimage_sha256"] == context[
        "review_preimage_sha256"
    ]

    forged = SimpleNamespace(**proposal.__dict__)
    forged_context, forged_reason = server._build_m110_fixed_reset_bundle(
        websocket=websocket,
        proposal=forged,
        control=control,
    )
    assert forged_context is None
    assert forged_reason == "m110_fixed_reset_private_context_required"

    server._m110_launch_authority = MagicMock()
    server._m110_launch_authority.verify_and_tombstone_request.return_value = (
        True,
        "verified",
    )
    issued = await server.issue_m110_supervised_capability(
        websocket=websocket,
        request={
            "type": "issue_supervised_capability",
            "contract_version": 1,
            "capability_kind": "restore_only_proof",
            "proposal_id": proposal.id,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": proposal.review_content_digest,
            "expected_operation_digest": context["operation_digest"],
            "approval_id": "m110-fixed-reset-proof-approval",
            "confirmation": "ISSUE_RESTORE_ONLY_PROOF_CAPABILITY",
            "request_nonce": "a" * 32,
            "authority_proof": "hmac-sha256:" + "b" * 64,
        },
    )
    assert issued["status"] == "issued"
    apply_request = {
        "type": "apply_operator_proposal",
        "proposal_id": proposal.id,
        "approved": True,
        "approval_id": "m110-fixed-reset-proof-approval",
        "idempotency_key": issued["idempotency_key"],
        "expected_review_contract_version": 2,
        "expected_review_content_digest": proposal.review_content_digest,
        "expected_operation_digest": context["operation_digest"],
    }
    consumed = server.consume_m110_capability_for_apply(
        websocket=websocket,
        request=apply_request,
        token=issued["supervised_capability_token"],
        proposal=proposal,
    )
    assert consumed["consumed"] is True, consumed
    grant = consumed["grant"]
    binding = grant.binding
    reservation_result = server.operator_proposal_queue.reserve_result_evidence(
        proposal
    )
    assert reservation_result["reserved"] is True
    reservation = reservation_result["reservation_binding"]
    claim = server.operator_proposal_queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=2,
        expected_review_content_digest=proposal.review_content_digest,
    )
    assert claim["claimed"] is True
    server._m110_consumed_grant_forward_current = MagicMock(return_value=True)
    server._m110_reset_contained_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "passed",
            evidence={
                "capture_status_after": {"captured_sample_count": 1_000_000}
            },
        )
    )
    server.activate_system_emergency_stop = AsyncMock(
        return_value={"success": True, "teardown_complete": True}
    )
    wing.query_calls.clear()

    result = await server.apply_m110_capability_transaction(
        websocket=websocket,
        request=apply_request,
        capability_grant=grant,
        proposal=proposal,
        plan_payload=None,
        result_evidence_reservation_binding=reservation,
    )

    assert result["status"] == "proof_passed"
    assert result["success"] is True
    assert result["bundle_kind"] == "maintenance_dsp_reset_bundle"
    assert result["review_preimage_sha256"] == context[
        "review_preimage_sha256"
    ]
    assert result["durable_result_projection_valid"] is True
    assert result["physical_state_restored"] is True
    assert [address for address, _value in wing.send_calls[:64]] == list(
        M110_RESET_ADDRESSES
    )
    server._m110_reset_contained_audio_recheck.assert_awaited()
    server.activate_system_emergency_stop.assert_not_awaited()

    finalized = server.operator_proposal_queue.mark_apply_result(
        proposal,
        result,
        evidence_reservation_binding=reservation,
    )
    assert finalized["status"] == "proof_passed"
    assert proposal.status == "proof_passed"
    receipt_result = server.record_m110_restored_proof_receipt(
        websocket=websocket,
        proposal=proposal,
        capability_grant=grant,
        terminal_result=proposal.result,
        result_digest=proposal.result_digest,
    )
    assert receipt_result == {
        "recorded": True,
        "reason": "m110_restored_proof_receipt_recorded",
    }
    receipt = server._m110_restored_proof_receipts[
        binding.operation_digest
    ]
    assert receipt.bundle_kind == "maintenance_dsp_reset_bundle"
    assert receipt.canonical_bundle_bytes == binding.canonical_bundle_bytes
    assert receipt.review_preimage_sha256 == binding.review_preimage_sha256

    wing.send_calls.clear()
    wing.query_calls.clear()
    receipt_created_at = receipt.created_at_monotonic
    object.__setattr__(
        receipt,
        "created_at_monotonic",
        time.monotonic() - 91.0,
    )
    insufficient_window = await server.create_m110_maintenance_reset_proposal(
        websocket=websocket,
        request={
            "type": "create_m110_maintenance_reset_commit_proposal",
            "contract_version": 1,
            "bundle_selector": "fixed_channels_1_16_dsp_bypass",
        },
    )
    assert insufficient_window["status"] == "blocked"
    assert insufficient_window["reason"] == (
        "m110_restored_proof_receipt_capture_window_expired"
    )
    assert wing.query_calls == []
    object.__setattr__(
        receipt,
        "created_at_monotonic",
        receipt_created_at,
    )

    with patch(
        "server.capture_m110_reset_preimage",
        wraps=capture_m110_reset_preimage,
    ) as commit_capture:
        commit_created = await server.create_m110_maintenance_reset_proposal(
            websocket=websocket,
            request={
                "type": "create_m110_maintenance_reset_commit_proposal",
                "contract_version": 1,
                "bundle_selector": "fixed_channels_1_16_dsp_bypass",
            },
        )
    assert commit_created["status"] == "created"
    assert commit_created["required_capability_kind"] == (
        "maintenance_dsp_reset_commit_once"
    )
    assert commit_created["matching_restored_proof_receipt"] is True
    assert wing.send_calls == []
    assert len(wing.query_calls) == (
        len(M110_RESET_ADDRESSES) + len(M110_PRESERVATION_ADDRESSES)
    )
    commit_total_deadline = commit_capture.await_args.kwargs[
        "total_deadline_seconds"
    ]
    assert 0.0 < commit_total_deadline <= 90.0
    commit_proposal = server.operator_proposal_queue.find(
        proposal_id=commit_created["proposal_id"]
    )
    assert commit_proposal is not None
    commit_context = server._m110_fixed_reset_proposal_contexts[
        commit_proposal.id
    ]
    assert commit_context["_proof_receipt_ref"] is receipt
    commit_prepared = await server.prepare_m110_restore_only_proof(
        websocket=websocket,
        request={
            "type": "prepare_m110_restore_only_proof",
            "contract_version": 1,
            "proposal_id": commit_proposal.id,
            "bundle_selector": "reviewed_supervised_transaction",
        },
    )
    assert commit_prepared["status"] == "prepared"
    assert commit_prepared["required_capability_kind"] == (
        "maintenance_dsp_reset_commit_once"
    )
    commit_issued = await server.issue_m110_supervised_capability(
        websocket=websocket,
        request={
            "type": "issue_supervised_capability",
            "contract_version": 1,
            "capability_kind": "maintenance_dsp_reset_commit_once",
            "proposal_id": commit_proposal.id,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": (
                commit_proposal.review_content_digest
            ),
            "expected_operation_digest": commit_context[
                "operation_digest"
            ],
            "approval_id": "m110-fixed-reset-commit-approval",
            "confirmation": "ISSUE_MAINTENANCE_DSP_RESET_COMMIT_ONCE",
            "request_nonce": "c" * 32,
            "authority_proof": "hmac-sha256:" + "d" * 64,
        },
    )
    assert commit_issued["status"] == "issued", commit_issued
    commit_request = {
        "type": "apply_operator_proposal",
        "proposal_id": commit_proposal.id,
        "approved": True,
        "approval_id": "m110-fixed-reset-commit-approval",
        "idempotency_key": commit_issued["idempotency_key"],
        "expected_review_contract_version": 2,
        "expected_review_content_digest": (
            commit_proposal.review_content_digest
        ),
        "expected_operation_digest": commit_context["operation_digest"],
    }
    commit_consumed = server.consume_m110_capability_for_apply(
        websocket=websocket,
        request=commit_request,
        token=commit_issued["supervised_capability_token"],
        proposal=commit_proposal,
    )
    assert commit_consumed["consumed"] is True, commit_consumed
    commit_grant = commit_consumed["grant"]
    commit_binding = commit_grant.binding
    assert commit_binding.capability_kind == (
        "maintenance_dsp_reset_commit_once"
    )
    assert commit_binding._proof_receipt_ref is receipt
    commit_reservation_result = (
        server.operator_proposal_queue.reserve_result_evidence(
            commit_proposal
        )
    )
    assert commit_reservation_result["reserved"] is True
    commit_reservation = commit_reservation_result["reservation_binding"]
    commit_claim = server.operator_proposal_queue.claim_verified_for_apply(
        commit_proposal,
        expected_review_contract_version=2,
        expected_review_content_digest=(
            commit_proposal.review_content_digest
        ),
    )
    assert commit_claim["claimed"] is True
    wing.send_calls.clear()

    commit_result = await server.apply_m110_capability_transaction(
        websocket=websocket,
        request=commit_request,
        capability_grant=commit_grant,
        proposal=commit_proposal,
        plan_payload=None,
        result_evidence_reservation_binding=commit_reservation,
    )

    assert commit_result["status"] == "applied"
    assert commit_result["success"] is True
    assert commit_result["wal_terminal"] == "committed"
    assert commit_result["physical_state_restored"] is False
    assert commit_result["rollback_state_unknown"] is False
    assert commit_result["manual_review_required"] is False
    assert [address for address, _value in wing.send_calls] == list(
        M110_RESET_ADDRESSES
    )
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    commit_finalized = server.operator_proposal_queue.mark_apply_result(
        commit_proposal,
        commit_result,
        evidence_reservation_binding=commit_reservation,
    )
    assert commit_finalized["status"] == "applied"
    assert commit_proposal.status == "applied"
    assert commit_proposal not in (
        server.operator_proposal_queue.iter_terminal_hazards()
    )


@pytest.mark.asyncio
async def test_server_fixed_reset_proof_uses_dedicated_executor_and_strict_wal(
    tmp_path,
    monkeypatch,
):
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
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
    wing, _hooks, _executor, approved, authority = await _executor_fixture(
        proof_run=True
    )
    original = dict(wing.values)
    grant = authority[0]
    binding = grant.binding
    server.mixer_client = wing
    server.operator_proposal_queue = MagicMock()
    server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred.return_value = {
        "transitioned": True,
        "reason": None,
    }
    server._m110_consumed_grant_forward_current = MagicMock(return_value=True)
    server._m110_reset_contained_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "passed",
            evidence={
                "capture_status_after": {"captured_sample_count": 1_000_000}
            },
        )
    )
    server.activate_system_emergency_stop = AsyncMock(
        return_value={"success": True, "teardown_complete": True}
    )
    reservation = ResultEvidenceReservationBinding(
        token="a" * 32,
        proposal_id=binding.proposal_id,
        review_content_digest=binding.review_content_digest,
    )
    transaction_id = "m110-server-fixed-reset-proof"

    result = await server._execute_m110_maintenance_dsp_reset_grant(
        capability_grant=grant,
        approved_preimage=approved,
        transaction_id=transaction_id,
        result_evidence_reservation_binding=reservation,
    )

    lifecycle = server._manual_write_wal_lifecycles[transaction_id]
    assert result["status"] == "proof_passed"
    assert result["success"] is True
    assert result["durable_result_projection_valid"] is True
    assert result["wal_terminal"] == "proof_restored"
    assert result["physical_verified"] is True
    assert result["physical_state_restored"] is True
    assert result["manual_review_required"] is False
    assert result["no_console_send_proven"] is False
    assert lifecycle["phases"] == [
        "prepared",
        "verification_pending",
        "verified_state_pending_cleanup",
        "proof_restored",
    ]
    assert classify_m110_lifecycle(lifecycle) == M110_LIFECYCLE_RESOLVED
    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id not in server._manual_write_prepared_transactions
    assert transaction_id not in server._manual_write_recovery_transactions
    assert [address for address, _value in wing.send_calls[:64]] == list(
        M110_RESET_ADDRESSES
    )
    assert {value for _address, value in wing.send_calls[:64]} == {0}
    assert wing.values == original
    assert result["applied_operation_ids"] == [
        row["operation_id"] for row in M110_RESET_OPERATION_DEFINITIONS
    ]
    assert result["rollback_operation_ids"] == [
        row["operation_id"]
        for row in reversed(M110_RESET_OPERATION_DEFINITIONS)
    ]
    server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred.assert_called_once_with(
        reservation
    )
    assert server._m110_reset_contained_audio_recheck.await_count == 2
    server.activate_system_emergency_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_server_fixed_reset_commit_uses_dedicated_executor_and_keeps_reset(
    tmp_path,
    monkeypatch,
):
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
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
    wing, _hooks, _executor, approved, authority = await _executor_fixture()
    grant, restored_receipt = authority
    binding = grant.binding
    server.mixer_client = wing
    server.operator_proposal_queue = MagicMock()
    server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred.return_value = {
        "transitioned": True,
        "reason": None,
    }
    server._m110_consumed_grant_forward_current = MagicMock(return_value=True)
    server._m110_reset_contained_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "passed",
            evidence={
                "capture_status_after": {"captured_sample_count": 1_000_000}
            },
        )
    )
    server.activate_system_emergency_stop = AsyncMock(
        return_value={"success": True, "teardown_complete": True}
    )
    reservation = ResultEvidenceReservationBinding(
        token="b" * 32,
        proposal_id=binding.proposal_id,
        review_content_digest=binding.review_content_digest,
    )
    transaction_id = "m110-server-fixed-reset-commit"

    result = await server._execute_m110_maintenance_dsp_reset_grant(
        capability_grant=grant,
        approved_preimage=approved,
        transaction_id=transaction_id,
        result_evidence_reservation_binding=reservation,
        restored_proof_receipt=restored_receipt,
    )

    lifecycle = server._manual_write_wal_lifecycles[transaction_id]
    assert result["status"] == "applied"
    assert result["success"] is True
    assert result["durable_result_projection_valid"] is True
    assert result["wal_terminal"] == "committed"
    assert result["physical_verified"] is True
    assert result["physical_state_restored"] is False
    assert result["rollback_state_unknown"] is False
    assert result["manual_review_required"] is False
    assert lifecycle["phases"] == [
        "prepared",
        "verification_pending",
        "verified_state_pending_cleanup",
        "committed",
    ]
    assert classify_m110_lifecycle(lifecycle) == M110_LIFECYCLE_RESOLVED
    assert [address for address, _value in wing.send_calls] == list(
        M110_RESET_ADDRESSES
    )
    assert {value for _address, value in wing.send_calls} == {0}
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert transaction_id not in server._manual_write_prepared_transactions
    assert transaction_id not in server._manual_write_recovery_transactions
    server.activate_system_emergency_stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("address", "changed_value", "expected_reason"),
    [
        (
            M110_RESET_ADDRESSES[0],
            0,
            "m110_review_reset_preimage_changed",
        ),
        (
            "/ch/1/in/set/trim",
            -8.0,
            "m110_review_preservation_manifest_changed",
        ),
    ],
)
async def test_next_commit_fresh_recapture_blocks_post_proof_drift(
    address,
    changed_value,
    expected_reason,
):
    wing, _hooks, proof_executor, approved, proof_authority = (
        await _executor_fixture(proof_run=True)
    )
    proof = await proof_executor.execute(
        transaction_id="m110-reset-proof-before-fresh-recapture",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=proof_authority[0],
        restored_proof_receipt=proof_authority[1],
    )
    assert proof.status == "proof_passed"
    sends_before_commit = list(wing.send_calls)
    wing.values[address] = changed_value
    commit_authority = _reset_authority(
        proof_run=False,
        approved_preimage=approved,
        wing_client=wing,
    )
    commit_hooks = _Hooks()
    commit_executor = M110MaintenanceDspResetExecutor(
        wing,
        commit_hooks.values(),
    )

    commit = await commit_executor.execute(
        transaction_id=f"m110-reset-fresh-recapture-{address}",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=commit_authority[0],
        restored_proof_receipt=commit_authority[1],
    )

    assert commit.status == "blocked"
    assert commit.reason == expected_reason
    assert wing.send_calls == sends_before_commit
    assert commit.wal_terminal == "not_prepared"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [1, 32, 64])
async def test_send_failure_rolls_back_attempted_prefix_and_keeps_known_state(fail_at):
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    wing.fail_send_at = fail_at

    result = await executor.execute(
        transaction_id=f"m110-reset-fail-{fail_at}",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "rolled_back"
    assert result.wal_terminal == "rolled_back"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert len(result.rollback_operation_ids) == fail_at
    assert hooks.emergencies == []


@pytest.mark.asyncio
async def test_prepare_and_result_boundary_failures_make_zero_value_sends():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    hooks.prepare_ok = False
    first = await executor.execute(
        transaction_id="m110-reset-no-wal",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )
    assert first.status == "blocked"
    assert wing.send_calls == []

    wing2, hooks2, executor2, approved2, authority2 = await _executor_fixture()
    hooks2.boundary_ok = False
    second = await executor2.execute(
        transaction_id="m110-reset-no-result-boundary",
        approved_preimage=approved2,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority2[0],
        restored_proof_receipt=authority2[1],
    )
    assert second.status == "blocked"
    assert second.wal_terminal == "aborted_before_send"
    assert wing2.send_calls == []


@pytest.mark.asyncio
async def test_outside_mutation_attempt_is_blocked_before_transport_and_rolls_back():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def attempt_outside(client, call_number):
        if call_number == 4:
            assert client.send("/ch/1/fdr", -10.0) is False

    wing.after_value_send = attempt_outside
    result = await executor.execute(
        transaction_id="m110-reset-outsider",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "rolled_back"
    assert result.reason == "m110_outside_mutation_attempt_detected"
    assert wing.values == original
    assert ("/ch/1/fdr", -10.0) not in wing.send_calls
    assert result.mutation_audit["denied_attempts"]


@pytest.mark.asyncio
async def test_delivered_outside_event_blocks_and_rolls_back_without_claiming_equality():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def deliver_outside(client, call_number):
        if call_number == 2:
            client._deliver("/bus/1/fdr", -20.0)

    wing.after_value_send = deliver_outside
    result = await executor.execute(
        transaction_id="m110-reset-delivered-outsider",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_delivered_outside_event_detected"
    assert wing.values == original
    assert any(audit["unexpected_events"] for audit in result.observer_audits)
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_explicit_false_emergency_result_is_not_reported_as_activated():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    hooks.emergency_result = False

    def deliver_outside(client, call_number):
        if call_number == 1:
            client._deliver("/bus/1/fdr", -20.0)

    wing.after_value_send = deliver_outside
    result = await executor.execute(
        transaction_id="m110-reset-emergency-false",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.emergency_stop_activated is False
    assert result.emergency_stop_activation_failed is True


@pytest.mark.asyncio
async def test_terminal_wal_failure_restores_commit_before_emergency_latch():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    hooks.terminal_ok = False

    result = await executor.execute(
        transaction_id="m110-reset-terminal-fail",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_reset_terminal_wal_failed"
    assert result.physical_state_restored is True
    assert result.rollback_state_unknown is False
    assert wing.values == original
    assert not any(
        event.get("phase") == "recovery_gate_installed"
        for event in result.events
    )
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_ambiguous_success_append_never_rolls_back_linearized_state():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()

    def ambiguous_terminal(phase, payload):
        hooks.terminals.append((phase, copy.deepcopy(dict(payload))))
        return None if phase == "committed" else "wal.jsonl"

    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=ambiguous_terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=base_hooks.emergency_stop,
    )
    result = await executor.execute(
        transaction_id="m110-reset-ambiguous-success-append",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_reset_terminal_wal_failed"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is False
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    incident = hooks.terminals[-1][1]
    assert incident["post_linearization_incident"] is True
    assert incident["success_terminal_attempted"] is True
    assert incident["success_terminal_durable_confirmed"] is False
    assert incident["compensation_attempted"] is False
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_cancellation_during_send_waits_for_transport_then_rolls_back():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    wing.block_send_at = 1
    task = asyncio.create_task(
        executor.execute(
            transaction_id="m110-reset-cancel-send",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
        restored_proof_receipt=authority[1],
        )
    )
    assert await asyncio.to_thread(wing.send_started.wait, 2.0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    wing.release_send.set()

    result = await asyncio.wait_for(task, timeout=5.0)

    assert result.status == "manual_review_required"
    assert result.reason.startswith("m110_forward_send_cancelled")
    assert result.physical_state_restored is True
    assert result.write_state_unknown is False
    assert wing.values == original


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_proof_rollback():
    wing, hooks, executor, approved, authority = await _executor_fixture(proof_run=True)
    original = dict(wing.values)
    wing.block_send_at = 65
    task = asyncio.create_task(
        executor.execute(
            transaction_id="m110-reset-cancel-rollback",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
        restored_proof_receipt=authority[1],
        )
    )
    assert await asyncio.to_thread(wing.send_started.wait, 3.0)
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    wing.release_send.set()

    result = await asyncio.wait_for(task, timeout=5.0)

    assert result.status == "rolled_back"
    assert result.reason == "m110_reset_cancelled"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert [phase for phase, _payload in hooks.terminals] == [
        "verified_state_pending_cleanup",
        "rolled_back",
    ]
    assert hooks.terminals[0][1]["cancellation_requested"] is True


@pytest.mark.asyncio
async def test_incremental_safety_catches_silent_reenable_and_restores():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def silent_drift():
        wing.values["/ch/1/eq/on"] = 1

    hooks.on_audio = silent_drift
    result = await executor.execute(
        transaction_id="m110-reset-silent-late-drift",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_preterminal_reset_surface_mismatch"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert hooks.terminals[0][0] == "manual_review_required"
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_audio_not_applicable_is_not_a_real_reset_recheck():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    hooks.audio_status = "not_applicable"

    result = await executor.execute(
        transaction_id="m110-reset-audio-na",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "blocked"
    assert result.reason == "m110_prewrite_audio_recheck_failed"
    assert result.audio_recheck_passed is False
    assert wing.values == original


@pytest.mark.asyncio
async def test_reset_requires_verified_main_mute_before_any_send():
    values = _physical_values()
    values["/main/1/mute"] = 0
    wing = _FakeWing(values)
    approved = await capture_m110_reset_preimage(
        wing.query_parameter,
        expected_identity_values=IDENTITY_VALUES,
    )
    wing.query_calls.clear()
    hooks = _Hooks()
    executor = M110MaintenanceDspResetExecutor(wing, hooks.values())
    grant, receipt = _reset_authority(
        proof_run=False,
        approved_preimage=approved,
        wing_client=wing,
    )

    result = await executor.execute(
        transaction_id="m110-reset-uncontained-main",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=grant,
        restored_proof_receipt=receipt,
    )

    assert result.status == "blocked"
    assert result.reason == "m110_reset_main_output_containment_missing"
    assert wing.send_calls == []
    assert hooks.prepared == []


@pytest.mark.asyncio
async def test_attempt_after_commit_is_incident_without_rewriting_commit():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def terminal_attempt(phase):
        if phase == "committed":
            assert wing.send("/ch/1/fdr", -10.0) is False

    hooks.on_terminal = terminal_attempt
    result = await executor.execute(
        transaction_id="m110-reset-terminal-outsider",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_post_terminal_safety_audit_failed"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is False
    assert wing.values != original
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "committed",
        "manual_review_required",
    ]
    incident = hooks.terminals[-1][1]
    assert incident["no_outside_mutation_attempt_proven"] is False
    assert incident["post_linearization_incident"] is True
    assert incident["linearized_terminal_phase"] == "committed"
    assert incident["compensation_attempted"] is False
    assert incident["rollback_attempted_after_linearization"] is False
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_sync_terminal_self_cancel_becomes_no_write_incident():
    wing, hooks, executor, approved, authority = await _executor_fixture()

    def cancel_after_terminal_append(phase):
        if phase == "committed":
            task = asyncio.current_task()
            assert task is not None
            task.cancel()

    hooks.on_terminal = cancel_after_terminal_append
    result = await executor.execute(
        transaction_id="m110-reset-sync-terminal-self-cancel",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_reset_cancelled"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is False
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "committed",
        "manual_review_required",
    ]
    incident = hooks.terminals[-1][1]
    assert incident["post_linearization_incident"] is True
    assert incident["compensation_attempted"] is False
    assert incident["cancellation_requested"] is True


@pytest.mark.asyncio
async def test_denied_writer_between_detach_audit_and_terminal_never_commits(
    monkeypatch,
):
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    original_snapshot = M110MaintenanceMutationGate.snapshot
    injected = False

    def snapshot_with_gap_attempt(gate):
        nonlocal injected
        audit = original_snapshot(gate)
        if (
            not injected
            and audit.get("active") is True
            and audit.get("sealed") is True
            and "*" not in wing.callbacks
            and hooks.terminals
            and hooks.terminals[-1][0] == "verified_state_pending_cleanup"
        ):
            injected = True
            assert wing.send("/ch/2/fdr", -7.0) is False
        return audit

    monkeypatch.setattr(
        M110MaintenanceMutationGate,
        "snapshot",
        snapshot_with_gap_attempt,
    )
    result = await executor.execute(
        transaction_id="m110-reset-final-gate-baseline-race",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert injected is True
    assert result.status == "manual_review_required"
    assert result.reason == "m110_preterminal_success_invariant_failed"
    assert result.physical_state_restored is True
    assert wing.values == original
    phases = [phase for phase, _payload in hooks.terminals]
    assert "committed" not in phases
    assert phases[-1] == "manual_review_required"


@pytest.mark.asyncio
async def test_observer_event_between_candidate_and_detach_never_commits(
    monkeypatch,
):
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original_snapshot = M110DeliveredMutationObserver.snapshot
    injected = False

    def snapshot_with_gap_event(observer):
        nonlocal injected
        audit = original_snapshot(observer)
        if (
            not injected
            and audit.get("active") is True
            and hooks.terminals
            and hooks.terminals[-1][0] == "verified_state_pending_cleanup"
        ):
            injected = True
            wing._deliver("/bus/1/fdr", -20.0)
        return audit

    monkeypatch.setattr(
        M110DeliveredMutationObserver,
        "snapshot",
        snapshot_with_gap_event,
    )
    result = await executor.execute(
        transaction_id="m110-reset-final-observer-baseline-race",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert injected is True
    assert result.status == "manual_review_required"
    assert result.reason == "m110_observer_detach_safety_audit_failed"
    phases = [phase for phase, _payload in hooks.terminals]
    assert "committed" not in phases
    assert phases[-1] == "manual_review_required"


@pytest.mark.asyncio
async def test_boundary_remove_failure_leaves_commit_and_gate_fail_closed():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    wing.remove_boundary_ok = False

    result = await executor.execute(
        transaction_id="m110-reset-boundary-remove-fail",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_mutation_boundary_remove_failed"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is False
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert wing.boundary is not None
    assert result.mutation_audit["active"] is True
    assert result.mutation_audit["sealed"] is True
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "committed",
        "manual_review_required",
    ]
    incident = hooks.terminals[-1][1]
    assert incident["post_linearization_incident"] is True
    assert incident["compensation_attempted"] is False
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_observer_unsubscribe_failure_is_durable_manual_and_not_success():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    wing.unsubscribe_ok = False

    result = await executor.execute(
        transaction_id="m110-reset-observer-unsubscribe-fail",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_delivered_observer_unsubscribe_failed"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is True
    assert wing.values == original
    assert wing.boundary is None
    assert result.observer_audits[-1]["unsubscribe_failed"] is True
    assert hooks.terminals[-1][0] == "manual_review_required"
    assert hooks.terminals[-1][1]["delivered_observer_clean"] is False
    assert hooks.emergencies


@pytest.mark.asyncio
async def test_observer_detach_drift_is_caught_before_gate_removal():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    trim_address = "/ch/1/in/set/trim"

    def drift_during_unsubscribe(client, _pattern, _callback):
        client.values[trim_address] = -8.0
        client._deliver(trim_address, -8.0)

    wing.on_unsubscribe = drift_during_unsubscribe
    result = await executor.execute(
        transaction_id="m110-reset-observer-detach-drift",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_observer_detach_safety_audit_failed"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is True
    assert wing.values[trim_address] == -8.0
    assert result.preservation_mismatches
    assert hooks.terminals[-1][0] == "manual_review_required"
    assert hooks.terminals[-1][1]["preservation_mismatches"]
    detach_phases = [phase for phase, _payload in hooks.terminals]
    assert "committed" not in detach_phases
    assert detach_phases[:2] == [
        "verification_pending",
        "verified_state_pending_cleanup",
    ]
    assert detach_phases[-1] == "manual_review_required"
    assert wing.boundary is None


@pytest.mark.asyncio
async def test_gate_close_attempt_is_audited_after_irrevocable_commit():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    attempts = []

    def attempt_during_remove(client):
        client.on_remove_boundary = None
        attempts.append(client.send("/ch/2/fdr", -7.0))

    wing.on_remove_boundary = attempt_during_remove
    result = await executor.execute(
        transaction_id="m110-reset-gate-close-attempt",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert attempts == [False]
    assert result.status == "manual_review_required"
    assert result.reason == "m110_mutation_boundary_close_audit_failed"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is False
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert hooks.terminals[-1][0] == "manual_review_required"
    incident = hooks.terminals[-1][1]
    assert incident["no_outside_mutation_attempt_proven"] is False
    assert incident["post_linearization_incident"] is True
    assert incident["compensation_attempted"] is False
    assert [phase for phase, _payload in hooks.terminals][-2:] == [
        "committed",
        "manual_review_required",
    ]
    assert wing.boundary is not None
    assert result.mutation_audit["active"] is True
    assert result.mutation_audit["sealed"] is True
    assert wing.send("/ch/2/fdr", -7.0) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("address", "changed_value"),
    [
        (M110_RESET_ADDRESSES[0], 1),
        ("/ch/1/in/set/trim", -8.0),
    ],
)
async def test_physical_change_after_verification_cut_is_a_later_event(
    address,
    changed_value,
):
    wing, hooks, executor, approved, authority = await _executor_fixture()

    def drift_during_remove(client):
        client.on_remove_boundary = None
        client.values[address] = changed_value
        client._deliver(address, changed_value)

    wing.on_remove_boundary = drift_during_remove
    result = await executor.execute(
        transaction_id=f"m110-reset-gate-remove-drift-{address}",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    # The final 64+184 verification cut defines the historical physical truth.
    # A later console event must be caught by the next mandatory fresh readback
    # rather than causing an unsafe compensating write after terminal attempt.
    assert result.status == "applied"
    assert result.reason == "m110_maintenance_dsp_reset_committed"
    assert result.physical_verified is True
    assert result.wal_terminal == "committed"
    assert result.physical_truth_scope == (
        "verified_as_of_m110_verification_cut"
    )
    assert result.current_state_guaranteed_at_response is False
    assert result.fresh_readback_required_before_next_stage is True
    assert result.attended_no_touch_required_during_execution is True
    assert wing.values[address] == changed_value
    phases = [phase for phase, _payload in hooks.terminals]
    assert phases[-1] == "committed"
    assert "manual_review_required" not in phases
    assert hooks.emergencies == []
    assert wing.boundary is None
    assert "*" not in wing.callbacks


@pytest.mark.asyncio
async def test_verification_pending_silent_reset_drift_never_commits():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def drift_while_pending(phase):
        if phase == "verification_pending":
            wing.values[M110_RESET_ADDRESSES[0]] = 1

    hooks.on_terminal = drift_while_pending
    result = await executor.execute(
        transaction_id="m110-reset-pending-silent-drift",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_final_reset_surface_mismatch"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "manual_review_required",
    ]
    assert "committed" not in [phase for phase, _payload in hooks.terminals]


@pytest.mark.asyncio
async def test_compensating_manual_wal_keeps_gate_sealed_until_it_finishes():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    attempts = []

    def attempt_during_wal(phase):
        if phase in {
            "verified_state_pending_cleanup",
            "manual_review_required",
        }:
            attempts.append((phase, wing.boundary is not None))
            assert wing.send("/ch/2/fdr", -7.0) is False

    hooks.on_terminal = attempt_during_wal
    result = await executor.execute(
        transaction_id="m110-reset-compensating-wal-gate",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert attempts
    assert all(boundary_active for _phase, boundary_active in attempts)
    assert hooks.terminals[-1][1]["no_outside_mutation_attempt_proven"] is False
    assert wing.boundary is None
    assert "*" not in wing.callbacks


@pytest.mark.asyncio
async def test_proof_drift_after_rollback_is_restored_again_before_terminal():
    wing, hooks, executor, approved, authority = await _executor_fixture(proof_run=True)
    original = dict(wing.values)
    injected = False

    def drift_after_manifest_query(client, address):
        nonlocal injected
        if not injected and address == "/main/1/mute" and len(client.send_calls) > 64:
            injected = True
            client.values["/ch/1/eq/on"] = 0

    wing.after_query = drift_after_manifest_query
    result = await executor.execute(
        transaction_id="m110-proof-post-rollback-drift",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert injected is True
    assert result.status == "manual_review_required"
    assert result.reason == "m110_preterminal_reset_surface_mismatch"
    assert result.physical_state_restored is True
    assert result.rollback_state_unknown is False
    assert wing.values == original
    assert hooks.terminals[0][0] == "manual_review_required"


@pytest.mark.asyncio
async def test_proof_change_after_verification_cut_is_a_later_console_event():
    wing, hooks, executor, approved, authority = await _executor_fixture(proof_run=True)
    original = dict(wing.values)

    def proof_terminal_drift(phase):
        if phase == "proof_restored":
            wing.values["/ch/1/eq/on"] = 0
            wing._deliver("/ch/1/eq/on", 0)

    hooks.on_terminal = proof_terminal_drift
    result = await executor.execute(
        transaction_id="m110-proof-terminal-drift",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "proof_passed"
    assert result.physical_state_restored is True
    assert result.rollback_state_unknown is False
    assert result.physical_truth_scope == (
        "verified_as_of_m110_verification_cut"
    )
    assert result.current_state_guaranteed_at_response is False
    assert result.fresh_readback_required_before_next_stage is True
    assert result.attended_no_touch_required_during_execution is True
    assert wing.values != original
    assert wing.values["/ch/1/eq/on"] == 0
    proof_phases = [phase for phase, _payload in hooks.terminals]
    assert proof_phases == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "proof_restored",
    ]
    assert hooks.emergencies == []


@pytest.mark.asyncio
async def test_proof_wal_terminal_has_strict_restored_truth_fields():
    wing, hooks, executor, approved, authority = await _executor_fixture(proof_run=True)

    result = await executor.execute(
        transaction_id="m110-proof-strict-wal",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "proof_passed"
    assert hooks.terminals[0][0] == "verification_pending"
    phase, terminal = hooks.terminals[-1]
    assert phase == "proof_restored"
    assert terminal["proof_target_verified"] is True
    assert terminal["terminal_after_console_send"] is True
    assert terminal["rollback_attempted"] is True
    assert terminal["rollback_succeeded"] is True
    assert terminal["emergency_stop_activated"] is False
    assert terminal["emergency_stop_required"] is False
    assert terminal["accepted"] is False
    assert terminal["owner_review_required"] is True
    assert terminal["live_authority_granted"] is False
    assert terminal["readiness_closure_allowed"] is False
    assert terminal["before_manifest"]["row_count"] == 184
    assert terminal["after_manifest"]["row_count"] == 184
    assert terminal["mutation_audit"]["active"] is True
    assert terminal["mutation_audit"]["sealed"] is True
    assert terminal["mutation_audit"]["denied_attempts"] == []
    assert terminal["delivered_observer_audit"]["active"] is False
    assert terminal["delivered_observer_audit"]["unsubscribe_failed"] is False
    assert terminal["physical_truth_scope"] == (
        "verified_as_of_m110_verification_cut"
    )
    assert terminal["current_state_guaranteed_at_response"] is False
    assert terminal["fresh_readback_required_before_next_stage"] is True
    assert terminal["attended_no_touch_required_during_execution"] is True
    assert terminal[
        "later_console_event_invalidates_readiness_freshness"
    ] is True
    assert terminal["verification_cut_epoch"] > 0
    assert terminal["verification_cut_reset_surface_sha256"].startswith(
        "sha256:"
    )
    assert terminal["mutation_audit_sha256"].startswith("sha256:")
    assert terminal["delivered_observer_audit_sha256"].startswith("sha256:")


@pytest.mark.asyncio
async def test_executor_requires_authentic_typed_capability_grant():
    wing, hooks, executor, approved, authority = await _executor_fixture()

    with pytest.raises(TypeError, match="typed_capability_grant_required"):
        await executor.execute(
            transaction_id="m110-reset-binding-injection",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=None,
        )

    assert wing.send_calls == []
    assert hooks.prepared == []


@pytest.mark.asyncio
async def test_untrusted_outside_value_is_json_safe_and_cannot_escape_cleanup():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def object_attempt(client, call_number):
        if call_number == 2:
            assert client.send("/ch/1/fdr", object()) is False

    wing.after_value_send = object_attempt
    result = await executor.execute(
        transaction_id="m110-reset-object-outsider",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "rolled_back"
    assert result.physical_state_restored is True
    assert wing.values == original
    denied = result.mutation_audit["denied_attempts"]
    assert denied[0]["values"] == [{"type": "object"}]
    assert hooks.terminals[-1][0] == "rolled_back"


@pytest.mark.asyncio
async def test_authority_exception_at_preterminal_check_forces_known_rollback():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    hooks.raise_authority_at = 323

    result = await executor.execute(
        transaction_id="m110-reset-final-authority-exception",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "rolled_back"
    assert result.reason == "m110_write_authority_expired_before_terminal"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert hooks.terminals[-1][0] == "rolled_back"


@pytest.mark.asyncio
async def test_post_cas_authority_fence_blocks_first_forward_send():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    seen_first_address = 0

    def expire_after_forward_cas(_client, address):
        nonlocal seen_first_address
        if address == M110_RESET_ADDRESSES[0]:
            seen_first_address += 1
            if seen_first_address == 2:
                hooks.authority = False

    wing.after_query = expire_after_forward_cas
    result = await executor.execute(
        transaction_id="m110-reset-post-cas-authority",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert wing.send_calls == []
    assert result.reason == "m110_write_authority_expired_before_send"
    assert result.status == "blocked"
    assert result.wal_terminal == "aborted_before_send"


@pytest.mark.asyncio
async def test_grant_is_burned_after_preimage_binding_rejection():
    wing, _hooks, executor, approved, authority = await _executor_fixture()
    changed = copy.deepcopy(approved)
    changed["reset_rows"][0]["value"] = 1 - changed["reset_rows"][0]["value"]

    with pytest.raises(
        ValueError,
        match="m110_reset_reviewed_preimage_binding_mismatch",
    ):
        await executor.execute(
            transaction_id="m110-reset-wrong-preimage",
            approved_preimage=changed,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    assert authority[0].execution_status() == "terminal"
    assert wing.send_calls == []

    with pytest.raises(
        ValueError,
        match="m110_reset_capability_grant_not_executable",
    ):
        await executor.execute(
            transaction_id="m110-reset-replay-after-rejection",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )


@pytest.mark.asyncio
async def test_commit_grant_is_burned_when_restored_receipt_is_missing():
    wing, _hooks, executor, approved, authority = await _executor_fixture()

    with pytest.raises(
        ValueError,
        match="m110_reset_restored_proof_receipt_invalid",
    ):
        await executor.execute(
            transaction_id="m110-reset-missing-receipt",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=None,
        )
    assert authority[0].execution_status() == "terminal"
    assert wing.send_calls == []


@pytest.mark.asyncio
async def test_identity_values_must_match_authenticated_grant():
    wing, _hooks, executor, approved, authority = await _executor_fixture()
    wrong_identity = dict(IDENTITY_VALUES)
    wrong_identity["/$syscfg/$serial"] = "DIFFERENT"

    with pytest.raises(
        ValueError,
        match="m110_reset_expected_identity_binding_mismatch",
    ):
        await executor.execute(
            transaction_id="m110-reset-wrong-identity",
            approved_preimage=approved,
            expected_identity_values=wrong_identity,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    assert authority[0].execution_status() == "terminal"
    assert wing.send_calls == []


def test_mutation_gate_audit_is_bounded_and_overflow_fails_proof():
    wing = _FakeWing(_physical_values())
    preimage = {
        address: int(wing.values[address]) for address in M110_RESET_ADDRESSES
    }
    gate = M110MaintenanceMutationGate(
        wing,
        preimage,
        authority_current=lambda: True,
        max_events=2,
    )
    gate.start()

    assert wing.send("/ch/1/fdr", -10.0) is False
    assert wing.send("/ch/2/fdr", -10.0) is False
    assert wing.send("/ch/3/fdr", -10.0) is False
    audit = gate.snapshot()

    assert audit["event_count"] == 3
    assert audit["retained_event_count"] == 2
    assert audit["overflow"] is True
    assert audit["no_outside_mutation_attempt_proven"] is False
    gate.close()


def test_mutation_gate_audit_projection_handles_cycles_and_large_mappings():
    wing = _FakeWing(_physical_values())
    preimage = {
        address: int(wing.values[address]) for address in M110_RESET_ADDRESSES
    }
    gate = M110MaintenanceMutationGate(
        wing,
        preimage,
        authority_current=lambda: True,
        max_events=8,
    )
    gate.start()
    cyclic = {}
    cyclic["self"] = cyclic
    large = {f"key-{index}": index for index in range(10_000)}

    assert wing.send("/ch/1/fdr", cyclic) is False
    assert wing.send("/ch/2/fdr", large) is False
    audit = gate.snapshot()
    serialized = json.dumps(audit, allow_nan=False)

    assert audit["event_count"] == 2
    assert len(audit["denied_attempts"]) == 2
    assert audit["no_outside_mutation_attempt_proven"] is False
    assert '"type": "cycle"' in serialized
    assert "__audit_truncated__" in serialized
    assert len(serialized) < 20_000
    gate.close()


@pytest.mark.asyncio
async def test_reviewed_preimage_is_frozen_before_waiting_for_executor_lock():
    _wing, _hooks, executor, approved, authority = await _executor_fixture()
    await executor._lock.acquire()
    task = asyncio.create_task(
        executor.execute(
            transaction_id="m110-reset-frozen-review",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    )
    await asyncio.sleep(0)
    assert authority[0].execution_status() == "executing"
    assert authority[0].mark_terminal("outer_cancel_handler") is False
    approved["reset_rows"][0]["value"] = 1 - approved["reset_rows"][0]["value"]
    executor._lock.release()

    result = await task

    assert result.status == "applied"


@pytest.mark.asyncio
async def test_terminal_persist_cancellation_is_true_in_compensating_wal():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    terminal_started = asyncio.Event()
    release_terminal = asyncio.Event()
    terminal_calls = 0

    async def terminal(phase, payload):
        nonlocal terminal_calls
        terminal_calls += 1
        recorded = copy.deepcopy(dict(payload))
        hooks.terminals.append((phase, recorded))
        if terminal_calls == 1:
            terminal_started.set()
            await release_terminal.wait()
        return {
            "path": "wal.jsonl",
            "payload_sha256": _payload_sha256(recorded),
        }

    base_hooks = hooks.values()
    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=base_hooks.emergency_stop,
    )
    task = asyncio.create_task(
        executor.execute(
            transaction_id="m110-reset-terminal-cancel-truth",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    )
    await terminal_started.wait()
    task.cancel()
    release_terminal.set()

    result = await task

    assert result.status == "manual_review_required"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert [phase for phase, _payload in hooks.terminals] == [
        "verification_pending",
        "manual_review_required",
    ]
    assert hooks.terminals[-1][1]["cancellation_requested"] is True


@pytest.mark.asyncio
async def test_preservation_change_after_verification_cut_is_a_later_event():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def drift_during_terminal(phase):
        if phase == "committed":
            wing.values["/ch/1/fdr"] = -10.0
            wing._deliver("/ch/1/fdr", -10.0)

    hooks.on_terminal = drift_during_terminal
    result = await executor.execute(
        transaction_id="m110-reset-terminal-preservation-drift",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert wing.values != original
    assert wing.values["/ch/1/fdr"] == -10.0
    assert result.status == "applied"
    assert result.physical_state_restored is False
    assert result.rollback_state_unknown is False
    assert hooks.terminals[-1][0] == "committed"
    assert hooks.terminals[-1][1]["physical_state_restored"] is False
    assert hooks.terminals[-1][1]["rollback_state_unknown"] is False


@pytest.mark.asyncio
async def test_boundary_authority_sample_blocks_revoke_after_post_cas_check():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()
    authority_calls = 0
    authority_live = True

    def expiring_authority():
        nonlocal authority_calls, authority_live
        authority_calls += 1
        if authority_calls == 3:
            authority_live = False
            return True
        return authority_live

    executor.hooks = M110ResetHooks(
        authority_current=expiring_authority,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=base_hooks.persist_terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=base_hooks.emergency_stop,
    )
    result = await executor.execute(
        transaction_id="m110-reset-boundary-authority-revoke",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert (M110_RESET_ADDRESSES[0], 0) not in wing.send_calls
    assert result.status == "blocked"
    assert result.reason == "m110_write_authority_expired_before_send"
    assert result.wal_terminal == "aborted_before_send"


@pytest.mark.asyncio
async def test_reset_address_toggle_away_from_target_is_observer_failure():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)

    def toggle_after_first_send(client, call_number):
        if call_number == 1:
            address = M110_RESET_ADDRESSES[0]
            client._deliver(address, 1)
            client._deliver(address, 0)

    wing.after_value_send = toggle_after_first_send
    result = await executor.execute(
        transaction_id="m110-reset-target-toggle",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_delivered_outside_event_detected"
    assert wing.values == original
    assert any(audit["unexpected_events"] for audit in result.observer_audits)


@pytest.mark.asyncio
async def test_proof_authority_expiry_after_audio_never_records_proof_success():
    wing, hooks, executor, approved, authority = await _executor_fixture(
        proof_run=True
    )
    original = dict(wing.values)
    def expire_after_final_audio():
        if hooks.audio_calls == 2:
            hooks.authority = False

    hooks.on_audio = expire_after_final_audio

    result = await executor.execute(
        transaction_id="m110-proof-authority-expired-after-audio",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "rolled_back"
    assert result.reason == "m110_write_authority_expired_before_terminal"
    assert result.physical_state_restored is True
    assert wing.values == original
    assert "proof_restored" not in [phase for phase, _ in hooks.terminals]


@pytest.mark.asyncio
async def test_proof_authority_expiry_before_success_terminal_never_succeeds():
    wing, hooks, executor, approved, authority = await _executor_fixture(
        proof_run=True
    )
    original = dict(wing.values)
    sends_at_expiry = []

    def expire_at_candidate(phase):
        if phase == "verified_state_pending_cleanup":
            sends_at_expiry.append(len(wing.send_calls))
            hooks.authority = False

    hooks.on_terminal = expire_at_candidate
    result = await executor.execute(
        transaction_id="m110-proof-authority-expired-before-success",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_write_authority_expired_before_terminal"
    assert wing.values == original
    assert sends_at_expiry == [len(wing.send_calls)]
    phases = [phase for phase, _ in hooks.terminals]
    assert "proof_restored" not in phases
    assert phases[-1] == "manual_review_required"


@pytest.mark.asyncio
async def test_proof_authority_expiry_during_success_is_no_write_incident():
    wing, hooks, executor, approved, authority = await _executor_fixture(
        proof_run=True
    )
    original = dict(wing.values)
    sends_at_success = []

    def expire_at_success(phase):
        if phase == "proof_restored":
            sends_at_success.append(len(wing.send_calls))
            hooks.authority = False

    hooks.on_terminal = expire_at_success
    result = await executor.execute(
        transaction_id="m110-proof-authority-expired-during-success",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_write_authority_expired_before_terminal"
    assert wing.values == original
    assert sends_at_success == [len(wing.send_calls)]
    phases = [phase for phase, _ in hooks.terminals]
    assert phases[-2:] == ["proof_restored", "manual_review_required"]
    incident = hooks.terminals[-1][1]
    assert incident["success_terminal_attempted"] is True
    assert incident["success_terminal_durable_confirmed"] is True
    assert incident["compensation_attempted"] is False
    assert hooks.emergencies


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_run", [False, True])
async def test_authority_expiry_during_gate_release_is_no_write_incident(
    proof_run,
):
    wing, hooks, executor, approved, authority = await _executor_fixture(
        proof_run=proof_run
    )
    sends_at_close = []

    def expire_during_remove(client):
        hooks.authority = False
        sends_at_close.append(len(client.send_calls))

    wing.on_remove_boundary = expire_during_remove
    result = await executor.execute(
        transaction_id=f"m110-authority-expired-gate-release-{proof_run}",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_write_authority_expired_after_terminal"
    assert sends_at_close == [len(wing.send_calls)]
    assert hooks.terminals[-2][0] == (
        "proof_restored" if proof_run else "committed"
    )
    assert hooks.terminals[-1][0] == "manual_review_required"
    assert hooks.terminals[-1][1]["compensation_attempted"] is False
    assert wing.boundary is not None
    assert wing.send("/ch/2/fdr", -7.0) is False


@pytest.mark.asyncio
async def test_final_observed_sweep_rejects_change_to_already_read_reset_row():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original = dict(wing.values)
    injected = False
    candidate_manifest_passes = 0

    def mutate_during_last_manifest_query(client, address):
        nonlocal injected, candidate_manifest_passes
        if (
            hooks.terminals
            and hooks.terminals[-1][0]
            == "verified_state_pending_cleanup"
            and address == M110_PRESERVATION_ADDRESSES[-1]
        ):
            candidate_manifest_passes += 1
        if (
            not injected
            and candidate_manifest_passes == 2
            and address == M110_PRESERVATION_ADDRESSES[-1]
            and "*" in client.callbacks
        ):
            injected = True
            reset_address = M110_RESET_ADDRESSES[0]
            client.values[reset_address] = 1
            client._deliver(reset_address, 1)

    wing.after_query = mutate_during_last_manifest_query
    result = await executor.execute(
        transaction_id="m110-final-observed-sweep-race",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert injected is True
    assert result.status == "manual_review_required"
    assert result.reason == "m110_observer_detach_safety_audit_failed"
    assert wing.values == original
    assert "committed" not in [phase for phase, _ in hooks.terminals]


@pytest.mark.asyncio
async def test_success_terminal_receipt_must_bind_actual_persisted_payload():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()

    def terminal(phase, payload):
        if phase == "committed":
            payload["physical_verified"] = False
            recorded = copy.deepcopy(dict(payload))
            hooks.terminals.append((phase, recorded))
            payload["physical_verified"] = True
        else:
            recorded = copy.deepcopy(dict(payload))
            hooks.terminals.append((phase, recorded))
        return {
            "path": "wal.jsonl",
            "payload_sha256": _payload_sha256(recorded),
        }

    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=base_hooks.emergency_stop,
    )
    result = await executor.execute(
        transaction_id="m110-success-terminal-content-bound-receipt",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_success_terminal_receipt_unbound"
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert hooks.terminals[-2][0] == "committed"
    assert hooks.terminals[-2][1]["physical_verified"] is False
    assert hooks.terminals[-1][0] == "manual_review_required"
    assert hooks.terminals[-1][1]["compensation_attempted"] is False


@pytest.mark.asyncio
async def test_ambiguous_boundary_remove_reinstalls_sealed_no_write_gate():
    wing, hooks, executor, approved, authority = await _executor_fixture()

    def remove_then_raise(client):
        client.boundary = None
        client.boundary_token = None
        client.on_remove_boundary = None
        raise RuntimeError("removed_then_failed")

    wing.on_remove_boundary = remove_then_raise
    result = await executor.execute(
        transaction_id="m110-ambiguous-boundary-remove",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_mutation_boundary_remove_failed"
    assert result.mutation_audit["recovery_boundary_installed"] is True
    assert result.mutation_audit["boundary_state_unknown"] is False
    assert wing.boundary is not None
    assert wing.send("/ch/2/fdr", -7.0) is False


@pytest.mark.asyncio
async def test_close_fault_emergency_hook_cannot_send_after_success_attempt():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()
    emergency_send_results = []

    def denied_attempt_during_remove(client):
        emergency_send_results.append(client.send("/ch/2/fdr", -7.0))

    async def emergency(_reason):
        emergency_send_results.append(wing.send("/ch/3/fdr", -9.0))
        return {"success": True, "teardown_complete": True}

    wing.on_remove_boundary = denied_attempt_during_remove
    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=base_hooks.persist_terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=emergency,
    )
    result = await executor.execute(
        transaction_id="m110-close-fault-no-write-emergency",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert emergency_send_results == [False, False]
    assert all(
        address not in {"/ch/2/fdr", "/ch/3/fdr"}
        for address, _value in wing.send_calls
    )
    assert wing.boundary is not None


@pytest.mark.asyncio
async def test_success_terminal_payload_mutation_is_no_write_incident():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()

    def terminal(phase, payload):
        recorded = copy.deepcopy(dict(payload))
        hooks.terminals.append((phase, recorded))
        if phase == "committed":
            payload["physical_verified"] = False
        return {
            "path": "wal.jsonl",
            "payload_sha256": _payload_sha256(recorded),
        }

    async def emergency(_reason):
        hooks.authority = False
        hooks.emergencies.append("payload_mutation")
        return {"success": True, "teardown_complete": True}

    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=emergency,
    )
    result = await executor.execute(
        transaction_id="m110-success-terminal-payload-mutated",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.reason == "m110_success_terminal_payload_mutated"
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    phases = [phase for phase, _ in hooks.terminals]
    assert phases[-2:] == ["committed", "manual_review_required"]
    assert phases.count("manual_review_required") == 1
    assert hooks.emergencies == ["payload_mutation"]


@pytest.mark.asyncio
async def test_no_callbacks_run_when_recovery_gate_cannot_be_installed():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()
    terminal_send_results = []
    emergency_send_results = []
    committed_seen = False

    def poison_boundary_recovery(client):
        client.boundary = None
        client.boundary_token = None
        client.on_remove_boundary = None

    def terminal(phase, payload):
        nonlocal committed_seen
        recorded = copy.deepcopy(dict(payload))
        hooks.terminals.append((phase, recorded))
        if phase == "committed":
            committed_seen = True
            wing.mutation_boundary_matches = lambda _token: False

            def fail_install(_boundary):
                raise RuntimeError("recovery_install_failed")

            wing.install_mutation_boundary = fail_install
        elif committed_seen:
            terminal_send_results.append(wing.send("/ch/4/fdr", -5.0))
        return {
            "path": "wal.jsonl",
            "payload_sha256": _payload_sha256(recorded),
        }

    async def emergency(_reason):
        emergency_send_results.append(wing.send("/ch/3/fdr", -6.0))
        return {"success": True, "teardown_complete": True}

    wing.on_remove_boundary = poison_boundary_recovery
    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=emergency,
    )
    result = await executor.execute(
        transaction_id="m110-recovery-gate-unavailable",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert result.status == "manual_review_required"
    assert result.wal_terminal == "terminal_record_failed"
    assert result.emergency_stop_activation_failed is True
    assert terminal_send_results == []
    assert emergency_send_results == []
    assert [phase for phase, _ in hooks.terminals][-1] == "committed"
    assert any(
        event.get("phase") == "post_terminal_callbacks_suppressed"
        for event in result.events
    )
    assert all(
        address not in {"/ch/3/fdr", "/ch/4/fdr"}
        for address, _value in wing.send_calls
    )


@pytest.mark.asyncio
async def test_preclose_audit_catches_attempt_after_stale_post_success_snapshot(
    monkeypatch,
):
    wing, hooks, executor, approved, authority = await _executor_fixture()
    original_snapshot = M110MaintenanceMutationGate.snapshot
    attempts = []
    injected = False

    def snapshot_then_attempt(gate):
        nonlocal injected
        audit = original_snapshot(gate)
        if (
            not injected
            and audit.get("active") is True
            and audit.get("sealed") is True
            and hooks.terminals
            and hooks.terminals[-1][0] == "committed"
        ):
            injected = True
            attempts.append(wing.send("/ch/5/fdr", -4.0))
        return audit

    monkeypatch.setattr(
        M110MaintenanceMutationGate,
        "snapshot",
        snapshot_then_attempt,
    )
    result = await executor.execute(
        transaction_id="m110-preclose-stale-snapshot-attempt",
        approved_preimage=approved,
        expected_identity_values=IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )

    assert injected is True
    assert attempts == [False]
    assert result.status == "manual_review_required"
    assert result.reason == "m110_mutation_boundary_close_audit_failed"
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
    assert [phase for phase, _ in hooks.terminals][-2:] == [
        "committed",
        "manual_review_required",
    ]
    assert hooks.terminals[-1][1]["compensation_attempted"] is False
    assert wing.boundary is not None
    assert wing.send("/ch/6/fdr", -3.0) is False


@pytest.mark.asyncio
async def test_sync_incident_hook_self_cancel_gets_corrective_truth_row():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()
    cancelled_in_incident = False

    def attempt_during_remove(client):
        assert client.send("/ch/2/fdr", -7.0) is False

    def terminal(phase, payload):
        nonlocal cancelled_in_incident
        recorded = copy.deepcopy(dict(payload))
        hooks.terminals.append((phase, recorded))
        if phase == "manual_review_required" and not cancelled_in_incident:
            cancelled_in_incident = True
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
        return {
            "path": "wal.jsonl",
            "payload_sha256": _payload_sha256(recorded),
        }

    wing.on_remove_boundary = attempt_during_remove
    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=base_hooks.emergency_stop,
    )
    task = asyncio.create_task(
        executor.execute(
            transaction_id="m110-sync-incident-self-cancel",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    )
    result = await task

    assert result.status == "manual_review_required"
    manual_payloads = [
        payload
        for phase, payload in hooks.terminals
        if phase == "manual_review_required"
    ]
    assert [row["cancellation_requested"] for row in manual_payloads] == [
        False,
        True,
    ]
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)


@pytest.mark.asyncio
async def test_sync_emergency_self_cancel_is_true_in_first_incident_row():
    wing, hooks, executor, approved, authority = await _executor_fixture()
    base_hooks = hooks.values()

    def attempt_during_remove(client):
        assert client.send("/ch/2/fdr", -7.0) is False

    def emergency(_reason):
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        return {"success": True, "teardown_complete": True}

    wing.on_remove_boundary = attempt_during_remove
    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=base_hooks.mark_send_may_have_occurred,
        persist_terminal=base_hooks.persist_terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=emergency,
    )
    task = asyncio.create_task(
        executor.execute(
            transaction_id="m110-sync-emergency-self-cancel",
            approved_preimage=approved,
            expected_identity_values=IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    )
    result = await task

    assert result.status == "manual_review_required"
    manual_payloads = [
        payload
        for phase, payload in hooks.terminals
        if phase == "manual_review_required"
    ]
    assert len(manual_payloads) == 1
    assert manual_payloads[0]["cancellation_requested"] is True
    assert all(wing.values[address] == 0 for address in M110_RESET_ADDRESSES)
