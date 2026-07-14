from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import pwd
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from server import AutoMixerServer
from m110_manual_write_wal import (
    M110_WAL_CONTRACT,
    canonical_sha256 as m110_wal_payload_sha256,
    freeze_wal_payload,
)
from tests.test_maintenance_dsp_reset import (
    IDENTITY_VALUES as M110_IDENTITY_VALUES,
    _executor_fixture as m110_executor_fixture,
)
from maintenance_dsp_reset import M110ResetHooks


class PhysicalMixer:
    def __init__(self):
        self.is_connected = True
        self.state = {"/ch/1/fdr": -8.0}
        self.state_timestamps = {"/ch/1/fdr": time.time()}
        self.sent = []

    def set_channel_fader(self, channel, value):
        self.sent.append((channel, value))
        address = f"/ch/{channel}/fdr"
        self.state[address] = float(value)
        self.state_timestamps[address] = time.time()
        return True

    def query_parameter(self, address, timeout=0.5):
        return {
            "status": "received",
            "address": address,
            "value": self.state.get(address),
            "confirmed_round_trips": 2,
        }


def build_server(tmp_path, monkeypatch, *, runtime_dir=None):
    monkeypatch.setenv("AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(runtime_dir or tmp_path),
    )
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch(
        "server.BleedService"
    ):
        return AutoMixerServer(ws_host="localhost", ws_port=8765)


def configure_physical_fader(server):
    mixer = PhysicalMixer()
    server.connection_mode = "offline"
    server.operator_mode = "auto"
    server._auto_mode_unlocked = True
    server._manual_write_cooldown_sec = 0.0
    server.mixer_client = mixer

    async def physical_read(*, address, **_kwargs):
        return {
            "value": mixer.state.get(address),
            "status": "verified_from_physical_readback",
            "source": "physical_readback",
            "cache_freshness": {"status": "fresh"},
        }

    server._manual_write_read_value = physical_read
    return mixer


def wal_phases(path):
    return [
        json.loads(line)["phase"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wal_event(transaction_id, phase, **extra):
    return {
        "schema": "automixer.manual_write_wal.v1",
        "timestamp": "2026-07-13T10:00:00Z",
        "transaction_id": transaction_id,
        "phase": phase,
        **extra,
    }


def restored_proof_wal_payloads(transaction_id):
    identity = {
        "proof_run": True,
        "restore_after_verify": True,
        "capability_kind": "restore_only_proof",
        "proposal_id": "m110-proof-proposal",
        "review_content_digest": "sha256:" + "1" * 64,
        "operation_digest": "sha256:" + "2" * 64,
    }
    prepared = {
        "scope": "supervised_transaction",
        **identity,
        "expected_operation_ids": ["m110-proof-operation"],
    }
    terminal = {
        "result_status": "proof_passed",
        **identity,
        "applied_operation_ids": ["m110-proof-operation"],
        "rollback_operation_ids": ["m110-proof-operation"],
        "physical_verified": True,
        "proof_target_verified": True,
        "audio_recheck_passed": True,
        "physical_state_restored": True,
        "terminal_after_console_send": True,
        "no_console_send_proven": False,
        "rollback_attempted": True,
        "rollback_succeeded": True,
        "rollback_status": "verified",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "accepted": False,
        "owner_review_required": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }
    return prepared, terminal


async def real_m110_wal_payloads(
    *,
    proof_run=False,
    transaction_id,
    boundary_ok=True,
    fail_send_at=None,
    raise_authority_at=None,
    raise_authority_after_terminal_phase=None,
    denied_outside_mutation=False,
    denied_outside_before_send=False,
    remove_boundary_ok=True,
):
    wing, hooks, executor, approved, authority = await m110_executor_fixture(
        proof_run=proof_run,
    )
    hooks.audio_queries = wing
    hooks.boundary_ok = boundary_ok
    hooks.raise_authority_at = raise_authority_at
    if raise_authority_after_terminal_phase is not None:
        def arm_authority_failure(phase):
            if phase == raise_authority_after_terminal_phase:
                hooks.raise_authority_at = hooks.authority_calls + 1

        hooks.on_terminal = arm_authority_failure
    wing.fail_send_at = fail_send_at
    wing.remove_boundary_ok = remove_boundary_ok
    if denied_outside_mutation:
        def attempt_outside_mutation(client, call_number):
            if call_number == 4:
                assert client.send("/ch/1/fdr", -10.0) is False

        wing.after_value_send = attempt_outside_mutation
    if denied_outside_before_send:
        attempted = False

        def attempt_outside_during_recapture(client, _address):
            nonlocal attempted
            if not attempted:
                attempted = True
                assert client.send("/ch/1/fdr", -10.0) is False

        wing.after_query = attempt_outside_during_recapture
    result = await executor.execute(
        transaction_id=transaction_id,
        approved_preimage=approved,
        expected_identity_values=M110_IDENTITY_VALUES,
        capability_grant=authority[0],
        restored_proof_receipt=authority[1],
    )
    return result, copy.deepcopy(hooks.prepared[0]), copy.deepcopy(hooks.terminals)


def append_m110_payload_sequence(server, transaction_id, prepared, terminals):
    receipts = [
        server._append_m110_manual_write_wal(
            "prepared",
            transaction_id,
            prepared,
        )
    ]
    receipts.extend(
        server._append_m110_manual_write_wal(
            phase,
            transaction_id,
            payload,
        )
        for phase, payload in terminals
    )
    return receipts


def m110_post_success_incident(success_payload, success_phase):
    payload = copy.deepcopy(success_payload)
    payload.update(
        {
            "result_status": "manual_review_required",
            "result_reason": "m110_write_authority_expired_before_terminal",
            "manual_review_required": True,
            "physical_verified": False,
            "emergency_stop_activated": True,
            "emergency_stop_activation_failed": False,
            "emergency_stop_required": True,
            "no_outside_mutation_attempt_proven": False,
            "accepted": False,
            "owner_review_required": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "post_linearization_incident": True,
            "linearized_terminal_phase": success_phase,
            "linearized_state_preserved_by_executor": True,
            "success_terminal_attempted": True,
            "success_terminal_durable_confirmed": True,
            "compensation_attempted": False,
            "rollback_attempted_after_linearization": False,
        }
    )
    return payload


def test_exact_restored_proof_wal_is_terminal_across_restart(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "proof-restored-runtime"
    server = build_server(
        tmp_path / "proof-restored-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    transaction_id = "m110-proof-restored-golden"
    prepared, terminal = restored_proof_wal_payloads(transaction_id)

    assert server._append_manual_write_wal(
        "prepared",
        transaction_id,
        **prepared,
    )
    assert server._append_manual_write_wal(
        "proof_restored",
        transaction_id,
        **terminal,
    )
    lifecycle = server._manual_write_wal_lifecycles[transaction_id]
    assert server._wal_lifecycle_proves_restored_proof(lifecycle) is True
    assert [event["phase"] for event in lifecycle["events"]] == [
        "prepared",
        "proof_restored",
    ]
    assert lifecycle["events"][0] is lifecycle["prepared"]
    assert lifecycle["events"][-1] is lifecycle["latest_event"]
    assert transaction_id not in server._manual_write_recovery_transactions

    restarted = build_server(
        tmp_path / "proof-restored-events-after-restart",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert restarted._manual_write_recovery_transactions == {}
    assert restarted._manual_write_emergency_stopped is False
    restarted_lifecycle = restarted._manual_write_wal_lifecycles[transaction_id]
    assert restarted_lifecycle["events"] == lifecycle["events"]


def test_already_satisfied_restored_proof_wal_is_terminal_without_send(
    tmp_path,
    monkeypatch,
):
    server = build_server(
        tmp_path / "proof-already-satisfied-events",
        monkeypatch,
        runtime_dir=tmp_path / "proof-already-satisfied-runtime",
    )
    transaction_id = "m110-proof-already-satisfied"
    prepared, terminal = restored_proof_wal_payloads(transaction_id)
    terminal.update(
        {
            "applied_operation_ids": [],
            "rollback_operation_ids": [],
            "already_satisfied_operation_ids": ["m110-proof-operation"],
            "terminal_after_console_send": False,
            "no_console_send_proven": True,
            "rollback_attempted": False,
            "rollback_succeeded": False,
            "rollback_status": "not_required_already_satisfied",
        }
    )

    assert server._append_manual_write_wal(
        "prepared",
        transaction_id,
        **prepared,
    )
    assert server._append_manual_write_wal(
        "proof_restored",
        transaction_id,
        **terminal,
    )

    lifecycle = server._manual_write_wal_lifecycles[transaction_id]
    assert server._wal_lifecycle_proves_restored_proof(lifecycle) is True
    assert transaction_id not in server._manual_write_recovery_transactions


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("audio_recheck_passed", False),
        ("physical_state_restored", False),
        ("rollback_succeeded", False),
        ("manual_review_required", True),
        ("emergency_stop_activated", True),
        ("accepted", True),
        ("owner_review_required", False),
        ("live_authority_granted", True),
        ("readiness_closure_allowed", True),
    ],
)
def test_near_miss_restored_proof_wal_remains_recovery_hazard(
    tmp_path,
    monkeypatch,
    field_name,
    unsafe_value,
):
    runtime_dir = tmp_path / f"near-miss-{field_name}"
    server = build_server(
        tmp_path / f"near-miss-events-{field_name}",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    transaction_id = f"m110-proof-near-miss-{field_name}"
    prepared, terminal = restored_proof_wal_payloads(transaction_id)
    terminal[field_name] = unsafe_value

    assert server._append_manual_write_wal(
        "prepared",
        transaction_id,
        **prepared,
    )
    assert server._append_manual_write_wal(
        "proof_restored",
        transaction_id,
        **terminal,
    )
    lifecycle = server._manual_write_wal_lifecycles[transaction_id]
    assert server._wal_lifecycle_proves_restored_proof(lifecycle) is False
    assert transaction_id in server._manual_write_recovery_transactions

    restarted = build_server(
        tmp_path / f"near-miss-restart-{field_name}",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_run", [False, True])
async def test_strict_m110_success_receipt_and_restart_are_content_bound(
    tmp_path,
    monkeypatch,
    proof_run,
):
    transaction_id = f"strict-m110-{'proof' if proof_run else 'commit'}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        proof_run=proof_run,
        transaction_id=transaction_id,
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    lifecycle = server._manual_write_wal_lifecycles[transaction_id]
    assert server._wal_lifecycle_terminal_resolved(lifecycle) is True
    assert transaction_id not in server._manual_write_recovery_transactions
    assert lifecycle["phases"] == [
        "prepared",
        "verification_pending",
        "verified_state_pending_cleanup",
        "proof_restored" if proof_run else "committed",
    ]
    expected_payloads = [prepared, *[payload for _phase, payload in terminals]]
    raw_rows = server._manual_write_wal_path().read_bytes().splitlines(
        keepends=True
    )
    for receipt, event, expected_payload, raw_row in zip(
        receipts,
        lifecycle["events"],
        expected_payloads,
        raw_rows,
    ):
        assert receipt["payload_sha256"] == m110_wal_payload_sha256(
            expected_payload
        )
        assert event["wal_contract"] == M110_WAL_CONTRACT
        assert event["wal_payload"] == freeze_wal_payload(expected_payload)
        assert event["wal_payload_sha256"] == receipt["payload_sha256"]
        assert receipt["record_sha256"] == (
            "sha256:" + hashlib.sha256(raw_row).hexdigest()
        )
        assert receipt["record_size_bytes"] == len(raw_row)

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert restarted._manual_write_recovery_transactions == {}
    assert restarted._manual_write_emergency_stopped is False
    assert restarted._manual_write_wal_lifecycles[transaction_id][
        "events"
    ] == lifecycle["events"]
    assert result.wal_terminal == (
        "proof_restored" if proof_run else "committed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix_length", [1, 2, 3])
async def test_strict_m110_valid_prefix_is_integral_but_unresolved(
    tmp_path,
    monkeypatch,
    prefix_length,
):
    transaction_id = f"strict-m110-prefix-{prefix_length}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )
    sequence = [("prepared", prepared), *terminals]

    for phase, payload in sequence[:prefix_length]:
        assert server._append_m110_manual_write_wal(
            phase,
            transaction_id,
            payload,
        )

    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id in server._manual_write_recovery_transactions
    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
async def test_strict_m110_real_abort_before_send_is_resolved_across_restart(
    tmp_path,
    monkeypatch,
):
    transaction_id = "strict-m110-real-abort-before-send"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        boundary_ok=False,
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    assert result.status == "blocked"
    assert result.applied_operation_ids == []
    assert result.wal_terminal == "aborted_before_send"
    assert [phase for phase, _payload in terminals] == ["aborted_before_send"]
    assert transaction_id not in server._manual_write_recovery_transactions
    assert server._manual_write_wal_integrity_ok is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert restarted._manual_write_recovery_transactions == {}
    assert restarted._manual_write_emergency_stopped is False


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_run", [False, True])
async def test_strict_m110_denied_outsider_before_send_is_safe_abort(
    tmp_path,
    monkeypatch,
    proof_run,
):
    transaction_id = f"strict-m110-denied-outsider-before-send-{proof_run}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        proof_run=proof_run,
        denied_outside_before_send=True,
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    assert result.status == "blocked"
    assert result.reason == "m110_outside_mutation_attempt_detected"
    assert result.applied_operation_ids == []
    assert [phase for phase, _payload in terminals] == ["aborted_before_send"]
    assert terminals[-1][1]["no_outside_mutation_attempt_proven"] is False
    assert terminals[-1][1]["mutation_audit"]["denied_attempts"]
    assert server._manual_write_wal_integrity_ok is True
    assert server._manual_write_recovery_transactions == {}

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert restarted._manual_write_recovery_transactions == {}
    assert restarted._manual_write_emergency_stopped is False


@pytest.mark.asyncio
async def test_strict_m110_cancelled_boundary_abort_is_valid_zero_send(
    tmp_path,
    monkeypatch,
):
    transaction_id = "strict-m110-cancelled-boundary-abort"
    wing, hooks, executor, approved, authority = await m110_executor_fixture()
    base_hooks = hooks.values()
    boundary_started = asyncio.Event()
    release_boundary = asyncio.Event()

    async def boundary():
        boundary_started.set()
        await release_boundary.wait()
        return True

    executor.hooks = M110ResetHooks(
        authority_current=base_hooks.authority_current,
        persist_prepared=base_hooks.persist_prepared,
        mark_send_may_have_occurred=boundary,
        persist_terminal=base_hooks.persist_terminal,
        audio_recheck=base_hooks.audio_recheck,
        emergency_stop=base_hooks.emergency_stop,
    )
    execution = asyncio.create_task(
        executor.execute(
            transaction_id=transaction_id,
            approved_preimage=approved,
            expected_identity_values=M110_IDENTITY_VALUES,
            capability_grant=authority[0],
            restored_proof_receipt=authority[1],
        )
    )
    await boundary_started.wait()
    execution.cancel()
    release_boundary.set()
    result = await execution
    server = build_server(tmp_path / "events", monkeypatch)

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        copy.deepcopy(hooks.prepared[0]),
        copy.deepcopy(hooks.terminals),
    )

    assert all(receipts)
    assert result.status == "blocked"
    assert result.reason == "m110_reset_cancelled"
    assert result.wal_terminal == "aborted_before_send"
    assert wing.send_calls == []
    assert hooks.terminals[-1][1]["cancellation_requested"] is True
    assert server._manual_write_wal_integrity_ok is True
    assert server._manual_write_recovery_transactions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsafe_field", "unsafe_value"),
    [
        ("emergency_stop_activation_failed", True),
        ("accepted", True),
        ("live_authority_granted", True),
        ("readiness_closure_allowed", True),
        ("delivered_observer_clean", False),
        ("no_outside_mutation_attempt_proven", False),
    ],
)
async def test_strict_m110_malformed_abort_before_send_is_invalid(
    tmp_path,
    monkeypatch,
    unsafe_field,
    unsafe_value,
):
    transaction_id = f"strict-m110-malformed-abort-{unsafe_field}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        boundary_ok=False,
    )
    assert server._append_m110_manual_write_wal(
        "prepared",
        transaction_id,
        prepared,
    )
    terminal_phase, terminal_payload = terminals[-1]
    terminal_payload[unsafe_field] = unsafe_value

    assert server._append_m110_manual_write_wal(
        terminal_phase,
        transaction_id,
        terminal_payload,
    ) is None
    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is False
    assert transaction_id in restarted._manual_write_recovery_transactions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_id", "executor_options"),
    [
        ("send-failure", {"fail_send_at": 32}),
        ("late-authority-expiry", {"raise_authority_at": 195}),
        ("denied-outside-mutation", {"denied_outside_mutation": True}),
    ],
)
async def test_strict_m110_real_rollback_is_integral_but_unresolved(
    tmp_path,
    monkeypatch,
    case_id,
    executor_options,
):
    transaction_id = f"strict-m110-real-rollback-{case_id}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        **executor_options,
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    assert result.status == "rolled_back"
    assert [phase for phase, _payload in terminals] == [
        "verified_state_pending_cleanup",
        "rolled_back",
    ]
    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_run", [False, True])
async def test_strict_m110_post_verification_compensation_is_unresolved(
    tmp_path,
    monkeypatch,
    proof_run,
):
    transaction_id = f"strict-m110-post-verification-rollback-{proof_run}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        proof_run=proof_run,
        raise_authority_after_terminal_phase="verification_pending",
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    assert result.status == "rolled_back"
    assert result.reason == "m110_write_authority_expired_before_terminal"
    assert [phase for phase, _payload in terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "rolled_back",
    ]
    assert terminals[-1][1]["verification_candidate_phase"] == (
        "proof_restored" if proof_run else "committed"
    )
    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_run", [False, True])
@pytest.mark.parametrize(
    ("case_id", "executor_options", "expected_phases"),
    [
        (
            "abort",
            {"boundary_ok": False, "remove_boundary_ok": False},
            ["aborted_before_send", "manual_review_required"],
        ),
        (
            "rollback",
            {"fail_send_at": 32, "remove_boundary_ok": False},
            [
                "verified_state_pending_cleanup",
                "rolled_back",
                "manual_review_required",
            ],
        ),
    ],
)
async def test_strict_m110_completed_terminal_then_manual_is_unresolved(
    tmp_path,
    monkeypatch,
    case_id,
    executor_options,
    expected_phases,
    proof_run,
):
    transaction_id = f"strict-m110-{case_id}-then-manual-{proof_run}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        proof_run=proof_run,
        **executor_options,
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    assert result.status == "manual_review_required"
    assert [phase for phase, _payload in terminals] == expected_phases
    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_run", [False, True])
async def test_strict_m110_pre_success_manual_is_unresolved(
    tmp_path,
    monkeypatch,
    proof_run,
):
    transaction_id = f"strict-m110-pre-success-manual-{proof_run}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        proof_run=proof_run,
        raise_authority_after_terminal_phase=(
            "verified_state_pending_cleanup"
        ),
    )

    receipts = append_m110_payload_sequence(
        server,
        transaction_id,
        prepared,
        terminals,
    )

    assert all(receipts)
    assert result.status == "manual_review_required"
    assert [phase for phase, _payload in terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "manual_review_required",
    ]
    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "candidate_status",
        "shortened_operation_history",
        "coordinated_operation_erasure",
        "arbitrary_cut_sha",
        "audit_no_outside_mismatch",
    ],
)
async def test_strict_m110_pre_success_manual_rejects_semantic_forgery(
    tmp_path,
    monkeypatch,
    corruption,
):
    transaction_id = f"strict-m110-pre-success-manual-{corruption}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        raise_authority_after_terminal_phase=(
            "verified_state_pending_cleanup"
        ),
    )
    assert [phase for phase, _payload in terminals] == [
        "verification_pending",
        "verified_state_pending_cleanup",
        "manual_review_required",
    ]
    for phase, payload in [("prepared", prepared), *terminals[:-1]]:
        assert server._append_m110_manual_write_wal(
            phase,
            transaction_id,
            payload,
        )
    manual_phase, manual_payload = terminals[-1]
    if corruption == "candidate_status":
        manual_payload["verification_candidate_status"] = "fabricated"
    elif corruption == "shortened_operation_history":
        manual_payload["applied_operation_ids"] = []
    elif corruption == "coordinated_operation_erasure":
        manual_payload.update(
            {
                "applied_operation_ids": [],
                "rollback_operation_ids": [],
                "rollback_attempted": False,
                "rollback_succeeded": False,
                "physical_state_restored": False,
            }
        )
    elif corruption == "arbitrary_cut_sha":
        manual_payload.update(
            {
                "verification_cut_epoch": 1.0,
                "verification_cut_reset_surface_sha256": (
                    "sha256:" + "0" * 64
                ),
                "verification_cut_preservation_manifest_sha256": (
                    manual_payload["after_manifest_sha256"]
                ),
                "physical_truth_scope": (
                    "verified_as_of_m110_verification_cut"
                ),
            }
        )
    else:
        manual_payload["mutation_audit"][
            "no_outside_mutation_attempt_proven"
        ] = False
        manual_payload["mutation_audit_sha256"] = m110_wal_payload_sha256(
            manual_payload["mutation_audit"]
        )

    assert server._append_m110_manual_write_wal(
        manual_phase,
        transaction_id,
        manual_payload,
    ) is None
    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is False
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["binding_drift", "duplicate_prepared"])
async def test_strict_m110_rollback_rejects_binding_drift_and_duplicate_prepare(
    tmp_path,
    monkeypatch,
    corruption,
):
    transaction_id = f"strict-m110-rollback-{corruption}"
    server = build_server(tmp_path / "events", monkeypatch)
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
        fail_send_at=32,
    )
    assert server._append_m110_manual_write_wal(
        "prepared",
        transaction_id,
        prepared,
    )
    if corruption == "duplicate_prepared":
        assert server._append_m110_manual_write_wal(
            "prepared",
            transaction_id,
            prepared,
        ) is None
    else:
        candidate_phase, candidate_payload = terminals[0]
        assert server._append_m110_manual_write_wal(
            candidate_phase,
            transaction_id,
            candidate_payload,
        )
        final_phase, final_payload = terminals[-1]
        final_payload["proposal_id"] = "different-proposal"
        assert server._append_m110_manual_write_wal(
            final_phase,
            transaction_id,
            final_payload,
        ) is None

    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsafe_field", "unsafe_value"),
    [
        ("physical_verified", False),
        ("emergency_stop_activation_failed", True),
    ],
)
async def test_strict_m110_malformed_success_is_integrity_failure_live_and_restart(
    tmp_path,
    monkeypatch,
    unsafe_field,
    unsafe_value,
):
    transaction_id = f"strict-m110-malformed-success-{unsafe_field}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )
    for phase, payload in [("prepared", prepared), *terminals[:-1]]:
        assert server._append_m110_manual_write_wal(
            phase,
            transaction_id,
            payload,
        )
    final_phase, final_payload = terminals[-1]
    final_payload[unsafe_field] = unsafe_value

    assert server._append_m110_manual_write_wal(
        final_phase,
        transaction_id,
        final_payload,
    ) is None
    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is False
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsafe_field", "unsafe_value"),
    [
        ("write_state_unknown", True),
        ("emergency_stop_required", True),
        ("cancellation_requested", True),
        ("delivered_observer_clean", False),
    ],
)
async def test_strict_m110_malformed_pending_cannot_be_healed_by_valid_commit(
    tmp_path,
    monkeypatch,
    unsafe_field,
    unsafe_value,
):
    transaction_id = f"strict-m110-malformed-pending-{unsafe_field}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )
    assert server._append_m110_manual_write_wal(
        "prepared",
        transaction_id,
        prepared,
    )
    pending_phase, pending_payload = terminals[0]
    pending_payload[unsafe_field] = unsafe_value

    assert server._append_m110_manual_write_wal(
        pending_phase,
        transaction_id,
        pending_payload,
    ) is None
    for phase, payload in terminals[1:]:
        assert server._append_m110_manual_write_wal(
            phase,
            transaction_id,
            payload,
        ) is None
    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._wal_lifecycle_terminal_resolved(
        server._manual_write_wal_lifecycles[transaction_id]
    ) is False

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is False
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
async def test_strict_m110_success_followed_by_incident_is_unresolved_not_corrupt(
    tmp_path,
    monkeypatch,
):
    transaction_id = "strict-m110-post-success-incident"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )
    assert all(
        append_m110_payload_sequence(
            server,
            transaction_id,
            prepared,
            terminals,
        )
    )
    success_phase, success_payload = terminals[-1]
    incident = m110_post_success_incident(success_payload, success_phase)

    assert server._append_m110_manual_write_wal(
        "manual_review_required",
        transaction_id,
        incident,
    )
    assert server._manual_write_wal_integrity_ok is True
    assert transaction_id in server._manual_write_recovery_transactions

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsafe_field", "unsafe_value"),
    [
        ("verification_candidate_status", "fabricated"),
        ("applied_operation_ids", []),
        ("after_manifest_sha256", "sha256:" + "0" * 64),
        ("mutation_audit_sha256", "sha256:" + "0" * 64),
        ("post_linearization_incident", False),
    ],
)
async def test_strict_m110_post_success_incident_is_bound_to_success_history(
    tmp_path,
    monkeypatch,
    unsafe_field,
    unsafe_value,
):
    transaction_id = f"strict-m110-incident-history-{unsafe_field}"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )
    assert all(
        append_m110_payload_sequence(
            server,
            transaction_id,
            prepared,
            terminals,
        )
    )
    success_phase, success_payload = terminals[-1]
    incident = m110_post_success_incident(success_payload, success_phase)
    incident[unsafe_field] = unsafe_value

    assert server._append_m110_manual_write_wal(
        "manual_review_required",
        transaction_id,
        incident,
    ) is None
    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is False
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


@pytest.mark.asyncio
async def test_strict_m110_directory_fsync_ambiguity_keeps_row_then_incident(
    tmp_path,
    monkeypatch,
):
    transaction_id = "strict-m110-directory-fsync-ambiguity"
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )
    for phase, payload in [("prepared", prepared), *terminals[:-1]]:
        assert server._append_m110_manual_write_wal(
            phase,
            transaction_id,
            payload,
        )
    success_phase, success_payload = terminals[-1]
    original_fsync_directory = server._fsync_directory
    server._fsync_directory = MagicMock(
        side_effect=OSError("injected_directory_fsync_failure")
    )

    assert server._append_m110_manual_write_wal(
        success_phase,
        transaction_id,
        success_payload,
    ) is None
    assert server._manual_write_wal_lifecycles[transaction_id]["phases"][-1] == (
        success_phase
    )
    server._fsync_directory = original_fsync_directory
    incident = m110_post_success_incident(success_payload, success_phase)
    assert server._append_m110_manual_write_wal(
        "manual_review_required",
        transaction_id,
        incident,
    )

    rows = [
        json.loads(line)
        for line in server._manual_write_wal_path()
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["phase"] for row in rows][-2:] == [
        success_phase,
        "manual_review_required",
    ]
    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is True
    assert transaction_id in restarted._manual_write_recovery_transactions


@pytest.mark.asyncio
async def test_strict_m110_file_fsync_failure_latches_integrity_before_receipt(
    tmp_path,
    monkeypatch,
):
    transaction_id = "strict-m110-file-fsync-failure"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True)
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    _result, prepared, _terminals = await real_m110_wal_payloads(
        transaction_id=transaction_id,
    )

    with patch("server.os.fsync", side_effect=OSError("injected_file_fsync")):
        receipt = server._append_m110_manual_write_wal(
            "prepared",
            transaction_id,
            prepared,
        )

    assert receipt is None
    assert server._manual_write_wal_integrity_ok is False
    assert server._manual_write_emergency_stopped is True
    assert server._manual_write_emergency_generation == 1


def test_strict_m110_rejects_reserved_payload_and_nonfinite_number(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )

    assert server._append_m110_manual_write_wal(
        "prepared",
        "strict-m110-reserved",
        {"schema": "caller-controlled"},
    ) is None
    assert server._append_m110_manual_write_wal(
        "prepared",
        "strict-m110-nan",
        {"unsafe": float("nan")},
    ) is None
    assert not server._manual_write_wal_path().exists()


def test_strict_m110_nonfinite_row_fails_closed_during_restart(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True)
    transaction_id = "strict-m110-nan-restart"
    row = {
        "schema": "automixer.manual_write_wal.v1",
        "timestamp": "2026-07-13T10:00:00Z",
        "transaction_id": transaction_id,
        "phase": "prepared",
        "scope": "supervised_transaction",
        "maintenance_dsp_reset": True,
        "wal_contract": M110_WAL_CONTRACT,
        "wal_payload": {"unsafe": float("nan")},
        "wal_payload_sha256": "sha256:" + "1" * 64,
    }
    (runtime_dir / "manual_write_wal.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )

    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True


def test_flat_m110_marker_cannot_downgrade_into_legacy_terminal_acceptance(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "runtime"
    server = build_server(
        tmp_path / "events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    transaction_id = "flat-m110-downgrade"

    assert server._append_manual_write_wal(
        "prepared",
        transaction_id,
        scope="supervised_transaction",
        bundle_kind="maintenance_dsp_reset_bundle",
    )
    assert server._append_manual_write_wal(
        "committed",
        transaction_id,
        result_status="applied",
    )

    assert server._manual_write_wal_integrity_ok is False
    assert transaction_id in server._manual_write_recovery_transactions

    restarted = build_server(
        tmp_path / "restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert restarted._manual_write_wal_integrity_ok is False
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True


def test_startup_latches_on_unresolved_wal_and_reset_cannot_clear_it(
    tmp_path,
    monkeypatch,
):
    transaction_id = "crashed-transaction"
    (tmp_path / "manual_write_wal.jsonl").write_text(
        json.dumps(
            {
                "schema": "automixer.manual_write_wal.v1",
                "timestamp": "2026-07-11T20:00:00Z",
                "transaction_id": transaction_id,
                "phase": "send_started",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    server = build_server(tmp_path, monkeypatch)

    assert server._manual_write_emergency_stopped is True
    assert transaction_id in server._manual_write_recovery_transactions
    status = server.get_manual_write_supervision_status()
    assert status["wal_recovery_required"] is True
    reset = server.reset_manual_write_emergency_stop(
        approved=True,
        approval_id="operator-reviewed",
        reason="text_ack_is_not_physical_reconciliation",
    )
    assert reset["status"] == "blocked"
    assert reset["reason"] == "console_write_recovery_pending"
    assert server._manual_write_emergency_stopped is True


def test_corrupt_wal_fails_closed_at_startup(tmp_path, monkeypatch):
    (tmp_path / "manual_write_wal.jsonl").write_text(
        '{"transaction_id":"cut-off",',
        encoding="utf-8",
    )

    server = build_server(tmp_path, monkeypatch)

    assert server._manual_write_emergency_stopped is True
    assert any(
        key.startswith("wal_parse_error:")
        for key in server._manual_write_recovery_transactions
    )


def test_console_writer_lock_is_global_across_configured_artifact_roots(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "shared-console-runtime"
    first = build_server(
        tmp_path / "first-evidence-root",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    first_lock = first._console_writer_lock_path
    second = build_server(
        tmp_path / "second-evidence-root",
        monkeypatch,
        runtime_dir=runtime_dir,
    )

    assert first._manual_write_artifacts_dir != second._manual_write_artifacts_dir
    assert first_lock == second._console_writer_lock_path
    assert str(first_lock) == first.get_manual_write_supervision_status()["console_writer_lock"]
    assert first.get_manual_write_supervision_status()["console_writer_lock_scope"] == (
        "per_user_global"
    )
    assert first._manual_write_wal_path() == second._manual_write_wal_path()
    assert first._manual_write_wal_path().parent == runtime_dir
    monkeypatch.delenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR")
    assert AutoMixerServer._resolve_console_runtime_dir() == (
        Path(pwd.getpwuid(os.getuid()).pw_dir) / ".automixer" / "runtime"
    )


def test_cross_root_process_recovers_unresolved_global_wal(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "shared-console-runtime"
    first = build_server(
        tmp_path / "first-evidence-root",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    transaction_id = "cross-root-crashed-transaction"
    assert first._append_manual_write_wal(
        "prepared",
        transaction_id,
        scope="manual_write",
        address="/ch/1/fdr",
        value_type="fader",
        old_value=-8.0,
        effective_value=-9.0,
    )

    second = build_server(
        tmp_path / "second-evidence-root",
        monkeypatch,
        runtime_dir=runtime_dir,
    )

    assert transaction_id in second._manual_write_recovery_transactions
    assert second._manual_write_emergency_stopped is True
    assert second.get_manual_write_supervision_status()["wal_recovery_required"] is True


def test_fresh_nested_runtime_fsyncs_each_created_parent_entry(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "outer" / "inner"
    server = build_server(
        tmp_path / "evidence",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    fsynced = []
    original_fsync_directory = AutoMixerServer._fsync_directory

    def record_fsync(path):
        fsynced.append(Path(path))
        original_fsync_directory(path)

    with patch.object(
        AutoMixerServer,
        "_fsync_directory",
        side_effect=record_fsync,
    ):
        assert server._append_manual_write_wal(
            "prepared",
            "nested-runtime-transaction",
            scope="manual_write",
        )

    assert fsynced[:3] == [tmp_path, tmp_path / "outer", runtime_dir]


@pytest.mark.asyncio
async def test_physical_reconciliation_closes_recovery_before_operator_reset(
    tmp_path,
    monkeypatch,
):
    transaction_id = "reconcile-target"
    rows = [
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:00Z",
            "transaction_id": transaction_id,
            "phase": "prepared",
            "scope": "manual_write",
            "address": "/ch/1/fdr",
            "value_type": "fader",
            "old_value": -8.0,
            "effective_value": -9.0,
        },
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:01Z",
            "transaction_id": transaction_id,
            "phase": "send_started",
        },
    ]
    (tmp_path / "manual_write_wal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    server = build_server(tmp_path, monkeypatch)
    mixer = PhysicalMixer()
    mixer.state["/ch/1/fdr"] = -9.0
    server.mixer_client = mixer

    result = await server.reconcile_console_write_recovery(
        transaction_id=transaction_id,
        approved=True,
        approval_id="operator-physical-review",
    )

    assert result["status"] == "reconciled"
    assert result["reason"] == "reconciled_committed"
    assert server._manual_write_recovery_transactions == {}
    assert server._manual_write_emergency_stopped is True
    reset = server.reset_manual_write_emergency_stop(
        approved=True,
        approval_id="operator-reset-after-reconcile",
    )
    assert reset["status"] == "reset"
    restarted = build_server(tmp_path, monkeypatch)
    assert restarted._manual_write_emergency_stopped is False


@pytest.mark.parametrize(
    ("query_result", "case_id"),
    [
        (
            {
                "status": "received",
                "address": "/ch/48/fdr",
                "value": -9.0,
                "confirmed_round_trips": 2,
            },
            "wrong_address",
        ),
        (
            {
                "status": "received",
                "value": -9.0,
                "confirmed_round_trips": 2,
            },
            "missing_address",
        ),
        (
            {
                "status": "received",
                "address": "/ch/1/fdr",
                "value": -9.0,
                "confirmed_round_trips": 2.9,
            },
            "fractional_round_trips",
        ),
        (
            {
                "status": "received",
                "address": "/ch/1/fdr",
                "value": -9.0,
                "confirmed_round_trips": "2",
            },
            "string_round_trips",
        ),
        (
            {
                "status": "received",
                "address": "/ch/1/fdr",
                "value": -9.0,
                "confirmed_round_trips": True,
            },
            "boolean_round_trips",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.asyncio
async def test_reconciliation_requires_address_bound_integer_round_trip_proof(
    tmp_path,
    monkeypatch,
    query_result,
    case_id,
):
    transaction_id = f"reconcile-unverified-{case_id}"
    rows = [
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:00Z",
            "transaction_id": transaction_id,
            "phase": "prepared",
            "scope": "manual_write",
            "address": "/ch/1/fdr",
            "value_type": "fader",
            "old_value": -8.0,
            "effective_value": -9.0,
        },
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:01Z",
            "transaction_id": transaction_id,
            "phase": "send_started",
        },
    ]
    wal_path = tmp_path / "manual_write_wal.jsonl"
    wal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    server = build_server(tmp_path, monkeypatch)
    mixer = PhysicalMixer()
    mixer.query_parameter = MagicMock(return_value=query_result)
    server.mixer_client = mixer

    result = await server.reconcile_console_write_recovery(
        transaction_id=transaction_id,
        approved=True,
        approval_id="operator-physical-review",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "physical_recovery_read_unverified"
    assert result["address"] == "/ch/1/fdr"
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True
    assert "reconciled_committed" not in wal_phases(wal_path)


@pytest.mark.asyncio
async def test_cancelled_reconciliation_drains_query_before_releasing_writer_scope(
    tmp_path,
    monkeypatch,
):
    transaction_id = "reconcile-cancelled-query"
    rows = [
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:00Z",
            "transaction_id": transaction_id,
            "phase": "prepared",
            "scope": "manual_write",
            "address": "/ch/1/fdr",
            "value_type": "fader",
            "old_value": -8.0,
            "effective_value": -9.0,
        },
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:01Z",
            "transaction_id": transaction_id,
            "phase": "send_started",
        },
    ]
    wal_path = tmp_path / "manual_write_wal.jsonl"
    wal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    server = build_server(tmp_path, monkeypatch)
    query_started = threading.Event()
    release_query = threading.Event()
    mixer = PhysicalMixer()

    def blocking_query(address, timeout=0.5):
        query_started.set()
        if not release_query.wait(timeout=2.0):
            raise TimeoutError("test query release timed out")
        return {
            "status": "received",
            "address": address,
            "value": -9.0,
            "confirmed_round_trips": 2,
        }

    mixer.query_parameter = blocking_query
    server.mixer_client = mixer
    reconcile_task = asyncio.create_task(
        server.reconcile_console_write_recovery(
            transaction_id=transaction_id,
            approved=True,
            approval_id="operator-physical-review",
        )
    )
    contender_acquired = asyncio.Event()

    async def contend_for_writer_scope():
        async with server._console_write_scope():
            contender_acquired.set()

    try:
        assert await asyncio.to_thread(query_started.wait, 1.0)
        reconcile_task.cancel()
        await asyncio.sleep(0)
        assert reconcile_task.done() is False

        contender = asyncio.create_task(contend_for_writer_scope())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(contender_acquired.wait(), timeout=0.05)
        assert contender.done() is False

        release_query.set()
        with pytest.raises(asyncio.CancelledError):
            await reconcile_task
        await asyncio.wait_for(contender_acquired.wait(), timeout=1.0)
        await contender
    finally:
        release_query.set()
        if not reconcile_task.done():
            reconcile_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reconcile_task

    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True
    assert "reconciled_committed" not in wal_phases(wal_path)


@pytest.mark.asyncio
async def test_reconciliation_refuses_console_value_outside_target_and_preimage(
    tmp_path,
    monkeypatch,
):
    transaction_id = "reconcile-mismatch"
    rows = [
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:00Z",
            "transaction_id": transaction_id,
            "phase": "prepared",
            "scope": "manual_write",
            "address": "/ch/1/fdr",
            "value_type": "fader",
            "old_value": -8.0,
            "effective_value": -9.0,
        },
        {
            "schema": "automixer.manual_write_wal.v1",
            "timestamp": "2026-07-11T20:00:01Z",
            "transaction_id": transaction_id,
            "phase": "manual_review_required",
        },
    ]
    (tmp_path / "manual_write_wal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    server = build_server(tmp_path, monkeypatch)
    mixer = PhysicalMixer()
    mixer.state["/ch/1/fdr"] = -7.0
    server.mixer_client = mixer

    result = await server.reconcile_console_write_recovery(
        transaction_id=transaction_id,
        approved=True,
        approval_id="operator-physical-review",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "recovery_state_mixed_or_changed"
    assert transaction_id in server._manual_write_recovery_transactions
    assert server._manual_write_emergency_stopped is True


@pytest.mark.asyncio
async def test_restore_only_proof_target_cannot_reconcile_as_committed(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "proof-target-runtime"
    transaction_id = "proof-target-retained"
    server = build_server(
        tmp_path / "proof-target-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    assert server._append_manual_write_wal(
        "prepared",
        transaction_id,
        scope="supervised_transaction",
        proof_run=True,
        restore_after_verify=True,
        capability_kind="restore_only_proof",
        proposal_id="proof-target-proposal",
        idempotency_key="proof-target-idempotency",
        operations=[
            {
                "operation_id": "proof-target-operation",
                "family_id": "fader",
                "write_kind": "fader",
                "writes": [
                    {
                        "address": "/ch/1/fdr",
                        "target_value": -9.0,
                        "tolerance": 0.05,
                    }
                ],
                "preimage_values": {"/ch/1/fdr": -8.0},
            }
        ],
    )
    assert server._append_manual_write_wal(
        "manual_review_required",
        transaction_id,
        result_status="manual_review_required",
        safe_target_retention_attempted=True,
        contained_operation_ids=["proof-target-operation"],
        contained_target_physical_verified=True,
        physical_state_restored=False,
        manual_review_required=True,
    )

    restarted = build_server(
        tmp_path / "proof-target-restart-events",
        monkeypatch,
        runtime_dir=runtime_dir,
    )
    mixer = PhysicalMixer()
    mixer.state["/ch/1/fdr"] = -9.0
    restarted.mixer_client = mixer

    result = await restarted.reconcile_console_write_recovery(
        transaction_id=transaction_id,
        approved=True,
        approval_id="operator-proof-recovery-review",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == (
        "restore_only_proof_target_cannot_be_reconciled_committed"
    )
    assert transaction_id in restarted._manual_write_recovery_transactions
    assert restarted._manual_write_emergency_stopped is True
    assert "reconciled_committed" not in wal_phases(
        runtime_dir / "manual_write_wal.jsonl"
    )


@pytest.mark.asyncio
async def test_verified_manual_write_commits_durably_and_restart_is_clean(
    tmp_path,
    monkeypatch,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)

    result = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:wal_commit",
        approved=True,
        approval_id="wal-commit-approval",
        idempotency_key="wal-commit-idempotency",
        proposal_id="wal-commit-proposal",
    )

    assert result["status"] == "applied"
    assert mixer.sent == [(1, -9.0)]
    wal_path = tmp_path / "manual_write_wal.jsonl"
    assert wal_phases(wal_path) == [
        "prepared",
        "send_started",
        "send_completed",
        "committed",
    ]
    restarted = build_server(tmp_path, monkeypatch)
    assert restarted._manual_write_emergency_stopped is False
    assert restarted._manual_write_recovery_transactions == {}
    assert "wal-commit-proposal" in restarted._durable_operator_proposal_ids
    duplicate = restarted.reserve_operator_apply_idempotency_key(
        "wal-commit-idempotency",
        proposal_id="replayed-proposal",
        approval_id="replayed-approval",
    )
    assert duplicate["reserved"] is False
    assert duplicate["reason"] == "duplicate_idempotency_key"
    assert duplicate["first_seen"]["restored_from_wal"] is True


@pytest.mark.asyncio
async def test_exact_prepared_abort_releases_manual_key_in_process_and_restart(
    tmp_path,
    monkeypatch,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    key = "retry-after-proven-no-send"
    proposal_id = "retry-proposal"
    assert server.reserve_operator_apply_idempotency_key(
        key,
        proposal_id=proposal_id,
        approval_id="first-attempt",
    )["reserved"] is True
    append = server._append_manual_write_wal

    def change_console_after_prepare(phase, transaction_id, **payload):
        path = append(phase, transaction_id, **payload)
        if phase == "prepared":
            mixer.state["/ch/1/fdr"] = -7.0
        return path

    server._append_manual_write_wal = change_console_after_prepare
    result = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:proven-no-send-retry",
        approved=True,
        approval_id="first-attempt",
        idempotency_key=key,
        proposal_id=proposal_id,
        expected_current_value=-8.0,
    )

    assert result["reason"] == "manual_write_cas_target_changed_or_unverified"
    assert mixer.sent == []
    assert wal_phases(tmp_path / "manual_write_wal.jsonl") == [
        "prepared",
        "aborted_before_send",
    ]
    assert key not in server._durable_operator_apply_idempotency_keys
    assert proposal_id not in server._durable_operator_proposal_ids
    assert server.release_operator_apply_idempotency_key_if_no_send(
        key,
        proposal_id=proposal_id,
    ) is True
    assert server.reserve_operator_apply_idempotency_key(
        key,
        proposal_id=proposal_id,
        approval_id="same-process-retry",
    )["reserved"] is True

    restarted = build_server(tmp_path, monkeypatch)
    assert restarted._manual_write_recovery_transactions == {}
    assert restarted._manual_write_emergency_stopped is False
    assert key not in restarted._durable_operator_apply_idempotency_keys
    assert proposal_id not in restarted._durable_operator_proposal_ids
    assert restarted.reserve_operator_apply_idempotency_key(
        key,
        proposal_id=proposal_id,
        approval_id="restart-retry",
    )["reserved"] is True


def test_send_started_then_abort_stays_locked_and_requires_recovery(
    tmp_path,
    monkeypatch,
):
    key = "contradictory-abort-key"
    proposal_id = "contradictory-abort-proposal"
    rows = [
        wal_event(
            "contradictory-abort",
            "prepared",
            scope="operator_proposal",
            idempotency_key=key,
            proposal_id=proposal_id,
        ),
        wal_event("contradictory-abort", "send_started"),
        wal_event("contradictory-abort", "aborted_before_send"),
    ]
    (tmp_path / "manual_write_wal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    server = build_server(tmp_path, monkeypatch)

    assert server._manual_write_emergency_stopped is True
    assert "contradictory-abort" in server._manual_write_recovery_transactions
    assert key in server._durable_operator_apply_idempotency_keys
    assert proposal_id in server._durable_operator_proposal_ids
    assert server.reserve_operator_apply_idempotency_key(
        key,
        proposal_id=proposal_id,
    )["reason"] == "duplicate_idempotency_key"


def test_duplicate_key_unlocks_only_when_every_transaction_proves_no_send(
    tmp_path,
    monkeypatch,
):
    key = "duplicate-lifecycle-key"
    proposal_id = "duplicate-lifecycle-proposal"
    rows = [
        wal_event(
            "safe-transaction",
            "prepared",
            scope="operator_proposal",
            idempotency_key=key,
            proposal_id=proposal_id,
        ),
        wal_event("safe-transaction", "aborted_before_send"),
        wal_event(
            "committed-transaction",
            "prepared",
            scope="operator_proposal",
            idempotency_key=key,
            proposal_id=proposal_id,
        ),
        wal_event("committed-transaction", "send_started"),
        wal_event("committed-transaction", "send_completed"),
        wal_event("committed-transaction", "committed"),
    ]
    (tmp_path / "manual_write_wal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    server = build_server(tmp_path, monkeypatch)

    assert server._manual_write_recovery_transactions == {}
    assert key in server._durable_operator_apply_idempotency_keys
    assert proposal_id in server._durable_operator_proposal_ids


@pytest.mark.parametrize("reverse_order", [False, True])
def test_mixed_scope_same_key_recovery_is_order_independent(
    tmp_path,
    monkeypatch,
    reverse_order,
):
    key = "mixed-scope-key"
    operator_rows = [
        wal_event(
            "operator-safe",
            "prepared",
            scope="operator_proposal",
            idempotency_key=key,
            proposal_id="operator-safe-proposal",
        ),
        wal_event("operator-safe", "aborted_before_send"),
    ]
    transaction_rows = [
        wal_event(
            "transaction-committed",
            "prepared",
            scope="supervised_transaction",
            idempotency_key=key,
        ),
        wal_event("transaction-committed", "committed"),
    ]
    groups = [operator_rows, transaction_rows]
    if reverse_order:
        groups.reverse()
    rows = [row for group in groups for row in group]
    (tmp_path / "manual_write_wal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    server = build_server(tmp_path, monkeypatch)

    assert key not in server._durable_operator_apply_idempotency_keys
    assert "operator-safe-proposal" not in server._durable_operator_proposal_ids
    assert server._supervised_transaction_idempotency_keys[key] == (
        "transaction-committed"
    )
    assert server._supervised_transaction_ids["transaction-committed"] == key


def test_corrupt_tail_never_unlocks_otherwise_safe_no_send_key(
    tmp_path,
    monkeypatch,
):
    key = "safe-before-corrupt-tail"
    rows = [
        wal_event(
            "safe-before-corrupt",
            "prepared",
            scope="operator_proposal",
            idempotency_key=key,
            proposal_id="safe-before-corrupt-proposal",
        ),
        wal_event("safe-before-corrupt", "aborted_before_send"),
    ]
    (tmp_path / "manual_write_wal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + '{"cut_off":',
        encoding="utf-8",
    )

    server = build_server(tmp_path, monkeypatch)

    assert server._manual_write_emergency_stopped is True
    assert key in server._durable_operator_apply_idempotency_keys
    assert "safe-before-corrupt-proposal" in server._durable_operator_proposal_ids


@pytest.mark.asyncio
async def test_two_running_servers_cannot_apply_same_durable_idempotency_key(
    tmp_path,
    monkeypatch,
):
    first = build_server(tmp_path, monkeypatch)
    second = build_server(tmp_path, monkeypatch)
    first_mixer = configure_physical_fader(first)
    second_mixer = configure_physical_fader(second)
    key = "cross-process-stale-reservation"

    assert first.reserve_operator_apply_idempotency_key(
        key,
        proposal_id="proposal-a",
        approval_id="approval-a",
    )["reserved"] is True
    assert second.reserve_operator_apply_idempotency_key(
        key,
        proposal_id="proposal-b",
        approval_id="approval-b",
    )["reserved"] is True

    first_result = await first._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:first-process",
        approved=True,
        approval_id="approval-a",
        idempotency_key=key,
        proposal_id="proposal-a",
    )
    second_result = await second._apply_manual_console_write(
        None,
        channel=1,
        value=-10.0,
        value_type="fader",
        source="unit:second-process",
        approved=True,
        approval_id="approval-b",
        idempotency_key=key,
        proposal_id="proposal-b",
    )

    assert first_result["status"] == "applied"
    assert second_result["status"] == "blocked"
    assert second_result["reason"] == "duplicate_idempotency_key"
    assert first_mixer.sent == [(1, -9.0)]
    assert second_mixer.sent == []
    assert wal_phases(tmp_path / "manual_write_wal.jsonl").count("prepared") == 1


@pytest.mark.asyncio
async def test_apply_rechecks_physical_cas_after_durable_prepare(
    tmp_path,
    monkeypatch,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    append_wal = server._append_manual_write_wal

    def change_console_after_prepare(phase, transaction_id, **payload):
        path = append_wal(phase, transaction_id, **payload)
        if phase == "prepared" and payload.get("expected_current_value") == -8.0:
            mixer.state["/ch/1/fdr"] = -7.0
        return path

    server._append_manual_write_wal = change_console_after_prepare
    result = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:cas-apply",
        approved=True,
        approval_id="cas-apply",
        expected_current_value=-8.0,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "manual_write_cas_target_changed_or_unverified"
    assert result["expected_current_value"] == -8.0
    assert result["current_readback_value"] == -7.0
    assert mixer.state["/ch/1/fdr"] == -7.0
    assert mixer.sent == []
    assert wal_phases(tmp_path / "manual_write_wal.jsonl")[-1] == "aborted_before_send"


@pytest.mark.asyncio
async def test_rollback_rechecks_physical_cas_after_durable_prepare(
    tmp_path,
    monkeypatch,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    applied = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:cas-apply",
        approved=True,
        approval_id="cas-apply",
    )
    assert applied["status"] == "applied"

    append_wal = server._append_manual_write_wal

    def change_console_after_prepare(phase, transaction_id, **payload):
        path = append_wal(phase, transaction_id, **payload)
        if phase == "prepared" and payload.get("expected_current_value") == -9.0:
            mixer.state["/ch/1/fdr"] = -7.0
        return path

    server._append_manual_write_wal = change_console_after_prepare
    rollback = await server._rollback_last_manual_console_write(
        None,
        approved=True,
        approval_id="cas-rollback",
        transaction_id=applied["transaction_id"],
    )

    assert rollback["status"] == "blocked"
    assert rollback["reason"] == "manual_write_cas_target_changed_or_unverified"
    assert rollback["expected_current_value"] == -9.0
    assert rollback["current_readback_value"] == -7.0
    assert mixer.state["/ch/1/fdr"] == -7.0
    assert mixer.sent == [(1, -9.0)]
    assert wal_phases(tmp_path / "manual_write_wal.jsonl")[-1] == "aborted_before_send"


@pytest.mark.asyncio
async def test_wal_prepare_failure_blocks_before_physical_send(tmp_path, monkeypatch):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    server._append_manual_write_wal = MagicMock(return_value=None)

    result = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:wal_prepare_failure",
        approved=True,
        approval_id="wal-prepare-failure",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "manual_write_wal_prepare_failed"
    assert mixer.sent == []
    assert server._manual_write_emergency_stopped is True


@pytest.mark.asyncio
async def test_wal_directory_fsync_failure_blocks_before_physical_send(
    tmp_path,
    monkeypatch,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    server._fsync_directory = MagicMock(side_effect=OSError("directory fsync failed"))

    result = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:wal_directory_fsync_failure",
        approved=True,
        approval_id="wal-directory-fsync-failure",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "manual_write_wal_prepare_failed"
    assert mixer.sent == []
    assert server._manual_write_emergency_stopped is True
    assert wal_phases(tmp_path / "manual_write_wal.jsonl") == ["prepared"]


@pytest.mark.asyncio
async def test_terminal_wal_failure_never_reports_applied(tmp_path, monkeypatch):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    append = server._append_manual_write_wal

    def fail_commit(phase, transaction_id, **payload):
        if phase == "committed":
            return None
        return append(phase, transaction_id, **payload)

    server._append_manual_write_wal = fail_commit

    result = await server._apply_manual_console_write(
        None,
        channel=1,
        value=-9.0,
        value_type="fader",
        source="unit:wal_commit_failure",
        approved=True,
        approval_id="wal-commit-failure",
    )

    assert mixer.sent == [(1, -9.0)]
    assert result["status"] == "sent_unverified"
    assert result["success"] is False
    assert result["reason"] == "manual_write_wal_commit_failed"
    assert server._manual_write_emergency_stopped is True
    assert server._manual_write_recovery_transactions


@pytest.mark.asyncio
async def test_cancellation_after_send_latches_and_survives_restart(
    tmp_path,
    monkeypatch,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    readback_started = asyncio.Event()
    release_readback = asyncio.Event()
    calls = 0

    async def cancellable_read(*, address, **_kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            readback_started.set()
            await release_readback.wait()
        return {
            "value": mixer.state.get(address),
            "status": "verified_from_physical_readback",
            "source": "physical_readback",
            "cache_freshness": {"status": "fresh"},
        }

    server._manual_write_read_value = cancellable_read
    task = asyncio.create_task(
        server._apply_manual_console_write(
            None,
            channel=1,
            value=-9.0,
            value_type="fader",
            source="unit:wal_cancelled",
            approved=True,
            approval_id="wal-cancelled",
        )
    )
    await readback_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mixer.sent == [(1, -9.0)]
    assert server._manual_write_emergency_stopped is True
    assert wal_phases(tmp_path / "manual_write_wal.jsonl")[-1] == (
        "manual_review_required"
    )
    restarted = build_server(tmp_path, monkeypatch)
    assert restarted._manual_write_emergency_stopped is True
    assert restarted._manual_write_recovery_transactions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel", "value"),
    [
        (True, -9.0),
        (1.5, -9.0),
        (0, -9.0),
        (49, -9.0),
        (1, float("nan")),
        (1, float("inf")),
        (1, True),
    ],
)
async def test_invalid_manual_targets_block_before_query_or_send(
    tmp_path,
    monkeypatch,
    channel,
    value,
):
    server = build_server(tmp_path, monkeypatch)
    mixer = configure_physical_fader(server)
    server._manual_write_read_value = MagicMock()

    result = await server._apply_manual_console_write(
        None,
        channel=channel,
        value=value,
        value_type="fader",
        source="unit:invalid_target",
        approved=True,
        approval_id="invalid-target",
    )

    assert result["status"] == "blocked"
    assert result["reason"] in {"invalid_channel", "invalid_value"}
    server._manual_write_read_value.assert_not_called()
    assert mixer.sent == []
