from __future__ import annotations

import asyncio
import json
import os
import pwd
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from server import AutoMixerServer


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
