"""Integration tests for snapshot handler maintenance quarantine."""

import pytest

import handlers.snapshot_handlers as snapshot_handlers


class FakeMixer:
    def __init__(self, *, connected: bool = True):
        self.is_connected = connected
        self.load_calls = []
        self.save_calls = []

    def load_snap(self, snap_name):
        self.load_calls.append(snap_name)
        return True

    def save_snap(self, snap_name):
        self.save_calls.append(snap_name)
        return True


class FakeMaintenanceResult:
    def __init__(self, *, accepted, dry_run=False, blocked_reason=None, send_status="not_sent", rollback_plan_required=False, rollback_plan_present=False):
        self.operation = "load_snapshot"
        self.source = "manual_ui"
        self.accepted = accepted
        self.dry_run = dry_run
        self.send_attempted = False
        self.send_result = None
        self.audit_id = "audit-1"
        self.send_status = send_status
        self.blocked_reason = blocked_reason
        self.failure_reason = blocked_reason
        self.confirm_maintenance = False
        self.rollback_plan_required = rollback_plan_required
        self.rollback_plan_present = rollback_plan_present
        self.target = {}
        self.audit_event = {"rollback_snapshot_path": None}


class FakeMaintenanceGate:
    def __init__(self, result):
        self.result = result
        self.requests = []
        self.executed = []

    def evaluate(self, request, *, mixer_client):
        self.requests.append((request, mixer_client))
        self.result.operation = request.operation
        self.result.target = dict(request.target)
        self.result.confirm_maintenance = request.confirm_maintenance
        self.result.audit_event = {"rollback_snapshot_path": request.rollback_snapshot_path}
        return self.result

    def mark_execution(self, result, *, send_result, failure_reason=None):
        self.executed.append((result.operation, send_result, failure_reason))
        result.send_attempted = True
        result.send_result = send_result
        result.send_status = "sent" if send_result else "failed"
        result.failure_reason = failure_reason
        if not send_result:
            result.accepted = False
            result.blocked_reason = "maintenance_execution_failed"
        return result


class FakeServer:
    def __init__(self, maintenance_result, *, mixer=None):
        self.mixer_client = mixer or FakeMixer()
        self.maintenance_gate = FakeMaintenanceGate(maintenance_result)
        self.messages = []
        self.restore_calls = []
        self.create_calls = []

    async def send_to_client(self, websocket, message):
        self.messages.append((websocket, message))

    async def restore_snapshot(self, websocket, snapshot_path=None, **kwargs):
        self.restore_calls.append((websocket, snapshot_path, kwargs))

    async def create_snapshot(self, websocket, channels):
        self.create_calls.append((websocket, channels))


@pytest.mark.asyncio
async def test_load_snap_is_blocked_without_confirmation():
    server = FakeServer(
        FakeMaintenanceResult(
            accepted=False,
            blocked_reason="maintenance_confirmation_required",
            rollback_plan_required=True,
        )
    )
    handlers = snapshot_handlers.register_handlers(server)

    await handlers["load_snap"]("ws", {"snap_name": "Scene A"})

    assert server.mixer_client.load_calls == []
    payload = server.messages[-1][1]
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "maintenance_confirmation_required"
    assert payload["rollback_plan_required"] is True


@pytest.mark.asyncio
async def test_load_snap_executes_only_after_approval_and_rollback():
    server = FakeServer(
        FakeMaintenanceResult(
            accepted=True,
            rollback_plan_required=True,
            rollback_plan_present=True,
        )
    )
    handlers = snapshot_handlers.register_handlers(server)

    await handlers["load_snap"](
        "ws",
        {
            "snap_name": "Scene A",
            "confirm_maintenance": True,
            "rollback_snapshot_path": "/tmp/rollback.json",
        },
    )

    assert server.mixer_client.load_calls == ["Scene A"]
    payload = server.messages[-1][1]
    assert payload["success"] is True
    assert payload["send_status"] == "sent"
    assert payload["rollback_plan_present"] is True


@pytest.mark.asyncio
async def test_restore_snapshot_handler_forwards_maintenance_metadata():
    server = FakeServer(FakeMaintenanceResult(accepted=True))
    handlers = snapshot_handlers.register_handlers(server)

    await handlers["undo_restore_snapshot"](
        "ws",
        {
            "snapshot_path": "/tmp/snapshot.json",
            "confirm_maintenance": True,
            "rollback_snapshot_path": "/tmp/rollback.json",
            "dry_run": True,
        },
    )

    assert server.restore_calls == [
        (
            "ws",
            "/tmp/snapshot.json",
            {
                "dry_run": True,
                "confirm_maintenance": True,
                "rollback_snapshot_path": "/tmp/rollback.json",
                "source": "manual_ui",
            },
        )
    ]
