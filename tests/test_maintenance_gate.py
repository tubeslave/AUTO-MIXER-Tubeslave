"""Safety tests for maintenance-only snapshot and reset gating."""

from maintenance_gate import MaintenanceActionRequest, MaintenanceGate


class FakeMixer:
    def __init__(self, *, connected: bool):
        self.is_connected = connected


def test_maintenance_gate_blocks_real_action_by_default():
    gate = MaintenanceGate(audit_sink=lambda event: None)

    result = gate.evaluate(
        MaintenanceActionRequest(
            source="manual_ui",
            operation="load_snapshot",
            target={"snap_name": "Scene A"},
        ),
        mixer_client=FakeMixer(connected=True),
    )

    assert result.accepted is False
    assert result.blocked_reason == "maintenance_confirmation_required"
    assert result.send_status == "not_sent"


def test_maintenance_gate_allows_explicit_dry_run_preview():
    gate = MaintenanceGate(audit_sink=lambda event: None)

    result = gate.evaluate(
        MaintenanceActionRequest(
            source="manual_ui",
            operation="load_snapshot",
            dry_run=True,
            target={"snap_name": "Scene A"},
        ),
        mixer_client=FakeMixer(connected=False),
    )

    assert result.accepted is True
    assert result.dry_run is True
    assert result.send_attempted is False
    assert result.rollback_plan_required is True


def test_maintenance_gate_requires_rollback_for_console_mutation():
    gate = MaintenanceGate(audit_sink=lambda event: None)

    result = gate.evaluate(
        MaintenanceActionRequest(
            source="manual_ui",
            operation="restore_snapshot",
            confirm_maintenance=True,
            target={"snapshot_path": "/tmp/snapshot.json"},
        ),
        mixer_client=FakeMixer(connected=True),
    )

    assert result.accepted is False
    assert result.blocked_reason == "rollback_plan_required"


def test_maintenance_gate_blocks_transport_when_disconnected():
    gate = MaintenanceGate(audit_sink=lambda event: None)

    result = gate.evaluate(
        MaintenanceActionRequest(
            source="manual_ui",
            operation="save_snapshot",
            confirm_maintenance=True,
        ),
        mixer_client=FakeMixer(connected=False),
    )

    assert result.accepted is False
    assert result.blocked_reason == "transport_disconnected"
    assert result.send_status == "disconnected"


def test_maintenance_gate_blocks_automation_owned_source():
    gate = MaintenanceGate(audit_sink=lambda event: None)

    result = gate.evaluate(
        MaintenanceActionRequest(
            source="auto_soundcheck_cycle",
            operation="reset_all_functions",
            confirm_maintenance=True,
            rollback_snapshot_path="/tmp/rollback.json",
        ),
        mixer_client=FakeMixer(connected=True),
    )

    assert result.accepted is False
    assert result.blocked_reason == "maintenance_source_not_allowed"
