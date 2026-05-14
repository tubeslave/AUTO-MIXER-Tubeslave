from backend.maintenance_cli_guard import (
    extract_maintenance_cli_args,
    validate_maintenance_cli_request,
    format_block_message,
)


def test_extract_maintenance_cli_args_removes_flags():
    args, ctx = extract_maintenance_cli_args(
        [
            "192.168.1.102",
            "SNAP_01",
            "200",
            "--confirm-maintenance",
            "--rollback-snapshot",
            "/backups/snap.snap",
            "--dry-run",
        ]
    )

    assert args == ["192.168.1.102", "SNAP_01", "200"]
    assert ctx["confirm_maintenance"] is True
    assert ctx["rollback_snapshot_path"] == "/backups/snap.snap"
    assert ctx["dry_run"] is True


def test_validate_requires_confirmation():
    ok, reason = validate_maintenance_cli_request(
        operation="load_snapshot",
        confirm_maintenance=False,
        rollback_snapshot_path="/tmp/snap.snap",
        dry_run=False,
    )
    assert ok is False
    assert reason == "maintenance_confirmation_required"
    assert format_block_message(reason) == "Maintenance action blocked: use --confirm-maintenance before execution."


def test_validate_requires_rollback_plan():
    ok, reason = validate_maintenance_cli_request(
        operation="load_snapshot",
        confirm_maintenance=True,
        rollback_snapshot_path=None,
        dry_run=False,
    )
    assert ok is False
    assert reason == "rollback_plan_required"


def test_validate_allows_dry_run_without_rollback():
    ok, reason = validate_maintenance_cli_request(
        operation="reset_all_functions",
        confirm_maintenance=False,
        rollback_snapshot_path=None,
        dry_run=True,
    )
    assert ok is True
    assert reason is None

