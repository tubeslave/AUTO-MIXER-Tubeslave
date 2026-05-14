"""Shared maintenance-mode guard for legacy CLI safety scripts."""

from __future__ import annotations

from typing import Any


def extract_maintenance_cli_args(argv: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Split maintenance approval flags from positional utility arguments."""
    remaining_args: list[str] = []
    maintenance_ctx = {
        "confirm_maintenance": False,
        "rollback_snapshot_path": None,
        "dry_run": False,
        "source": "maintenance_cli",
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--confirm-maintenance":
            maintenance_ctx["confirm_maintenance"] = True
            i += 1
            continue
        if arg == "--rollback-snapshot":
            if i + 1 >= len(argv):
                raise ValueError("--rollback-snapshot requires a snapshot path")
            maintenance_ctx["rollback_snapshot_path"] = argv[i + 1]
            i += 2
            continue
        if arg == "--dry-run":
            maintenance_ctx["dry_run"] = True
            i += 1
            continue
        if arg == "--maintenance-source":
            if i + 1 >= len(argv):
                raise ValueError("--maintenance-source requires a value")
            maintenance_ctx["source"] = argv[i + 1]
            i += 2
            continue
        if arg in {"-y", "--yes"}:
            maintenance_ctx["confirm_maintenance"] = True
            i += 1
            continue
        if arg in {"-h", "--help"}:
            raise ValueError("--help")

        remaining_args.append(arg)
        i += 1

    return remaining_args, maintenance_ctx


def validate_maintenance_cli_request(
    *,
    operation: str,
    confirm_maintenance: bool,
    rollback_snapshot_path: str | None,
    dry_run: bool,
) -> tuple[bool, str | None]:
    """Validate explicit approval and rollback requirements for CLI maintenance action."""
    if dry_run:
        return True, None

    if not confirm_maintenance:
        return False, "maintenance_confirmation_required"

    if not rollback_snapshot_path:
        return False, "rollback_plan_required"

    _ = operation
    return True, None


def format_block_message(reason: str) -> str:
    if reason == "maintenance_confirmation_required":
        return "Maintenance action blocked: use --confirm-maintenance before execution."
    if reason == "rollback_plan_required":
        return "Maintenance action blocked: provide --rollback-snapshot PATH before execution."
    return "Maintenance action blocked."

