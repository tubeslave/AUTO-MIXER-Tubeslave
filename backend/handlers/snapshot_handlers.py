"""Snapshot/scene message handlers."""

from maintenance_gate import (
    MaintenanceActionRequest,
    build_maintenance_operator_payload,
)


def register_handlers(server):
    async def handle_load_snap(websocket, data):
        snap_name = data.get("snap_name")
        request = MaintenanceActionRequest(
            source="manual_ui",
            operation="load_snapshot",
            dry_run=data.get("dry_run"),
            confirm_maintenance=bool(data.get("confirm_maintenance", False)),
            rollback_snapshot_path=data.get("rollback_snapshot_path"),
            target={"snap_name": snap_name},
        )
        result = server.maintenance_gate.evaluate(request, mixer_client=server.mixer_client)
        if result.accepted and not result.dry_run:
            success = bool(
                server.mixer_client
                and snap_name
                and hasattr(server.mixer_client, "load_snap")
                and server.mixer_client.load_snap(snap_name)
            )
            server.maintenance_gate.mark_execution(
                result,
                send_result=success,
                failure_reason=None if success else "maintenance_execution_failed",
            )
        await server.send_to_client(
            websocket,
            build_maintenance_operator_payload(
                result,
                event_type="load_snap_result",
                extra={"success": bool(result.send_result), "snap_name": snap_name},
            ),
        )

    async def handle_save_snap(websocket, data):
        snap_name = data.get("snap_name")
        request = MaintenanceActionRequest(
            source="manual_ui",
            operation="save_snapshot",
            dry_run=data.get("dry_run"),
            confirm_maintenance=bool(data.get("confirm_maintenance", False)),
            rollback_snapshot_path=data.get("rollback_snapshot_path"),
            target={"snap_name": snap_name},
        )
        result = server.maintenance_gate.evaluate(request, mixer_client=server.mixer_client)
        if result.accepted and not result.dry_run:
            success = bool(
                server.mixer_client
                and snap_name
                and hasattr(server.mixer_client, "save_snap")
                and server.mixer_client.save_snap(snap_name)
            )
            server.maintenance_gate.mark_execution(
                result,
                send_result=success,
                failure_reason=None if success else "maintenance_execution_failed",
            )
        await server.send_to_client(
            websocket,
            build_maintenance_operator_payload(
                result,
                event_type="save_snap_result",
                extra={"success": bool(result.send_result), "snap_name": snap_name},
            ),
        )

    async def handle_create_snapshot(websocket, data):
        await server.create_snapshot(websocket, data.get("channels"))

    async def handle_restore_snapshot(websocket, data):
        await server.restore_snapshot(
            websocket,
            data.get("snapshot_path"),
            dry_run=data.get("dry_run"),
            confirm_maintenance=bool(data.get("confirm_maintenance", False)),
            rollback_snapshot_path=data.get("rollback_snapshot_path"),
            source="manual_ui",
        )

    return {
        "load_snap": handle_load_snap,
        "save_snap": handle_save_snap,
        "create_snapshot": handle_create_snapshot,
        "undo_snapshot_create": handle_create_snapshot,
        "undo_restore_snapshot": handle_restore_snapshot,
    }
