"""Snapshot/scene message handlers."""

try:
    from maintenance_gate import MaintenanceActionRequest, build_maintenance_operator_payload
except ImportError:  # pragma: no cover - package import fallback
    from backend.maintenance_gate import MaintenanceActionRequest, build_maintenance_operator_payload

from wing_client import WingClient
from osc.enhanced_osc_client import EnhancedOSCClient


def register_handlers(server):
    async def _handle_snap_maintenance(websocket, data, *, operation: str, method_name: str, result_type: str):
        gate = getattr(server, "maintenance_gate", None)
        mixer = getattr(server, "mixer_client", None)
        snap_name = data.get("snap_name")
        if gate is not None:
            result = gate.evaluate(
                MaintenanceActionRequest(
                    source="manual_ui",
                    operation=operation,
                    dry_run=data.get("dry_run"),
                    confirm_maintenance=bool(data.get("confirm_maintenance", False)),
                    rollback_snapshot_path=data.get("rollback_snapshot_path"),
                    target={"snap_name": snap_name},
                    metadata={"handler": f"snapshot:{method_name}"},
                ),
                mixer_client=mixer,
            )
            if not result.accepted or result.dry_run:
                await server.send_to_client(
                    websocket,
                    build_maintenance_operator_payload(
                        result,
                        event_type=result_type,
                        extra={"success": False, "snap_name": snap_name},
                    ),
                )
                return True

            send_result = False
            failure_reason = None
            try:
                method = getattr(mixer, method_name)
                send_result = bool(method(snap_name))
            except Exception as exc:
                failure_reason = str(exc)
            result = gate.mark_execution(
                result,
                send_result=send_result,
                failure_reason=failure_reason,
            )
            await server.send_to_client(
                websocket,
                build_maintenance_operator_payload(
                    result,
                    event_type=result_type,
                    extra={"success": bool(send_result), "snap_name": snap_name},
                ),
            )
            return True

        blocker = getattr(server, "_block_quarantined_wing_write_surface", None)
        if callable(blocker):
            return await blocker(
                websocket,
                surface=method_name,
                result_type=result_type,
                request={"snap_name": snap_name},
            )
        await server.send_to_client(
            websocket,
            {
                "type": result_type,
                "success": False,
                "blocked": True,
                "blocked_reason": "maintenance_gate_unavailable",
                "snap_name": snap_name,
            },
        )
        return True

    async def handle_load_snap(websocket, data):
        if await _handle_snap_maintenance(
            websocket,
            data,
            operation="load_snapshot",
            method_name="load_snap",
            result_type="load_snap_result",
        ):
            return
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            snap_name = data.get("snap_name")
            if snap_name:
                success = server.mixer_client.load_snap(snap_name)
                await server.send_to_client(websocket, {
                    "type": "load_snap_result",
                    "success": success,
                    "snap_name": snap_name
                })

    async def handle_save_snap(websocket, data):
        if await _handle_snap_maintenance(
            websocket,
            data,
            operation="save_snapshot",
            method_name="save_snap",
            result_type="save_snap_result",
        ):
            return
        if server.mixer_client and isinstance(server.mixer_client, (WingClient, EnhancedOSCClient)):
            snap_name = data.get("snap_name")
            if snap_name:
                success = server.mixer_client.save_snap(snap_name)
                await server.send_to_client(websocket, {
                    "type": "save_snap_result",
                    "success": success,
                    "snap_name": snap_name
                })

    async def handle_create_snapshot(websocket, data):
        await server.create_snapshot(websocket, data.get("channels"))

    async def handle_restore_snapshot(websocket, data):
        await server.restore_snapshot(
            websocket,
            data.get("snapshot_path"),
            dry_run=data.get("dry_run"),
            confirm_maintenance=bool(data.get("confirm_maintenance", False)),
            rollback_snapshot_path=data.get("rollback_snapshot_path"),
            source=data.get("source") or "manual_ui",
        )

    return {
        "load_snap": handle_load_snap,
        "save_snap": handle_save_snap,
        "create_snapshot": handle_create_snapshot,
        "undo_snapshot_create": handle_create_snapshot,
        "undo_restore_snapshot": handle_restore_snapshot,
    }
