"""Snapshot/scene message handlers."""

from wing_client import WingClient
from osc.enhanced_osc_client import EnhancedOSCClient


def register_handlers(server):
    async def handle_load_snap(websocket, data):
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
        await server.restore_snapshot(websocket, data.get("snapshot_path"))

    return {
        "load_snap": handle_load_snap,
        "save_snap": handle_save_snap,
        "create_snapshot": handle_create_snapshot,
        "undo_snapshot_create": handle_create_snapshot,
        "undo_restore_snapshot": handle_restore_snapshot,
    }
