"""Auto soundcheck message handlers."""


def register_handlers(server):
    async def handle_start_auto_soundcheck(websocket, data):
        await server.start_auto_soundcheck(
            websocket,
            data.get("device_id"),
            data.get("channels", []),
            data.get("channel_settings", {}),
            data.get("channel_mapping", {}),
            data.get("timings", {})
        )

    async def handle_stop_auto_soundcheck(websocket, data):
        await server.stop_auto_soundcheck(websocket)

    async def handle_get_auto_soundcheck_status(websocket, data):
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": server.auto_soundcheck_running,
            "current_step": None,
            "step_progress": 0,
            "step_time_remaining": 0,
            "message": "Idle" if not server.auto_soundcheck_running else "Running"
        })

    return {
        "start_auto_soundcheck": handle_start_auto_soundcheck,
        "stop_auto_soundcheck": handle_stop_auto_soundcheck,
        "get_auto_soundcheck_status": handle_get_auto_soundcheck_status,
    }
