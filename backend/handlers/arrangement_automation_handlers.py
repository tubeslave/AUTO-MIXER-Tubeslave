"""Arrangement-aware level automation WebSocket handlers."""


def register_handlers(server):
    async def handle_start_arrangement_automation(websocket, data):
        await server.start_arrangement_automation(
            websocket,
            channels=data.get("channels", []),
            channel_mapping=data.get("channel_mapping", {}),
            channel_names=data.get("channel_names", {}),
            settings=data.get("settings", {}),
        )

    async def handle_stop_arrangement_automation(websocket, data):
        await server.stop_arrangement_automation(websocket)

    async def handle_get_arrangement_automation_status(websocket, data):
        await server.send_to_client(websocket, {
            "type": "arrangement_automation_status",
            "status_type": "status",
            "active": bool(getattr(server, "arrangement_automation_controller", None)),
            "arrangement_automation_state": server.get_arrangement_automation_status(),
        })

    async def handle_run_arrangement_automation_tick(websocket, data):
        await server.run_arrangement_automation_tick(websocket, data)

    async def handle_set_arrangement_automation_live_apply(websocket, data):
        await server.set_arrangement_automation_live_apply(
            websocket,
            enabled=bool(data.get("live_apply_enabled", False)),
            analysis_only_mode=data.get("analysis_only_mode"),
            confirm_live_apply=data.get("confirm_live_apply"),
        )

    return {
        "start_arrangement_automation": handle_start_arrangement_automation,
        "stop_arrangement_automation": handle_stop_arrangement_automation,
        "get_arrangement_automation_status": handle_get_arrangement_automation_status,
        "run_arrangement_automation_tick": handle_run_arrangement_automation_tick,
        "set_arrangement_automation_live_apply": handle_set_arrangement_automation_live_apply,
    }
