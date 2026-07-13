"""Automation control (live mode, freeze, emergency stop) message handlers."""


def register_handlers(server):
    async def handle_set_live_mode(websocket, data):
        server.live_mode = data.get("enabled") is True
        if server.gain_staging:
            server.gain_staging.apply_preset("live" if server.live_mode else "soundcheck")
        await server.send_to_client(websocket, {"type": "live_mode_set", "live_mode": server.live_mode})

    async def handle_emergency_stop(websocket, data):
        result = await server.activate_system_emergency_stop(
            data.get("reason") or "websocket:emergency_stop"
        )
        payload = {
            "type": "emergency_stop_applied",
            **result,
            "manual_write_supervision": server.get_manual_write_supervision_status(),
            "ready_for_live": server.get_ready_for_live_status(),
        }
        artifact_path = server._record_manual_write_event("manual_write_emergency_stop", payload)
        if artifact_path:
            payload["artifact_path"] = artifact_path
        await server.send_to_client(websocket, payload)

    async def handle_get_manual_write_supervision(websocket, data):
        await server.send_to_client(
            websocket,
            {
                "type": "manual_write_supervision_status",
                **server.get_manual_write_supervision_status(),
            },
        )

    async def handle_reset_manual_write_emergency_stop(websocket, data):
        result = server.reset_manual_write_emergency_stop(
            approved=data.get("approved") is True,
            approval_id=data.get("approval_id"),
            reason=data.get("reason") or "operator_requested_emergency_reset",
        )
        await server.send_to_client(websocket, result)

    async def handle_reconcile_console_write_recovery(websocket, data):
        result = await server.reconcile_console_write_recovery(
            transaction_id=data.get("transaction_id"),
            approved=data.get("approved") is True,
            approval_id=data.get("approval_id"),
        )
        await server.send_to_client(websocket, result)

    async def handle_set_automation_frozen(websocket, data):
        frozen = data.get("frozen") is True
        if server.auto_fader_controller:
            server.auto_fader_controller.set_automation_frozen(frozen)
        if server.gain_staging:
            server.gain_staging.automation_frozen = frozen
        if getattr(server, "auto_fader_v2_controller", None):
            server.auto_fader_v2_controller.set_automation_frozen(frozen)
        await server.send_to_client(websocket, {"type": "automation_frozen_set", "frozen": frozen})

    async def handle_set_channel_frozen(websocket, data):
        ch = data.get("channel")
        seconds = float(data.get("seconds", 10))
        if ch is not None and server.auto_fader_controller:
            server.auto_fader_controller.set_channel_frozen(int(ch), seconds)
        if getattr(server, "auto_fader_v2_controller", None) and ch is not None:
            server.auto_fader_v2_controller.set_channel_frozen(int(ch), seconds)
        await server.send_to_client(websocket, {"type": "channel_frozen_set", "channel": ch, "seconds": seconds})

    async def handle_get_freeze_status(websocket, data):
        status = {"live_mode": server.live_mode}
        if server.auto_fader_controller:
            status["auto_fader"] = server.auto_fader_controller.get_freeze_status()
        if getattr(server, "auto_fader_v2_controller", None):
            status["auto_fader_v2"] = server.auto_fader_v2_controller.get_freeze_status()
        if server.gain_staging:
            status["gain_staging_frozen"] = getattr(server.gain_staging, "automation_frozen", False)
        await server.send_to_client(websocket, {"type": "freeze_status", **status})

    async def handle_get_ready_for_live(websocket, data):
        await server.send_to_client(websocket, {"type": "ready_for_live", **server.get_ready_for_live_status()})

    return {
        "set_live_mode": handle_set_live_mode,
        "emergency_stop": handle_emergency_stop,
        "get_manual_write_supervision": handle_get_manual_write_supervision,
        "reset_manual_write_emergency_stop": handle_reset_manual_write_emergency_stop,
        "reconcile_console_write_recovery": handle_reconcile_console_write_recovery,
        "set_automation_frozen": handle_set_automation_frozen,
        "set_channel_frozen": handle_set_channel_frozen,
        "get_freeze_status": handle_get_freeze_status,
        "get_ready_for_live": handle_get_ready_for_live,
    }
