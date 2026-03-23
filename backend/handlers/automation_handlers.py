"""Automation control (live mode, freeze, emergency stop) message handlers."""


def register_handlers(server):
    async def handle_set_live_mode(websocket, data):
        server.live_mode = bool(data.get("enabled", False))
        if server.gain_staging:
            server.gain_staging.apply_preset(
                "live" if server.live_mode else "soundcheck"
            )
        await server.send_to_client(
            websocket, {"type": "live_mode_set", "live_mode": server.live_mode}
        )

    async def handle_emergency_stop(websocket, data):
        if server.gain_staging:
            server.gain_staging.automation_frozen = True
        if server.auto_fader_controller:
            server.auto_fader_controller.set_automation_frozen(True)
        await server.send_to_client(websocket, {"type": "emergency_stop_applied"})

    async def handle_set_automation_frozen(websocket, data):
        frozen = bool(data.get("frozen", False))
        if server.auto_fader_controller:
            server.auto_fader_controller.set_automation_frozen(frozen)
        if server.gain_staging:
            server.gain_staging.automation_frozen = frozen
        await server.send_to_client(
            websocket, {"type": "automation_frozen_set", "frozen": frozen}
        )

    async def handle_set_channel_frozen(websocket, data):
        ch = data.get("channel")
        seconds = float(data.get("seconds", 10))
        if ch is not None and server.auto_fader_controller:
            server.auto_fader_controller.set_channel_frozen(int(ch), seconds)
        await server.send_to_client(
            websocket, {"type": "channel_frozen_set", "channel": ch, "seconds": seconds}
        )

    async def handle_get_freeze_status(websocket, data):
        status = {"live_mode": server.live_mode}
        if server.auto_fader_controller:
            status["auto_fader"] = server.auto_fader_controller.get_freeze_status()
        if server.gain_staging:
            status["gain_staging_frozen"] = getattr(
                server.gain_staging, "automation_frozen", False
            )
        await server.send_to_client(websocket, {"type": "freeze_status", **status})

    async def handle_get_ready_for_live(websocket, data):
        await server.send_to_client(
            websocket, {"type": "ready_for_live", **server.get_ready_for_live_status()}
        )

    return {
        "set_live_mode": handle_set_live_mode,
        "emergency_stop": handle_emergency_stop,
        "set_automation_frozen": handle_set_automation_frozen,
        "set_channel_frozen": handle_set_channel_frozen,
        "get_freeze_status": handle_get_freeze_status,
        "get_ready_for_live": handle_get_ready_for_live,
    }
