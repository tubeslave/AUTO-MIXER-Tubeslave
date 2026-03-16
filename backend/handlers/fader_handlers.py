"""Auto fader message handlers."""


def register_handlers(server):
    async def handle_start_realtime_fader(websocket, data):
        await server.start_realtime_fader(
            websocket,
            data.get("device_id"),
            data.get("channels", []),
            data.get("channel_settings", {}),
            data.get("channel_mapping", {}),
            data.get("settings", {})
        )

    async def handle_stop_realtime_fader(websocket, data):
        await server.stop_realtime_fader(websocket)

    async def handle_start_auto_balance(websocket, data):
        await server.start_auto_balance(
            websocket,
            data.get("device_id"),
            data.get("channels", []),
            data.get("channel_settings", {}),
            data.get("channel_mapping", {}),
            data.get("duration", 15),
            data.get("bleed_threshold", -50)
        )

    async def handle_apply_auto_balance(websocket, data):
        await server.apply_auto_balance(websocket)

    async def handle_cancel_auto_balance(websocket, data):
        await server.cancel_auto_balance(websocket)

    async def handle_lock_channel(websocket, data):
        await server.lock_auto_balance_channel(websocket, data.get("channel"))

    async def handle_unlock_channel(websocket, data):
        await server.unlock_auto_balance_channel(websocket, data.get("channel"))

    async def handle_set_auto_fader_profile(websocket, data):
        await server.set_auto_fader_profile(websocket, data.get("profile", "custom"))

    async def handle_update_auto_fader_settings(websocket, data):
        await server.update_auto_fader_settings(websocket, data.get("settings", {}))

    async def handle_save_auto_fader_defaults(websocket, data):
        await server.save_auto_fader_defaults(websocket, data.get("settings", {}))

    async def handle_load_auto_fader_defaults(websocket, data):
        await server.load_auto_fader_defaults(websocket)

    async def handle_save_all_settings(websocket, data):
        await server.save_all_settings(websocket, data.get("settings", {}))

    async def handle_load_all_settings(websocket, data):
        await server.load_all_settings(websocket)

    async def handle_get_auto_fader_status(websocket, data):
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "auto_fader_status",
            **server.get_auto_fader_status()
        })

    return {
        "start_realtime_fader": handle_start_realtime_fader,
        "stop_realtime_fader": handle_stop_realtime_fader,
        "start_auto_balance": handle_start_auto_balance,
        "apply_auto_balance": handle_apply_auto_balance,
        "cancel_auto_balance": handle_cancel_auto_balance,
        "lock_channel": handle_lock_channel,
        "unlock_channel": handle_unlock_channel,
        "set_auto_fader_profile": handle_set_auto_fader_profile,
        "update_auto_fader_settings": handle_update_auto_fader_settings,
        "save_auto_fader_defaults": handle_save_auto_fader_defaults,
        "load_auto_fader_defaults": handle_load_auto_fader_defaults,
        "save_all_settings": handle_save_all_settings,
        "load_all_settings": handle_load_all_settings,
        "get_auto_fader_status": handle_get_auto_fader_status,
    }
