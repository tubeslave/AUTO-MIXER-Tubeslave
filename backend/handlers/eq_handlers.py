"""Auto-EQ and multi-channel Auto-EQ message handlers."""

from auto_eq import InstrumentProfiles


def register_handlers(server):
    async def handle_start_auto_eq(websocket, data):
        await server.start_auto_eq(
            websocket,
            device_id=data.get("device_id"),
            channel=data.get("channel"),
            profile=data.get("profile", "custom"),
            auto_apply=data.get("auto_apply", False),
            monitored_channels=data.get("monitored_channels", [])
        )

    async def handle_stop_auto_eq(websocket, data):
        await server.stop_auto_eq(websocket)

    async def handle_set_eq_profile(websocket, data):
        await server.set_eq_profile(websocket, data.get("profile", "custom"))

    async def handle_apply_eq_correction(websocket, data):
        await server.apply_eq_correction(websocket)

    async def handle_reset_eq(websocket, data):
        await server.reset_eq(websocket, data)

    async def handle_reset_all_eq(websocket, data):
        await server.reset_all_eq(websocket, data)

    async def handle_get_eq_profiles(websocket, data):
        await server.send_to_client(websocket, {
            "type": "eq_profiles",
            "profiles": InstrumentProfiles.get_all_profiles()
        })

    async def handle_get_auto_eq_status(websocket, data):
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "auto_eq_status",
            **server.get_auto_eq_status()
        })

    # Multi-channel Auto-EQ
    async def handle_start_multi_channel_auto_eq(websocket, data):
        await server.start_multi_channel_auto_eq(
            websocket,
            device_id=data.get("device_id"),
            channels_config=data.get("channels_config", []),
            mode=data.get("mode", "soundcheck")
        )

    async def handle_stop_multi_channel_auto_eq(websocket, data):
        await server.stop_multi_channel_auto_eq(websocket)

    async def handle_set_channel_profile(websocket, data):
        await server.set_channel_profile(
            websocket,
            channel=data.get("channel"),
            profile=data.get("profile")
        )

    async def handle_apply_channel_correction(websocket, data):
        await server.apply_channel_correction(
            websocket,
            channel=data.get("channel")
        )

    async def handle_apply_all_corrections(websocket, data):
        await server.apply_all_corrections(websocket)

    return {
        "start_auto_eq": handle_start_auto_eq,
        "stop_auto_eq": handle_stop_auto_eq,
        "set_eq_profile": handle_set_eq_profile,
        "apply_eq_correction": handle_apply_eq_correction,
        "reset_eq": handle_reset_eq,
        "reset_all_eq": handle_reset_all_eq,
        "get_eq_profiles": handle_get_eq_profiles,
        "get_auto_eq_status": handle_get_auto_eq_status,
        "start_multi_channel_auto_eq": handle_start_multi_channel_auto_eq,
        "stop_multi_channel_auto_eq": handle_stop_multi_channel_auto_eq,
        "set_channel_profile": handle_set_channel_profile,
        "apply_channel_correction": handle_apply_channel_correction,
        "apply_all_corrections": handle_apply_all_corrections,
    }
