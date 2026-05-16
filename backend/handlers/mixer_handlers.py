"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""


def register_handlers(server):
    async def handle_set_fader(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if server.mixer_client:
            await server._apply_manual_console_write(
                websocket,
                channel=channel,
                value=value,
                value_type="fader",
                source="websocket:set_fader",
                reason=data.get("reason"),
                approved=bool(data.get("approved", False)),
                approval_id=data.get("approval_id"),
            )

    async def handle_set_gain(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if server.mixer_client:
            await server._apply_manual_console_write(
                websocket,
                channel=channel,
                value=value,
                value_type="gain",
                source="websocket:set_gain",
                reason=data.get("reason"),
                approved=bool(data.get("approved", False)),
                approval_id=data.get("approval_id"),
            )

    async def handle_set_eq(websocket, data):
        channel = data.get("channel")
        band = data.get("band")
        freq = data.get("freq")
        gain = data.get("gain")
        q = data.get("q")
        if server.mixer_client:
            await server._block_console_write(
                websocket,
                source="websocket:set_eq",
                value_type="eq",
                reason="write_kind_not_allowed",
                channel=channel,
                extra={"band": band, "freq": freq, "gain": gain, "q": q},
            )

    async def handle_set_compressor(websocket, data):
        channel = data.get("channel")
        params = data.get("params", {})
        if server.mixer_client:
            await server._block_console_write(
                websocket,
                source="websocket:set_compressor",
                value_type="compressor",
                reason="write_kind_not_allowed",
                channel=channel,
                extra={"params": params},
            )

    async def handle_get_state(websocket, data):
        if server.mixer_client:
            await server.send_to_client(websocket, {
                "type": "state_update",
                "mode": server.connection_mode,
                "state": server.mixer_client.get_state()
            })

    return {
        "set_fader": handle_set_fader,
        "set_gain": handle_set_gain,
        "set_eq": handle_set_eq,
        "set_compressor": handle_set_compressor,
        "get_state": handle_get_state,
    }
