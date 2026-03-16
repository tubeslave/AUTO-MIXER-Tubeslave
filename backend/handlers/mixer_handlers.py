"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""


def register_handlers(server):
    async def handle_set_fader(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if server.mixer_client:
            server.mixer_client.set_channel_fader(channel, value)

    async def handle_set_gain(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if server.mixer_client:
            server.mixer_client.set_channel_gain(channel, value)

    async def handle_set_eq(websocket, data):
        channel = data.get("channel")
        band = data.get("band")
        freq = data.get("freq")
        gain = data.get("gain")
        q = data.get("q")
        if server.mixer_client:
            server.mixer_client.set_eq_band(channel, band, freq, gain, q)

    async def handle_set_compressor(websocket, data):
        channel = data.get("channel")
        params = data.get("params", {})
        if server.mixer_client:
            server.mixer_client.set_compressor(
                channel,
                params.get("threshold", 0),
                params.get("ratio", 1),
                params.get("attack", 0),
                params.get("release", 0)
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
