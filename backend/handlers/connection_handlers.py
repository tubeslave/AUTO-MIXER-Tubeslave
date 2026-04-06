"""Connection (Wing / dLive / Mixing Station) message handlers."""


def register_handlers(server):
    async def handle_connect_wing(websocket, data):
        await server.connect_wing(
            data.get("ip"),
            data.get("send_port") or 2223,
            data.get("receive_port") or 2223,
        )

    async def handle_connect_dlive(websocket, data):
        await server.connect_dlive(
            ip=data.get("ip", "192.168.1.70"),
            port=data.get("port", 51328),
            tls=data.get("tls", False),
            midi_base_channel=data.get("midi_base_channel", 0),
        )

    async def handle_connect_ableton(websocket, data):
        await server.connect_ableton(
            ip=data.get("ip", "127.0.0.1"),
            send_port=data.get("send_port", 11000),
            recv_port=data.get("recv_port", 11001),
            channel_offset=data.get("channel_offset", 0),
            utility_device_index=data.get("utility_device_index"),
            eq_eight_device_index=data.get("eq_eight_device_index"),
        )

    async def handle_connect_mixing_station(websocket, data):
        await server.connect_mixing_station(
            data.get("host", "127.0.0.1"),
            data.get("osc_port", 8000),
            data.get("rest_port", 8080)
        )

    async def handle_discover_mixing_station(websocket, data):
        await server.discover_mixing_station()

    async def handle_disconnect(websocket, data):
        await server.disconnect_mixer()

    return {
        "connect_wing": handle_connect_wing,
        "connect_dlive": handle_connect_dlive,
        "connect_ableton": handle_connect_ableton,
        "connect_mixing_station": handle_connect_mixing_station,
        "discover_mixing_station": handle_discover_mixing_station,
        "disconnect_wing": handle_disconnect,
        "disconnect_mixer": handle_disconnect,
    }
