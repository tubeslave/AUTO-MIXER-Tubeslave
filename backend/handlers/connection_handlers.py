"""Connection (Wing / Mixing Station) message handlers."""


def register_handlers(server):
    async def handle_connect_wing(websocket, data):
        await server.connect_wing(data.get("ip"), data.get("send_port"), data.get("receive_port"))

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
        "connect_mixing_station": handle_connect_mixing_station,
        "discover_mixing_station": handle_discover_mixing_station,
        "disconnect_wing": handle_disconnect,
        "disconnect_mixer": handle_disconnect,
    }
