"""Audio device message handlers."""

from audio_devices import get_audio_devices
from dante_routing_config import get_routing_as_dict, get_module_signal_info


def register_handlers(server):
    async def handle_get_audio_devices(websocket, data):
        devices = get_audio_devices()
        await server.send_to_client(websocket, {
            "type": "audio_devices",
            "devices": devices
        })

    async def handle_get_dante_routing(websocket, data):
        total_ch = data.get("total_channels", 64)
        await server.send_to_client(websocket, {
            "type": "dante_routing",
            "routing_scheme": get_routing_as_dict(total_ch),
            "module_signal_info": get_module_signal_info(),
        })

    return {
        "get_audio_devices": handle_get_audio_devices,
        "get_dante_routing": handle_get_dante_routing,
    }
