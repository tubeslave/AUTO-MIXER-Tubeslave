"""Auto compressor message handlers."""

import logging

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_start_auto_compressor(websocket, data):
        await server.start_auto_compressor(
            websocket,
            data.get("device_id"),
            data.get("channels", []),
            data.get("channel_mapping", {}),
            data.get("channel_names", {})
        )

    async def handle_stop_auto_compressor(websocket, data):
        await server.stop_auto_compressor(websocket)

    async def handle_get_auto_compressor_status(websocket, data):
        await server.get_auto_compressor_status(websocket, data.get("request_id"))

    async def handle_get_auto_gate_status(websocket, data):
        await server.get_auto_gate_status(websocket, data.get("request_id"))

    async def handle_get_auto_panner_status(websocket, data):
        await server.get_auto_panner_status(websocket, data.get("request_id"))

    async def handle_get_auto_reverb_status(websocket, data):
        await server.get_auto_reverb_status(websocket, data.get("request_id"))

    async def handle_get_auto_effects_status(websocket, data):
        await server.get_auto_effects_status(websocket, data.get("request_id"))

    async def handle_get_cross_adaptive_eq_status(websocket, data):
        await server.get_cross_adaptive_eq_status(websocket, data.get("request_id"))

    async def handle_start_auto_compressor_soundcheck(websocket, data):
        await server.start_auto_compressor_soundcheck(
            websocket,
            genre=data.get("genre", "unknown"),
            style=data.get("style", "live"),
            genre_factor=data.get("genre_factor", 1.0),
            mix_density_factor=data.get("mix_density_factor", 1.0),
            bpm=data.get("bpm"),
        )

    async def handle_stop_auto_compressor_soundcheck(websocket, data):
        await server.stop_auto_compressor_soundcheck(websocket)

    async def handle_start_auto_compressor_live(websocket, data):
        await server.start_auto_compressor_live(websocket, data.get("auto_correct", True))

    async def handle_stop_auto_compressor_live(websocket, data):
        await server.stop_auto_compressor_live(websocket)

    async def handle_set_auto_compressor_profile(websocket, data):
        await server.set_auto_compressor_profile(websocket, data.get("channel"), data.get("profile", "base"))

    async def handle_set_auto_compressor_manual(websocket, data):
        await server.set_auto_compressor_manual(websocket, data.get("channel"), data.get("params", {}))

    return {
        "start_auto_compressor": handle_start_auto_compressor,
        "stop_auto_compressor": handle_stop_auto_compressor,
        "get_auto_compressor_status": handle_get_auto_compressor_status,
        "get_auto_gate_status": handle_get_auto_gate_status,
        "get_auto_panner_status": handle_get_auto_panner_status,
        "get_auto_reverb_status": handle_get_auto_reverb_status,
        "get_auto_effects_status": handle_get_auto_effects_status,
        "get_cross_adaptive_eq_status": handle_get_cross_adaptive_eq_status,
        "start_auto_compressor_soundcheck": handle_start_auto_compressor_soundcheck,
        "stop_auto_compressor_soundcheck": handle_stop_auto_compressor_soundcheck,
        "start_auto_compressor_live": handle_start_auto_compressor_live,
        "stop_auto_compressor_live": handle_stop_auto_compressor_live,
        "set_auto_compressor_profile": handle_set_auto_compressor_profile,
        "set_auto_compressor_manual": handle_set_auto_compressor_manual,
    }
