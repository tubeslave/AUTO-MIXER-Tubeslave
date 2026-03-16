"""Channel scanning / recognition / bypass message handlers."""

import logging

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_scan_channel_names(websocket, data):
        await server.scan_and_recognize_channels(websocket, data.get("channels", []))

    async def handle_scan_mixer_channel_names(websocket, data):
        logger.info("=" * 60)
        logger.info("RECEIVED scan_mixer_channel_names MESSAGE")
        logger.info(f"Data: {data}")
        logger.info("=" * 60)
        await server.scan_mixer_channel_names(websocket)

    async def handle_reset_trim(websocket, data):
        await server.reset_trim(websocket, data.get("channels", []))

    async def handle_bypass_mixer(websocket, data):
        await server.bypass_mixer(websocket)

    return {
        "scan_channel_names": handle_scan_channel_names,
        "scan_mixer_channel_names": handle_scan_mixer_channel_names,
        "reset_trim": handle_reset_trim,
        "bypass_mixer": handle_bypass_mixer,
    }
