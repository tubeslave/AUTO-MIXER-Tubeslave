"""LUFS gain staging / real-time correction message handlers."""

import logging

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_start_realtime_correction(websocket, data):
        logger.info("=" * 60)
        logger.info("RECEIVED start_realtime_correction MESSAGE")
        logger.info(f"Data: {data}")
        logger.info("=" * 60)
        await server.start_realtime_correction(
            device_id=data.get("device_id"),
            channels=data.get("channels", []),
            channel_settings=data.get("channel_settings", {}),
            channel_mapping=data.get("channel_mapping"),
            mode=data.get("mode", "lufs"),
            learning_duration_sec=data.get("learning_duration_sec")
        )

    async def handle_stop_realtime_correction(websocket, data):
        logger.info("=" * 60)
        logger.info("RECEIVED stop_realtime_correction MESSAGE")
        logger.info(f"Data: {data}")
        logger.info("=" * 60)
        await server.stop_realtime_correction()

    async def handle_get_gain_staging_status(websocket, data):
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "gain_staging_status",
            **server.get_gain_staging_status()
        })

    async def handle_update_safe_gain_settings(websocket, data):
        settings = data.get("settings", {})
        await server.update_safe_gain_settings(settings)

    return {
        "start_realtime_correction": handle_start_realtime_correction,
        "stop_realtime_correction": handle_stop_realtime_correction,
        "get_gain_staging_status": handle_get_gain_staging_status,
        "update_safe_gain_settings": handle_update_safe_gain_settings,
    }
