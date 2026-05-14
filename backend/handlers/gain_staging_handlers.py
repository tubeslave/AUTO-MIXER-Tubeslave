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
            learning_duration_sec=data.get("learning_duration_sec"),
            dry_run=bool(data.get("dry_run", True)),
            confirm_live_apply=bool(data.get("confirm_live_apply", False)),
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

    async def handle_get_gain_staging_recommendations(websocket, data):
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "gain_staging_recommendations",
            **server.get_gain_staging_recommendations(),
        })

    async def handle_update_safe_gain_settings(websocket, data):
        settings = data.get("settings", {})
        await server.update_safe_gain_settings(settings)

    async def handle_start_live_input_trim(websocket, data):
        logger.info("RECEIVED start_live_input_trim MESSAGE")
        await server.start_live_input_trim(
            websocket,
            channels=data.get("channels", []),
            channel_mapping=data.get("channel_mapping", {}),
            channel_names=data.get("channel_names", {}),
            settings=data.get("settings", {}),
            device_id=data.get("device_id"),
        )

    async def handle_stop_live_input_trim(websocket, data):
        logger.info("RECEIVED stop_live_input_trim MESSAGE")
        await server.stop_live_input_trim(websocket)

    async def handle_get_live_input_trim_status(websocket, data):
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "live_input_trim_status",
            "status_type": "status",
            "live_input_trim_state": server.get_live_input_trim_status(),
        })

    async def handle_get_live_input_trim_recommendations(websocket, data):
        await server.send_to_client(websocket, {
            "request_id": data.get("request_id"),
            "type": "live_input_trim_recommendations",
            **server.get_live_input_trim_recommendations(),
        })

    async def handle_run_live_input_trim_tick(websocket, data):
        await server.run_live_input_trim_tick(websocket, data)

    async def handle_set_live_input_trim_live_apply(websocket, data):
        await server.set_live_input_trim_live_apply(
            websocket,
            enabled=bool(data.get("live_apply_enabled", False)),
            analysis_only_mode=data.get("analysis_only_mode"),
            confirm_live_apply=data.get("confirm_live_apply"),
        )

    return {
        "start_realtime_correction": handle_start_realtime_correction,
        "stop_realtime_correction": handle_stop_realtime_correction,
        "get_gain_staging_status": handle_get_gain_staging_status,
        "get_gain_staging_recommendations": handle_get_gain_staging_recommendations,
        "update_safe_gain_settings": handle_update_safe_gain_settings,
        "start_live_input_trim": handle_start_live_input_trim,
        "stop_live_input_trim": handle_stop_live_input_trim,
        "get_live_input_trim_status": handle_get_live_input_trim_status,
        "get_live_input_trim_recommendations": handle_get_live_input_trim_recommendations,
        "run_live_input_trim_tick": handle_run_live_input_trim_tick,
        "set_live_input_trim_live_apply": handle_set_live_input_trim_live_apply,
    }
