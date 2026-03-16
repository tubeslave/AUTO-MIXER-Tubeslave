"""Voice control message handlers."""

import logging

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_start_voice_control(websocket, data):
        logger.info("=" * 60)
        logger.info("RECEIVED start_voice_control MESSAGE")
        logger.info(f"Data: {data}")
        logger.info("=" * 60)
        language_param = data.get("language", "ru")
        if language_param == "":
            language_param = None
        logger.info(f"Calling start_voice_control with: model_size={data.get('model_size')}, language={language_param}, device_id={data.get('device_id')}, channel={data.get('channel')}")

        await server.send_to_client(websocket, {
            "type": "voice_control_status",
            "active": False,
            "message": "Starting voice control..."
        })
        logger.info("Sent initial acknowledgment to client")

        try:
            await server.start_voice_control(
                model_size=data.get("model_size", "small"),
                language=language_param,
                device_id=data.get("device_id"),
                channel=data.get("channel", 0)
            )
            logger.info("start_voice_control completed")
        except Exception as e:
            logger.error(f"Exception in start_voice_control: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            await server.send_to_client(websocket, {
                "type": "voice_control_status",
                "active": False,
                "error": str(e)
            })
            logger.info("Sent error response to client")

    async def handle_stop_voice_control(websocket, data):
        await server.stop_voice_control()

    async def handle_rescan_channel_names(websocket, data):
        await server.rescan_channel_names()
        await server.send_to_client(websocket, {
            "type": "rescan_complete",
            "message": "Channel names rescanned successfully"
        })

    async def handle_get_voice_control_status(websocket, data):
        await server.send_to_client(websocket, {
            "type": "voice_control_status",
            "active": server.voice_control.is_listening if server.voice_control else False
        })

    return {
        "start_voice_control": handle_start_voice_control,
        "stop_voice_control": handle_stop_voice_control,
        "rescan_channel_names": handle_rescan_channel_names,
        "get_voice_control_status": handle_get_voice_control_status,
    }
