"""Feedback detection message handlers."""


def register_handlers(server):
    async def handle_start_feedback_detection(websocket, data):
        device_id = data.get("device_id")
        channels = data.get("channels", [1])
        channel_mapping = data.get("channel_mapping") or {int(c): int(c) for c in channels}
        await server.start_feedback_detection(
            websocket, device_id=device_id, channels=channels, channel_mapping=channel_mapping
        )

    async def handle_stop_feedback_detection(websocket, data):
        await server.stop_feedback_detection(websocket)

    async def handle_get_feedback_status(websocket, data):
        status = server.get_feedback_detector_status()
        await server.send_to_client(websocket, {"type": "feedback_detector_status", **status})

    return {
        "start_feedback_detection": handle_start_feedback_detection,
        "stop_feedback_detection": handle_stop_feedback_detection,
        "get_feedback_status": handle_get_feedback_status,
    }
