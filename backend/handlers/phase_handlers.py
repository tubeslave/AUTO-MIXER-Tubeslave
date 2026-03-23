"""Phase alignment message handlers."""

import logging

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_start_phase_alignment(websocket, data):
        await server.start_phase_alignment(
            websocket,
            device_id=data.get("device_id"),
            reference_channel=data.get("reference_channel"),
            channels=data.get("channels", []),
            settings=data.get("settings", {})
        )

    async def handle_stop_phase_alignment(websocket, data):
        await server.stop_phase_alignment(websocket)

    async def handle_apply_phase_corrections(websocket, data):
        logger.info(f"Received apply_phase_corrections command. Measurements: {data.get('measurements')}")
        await server.apply_phase_corrections(websocket, data.get("measurements"))

    async def handle_reset_phase_delay(websocket, data):
        await server.reset_phase_delay(websocket, data.get("channels"))

    async def handle_get_phase_alignment_status(websocket, data):
        await server.send_to_client(websocket, {
            "type": "phase_alignment_status",
            **server.get_phase_alignment_status()
        })

    return {
        "start_phase_alignment": handle_start_phase_alignment,
        "stop_phase_alignment": handle_stop_phase_alignment,
        "apply_phase_corrections": handle_apply_phase_corrections,
        "reset_phase_delay": handle_reset_phase_delay,
        "get_phase_alignment_status": handle_get_phase_alignment_status,
    }
