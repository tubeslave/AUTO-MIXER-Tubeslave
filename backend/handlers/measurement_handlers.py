"""System measurement / master EQ message handlers."""


def register_handlers(server):
    async def handle_start_system_measurement(websocket, data):
        await server.start_system_measurement(
            websocket,
            device_id=data.get("device_id"),
            reference_channel=data.get("reference_channel"),
            measurement_channel=data.get("measurement_channel"),
            duration_sec=data.get("duration_sec", 6.0),
            target_bus=data.get("target_bus", "master"),
            target_id=data.get("target_id", 1),
            correction_mode=data.get("correction_mode", "flat"),
            reference_curve=data.get("reference_curve", "pink_noise_live_pa"),
        )

    async def handle_apply_system_measurement(websocket, data):
        await server.apply_system_measurement(
            websocket,
            target_bus=data.get("target_bus", "master"),
            target_id=data.get("target_id", 1),
        )

    async def handle_reset_system_measurement(websocket, data):
        await server.reset_system_measurement(
            websocket,
            target_bus=data.get("target_bus", "master"),
            target_id=data.get("target_id", 1),
        )

    async def handle_get_system_measurement_status(websocket, data):
        await server.send_to_client(websocket, {
            "type": "system_measurement_status",
            **server.get_system_measurement_status(),
        })

    return {
        "start_system_measurement": handle_start_system_measurement,
        "apply_system_measurement": handle_apply_system_measurement,
        "reset_system_measurement": handle_reset_system_measurement,
        "get_system_measurement_status": handle_get_system_measurement_status,
    }
