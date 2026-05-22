"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""


def register_handlers(server):
    async def handle_set_fader(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if server._is_wing_deployment_write_boundary_active():
            result = await server._apply_manual_console_write(
                websocket,
                channel=channel,
                value=value,
                value_type="fader",
                source="websocket:set_fader",
                reason=data.get("reason") or "websocket_set_fader",
                approved=bool(data.get("approved", False)),
                approval_id=data.get("approval_id"),
            )
            result_type = "manual_write_applied" if result.get("status") == "applied" else "manual_write_blocked"
            if result.get("status") == "error":
                result_type = "manual_write_error"
            await server.send_to_client(websocket, {"type": result_type, **result})
            return
        if server.mixer_client:
            server.mixer_client.set_channel_fader(channel, value)

    async def handle_set_gain(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if server._is_wing_deployment_write_boundary_active():
            result = await server._apply_manual_console_write(
                websocket,
                channel=channel,
                value=value,
                value_type="gain",
                source="websocket:set_gain",
                reason=data.get("reason") or "websocket_set_gain",
                approved=bool(data.get("approved", False)),
                approval_id=data.get("approval_id"),
            )
            result_type = "manual_write_applied" if result.get("status") == "applied" else "manual_write_blocked"
            if result.get("status") == "error":
                result_type = "manual_write_error"
            await server.send_to_client(websocket, {"type": result_type, **result})
            return
        if server.mixer_client:
            server.mixer_client.set_channel_gain(channel, value)

    async def handle_set_eq(websocket, data):
        channel = data.get("channel")
        band = data.get("band")
        freq = data.get("freq")
        gain = data.get("gain")
        q = data.get("q")
        blocked = await server._block_quarantined_wing_write_surface(
            websocket,
            surface="set_eq",
            result_type="set_eq_result",
            request=data,
        )
        if blocked:
            return
        if server.mixer_client:
            server.mixer_client.set_eq_band(channel, band, freq, gain, q)

    async def handle_set_compressor(websocket, data):
        channel = data.get("channel")
        params = data.get("params", {})
        blocked = await server._block_quarantined_wing_write_surface(
            websocket,
            surface="set_compressor",
            result_type="set_compressor_result",
            request=data,
        )
        if blocked:
            return
        if server.mixer_client:
            server.mixer_client.set_compressor(
                channel,
                params.get("threshold", 0),
                params.get("ratio", 1),
                params.get("attack", 0),
                params.get("release", 0)
            )

    async def handle_rollback_last_manual_write(websocket, data):
        result = await server._rollback_last_manual_console_write(
            websocket,
            source="websocket:rollback_last_manual_write",
            reason=data.get("reason") or "websocket_rollback_last_manual_write",
            approved=bool(data.get("approved", False)),
            approval_id=data.get("approval_id"),
        )
        result_type = (
            "manual_write_rollback_applied"
            if result.get("status") == "applied"
            else "manual_write_rollback_blocked"
        )
        if result.get("status") == "error":
            result_type = "manual_write_rollback_error"
        await server.send_to_client(websocket, {"type": result_type, **result})

    async def handle_get_state(websocket, data):
        if server.mixer_client:
            await server.send_to_client(websocket, {
                "type": "state_update",
                "mode": server.connection_mode,
                "state": server.mixer_client.get_state()
            })

    return {
        "set_fader": handle_set_fader,
        "set_gain": handle_set_gain,
        "set_eq": handle_set_eq,
        "set_compressor": handle_set_compressor,
        "rollback_last_manual_write": handle_rollback_last_manual_write,
        "get_state": handle_get_state,
    }
