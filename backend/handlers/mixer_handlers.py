"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""

from typing import Any, Optional


def _parse_readonly_parameter_request(server, data: dict) -> tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
    value_type = server._normalize_readonly_parameter_kind(data.get("value_type"))
    channel = data.get("channel")
    address = str(data.get("address") or "").strip()

    if value_type and channel is not None:
        try:
            channel_int = int(channel)
        except (TypeError, ValueError):
            return None, None, None, "invalid_channel"
        if channel_int < 1 or channel_int > 48:
            return None, None, None, "channel_out_of_range"
        expected_address = server._readonly_parameter_address(channel_int, value_type)
        if not expected_address:
            return None, None, None, "missing_read_target"
        if address and address != expected_address:
            return None, None, None, "address_value_type_mismatch"
        return channel_int, value_type, expected_address, None

    if address:
        parsed = server._parse_readonly_parameter_address(address)
        if parsed is None:
            return None, None, None, "unsupported_address"
        channel_int, parsed_type = parsed
        try:
            channel_int = int(channel_int)
        except (TypeError, ValueError):
            return None, None, None, "invalid_channel"
        if channel_int < 1 or channel_int > 48:
            return None, None, None, "channel_out_of_range"
        if value_type and value_type != parsed_type:
            return None, None, None, "address_value_type_mismatch"
        return channel_int, parsed_type, server._readonly_parameter_address(channel_int, parsed_type), None

    return None, None, None, "missing_read_target"


def _readonly_parameter_blocked_payload(data: dict, reason: str) -> dict[str, Any]:
    return {
        "type": "mixer_parameter_read_blocked",
        "request_id": data.get("request_id"),
        "status": "blocked",
        "reason": reason,
        "request": {
            "channel": data.get("channel"),
            "value_type": data.get("value_type"),
            "address": data.get("address"),
        },
    }


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

    async def handle_set_mute(websocket, data, surface: str = "set_mute"):
        channel = data.get("channel")
        value = data.get("muted", data.get("value"))
        if server._is_wing_deployment_write_boundary_active():
            blocked = await server._block_quarantined_wing_write_surface(
                websocket,
                surface=surface,
                result_type="manual_write_blocked",
                request=data,
                extra={
                    "channel": channel,
                    "value_type": "mute",
                    "requested_value": value,
                },
            )
            if blocked:
                return
        if server.mixer_client:
            server.mixer_client.set_channel_mute(channel, value)

    async def handle_set_pan(websocket, data):
        channel = data.get("channel")
        raw_value = data.get("value", data.get("pan", data.get("pan_percent", 0.0)))
        if server._is_wing_deployment_write_boundary_active():
            blocked = await server._block_quarantined_wing_write_surface(
                websocket,
                surface="set_pan",
                result_type="manual_write_blocked",
                request=data,
            )
            if blocked:
                return

        value = server._coerce_optional_float(raw_value)
        if value is None:
            value = 0.0
        value = server._clamp_manual_write_value(value, "pan")
        if server.mixer_client:
            server.mixer_client.set_channel_pan(channel, value)

    async def handle_mute_channel(websocket, data):
        payload = dict(data)
        if "muted" not in payload and "value" not in payload:
            payload["muted"] = True
        await handle_set_mute(websocket, payload, surface="mute_channel")

    async def handle_unmute_channel(websocket, data):
        payload = dict(data)
        payload["muted"] = False if payload.get("muted") is None else payload.get("muted")
        await handle_set_mute(websocket, payload, surface="unmute_channel")

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
        state_getter = getattr(server.mixer_client, "get_state", None)
        if callable(state_getter):
            payload = {
                "type": "state_update",
                "mode": server.connection_mode,
                "state": state_getter(),
            }
            await server.send_to_client(websocket, payload)
            return

        await server.send_to_client(
            websocket,
            {
                "type": "state_update",
                "mode": server.connection_mode,
                "state": {},
                "status": "state_unavailable",
                "reason": "mixer_client_state_source_unavailable",
            },
        )

    async def handle_query_mixer_parameter(websocket, data):
        channel, value_type, address, error = _parse_readonly_parameter_request(server, data)
        if error:
            await server.send_to_client(websocket, _readonly_parameter_blocked_payload(data, error))
            return

        result = await server._manual_write_read_value(
            channel=channel,
            value_type=value_type,
            address=address,
            prefer_physical=True,
        )
        await server.send_to_client(websocket, {
            "type": "mixer_parameter_value",
            "request_id": data.get("request_id"),
            "status": result.get("status") or "readback_unavailable",
            "channel": channel,
            "value_type": value_type,
            "address": address,
            "value": result.get("value"),
            "source": result.get("source") or "unavailable",
            "error": result.get("error"),
            "raw": result.get("raw"),
        })

    return {
        "set_fader": handle_set_fader,
        "set_gain": handle_set_gain,
        "set_mute": handle_set_mute,
        "set_pan": handle_set_pan,
        "mute_channel": handle_mute_channel,
        "unmute_channel": handle_unmute_channel,
        "set_eq": handle_set_eq,
        "set_compressor": handle_set_compressor,
        "rollback_last_manual_write": handle_rollback_last_manual_write,
        "get_state": handle_get_state,
        "query_mixer_parameter": handle_query_mixer_parameter,
    }
