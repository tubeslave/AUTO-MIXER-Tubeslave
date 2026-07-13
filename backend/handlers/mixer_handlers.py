"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""

import math
from typing import Any, Optional

try:
    from live_apply import LiveApplyRequest
except ImportError:  # pragma: no cover - package import fallback
    from backend.live_apply import LiveApplyRequest


def _wing_boundary_active(server) -> bool:
    checker = getattr(server, "_is_wing_deployment_write_boundary_active", None)
    policy_boundary = bool(checker()) if callable(checker) else False
    # Direct WebSocket writes are never a supported WING path, even when a
    # legacy deployment-boundary feature flag is disabled. The canonical
    # operator-proposal path owns WING authorization and transaction proof.
    return policy_boundary or str(getattr(server, "connection_mode", "")).lower() == "wing"


def _target_type(operation: str) -> str:
    return {
        "set_fader": "channel_fader",
        "set_gain": "channel_gain",
        "set_eq": "channel_eq",
        "set_compressor": "channel_compressor",
    }.get(operation, operation)


def _message_for_live_apply_result(result) -> str:
    if result.accepted and result.dry_run:
        return "Dry-run only: no OSC command was sent."
    if result.accepted and result.readback_status == "confirmed":
        return "Live apply confirmed by mixer readback."
    if result.blocked_reason == "confirm_live_apply_required":
        return "Live apply blocked: confirmation required."
    if result.blocked_reason == "panic_stop_active":
        return "Live apply blocked: panic stop is active for manual live apply."
    if result.blocked_reason == "unsupported_operation":
        return "Live apply blocked: operation is not gated for console writes."
    if result.blocked_reason == "mixer_client_unavailable":
        return "Mixer unavailable: command was not sent."
    if result.blocked_reason == "readback_timeout":
        return "Live apply sent, but readback timed out before confirmation."
    if result.blocked_reason == "readback_query_failed":
        return "Live apply sent, but readback query failed."
    return "Live apply blocked." if not result.accepted else "Live apply result recorded."


def _coerce_finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if str(type(value).__module__).startswith("unittest.mock"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _coerce_positive_int(value: Any, *, max_value: int | None = None) -> Optional[int]:
    parsed = _coerce_finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    integer = int(parsed)
    if integer < 1:
        return None
    if max_value is not None and integer > max_value:
        return None
    return integer


async def _send_invalid_live_apply_request(
    server,
    websocket,
    *,
    operation: str,
    channel: Any,
    value_type: str,
    requested_value: Any,
    data: dict,
    reason: str = "invalid_value",
):
    await server.send_to_client(
        websocket,
        {
            "type": "manual_live_apply_result",
            "request_id": data.get("request_id"),
            "operation": operation,
            "channel": channel,
            "accepted": False,
            "blocked": True,
            "blocked_reason": reason,
            "dry_run_only": True,
            "send_result": None,
            "send_status": "not_sent",
            "readback_status": "not_requested",
            "desired_value": None,
            "confirmed_value": None,
            "verification_id": None,
            "audit_id": "",
            "replay_correlation_id": str(data.get("replay_correlation_id") or ""),
            "failure_reason": reason,
            "timeout_ms": None,
            "panic_stop_active": False,
            "panic_stop_scope": None,
            "panic_stop_reason": None,
            "shadow_mode_active": False,
            "guard_reasons": [reason],
            "message_for_user": "Live apply blocked: invalid request.",
            "live_apply_source": "manual_ui",
            "target_type": _target_type(operation),
            "value_type": value_type,
            "requested_value": requested_value,
            "live_write": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        },
    )


async def _send_live_apply_result(
    server,
    websocket,
    *,
    operation: str,
    channel: Any,
    value: Any,
    data: dict,
    value_type: str = "value",
):
    channel_int = _coerce_positive_int(channel)
    if channel_int is None:
        await _send_invalid_live_apply_request(
            server,
            websocket,
            operation=operation,
            channel=channel,
            value_type="channel",
            requested_value=channel,
            data=data,
            reason="invalid_channel",
        )
        return None

    value_float = _coerce_finite_float(value)
    if value_float is None:
        await _send_invalid_live_apply_request(
            server,
            websocket,
            operation=operation,
            channel=channel_int,
            value_type=value_type,
            requested_value=value,
            data=data,
        )
        return None

    result = server.live_apply_service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation=operation,
            channel=channel_int,
            value=value_float,
            dry_run=data.get("dry_run"),
            confirm_live_apply=data.get("confirm_live_apply") is True,
            current_value=data.get("current_value"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            max_step=data.get("max_step"),
            replay_correlation_id=str(data.get("replay_correlation_id") or ""),
            metadata={"handler": f"mixer:{operation}"},
        ),
        getattr(server, "mixer_client", None),
    )
    payload = {
        "type": "manual_live_apply_result",
        "request_id": data.get("request_id"),
        "operation": operation,
        "accepted": bool(result.accepted),
        "blocked": bool(not result.accepted or result.blocked_reason),
        "blocked_reason": result.blocked_reason,
        "dry_run_only": bool(result.dry_run and not result.send_attempted),
        "send_result": result.send_result,
        "send_status": result.send_status,
        "readback_status": result.readback_status,
        "desired_value": result.desired_value,
        "confirmed_value": result.confirmed_value,
        "verification_id": result.verification_id,
        "audit_id": result.audit_id,
        "replay_correlation_id": result.replay_correlation_id,
        "failure_reason": result.failure_reason,
        "timeout_ms": result.timeout_ms,
        "panic_stop_active": result.panic_stop_active,
        "panic_stop_scope": result.panic_stop_scope,
        "panic_stop_reason": result.panic_stop_reason,
        "shadow_mode_active": result.blocked_reason == "shadow_mode_active",
        "guard_reasons": list(result.guard_reasons),
        "message_for_user": _message_for_live_apply_result(result),
        "live_apply_source": "manual_ui",
        "target_type": _target_type(operation),
    }
    await server.send_to_client(websocket, payload)
    return result


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
    async def send_direct_write_failed(
        websocket,
        data: dict,
        *,
        surface: str,
        value_type: str,
        requested_value: Any,
        reason: str,
        error: str | None = None,
    ):
        status = "error" if error else "blocked"
        payload = {
            "type": "manual_write_error" if error else "manual_write_blocked",
            "request_id": data.get("request_id"),
            "status": status,
            "success": False,
            "reason": reason,
            "surface": surface,
            "channel": data.get("channel"),
            "value_type": value_type,
            "requested_value": requested_value,
            "live_write": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        }
        if error:
            payload["error"] = error
        await server.send_to_client(websocket, payload)

    async def handle_set_fader(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if _wing_boundary_active(server):
            result = server._manual_write_blocked_result(
                reason="direct_wing_write_requires_operator_proposal",
                channel=channel,
                value=value,
                value_type="fader",
                source="websocket:set_fader",
                approval_id=data.get("approval_id"),
            )
            await server.send_to_client(
                websocket,
                {"type": "manual_write_blocked", **result},
            )
            return
        await _send_live_apply_result(
            server,
            websocket,
            operation="set_fader",
            channel=channel,
            value=value,
            data=data,
        )

    async def handle_set_gain(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        if _wing_boundary_active(server):
            result = server._manual_write_blocked_result(
                reason="direct_wing_write_requires_operator_proposal",
                channel=channel,
                value=value,
                value_type="gain",
                source="websocket:set_gain",
                approval_id=data.get("approval_id"),
            )
            await server.send_to_client(
                websocket,
                {"type": "manual_write_blocked", **result},
            )
            return
        await _send_live_apply_result(
            server,
            websocket,
            operation="set_gain",
            channel=channel,
            value=value,
            data=data,
        )

    async def handle_set_mute(websocket, data, surface: str = "set_mute"):
        channel = data.get("channel")
        value = data.get("muted", data.get("value"))
        if _wing_boundary_active(server):
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
            parsed_value = server._coerce_optional_bool(value)
            if parsed_value is None:
                await send_direct_write_failed(
                    websocket,
                    data,
                    surface=surface,
                    value_type="mute",
                    requested_value=value,
                    reason="invalid_mute",
                )
                return
            try:
                write_result = server.mixer_client.set_channel_mute(channel, int(parsed_value))
            except Exception as exc:
                await send_direct_write_failed(
                    websocket,
                    data,
                    surface=surface,
                    value_type="mute",
                    requested_value=bool(parsed_value),
                    reason="mixer_write_exception",
                    error=str(exc),
                )
                return
            if write_result is False:
                await send_direct_write_failed(
                    websocket,
                    data,
                    surface=surface,
                    value_type="mute",
                    requested_value=bool(parsed_value),
                    reason="mixer_write_failed",
                )

    async def handle_set_pan(websocket, data):
        channel = data.get("channel")
        raw_value = data.get("value", data.get("pan", data.get("pan_percent", 0.0)))
        if _wing_boundary_active(server):
            blocked = await server._block_quarantined_wing_write_surface(
                websocket,
                surface="set_pan",
                result_type="manual_write_blocked",
                request=data,
            )
            if blocked:
                return

        value = server._coerce_finite_float(raw_value)
        if value is None:
            await send_direct_write_failed(
                websocket,
                data,
                surface="set_pan",
                value_type="pan",
                requested_value=raw_value,
                reason="invalid_value",
            )
            return
        value = server._clamp_manual_write_value(value, "pan")
        if server.mixer_client:
            try:
                write_result = server.mixer_client.set_channel_pan(channel, value)
            except Exception as exc:
                await send_direct_write_failed(
                    websocket,
                    data,
                    surface="set_pan",
                    value_type="pan",
                    requested_value=value,
                    reason="mixer_write_exception",
                    error=str(exc),
                )
                return
            if write_result is False:
                await send_direct_write_failed(
                    websocket,
                    data,
                    surface="set_pan",
                    value_type="pan",
                    requested_value=value,
                    reason="mixer_write_failed",
                )

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
        if _wing_boundary_active(server):
            blocked = await server._block_quarantined_wing_write_surface(
                websocket,
                surface="set_eq",
                result_type="set_eq_result",
                request=data,
            )
            if blocked:
                return
        if _coerce_positive_int(band, max_value=4) is None:
            await _send_invalid_live_apply_request(
                server,
                websocket,
                operation="set_eq",
                channel=channel,
                value_type="eq_band",
                requested_value=band,
                data=data,
            )
            return
        freq_value = _coerce_finite_float(freq)
        if freq_value is None or freq_value <= 0:
            await _send_invalid_live_apply_request(
                server,
                websocket,
                operation="set_eq",
                channel=channel,
                value_type="eq_frequency",
                requested_value=freq,
                data=data,
            )
            return
        q_value = _coerce_finite_float(q)
        if q_value is None or q_value <= 0:
            await _send_invalid_live_apply_request(
                server,
                websocket,
                operation="set_eq",
                channel=channel,
                value_type="eq_q",
                requested_value=q,
                data=data,
            )
            return
        await _send_live_apply_result(
            server,
            websocket,
            operation="set_eq",
            channel=channel,
            value=gain,
            data=data,
            value_type="eq_gain",
        )

    async def handle_set_compressor(websocket, data):
        channel = data.get("channel")
        params = data.get("params", {})
        if _wing_boundary_active(server):
            blocked = await server._block_quarantined_wing_write_surface(
                websocket,
                surface="set_compressor",
                result_type="set_compressor_result",
                request=data,
            )
            if blocked:
                return
        if not isinstance(params, dict):
            await _send_invalid_live_apply_request(
                server,
                websocket,
                operation="set_compressor",
                channel=channel,
                value_type="compressor_params",
                requested_value=params,
                data=data,
            )
            return
        await _send_live_apply_result(
            server,
            websocket,
            operation="set_compressor",
            channel=channel,
            value=params.get("threshold"),
            data=data,
            value_type="compressor_threshold",
        )

    async def handle_rollback_last_manual_write(websocket, data):
        result = await server._rollback_last_manual_console_write(
            websocket,
            source="websocket:rollback_last_manual_write",
            reason=data.get("reason") or "websocket_rollback_last_manual_write",
            approved=data.get("approved") is True,
            approval_id=data.get("approval_id"),
            transaction_id=data.get("transaction_id"),
        )
        result_type = (
            "manual_write_rollback_applied"
            if result.get("status") == "applied"
            else "manual_write_rollback_blocked"
        )
        if result.get("status") == "error":
            result_type = "manual_write_rollback_error"
        elif result.get("status") == "sent_unverified":
            result_type = "manual_write_rollback_unverified"
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
            allow_cache_fallback=False,
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
