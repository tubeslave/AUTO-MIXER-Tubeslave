"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""

from live_apply import (
    LiveApplyRequest,
    LiveApplyService,
    build_live_apply_operator_payload,
)


def register_handlers(server):
    if getattr(server, "live_apply_service", None) is None:
        server.live_apply_service = LiveApplyService()

    def build_live_apply_response(result):
        return build_live_apply_operator_payload(
            result,
            event_type="live_apply_result",
        )

    async def handle_set_fader(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        result = server.live_apply_service.apply(
            LiveApplyRequest(
                source="manual_ui",
                operation="set_fader",
                channel=int(channel),
                value=float(value),
                dry_run=data.get("dry_run"),
                confirm_live_apply=bool(data.get("confirm_live_apply", False)),
                replay_correlation_id=str(data.get("replay_correlation_id", "")),
                metadata={"handler": "set_fader"},
            ),
            server.mixer_client,
        )
        await server.send_to_client(websocket, build_live_apply_response(result))

    async def handle_set_gain(websocket, data):
        channel = data.get("channel")
        value = data.get("value")
        result = server.live_apply_service.apply(
            LiveApplyRequest(
                source="manual_ui",
                operation="set_gain",
                channel=int(channel),
                value=float(value),
                dry_run=data.get("dry_run"),
                confirm_live_apply=bool(data.get("confirm_live_apply", False)),
                replay_correlation_id=str(data.get("replay_correlation_id", "")),
                metadata={"handler": "set_gain"},
            ),
            server.mixer_client,
        )
        await server.send_to_client(websocket, build_live_apply_response(result))

    async def handle_set_eq(websocket, data):
        channel = data.get("channel")
        band = data.get("band")
        freq = data.get("freq")
        gain = data.get("gain")
        q = data.get("q")
        result = server.live_apply_service.apply(
            LiveApplyRequest(
                source="manual_ui",
                operation="set_eq",
                channel=int(channel),
                value=0.0,
                dry_run=data.get("dry_run"),
                confirm_live_apply=bool(data.get("confirm_live_apply", False)),
                replay_correlation_id=str(data.get("replay_correlation_id", "")),
                metadata={
                    "handler": "set_eq",
                    "band": band,
                    "freq": freq,
                    "gain": gain,
                    "q": q,
                },
            ),
            server.mixer_client,
        )
        await server.send_to_client(websocket, build_live_apply_response(result))

    async def handle_set_compressor(websocket, data):
        channel = data.get("channel")
        params = data.get("params", {})
        result = server.live_apply_service.apply(
            LiveApplyRequest(
                source="manual_ui",
                operation="set_compressor",
                channel=int(channel),
                value=0.0,
                dry_run=data.get("dry_run"),
                confirm_live_apply=bool(data.get("confirm_live_apply", False)),
                replay_correlation_id=str(data.get("replay_correlation_id", "")),
                metadata={
                    "handler": "set_compressor",
                    "params": dict(params),
                },
            ),
            server.mixer_client,
        )
        await server.send_to_client(websocket, build_live_apply_response(result))

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
        "get_state": handle_get_state,
    }
