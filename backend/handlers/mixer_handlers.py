"""Mixer control message handlers (fader, gain, EQ band, compressor, state)."""

import logging

logger = logging.getLogger(__name__)

# WING channel range (other mixers may differ)
CHANNEL_MIN = 1
CHANNEL_MAX = 40


def _get_safety_limits(server):
    """Get safety limits from config. Returns (max_fader_db, max_gain_db) or None if disabled."""
    safety = server.config.get("safety", {})
    if not safety.get("enable_limits", False):
        return None
    max_fader = safety.get("max_fader", 0)  # Default 0 dBFS for safety
    max_gain = safety.get("max_gain", 18)
    return (max_fader, max_gain)


def _validate_channel(channel) -> tuple[bool, str]:
    """Validate channel number. Returns (ok, error_message)."""
    if channel is None:
        return False, "Missing channel"
    try:
        ch = int(channel)
    except (TypeError, ValueError):
        return False, f"Invalid channel type: {type(channel).__name__}"
    if not (CHANNEL_MIN <= ch <= CHANNEL_MAX):
        return False, f"Channel must be {CHANNEL_MIN}..{CHANNEL_MAX}, got {ch}"
    return True, ""


def _validate_float(value, param_name: str) -> tuple[bool, float | None, str]:
    """Validate and parse float value. Returns (ok, parsed_value, error_message)."""
    if value is None:
        return False, None, f"Missing {param_name}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, None, f"Invalid {param_name}: cannot convert to float"
    return True, v, ""


def register_handlers(server):
    async def handle_set_fader(websocket, data):
        if not server.mixer_client:
            await server.send_to_client(websocket, {"type": "error", "error": "Mixer not connected"})
            return
        ok, err = _validate_channel(data.get("channel"))
        if not ok:
            logger.warning(f"set_fader validation: {err}")
            await server.send_to_client(websocket, {"type": "error", "error": err})
            return
        ok, value, err = _validate_float(data.get("value"), "value")
        if not ok:
            logger.warning(f"set_fader validation: {err}")
            await server.send_to_client(websocket, {"type": "error", "error": err})
            return
        limits = _get_safety_limits(server)
        if limits:
            max_fader, _ = limits
            if value > max_fader:
                logger.info(f"set_fader: clamping {value:.2f} dB to safety max {max_fader} dB")
                value = min(value, max_fader)
        try:
            server.mixer_client.set_channel_fader(int(data.get("channel")), value)
        except Exception as e:
            logger.error(f"set_fader failed: {e}", exc_info=True)
            await server.send_to_client(websocket, {"type": "error", "error": str(e)})

    async def handle_set_gain(websocket, data):
        if not server.mixer_client:
            await server.send_to_client(websocket, {"type": "error", "error": "Mixer not connected"})
            return
        ok, err = _validate_channel(data.get("channel"))
        if not ok:
            logger.warning(f"set_gain validation: {err}")
            await server.send_to_client(websocket, {"type": "error", "error": err})
            return
        ok, value, err = _validate_float(data.get("value"), "value")
        if not ok:
            logger.warning(f"set_gain validation: {err}")
            await server.send_to_client(websocket, {"type": "error", "error": err})
            return
        limits = _get_safety_limits(server)
        if limits:
            _, max_gain = limits
            if value > max_gain:
                logger.info(f"set_gain: clamping {value:.2f} dB to safety max {max_gain} dB")
                value = min(value, max_gain)
        try:
            server.mixer_client.set_channel_gain(int(data.get("channel")), value)
        except Exception as e:
            logger.error(f"set_gain failed: {e}", exc_info=True)
            await server.send_to_client(websocket, {"type": "error", "error": str(e)})

    async def handle_set_eq(websocket, data):
        if not server.mixer_client:
            await server.send_to_client(websocket, {"type": "error", "error": "Mixer not connected"})
            return
        ok, err = _validate_channel(data.get("channel"))
        if not ok:
            await server.send_to_client(websocket, {"type": "error", "error": err})
            return
        try:
            channel = int(data.get("channel"))
            band = data.get("band")
            freq = data.get("freq")
            gain = data.get("gain")
            q = data.get("q")
            server.mixer_client.set_eq_band(channel, band, freq, gain, q)
        except Exception as e:
            logger.error(f"set_eq failed: {e}", exc_info=True)
            await server.send_to_client(websocket, {"type": "error", "error": str(e)})

    async def handle_set_compressor(websocket, data):
        if not server.mixer_client:
            await server.send_to_client(websocket, {"type": "error", "error": "Mixer not connected"})
            return
        ok, err = _validate_channel(data.get("channel"))
        if not ok:
            await server.send_to_client(websocket, {"type": "error", "error": err})
            return
        try:
            channel = int(data.get("channel"))
            params = data.get("params", {})
            server.mixer_client.set_compressor(
                channel,
                params.get("threshold", 0),
                params.get("ratio", 1),
                params.get("attack", 0),
                params.get("release", 0),
            )
        except Exception as e:
            logger.error(f"set_compressor failed: {e}", exc_info=True)
            await server.send_to_client(websocket, {"type": "error", "error": str(e)})

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
