"""Audio device message handlers."""

import asyncio
import logging

from audio_devices import get_audio_devices
from dante_routing_config import get_routing_as_dict, get_module_signal_info
from audio_device_scanner import (
    scan_audio_devices, select_best_device, detect_and_report, AudioProtocol,
)

logger = logging.getLogger(__name__)


def _normalize_device_id(device_id):
    if device_id in (None, "", "default"):
        return None
    if isinstance(device_id, str):
        value = device_id.strip()
        if value.lower() == "default" or value == "":
            return None
        if value.isdigit():
            return int(value)
        return value
    return device_id


def register_handlers(server):
    async def handle_get_audio_devices(websocket, data):
        devices = get_audio_devices()
        await server.send_to_client(websocket, {
            "type": "audio_devices",
            "devices": devices
        })

    async def handle_get_dante_routing(websocket, data):
        total_ch = data.get("total_channels", 64)
        await server.send_to_client(websocket, {
            "type": "dante_routing",
            "routing_scheme": get_routing_as_dict(total_ch),
            "module_signal_info": get_module_signal_info(),
        })

    async def handle_scan_audio_devices(websocket, data):
        """Scan all audio input devices with full classification."""
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, detect_and_report)
        await server.send_to_client(websocket, {
            "type": "audio_device_scan",
            **report,
        })
        logger.info(
            f"Audio device scan: {report['count']} device(s), "
            f"{report['multichannel_count']} multichannel"
        )

    async def handle_select_audio_device(websocket, data):
        """Select best audio device, optionally with preferences."""
        preferred_name = data.get("preferred_name", None)
        preferred_protocol = data.get("preferred_protocol", None)
        min_channels = data.get("min_channels", 2)

        loop = asyncio.get_event_loop()
        devices = await loop.run_in_executor(None, scan_audio_devices)

        proto = None
        if preferred_protocol:
            try:
                proto = AudioProtocol(preferred_protocol)
            except ValueError:
                pass

        best = select_best_device(
            devices,
            preferred_protocol=proto,
            preferred_name=preferred_name,
            min_channels=min_channels,
        )

        if best:
            await server.send_to_client(websocket, {
                "type": "audio_device_selected",
                "device": {
                    "index": best.index,
                    "name": best.name,
                    "channels": best.max_input_channels,
                    "protocol": best.protocol.value,
                    "samplerate": best.default_samplerate,
                    "latency_ms": round(best.latency_ms, 1),
                    "score": best.score,
                },
            })
        else:
            await server.send_to_client(websocket, {
                "type": "audio_device_selected",
                "device": None,
                "error": "No suitable audio device found",
            })

    async def handle_start_audio_capture_preview(websocket, data):
        """Start read-only audio capture for meters without starting automation."""
        try:
            device_id = _normalize_device_id(
                data.get("device_id", data.get("device"))
            )
            num_channels = int(data.get("num_channels", data.get("channels", 48)))
            sample_rate = int(data.get("sample_rate", 48000))
            buffer_seconds = float(data.get("buffer_seconds", 8.0))
        except (TypeError, ValueError) as exc:
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                "running": False,
                "error": f"Invalid audio capture preview request: {exc}",
            })
            return

        try:
            capture = server._ensure_audio_capture(
                device_id,
                num_channels=num_channels,
                sample_rate=sample_rate,
                buffer_seconds=buffer_seconds,
            )
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                "preview": True,
                "device_id": device_id,
                **capture.get_status(),
            })
        except Exception as exc:
            logger.exception("Failed to start audio capture preview")
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                "running": False,
                "preview": True,
                "device_id": device_id,
                "error": str(exc),
            })

    async def handle_stop_audio_capture_preview(websocket, data):
        """Stop read-only audio capture if it is running."""
        if server.audio_capture:
            try:
                server.audio_capture.stop()
            finally:
                server.audio_capture = None
        await server.send_to_client(websocket, {
            "type": "audio_capture_status",
            "running": False,
            "preview": True,
        })

    return {
        "get_audio_devices": handle_get_audio_devices,
        "get_dante_routing": handle_get_dante_routing,
        "scan_audio_devices": handle_scan_audio_devices,
        "select_audio_device": handle_select_audio_device,
        "start_audio_capture_preview": handle_start_audio_capture_preview,
        "stop_audio_capture_preview": handle_stop_audio_capture_preview,
    }
