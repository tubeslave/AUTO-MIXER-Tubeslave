"""Audio device message handlers."""

import asyncio
import logging

from audio_devices import get_audio_devices
from dante_routing_config import get_routing_as_dict, get_module_signal_info
from audio_device_scanner import (
    scan_audio_devices, select_best_device, detect_and_report, AudioProtocol,
)

logger = logging.getLogger(__name__)


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

    return {
        "get_audio_devices": handle_get_audio_devices,
        "get_dante_routing": handle_get_dante_routing,
        "scan_audio_devices": handle_scan_audio_devices,
        "select_audio_device": handle_select_audio_device,
    }
