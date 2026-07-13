"""Audio device message handlers."""

import asyncio
import logging

from audio_capture import validate_live_capture_dimensions
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
            num_channels, sample_rate, buffer_seconds = validate_live_capture_dimensions(
                num_channels,
                sample_rate,
                buffer_seconds,
            )
        except (TypeError, ValueError) as exc:
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                "running": False,
                "error": f"Invalid audio capture preview request: {exc}",
            })
            return

        try:
            measurement_task = getattr(server, "_system_measurement_task", None)
            if measurement_task is not None and not measurement_task.done():
                await server.send_to_client(websocket, {
                    "type": "audio_capture_status",
                    "running": False,
                    "preview": True,
                    "device_id": device_id,
                    "error": "audio_capture_in_use_by_system_measurement",
                })
                return
            async with server._audio_capture_lock:
                current_capture = getattr(server, "audio_capture", None)
                current_owner = getattr(server, "_audio_capture_preview_owner", None)
                engine = getattr(server, "auto_soundcheck_engine", None)
                engine_state = getattr(getattr(engine, "state", None), "value", None)
                if getattr(server, "auto_soundcheck_running", False) or (
                    engine is not None and engine_state not in ("stopped", "error")
                ):
                    payload = {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": False,
                        "error": "audio_capture_in_use_by_auto_soundcheck",
                    }
                elif current_capture is not None and current_owner is not websocket:
                    payload = {
                        "type": "audio_capture_status",
                        "running": bool(getattr(current_capture, "running", False)),
                        "preview": True,
                        "owned_by_requester": False,
                        "error": "audio_capture_in_use_by_another_runtime_owner",
                    }
                else:
                    capture = await server._ensure_audio_capture_async(
                        device_id,
                        num_channels=num_channels,
                        sample_rate=sample_rate,
                        buffer_seconds=buffer_seconds,
                    )
                    server._audio_capture_preview_owner = websocket
                    payload = {
                        "type": "audio_capture_status",
                        "preview": True,
                        "device_id": device_id,
                        "owned_by_requester": True,
                        **capture.get_status(),
                    }
            await server.send_to_client(websocket, {
                **payload,
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
        """Stop only the read-only preview owned by this WebSocket."""
        stop_owned = getattr(server, "_stop_owned_audio_capture_preview", None)
        if callable(stop_owned):
            stopped = bool(await stop_owned(websocket))
        else:
            owner = getattr(server, "_audio_capture_preview_owner", None)
            stopped = owner is websocket
            if stopped and server.audio_capture:
                try:
                    server.audio_capture.stop()
                finally:
                    server.audio_capture = None
                    server._audio_capture_preview_owner = None
        if not stopped:
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                "running": bool(
                    server.audio_capture and getattr(server.audio_capture, "running", False)
                ),
                "preview": True,
                "owned_by_requester": False,
                "error": "audio_capture_preview_not_owned_by_requester",
            })
            return
        await server.send_to_client(websocket, {
            "type": "audio_capture_status",
            "running": False,
            "preview": True,
            "owned_by_requester": True,
        })

    return {
        "get_audio_devices": handle_get_audio_devices,
        "get_dante_routing": handle_get_dante_routing,
        "scan_audio_devices": handle_scan_audio_devices,
        "select_audio_device": handle_select_audio_device,
        "start_audio_capture_preview": handle_start_audio_capture_preview,
        "stop_audio_capture_preview": handle_stop_audio_capture_preview,
    }
