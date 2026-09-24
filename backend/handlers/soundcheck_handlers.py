"""Live soundcheck message handlers.

The websocket/UI layer does not construct, start, stop or inspect the legacy
soundcheck engine directly. Engine lifecycle is routed through
``live_runtime`` so the old engine can be decomposed without leaving parallel
decision authorities in the composition root.
"""

import asyncio
import logging

from live_runtime.contracts import LiveMode
from live_runtime.feature_stream import USB_CHANNEL_COUNT
from live_runtime.service import LiveSoundcheckService, LiveStartRequest
from live_runtime.session_config import (
    LiveSessionConfigError,
    resolve_live_capture_bridge_config,
)

logger = logging.getLogger(__name__)


def register_handlers(server):
    def _service():
        service = getattr(server, "_live_soundcheck_service", None)
        if service is None:
            service = LiveSoundcheckService()
            server._live_soundcheck_service = service
        return service

    def _engine_is_active():
        return bool(_service().is_active())

    def _channel_selection(data):
        raw_channels = data.get("channels", [])
        if isinstance(raw_channels, int):
            selected_channels = []
            num_channels = max(raw_channels, 1)
        else:
            selected_channels = []
            for ch in raw_channels:
                try:
                    channel = int(ch)
                except (TypeError, ValueError):
                    continue
                if channel > 0:
                    selected_channels.append(channel)
            selected_channels = sorted(set(selected_channels))
            num_channels = max(selected_channels) if selected_channels else 48
        return selected_channels, num_channels

    def _resolve_mode(data):
        """Resolve only explicit live modes; never infer BENCH_TEST from hardware."""
        raw = str(data.get("mode", "")).strip().lower()
        if raw:
            aliases = {
                "test": LiveMode.BENCH_TEST,
                "bench": LiveMode.BENCH_TEST,
                "bench_test": LiveMode.BENCH_TEST,
                "observe": LiveMode.OBSERVE,
                "propose": LiveMode.PROPOSE,
                "supervised": LiveMode.SUPERVISED,
                "auto_safe": LiveMode.AUTO_SAFE,
                "emergency": LiveMode.EMERGENCY,
                "freeze": LiveMode.FREEZE,
            }
            mode = aliases.get(raw)
            if mode is None:
                raise ValueError(f"unsupported live mode: {raw}")
            return mode

        # Backward compatibility for old clients that explicitly sent
        # observe_only. Missing mode/flag is deliberately read-only.
        if data.get("observe_only") is False:
            return LiveMode.SUPERVISED
        return LiveMode.OBSERVE

    def _schedule(loop, coro, label):
        def _log_failure(task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("%s failed: %s", label, exc)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            task = loop.create_task(coro)
            task.add_done_callback(_log_failure)
            return

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.add_done_callback(_log_failure)

    async def _start_engine(websocket, data, *, soundcheck_events: bool):
        service = _service()
        if _engine_is_active():
            await server.send_to_client(websocket, {
                "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
                "is_running": True,
                "running": True,
                "error": "Live soundcheck engine already running",
            })
            return

        mixer_config = server.config.get("mixer", {})
        audio_config = server.config.get("audio", {})
        selected_channels, num_channels = _channel_selection(data)

        mixer_type = data.get("mixer_type", mixer_config.get("type", "dlive"))
        mixer_ip = data.get("mixer_ip", mixer_config.get("ip", "192.168.3.70"))
        mixer_port = data.get(
            "mixer_port",
            mixer_config.get("port", 51328 if mixer_type == "dlive" else 2223)
        )
        audio_device = data.get("audio_device", data.get("device_id", audio_config.get("device_name", "soundgrid")))
        try:
            mode = _resolve_mode(data)
        except ValueError as exc:
            await server.send_to_client(websocket, {
                "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
                "status": "blocked",
                "is_running": False,
                "running": False,
                "error": str(exc),
            })
            return

        try:
            capture_bridge = resolve_live_capture_bridge_config(
                server.config,
                mixer_type=mixer_type,
                selected_channels=selected_channels,
            )
        except LiveSessionConfigError as exc:
            await server.send_to_client(websocket, {
                "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
                "status": "blocked",
                "is_running": False,
                "running": False,
                "error": f"live capture config: {exc}",
            })
            return

        observe_only = mode in (LiveMode.OBSERVE, LiveMode.PROPOSE, LiveMode.FREEZE)
        loop = asyncio.get_running_loop()

        def on_state(state, msg):
            server.auto_soundcheck_running = state not in ("stopped", "error")
            if soundcheck_events:
                _schedule(loop, server.send_to_client(websocket, {
                    "type": "auto_soundcheck_status",
                    "is_running": server.auto_soundcheck_running,
                    "running": server.auto_soundcheck_running,
                    "observe_only": server.auto_soundcheck_observe_only,
                    "mode": mode.value,
                    "current_step": state,
                    "step_progress": 0,
                    "progress": 0,
                    "step_time_remaining": 0,
                    "message": msg or state,
                }), "live soundcheck status send")

            _schedule(loop, server.broadcast({
                "type": "auto_engine_state",
                "state": state,
                "message": msg,
                "mode": mode.value,
            }), "live engine state broadcast")

        def on_channel(ch, ch_data):
            if hasattr(server, "update_mixing_agent_channel"):
                server.update_mixing_agent_channel(ch, ch_data)

            if soundcheck_events:
                _schedule(loop, server.send_to_client(websocket, {
                    "type": "auto_soundcheck_channel_update",
                    "channel": ch,
                    "data": ch_data,
                }), "live soundcheck channel send")

            _schedule(loop, server.broadcast({
                "type": "auto_engine_channel",
                "channel": ch,
                "data": ch_data,
            }), "live engine channel broadcast")

        def on_observation(payload):
            payload = {
                "type": "auto_soundcheck_observation",
                "mode": mode.value,
                **payload,
            }
            _schedule(loop, server.send_to_client(websocket, payload), "live soundcheck observation send")

        # The canonical WING feature ingress is always 48 channels.  A smaller
        # UI selection controls which inputs have roles/decision authority; it
        # must not shrink the underlying AudioCapture transport and accidentally
        # remove the reserved post-console Main return channels.
        capture_channels = USB_CHANNEL_COUNT if capture_bridge is not None else num_channels
        request = LiveStartRequest(
            mixer_type=mixer_type,
            mixer_ip=mixer_ip,
            mixer_port=mixer_port,
            audio_device_name=audio_device,
            num_channels=capture_channels,
            selected_channels=selected_channels,
            mode=mode,
            capture_bridge=capture_bridge,
        )
        try:
            service.start(
                request,
                on_state_change=on_state,
                on_channel_update=on_channel,
                on_observation=on_observation,
            )
        except RuntimeError as exc:
            await server.send_to_client(websocket, {
                "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
                "status": "blocked",
                "is_running": service.is_active(),
                "running": service.is_active(),
                "error": str(exc),
            })
            return

        server.auto_soundcheck_running = True
        server.auto_soundcheck_observe_only = observe_only

        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
            "status": "started",
            "is_running": True,
            "running": True,
            "observe_only": observe_only,
            "mode": mode.value,
            "mixer_type": mixer_type,
            "mixer_ip": mixer_ip,
            "selected_channels": selected_channels,
            "capture_bridge_configured": capture_bridge is not None,
            "message": "live_runtime soundcheck service started",
        })

    async def _stop_engine(websocket, *, soundcheck_events: bool):
        _service().stop()
        server.auto_soundcheck_running = False
        server.auto_soundcheck_observe_only = False
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
            "status": "stopped",
            "is_running": False,
            "running": False,
            "observe_only": False,
            "message": "Live soundcheck stopped",
        })

    async def handle_start_auto_soundcheck(websocket, data):
        await _start_engine(websocket, data, soundcheck_events=True)

    async def handle_stop_auto_soundcheck(websocket, data):
        await _stop_engine(websocket, soundcheck_events=True)

    async def handle_get_auto_soundcheck_status(websocket, data):
        engine_status = _service().get_status()
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": server.auto_soundcheck_running,
            "running": server.auto_soundcheck_running,
            "observe_only": server.auto_soundcheck_observe_only,
            "current_step": engine_status.get("state"),
            "step_progress": 0,
            "progress": 0,
            "step_time_remaining": 0,
            "message": "Idle" if not server.auto_soundcheck_running else "Running",
            **engine_status,
        })

    async def handle_start_auto_engine(websocket, data):
        await _start_engine(websocket, data, soundcheck_events=False)

    async def handle_stop_auto_engine(websocket, data):
        await _stop_engine(websocket, soundcheck_events=False)

    async def handle_get_auto_engine_status(websocket, data):
        status = _service().get_status()
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "is_running": server.auto_soundcheck_running,
            "running": server.auto_soundcheck_running,
            **status,
        })

    return {
        "start_auto_soundcheck": handle_start_auto_soundcheck,
        "stop_auto_soundcheck": handle_stop_auto_soundcheck,
        "get_auto_soundcheck_status": handle_get_auto_soundcheck_status,
        "start_auto_engine": handle_start_auto_engine,
        "stop_auto_engine": handle_stop_auto_engine,
        "get_auto_engine_status": handle_get_auto_engine_status,
    }
