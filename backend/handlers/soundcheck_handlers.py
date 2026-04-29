"""Auto soundcheck message handlers."""

import asyncio
import logging

from auto_soundcheck_engine import AutoSoundcheckEngine

logger = logging.getLogger(__name__)


def register_handlers(server):
    def _engine_is_active():
        engine = server.auto_soundcheck_engine
        state = getattr(getattr(engine, "state", None), "value", None)
        return bool(engine and state not in ("stopped", "error"))

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
        """Start the new AutoSoundcheckEngine as the canonical soundcheck path."""
        if _engine_is_active():
            await server.send_to_client(websocket, {
                "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
                "is_running": True,
                "running": True,
                "error": "Auto soundcheck engine already running",
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
        observe_only = data.get("observe_only", soundcheck_events)
        loop = asyncio.get_running_loop()

        def on_state(state, msg):
            server.auto_soundcheck_running = state not in ("stopped", "error")
            if soundcheck_events:
                _schedule(loop, server.send_to_client(websocket, {
                    "type": "auto_soundcheck_status",
                    "is_running": server.auto_soundcheck_running,
                    "running": server.auto_soundcheck_running,
                    "observe_only": server.auto_soundcheck_observe_only,
                    "current_step": state,
                    "step_progress": 0,
                    "progress": 0,
                    "step_time_remaining": 0,
                    "message": msg or state,
                }), "auto soundcheck status send")

            _schedule(loop, server.broadcast({
                "type": "auto_engine_state",
                "state": state,
                "message": msg,
            }), "auto engine state broadcast")

        def on_channel(ch, ch_data):
            if hasattr(server, "update_mixing_agent_channel"):
                server.update_mixing_agent_channel(ch, ch_data)

            if soundcheck_events:
                _schedule(loop, server.send_to_client(websocket, {
                    "type": "auto_soundcheck_channel_update",
                    "channel": ch,
                    "data": ch_data,
                }), "auto soundcheck channel send")

            _schedule(loop, server.broadcast({
                "type": "auto_engine_channel",
                "channel": ch,
                "data": ch_data,
            }), "auto engine channel broadcast")

        def on_observation(payload):
            payload = {
                "type": "auto_soundcheck_observation",
                **payload,
            }
            _schedule(loop, server.send_to_client(websocket, payload), "auto soundcheck observation send")

        engine = AutoSoundcheckEngine(
            mixer_type=mixer_type,
            mixer_ip=mixer_ip,
            mixer_port=mixer_port,
            audio_device_name=audio_device,
            num_channels=num_channels,
            selected_channels=selected_channels,
            observe_only=observe_only,
            auto_apply=True,
            on_state_change=on_state,
            on_channel_update=on_channel,
            on_observation=on_observation,
        )
        server.auto_soundcheck_engine = engine
        server.auto_soundcheck_running = True
        server.auto_soundcheck_observe_only = observe_only
        engine.start_async()

        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status" if soundcheck_events else "auto_engine_status",
            "status": "started",
            "is_running": True,
            "running": True,
            "observe_only": observe_only,
            "mixer_type": mixer_type,
            "mixer_ip": mixer_ip,
            "selected_channels": selected_channels,
            "message": "New AutoSoundcheckEngine started",
        })

    async def handle_start_auto_soundcheck(websocket, data):
        await _start_engine(websocket, data, soundcheck_events=True)

    async def handle_stop_auto_soundcheck(websocket, data):
        if server.auto_soundcheck_engine:
            server.auto_soundcheck_engine.stop()
            server.auto_soundcheck_engine = None
        server.auto_soundcheck_running = False
        server.auto_soundcheck_observe_only = False
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": False,
            "running": False,
            "observe_only": False,
            "message": "Auto soundcheck stopped"
        })

    async def handle_get_auto_soundcheck_status(websocket, data):
        engine_status = server.auto_soundcheck_engine.get_status() if server.auto_soundcheck_engine else {}
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
        """Start the headless auto-soundcheck engine."""
        await _start_engine(websocket, data, soundcheck_events=False)

    async def handle_stop_auto_engine(websocket, data):
        """Stop the headless auto-soundcheck engine."""
        if server.auto_soundcheck_engine:
            server.auto_soundcheck_engine.stop()
            server.auto_soundcheck_engine = None
        server.auto_soundcheck_running = False
        server.auto_soundcheck_observe_only = False
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "status": "stopped",
            "is_running": False,
            "running": False,
        })

    async def handle_get_auto_engine_status(websocket, data):
        """Get auto-soundcheck engine status."""
        if server.auto_soundcheck_engine:
            status = server.auto_soundcheck_engine.get_status()
        else:
            status = {"state": "idle", "mixer_connected": False, "audio_running": False}
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "is_running": server.auto_soundcheck_running,
            "running": server.auto_soundcheck_running,
            **status
        })

    return {
        "start_auto_soundcheck": handle_start_auto_soundcheck,
        "stop_auto_soundcheck": handle_stop_auto_soundcheck,
        "get_auto_soundcheck_status": handle_get_auto_soundcheck_status,
        "start_auto_engine": handle_start_auto_engine,
        "stop_auto_engine": handle_stop_auto_engine,
        "get_auto_engine_status": handle_get_auto_engine_status,
    }
