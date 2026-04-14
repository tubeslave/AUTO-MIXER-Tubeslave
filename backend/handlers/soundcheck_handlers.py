"""Auto soundcheck message handlers."""

import asyncio
import logging

from auto_soundcheck_engine import AutoSoundcheckEngine

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_start_auto_soundcheck(websocket, data):
        await server.start_auto_soundcheck(
            websocket,
            data.get("device_id"),
            data.get("channels", []),
            data.get("channel_settings", {}),
            data.get("channel_mapping", {}),
            data.get("timings", {})
        )

    async def handle_stop_auto_soundcheck(websocket, data):
        await server.stop_auto_soundcheck(websocket)

    async def handle_get_auto_soundcheck_status(websocket, data):
        await server.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": server.auto_soundcheck_running,
            "running": server.auto_soundcheck_running,
            "current_step": None,
            "step_progress": 0,
            "progress": 0,
            "step_time_remaining": 0,
            "message": "Idle" if not server.auto_soundcheck_running else "Running"
        })

    async def handle_start_auto_engine(websocket, data):
        """Start the headless auto-soundcheck engine."""
        if server.auto_soundcheck_engine and server.auto_soundcheck_engine.state.value == "running":
            await server.send_to_client(websocket, {
                "type": "auto_engine_status",
                "error": "Engine already running"
            })
            return

        mixer_config = server.config.get("mixer", {})
        audio_config = server.config.get("audio", {})

        mixer_type = data.get("mixer_type", mixer_config.get("type", "dlive"))
        mixer_ip = data.get("mixer_ip", mixer_config.get("ip", "192.168.3.70"))
        mixer_port = data.get(
            "mixer_port",
            mixer_config.get("port", 51328 if mixer_type == "dlive" else 2223)
        )
        audio_device = data.get("audio_device", audio_config.get("device_name", "soundgrid"))
        num_channels = data.get("channels", 48)
        loop = asyncio.get_running_loop()

        def _schedule_broadcast(payload):
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is loop:
                task = loop.create_task(server.broadcast(payload))
                task.add_done_callback(
                    lambda t: logger.warning(
                        "Auto engine broadcast failed: %s",
                        t.exception(),
                    ) if t.exception() else None
                )
                return

            future = asyncio.run_coroutine_threadsafe(server.broadcast(payload), loop)
            future.add_done_callback(
                lambda f: logger.warning(
                    "Auto engine broadcast failed: %s",
                    f.exception(),
                ) if f.exception() else None
            )

        def on_state(state, msg):
            _schedule_broadcast({
                "type": "auto_engine_state",
                "state": state,
                "message": msg
            })

        def on_channel(ch, ch_data):
            _schedule_broadcast({
                "type": "auto_engine_channel",
                "channel": ch,
                "data": ch_data
            })

        engine = AutoSoundcheckEngine(
            mixer_type=mixer_type,
            mixer_ip=mixer_ip,
            mixer_port=mixer_port,
            audio_device_name=audio_device,
            num_channels=num_channels,
            auto_apply=True,
            on_state_change=on_state,
            on_channel_update=on_channel,
        )
        server.auto_soundcheck_engine = engine
        engine.start_async()

        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "status": "started",
            "mixer_type": mixer_type,
            "mixer_ip": mixer_ip,
        })

    async def handle_stop_auto_engine(websocket, data):
        """Stop the headless auto-soundcheck engine."""
        if server.auto_soundcheck_engine:
            server.auto_soundcheck_engine.stop()
            server.auto_soundcheck_engine = None
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
            "status": "stopped"
        })

    async def handle_get_auto_engine_status(websocket, data):
        """Get auto-soundcheck engine status."""
        if server.auto_soundcheck_engine:
            status = server.auto_soundcheck_engine.get_status()
        else:
            status = {"state": "idle", "mixer_connected": False, "audio_running": False}
        await server.send_to_client(websocket, {
            "type": "auto_engine_status",
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
