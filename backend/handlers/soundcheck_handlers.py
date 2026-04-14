"""Auto soundcheck message handlers."""

from auto_soundcheck_engine import AutoSoundcheckEngine


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
            "current_step": None,
            "step_progress": 0,
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

        mixer_type = data.get("mixer_type", "dlive")
        mixer_ip = data.get("mixer_ip", "192.168.3.70")
        mixer_port = data.get("mixer_port", 51328 if mixer_type == "dlive" else 2222)
        audio_device = data.get("audio_device", "soundgrid")
        num_channels = data.get("channels", 48)

        async def on_state(state, msg):
            await server.broadcast({
                "type": "auto_engine_state",
                "state": state,
                "message": msg
            })

        async def on_channel(ch, ch_data):
            await server.broadcast({
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
