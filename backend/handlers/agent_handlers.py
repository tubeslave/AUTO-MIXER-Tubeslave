"""AI MixingAgent WebSocket handlers."""


def register_handlers(server):
    def get_mixing_agent():
        return getattr(server, "mixing_agent", None)

    async def send_unavailable(websocket, response_type, command, **payload):
        await server.send_to_client(websocket, {
            "type": response_type,
            "status": "unavailable",
            "available": False,
            "reason": "mixing_agent_runtime_not_initialized",
            "command": command,
            "error": "Mixing agent runtime is not attached to AutoMixerServer",
            **payload,
        })

    async def handle_get_agent_status(websocket, data):
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            status = mixing_agent.get_status()
            status["channel_summary"] = mixing_agent.get_channel_summary()
            await server.send_to_client(websocket, {
                "type": "agent_status",
                **status,
            })
        else:
            await send_unavailable(
                websocket,
                "agent_status",
                "get_agent_status",
                is_running=False,
            )

    async def handle_set_agent_mode(websocket, data):
        mode = data.get("mode", "suggest")
        init_mixing_agent = getattr(server, "init_mixing_agent", None)
        if callable(init_mixing_agent):
            await init_mixing_agent(mode=mode)
            await server.send_to_client(websocket, {
                "type": "agent_mode_changed",
                "mode": mode,
                "success": True,
            })
        else:
            await send_unavailable(
                websocket,
                "agent_mode_changed",
                "set_agent_mode",
                mode=mode,
                success=False,
            )

    async def handle_get_pending_actions(websocket, data):
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            await server.send_to_client(websocket, {
                "type": "pending_actions",
                "actions": mixing_agent.get_pending_actions(),
            })
        else:
            await send_unavailable(
                websocket,
                "pending_actions",
                "get_pending_actions",
                actions=[],
            )

    async def handle_approve_action(websocket, data):
        idx = data.get("index", -1)
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            ok = mixing_agent.approve_action(idx)
            await server.send_to_client(websocket, {
                "type": "action_approved",
                "index": idx,
                "success": ok,
            })
        else:
            await send_unavailable(
                websocket,
                "action_approved",
                "approve_action",
                index=idx,
                success=False,
            )

    async def handle_approve_all(websocket, data):
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            count = mixing_agent.approve_all_pending()
            await server.send_to_client(websocket, {
                "type": "all_actions_approved",
                "count": count,
            })
        else:
            await send_unavailable(
                websocket,
                "all_actions_approved",
                "approve_all_actions",
                count=0,
                success=False,
            )

    async def handle_dismiss_action(websocket, data):
        idx = data.get("index", -1)
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            ok = mixing_agent.dismiss_action(idx)
            await server.send_to_client(websocket, {
                "type": "action_dismissed",
                "index": idx,
                "success": ok,
            })
        else:
            await send_unavailable(
                websocket,
                "action_dismissed",
                "dismiss_action",
                index=idx,
                success=False,
            )

    async def handle_dismiss_all(websocket, data):
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            mixing_agent.dismiss_all_pending()
            await server.send_to_client(websocket, {
                "type": "all_actions_dismissed",
                "success": True,
            })
        else:
            await send_unavailable(
                websocket,
                "all_actions_dismissed",
                "dismiss_all_actions",
                success=False,
            )

    async def handle_get_action_history(websocket, data):
        limit = data.get("limit", 50)
        mixing_agent = get_mixing_agent()
        if mixing_agent:
            await server.send_to_client(websocket, {
                "type": "action_history",
                "history": mixing_agent.get_action_history(limit),
            })
        else:
            await send_unavailable(
                websocket,
                "action_history",
                "get_action_history",
                history=[],
                limit=limit,
            )

    async def handle_get_audio_status(websocket, data):
        if server.audio_capture:
            status = server.audio_capture.get_status()
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                **status,
            })
        else:
            await server.send_to_client(websocket, {
                "type": "audio_capture_status",
                "running": False,
            })

    async def handle_get_channel_meters(websocket, data):
        if not server.audio_capture or not server.audio_capture.running:
            await server.send_to_client(websocket, {
                "type": "channel_meters",
                "channels": {},
            })
            return
        channels = {}
        for ch in range(1, server.audio_capture.num_channels + 1):
            rms = server.audio_capture.get_rms(ch)
            if rms > -90.0:
                channels[ch] = {
                    "rms_db": round(rms, 1),
                    "peak_db": round(server.audio_capture.get_peak(ch), 1),
                    "lufs": round(server.audio_capture.get_lufs(ch), 1),
                }
        await server.send_to_client(websocket, {
            "type": "channel_meters",
            "channels": channels,
        })

    return {
        "get_agent_status": handle_get_agent_status,
        "set_agent_mode": handle_set_agent_mode,
        "get_pending_actions": handle_get_pending_actions,
        "approve_action": handle_approve_action,
        "approve_all_actions": handle_approve_all,
        "dismiss_action": handle_dismiss_action,
        "dismiss_all_actions": handle_dismiss_all,
        "get_action_history": handle_get_action_history,
        "get_audio_capture_status": handle_get_audio_status,
        "get_channel_meters": handle_get_channel_meters,
    }
