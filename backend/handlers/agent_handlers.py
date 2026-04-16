"""AI MixingAgent WebSocket handlers."""


def register_handlers(server):
    def _agent():
        return getattr(server, "mixing_agent", None)

    async def _send_agent_status(websocket):
        agent = _agent()
        if agent:
            status = agent.get_status()
            status["channel_summary"] = agent.get_channel_summary()
            status["auto_apply_enabled"] = getattr(server, "agent_auto_apply_enabled", False)
            await server.send_to_client(websocket, {
                "type": "agent_status",
                **status,
            })
        else:
            await server.send_to_client(websocket, {
                "type": "agent_status",
                "is_running": False,
                "auto_apply_enabled": False,
                "error": "Agent not initialized",
            })

    async def handle_get_agent_status(websocket, data):
        await _send_agent_status(websocket)

    async def handle_start_agent(websocket, data):
        agent = await server.init_mixing_agent(
            mode=data.get("mode", "auto"),
            channels=data.get("channels"),
            use_llm=data.get("use_llm", True),
            allow_auto_apply=data.get("allow_auto_apply", True),
            start=True,
        )
        await server.send_to_client(websocket, {
            "type": "agent_started",
            **agent.get_status(),
            "auto_apply_enabled": getattr(server, "agent_auto_apply_enabled", False),
        })

    async def handle_stop_agent(websocket, data):
        await server.stop_mixing_agent()
        await server.send_to_client(websocket, {
            "type": "agent_stopped",
            "is_running": False,
        })

    async def handle_emergency_stop_agent(websocket, data):
        agent = _agent()
        if agent:
            agent.emergency_stop()
        await server.stop_mixing_agent()
        await server.send_to_client(websocket, {
            "type": "agent_emergency_stopped",
            "is_running": False,
            "emergency_stop": True,
        })

    async def handle_set_agent_mode(websocket, data):
        mode = data.get("mode", "auto")
        agent = await server.init_mixing_agent(
            mode=mode,
            channels=data.get("channels"),
            use_llm=data.get("use_llm", True),
            allow_auto_apply=data.get("allow_auto_apply", True),
            start=data.get("start", False),
        )
        await server.send_to_client(websocket, {
            "type": "agent_mode_changed",
            "mode": agent.state.mode.value,
            "auto_apply_enabled": getattr(server, "agent_auto_apply_enabled", False),
        })

    async def handle_get_pending_actions(websocket, data):
        agent = _agent()
        if agent:
            await server.send_to_client(websocket, {
                "type": "pending_actions",
                "actions": agent.get_pending_actions(),
            })
        else:
            await server.send_to_client(websocket, {"type": "pending_actions", "actions": []})

    async def handle_approve_action(websocket, data):
        idx = data.get("index", -1)
        agent = _agent()
        if agent:
            ok = agent.approve_action(idx)
            await server.send_to_client(websocket, {
                "type": "action_approved",
                "index": idx,
                "success": ok,
            })
            await handle_get_pending_actions(websocket, data)

    async def handle_approve_all(websocket, data):
        agent = _agent()
        if agent:
            count = agent.approve_all_pending()
            await server.send_to_client(websocket, {
                "type": "all_actions_approved",
                "count": count,
            })
            await handle_get_pending_actions(websocket, data)

    async def handle_dismiss_action(websocket, data):
        idx = data.get("index", -1)
        agent = _agent()
        if agent:
            ok = agent.dismiss_action(idx)
            await server.send_to_client(websocket, {
                "type": "action_dismissed",
                "index": idx,
                "success": ok,
            })
            await handle_get_pending_actions(websocket, data)

    async def handle_dismiss_all(websocket, data):
        agent = _agent()
        if agent:
            agent.dismiss_all_pending()
            await server.send_to_client(websocket, {
                "type": "all_actions_dismissed",
            })
            await handle_get_pending_actions(websocket, data)

    async def handle_get_action_history(websocket, data):
        limit = data.get("limit", 50)
        agent = _agent()
        if agent:
            await server.send_to_client(websocket, {
                "type": "action_history",
                "history": agent.get_action_history(limit),
                "audit": agent.get_action_audit_log(limit),
            })
        else:
            await server.send_to_client(websocket, {"type": "action_history", "history": [], "audit": []})

    async def handle_update_agent_state(websocket, data):
        agent = _agent()
        if agent is None:
            agent = await server.init_mixing_agent(
                mode=data.get("mode", "auto"),
                channels=data.get("channels"),
                use_llm=data.get("use_llm", True),
                allow_auto_apply=data.get("allow_auto_apply", True),
                start=False,
            )
        states = data.get("channel_states")
        if states:
            normalized = {int(ch): state for ch, state in states.items()}
            agent.update_channel_states_batch(normalized)
        else:
            agent.update_channel_states_batch(server.collect_agent_channel_states(data.get("channels")))
        await _send_agent_status(websocket)

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
        "start_agent": handle_start_agent,
        "stop_agent": handle_stop_agent,
        "emergency_stop_agent": handle_emergency_stop_agent,
        "get_agent_status": handle_get_agent_status,
        "set_agent_mode": handle_set_agent_mode,
        "update_agent_state": handle_update_agent_state,
        "get_pending_actions": handle_get_pending_actions,
        "approve_action": handle_approve_action,
        "approve_all_actions": handle_approve_all,
        "dismiss_action": handle_dismiss_action,
        "dismiss_all_actions": handle_dismiss_all,
        "get_action_history": handle_get_action_history,
        "get_audio_capture_status": handle_get_audio_status,
        "get_channel_meters": handle_get_channel_meters,
    }
