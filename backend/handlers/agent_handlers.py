"""AI MixingAgent WebSocket handlers."""

import json


def register_handlers(server):
    async def handle_get_agent_status(websocket, data):
        if server.mixing_agent:
            status = server.mixing_agent.get_status()
            status["channel_summary"] = server.mixing_agent.get_channel_summary()
            await server.send_to_client(websocket, {
                "type": "agent_status",
                **status,
            })
        else:
            await server.send_to_client(websocket, {
                "type": "agent_status",
                "is_running": False,
                "error": "Agent not initialized",
            })

    async def handle_set_agent_mode(websocket, data):
        mode = data.get("mode", "suggest")
        if server.mixing_agent:
            await server.init_mixing_agent(mode=mode)
            await server.send_to_client(websocket, {
                "type": "agent_mode_changed",
                "mode": mode,
            })

    async def handle_get_pending_actions(websocket, data):
        if server.mixing_agent:
            await server.send_to_client(websocket, {
                "type": "pending_actions",
                "actions": server.mixing_agent.get_pending_actions(),
            })

    async def handle_approve_action(websocket, data):
        idx = data.get("index", -1)
        if server.mixing_agent:
            ok = server.mixing_agent.approve_action(idx)
            await server.send_to_client(websocket, {
                "type": "action_approved",
                "index": idx,
                "success": ok,
            })

    async def handle_approve_all(websocket, data):
        if server.mixing_agent:
            count = server.mixing_agent.approve_all_pending()
            await server.send_to_client(websocket, {
                "type": "all_actions_approved",
                "count": count,
            })

    async def handle_dismiss_action(websocket, data):
        idx = data.get("index", -1)
        if server.mixing_agent:
            ok = server.mixing_agent.dismiss_action(idx)
            await server.send_to_client(websocket, {
                "type": "action_dismissed",
                "index": idx,
                "success": ok,
            })

    async def handle_dismiss_all(websocket, data):
        if server.mixing_agent:
            server.mixing_agent.dismiss_all_pending()
            await server.send_to_client(websocket, {
                "type": "all_actions_dismissed",
            })

    async def handle_get_action_history(websocket, data):
        limit = data.get("limit", 50)
        if server.mixing_agent:
            await server.send_to_client(websocket, {
                "type": "action_history",
                "history": server.mixing_agent.get_action_history(limit),
            })

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
