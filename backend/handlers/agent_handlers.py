"""WebSocket handlers for MixingAgent (RAG / kb_first autonomous suggestions)."""

import logging

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_start_mixing_agent(websocket, data):
        await server.start_mixing_agent(websocket, data)

    async def handle_stop_mixing_agent(websocket, data):
        await server.stop_mixing_agent(websocket, data)

    async def handle_set_mixing_agent_mode(websocket, data):
        await server.set_mixing_agent_mode(websocket, data)

    async def handle_get_mixing_agent_status(websocket, data):
        await server.get_mixing_agent_status(websocket, data)

    async def handle_mixing_agent_approve(websocket, data):
        await server.mixing_agent_approve(websocket, data)

    async def handle_mixing_agent_dismiss(websocket, data):
        await server.mixing_agent_dismiss(websocket, data)

    return {
        "start_mixing_agent": handle_start_mixing_agent,
        "stop_mixing_agent": handle_stop_mixing_agent,
        "set_mixing_agent_mode": handle_set_mixing_agent_mode,
        "get_mixing_agent_status": handle_get_mixing_agent_status,
        "mixing_agent_approve": handle_mixing_agent_approve,
        "mixing_agent_dismiss": handle_mixing_agent_dismiss,
    }
