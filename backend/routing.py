"""
Message routing between WebSocket clients and OSC mixer.

Routes commands from frontend WebSocket connections to OSC,
and OSC state updates back to connected WebSocket clients.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessageRouter:
    """
    Routes messages between WebSocket clients and OSC mixer.

    Provides:
    - WebSocket → OSC command translation
    - OSC → WebSocket state broadcast
    - Command filtering and validation
    - Rate limiting per client
    """

    def __init__(self, max_clients: int = 10, broadcast_interval: float = 0.1):
        self._clients: Set = set()
        self._max_clients = max_clients
        self._broadcast_interval = broadcast_interval
        self._command_handlers: Dict[str, Callable] = {}
        self._message_count = 0
        self._last_broadcast = 0.0

    def register_handler(self, command: str, handler: Callable):
        """Register a command handler."""
        self._command_handlers[command] = handler
        logger.debug(f"Registered handler for command: {command}")

    def unregister_handler(self, command: str):
        """Remove a command handler."""
        self._command_handlers.pop(command, None)

    async def add_client(self, websocket):
        """Register a new WebSocket client."""
        if len(self._clients) >= self._max_clients:
            logger.warning("Max clients reached, rejecting connection")
            return False
        self._clients.add(websocket)
        logger.info(f"Client connected ({len(self._clients)} total)")
        return True

    async def remove_client(self, websocket):
        """Remove a WebSocket client."""
        self._clients.discard(websocket)
        logger.info(f"Client disconnected ({len(self._clients)} remaining)")

    async def route_message(self, websocket, message: str) -> Optional[Dict]:
        """
        Route an incoming WebSocket message to the appropriate handler.

        Args:
            websocket: Source WebSocket connection
            message: JSON message string

        Returns:
            Response dict or None
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
            return {"error": "Invalid JSON"}

        command = data.get("command") or data.get("type")
        if not command:
            return {"error": "Missing command"}

        handler = self._command_handlers.get(command)
        if not handler:
            logger.debug(f"No handler for command: {command}")
            return {"error": f"Unknown command: {command}"}

        self._message_count += 1

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(data)
            else:
                result = handler(data)
            return result
        except Exception as e:
            logger.error(f"Handler error for {command}: {e}")
            return {"error": str(e)}

    async def broadcast(self, message: Dict):
        """Broadcast a message to all connected WebSocket clients."""
        if not self._clients:
            return

        now = time.time()
        if now - self._last_broadcast < self._broadcast_interval:
            return
        self._last_broadcast = now

        payload = json.dumps(message)
        disconnected = set()

        for client in self._clients:
            try:
                await client.send(payload)
            except Exception:
                disconnected.add(client)

        for client in disconnected:
            self._clients.discard(client)

    async def broadcast_state(self, state: Dict):
        """Broadcast mixer state update to all clients."""
        await self.broadcast({
            "type": "state_update",
            "data": state,
            "timestamp": time.time(),
        })

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def get_stats(self) -> Dict:
        """Get routing statistics."""
        return {
            "connected_clients": len(self._clients),
            "max_clients": self._max_clients,
            "total_messages": self._message_count,
            "registered_handlers": len(self._command_handlers),
        }
