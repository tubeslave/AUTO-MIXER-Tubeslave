"""
Message Routing between WebSocket Clients and OSC

Routes commands from the frontend (via WebSocket) to the Behringer Wing mixer
(via OSC), and routes mixer state changes back to connected WebSocket clients.

Key features:
- Bidirectional message translation (WS JSON <-> OSC)
- Command parsing and validation with schema enforcement
- Broadcast to all WS clients or targeted per-client messages
- Topic-based subscription filtering
- Pluggable command handlers for extensibility
"""

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class MessageDirection(Enum):
    WS_TO_OSC = "ws_to_osc"
    OSC_TO_WS = "osc_to_ws"


@dataclass
class WSClient:
    """Represents a connected WebSocket client."""
    client_id: str
    websocket: Any
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)


@dataclass
class WSCommand:
    """Parsed WebSocket command from a frontend client."""
    action: str                        # e.g. 'set_fader', 'get_state', 'mute'
    channel: Optional[int] = None      # Target channel (1-based), None for global
    params: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None   # Client-supplied correlation ID
    client_id: Optional[str] = None    # Originating WS client

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"action": self.action}
        if self.channel is not None:
            d["channel"] = self.channel
        if self.params:
            d["params"] = self.params
        if self.request_id:
            d["request_id"] = self.request_id
        return d


@dataclass
class WSResponse:
    """Response to send back to one or more WebSocket clients."""
    event: str                           # e.g. 'state_update', 'error', 'ack'
    data: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None     # Echo back client correlation ID
    target_client: Optional[str] = None  # None = broadcast to all

    def to_json(self) -> str:
        d: Dict[str, Any] = {"event": self.event, "data": self.data}
        if self.request_id:
            d["request_id"] = self.request_id
        return json.dumps(d, default=str)


@dataclass
class OSCAction:
    """An OSC message to send to the mixer."""
    address: str
    args: Tuple[Any, ...] = ()

    def __repr__(self) -> str:
        return f"OSC({self.address} {self.args})"


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------

# Allowed actions and their required/optional fields
_ACTION_SCHEMA: Dict[str, Dict[str, Any]] = {
    "set_fader": {
        "requires_channel": True,
        "required_params": ["value"],
        "param_types": {"value": (int, float)},
        "param_ranges": {"value": (-144.0, 10.0)},
    },
    "set_mute": {
        "requires_channel": True,
        "required_params": ["value"],
        "param_types": {"value": (bool, int)},
    },
    "set_pan": {
        "requires_channel": True,
        "required_params": ["value"],
        "param_types": {"value": (int, float)},
        "param_ranges": {"value": (-100.0, 100.0)},
    },
    "set_eq_band": {
        "requires_channel": True,
        "required_params": ["band", "gain"],
        "param_types": {"band": (int,), "gain": (int, float)},
        "param_ranges": {"band": (1, 6), "gain": (-15.0, 15.0)},
        "optional_params": {"frequency": (int, float), "q": (int, float)},
    },
    "set_compressor": {
        "requires_channel": True,
        "required_params": [],
        "optional_params": {
            "on": (bool, int), "threshold": (int, float),
            "ratio": (int, float), "attack": (int, float),
            "release": (int, float), "makeup_gain": (int, float),
        },
    },
    "set_gate": {
        "requires_channel": True,
        "required_params": [],
        "optional_params": {
            "on": (bool, int), "threshold": (int, float),
            "attack": (int, float), "release": (int, float),
            "hold": (int, float), "range": (int, float),
        },
    },
    "set_name": {
        "requires_channel": True,
        "required_params": ["value"],
        "param_types": {"value": (str,)},
    },
    "set_trim": {
        "requires_channel": True,
        "required_params": ["value"],
        "param_types": {"value": (int, float)},
        "param_ranges": {"value": (-18.0, 18.0)},
    },
    "get_state": {
        "requires_channel": False,
        "required_params": [],
    },
    "get_channel_state": {
        "requires_channel": True,
        "required_params": [],
    },
    "snapshot_save": {
        "requires_channel": False,
        "required_params": ["name"],
        "param_types": {"name": (str,)},
    },
    "snapshot_recall": {
        "requires_channel": False,
        "required_params": ["name"],
        "param_types": {"name": (str,)},
    },
    "snapshot_list": {
        "requires_channel": False,
        "required_params": [],
    },
    "ping": {
        "requires_channel": False,
        "required_params": [],
    },
}


def validate_command(cmd: WSCommand) -> Optional[str]:
    """
    Validate a parsed WebSocket command against the action schema.

    Returns None if valid, or an error message string if invalid.
    """
    schema = _ACTION_SCHEMA.get(cmd.action)
    if schema is None:
        return f"Unknown action: {cmd.action}"

    # Channel requirement
    if schema.get("requires_channel") and cmd.channel is None:
        return f"Action '{cmd.action}' requires a channel number"

    if cmd.channel is not None:
        if not isinstance(cmd.channel, int) or not (1 <= cmd.channel <= 40):
            return f"Invalid channel: {cmd.channel} (must be 1-40)"

    # Required params
    for param_name in schema.get("required_params", []):
        if param_name not in cmd.params:
            return f"Missing required parameter: {param_name}"

    # Type checking
    param_types = schema.get("param_types", {})
    for param_name, allowed_types in param_types.items():
        if param_name in cmd.params:
            val = cmd.params[param_name]
            if not isinstance(val, allowed_types):
                return (
                    f"Parameter '{param_name}' has wrong type: "
                    f"expected {allowed_types}, got {type(val).__name__}"
                )

    # Range checking
    param_ranges = schema.get("param_ranges", {})
    for param_name, (lo, hi) in param_ranges.items():
        if param_name in cmd.params:
            val = cmd.params[param_name]
            if isinstance(val, (int, float)) and not (lo <= val <= hi):
                return (
                    f"Parameter '{param_name}' out of range: "
                    f"{val} (must be {lo}..{hi})"
                )

    return None


def parse_ws_message(raw: str, client_id: Optional[str] = None) -> WSCommand:
    """
    Parse a raw WebSocket JSON message into a WSCommand.

    Accepts two JSON formats:
        Format A (action-based):
            {"action": "set_fader", "channel": 5, "params": {"value": -10.0}}
        Format B (type-based, legacy):
            {"type": "set_fader", "channel": 5, "value": -10.0}

    Raises:
        ValueError: if the message cannot be parsed.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}")

    if not isinstance(data, dict):
        raise ValueError("Message must be a JSON object")

    # Support both "action" and "type" fields
    action = data.get("action") or data.get("type")
    if not action or not isinstance(action, str):
        raise ValueError("Missing or invalid 'action'/'type' field")

    # Extract params: either from a "params" sub-dict or from top-level keys
    params = data.get("params", {})
    if not params:
        # Gather non-meta fields as params
        meta_keys = {"action", "type", "channel", "request_id", "client_id"}
        params = {k: v for k, v in data.items() if k not in meta_keys}

    return WSCommand(
        action=action,
        channel=data.get("channel"),
        params=params,
        request_id=data.get("request_id"),
        client_id=client_id,
    )


# ---------------------------------------------------------------------------
# Command-to-OSC translation
# ---------------------------------------------------------------------------

def command_to_osc(cmd: WSCommand) -> List[OSCAction]:
    """
    Translate a validated WSCommand into one or more OSCActions.

    Returns a list because some commands expand to multiple OSC messages
    (e.g., setting EQ band sends frequency, gain, and Q).
    """
    ch = cmd.channel
    params = cmd.params
    actions: List[OSCAction] = []

    if cmd.action == "set_fader":
        actions.append(OSCAction(f"/ch/{ch}/fdr", (float(params["value"]),)))

    elif cmd.action == "set_mute":
        val = 1 if params["value"] else 0
        actions.append(OSCAction(f"/ch/{ch}/mute", (val,)))

    elif cmd.action == "set_pan":
        actions.append(OSCAction(f"/ch/{ch}/pan", (float(params["value"]),)))

    elif cmd.action == "set_name":
        actions.append(OSCAction(f"/ch/{ch}/name", (str(params["value"]),)))

    elif cmd.action == "set_trim":
        actions.append(OSCAction(f"/ch/{ch}/in/set/trim", (float(params["value"]),)))

    elif cmd.action == "set_eq_band":
        band = int(params["band"])
        # Map band numbers: 0 = low shelf, 1-4 = parametric, 5 = high shelf
        band_prefix_map = {
            0: "l", 1: "1", 2: "2", 3: "3", 4: "4", 5: "h",
        }
        prefix = band_prefix_map.get(band, str(band))

        actions.append(OSCAction(f"/ch/{ch}/eq/{prefix}g", (float(params["gain"]),)))
        if "frequency" in params:
            actions.append(
                OSCAction(f"/ch/{ch}/eq/{prefix}f", (float(params["frequency"]),))
            )
        if "q" in params:
            actions.append(
                OSCAction(f"/ch/{ch}/eq/{prefix}q", (float(params["q"]),))
            )

    elif cmd.action == "set_compressor":
        field_to_osc = {
            "on": "on", "threshold": "thr", "ratio": "ratio",
            "attack": "att", "release": "rel", "hold": "hld",
            "makeup_gain": "gain", "knee": "knee", "detect": "det",
            "envelope": "env", "mix": "mix", "auto": "auto",
        }
        for field_name, osc_suffix in field_to_osc.items():
            if field_name in params:
                val = params[field_name]
                if field_name == "on":
                    val = 1 if val else 0
                actions.append(
                    OSCAction(f"/ch/{ch}/dyn/{osc_suffix}", (val,))
                )

    elif cmd.action == "set_gate":
        field_to_osc = {
            "on": "on", "threshold": "thr", "attack": "att",
            "release": "rel", "hold": "hld", "range": "range",
            "accent": "acc", "ratio": "ratio",
        }
        for field_name, osc_suffix in field_to_osc.items():
            if field_name in params:
                val = params[field_name]
                if field_name == "on":
                    val = 1 if val else 0
                actions.append(
                    OSCAction(f"/ch/{ch}/gate/{osc_suffix}", (val,))
                )

    # Query commands (get_state, snapshot_*, ping) produce no OSC actions

    return actions


# ---------------------------------------------------------------------------
# MessageRouter
# ---------------------------------------------------------------------------

class MessageRouter:
    """
    Bidirectional message router between WebSocket clients and OSC.

    Manages:
    - WS client registration with topic-based subscriptions
    - Command parsing, validation, and translation to OSC
    - OSC state change broadcasting back to WS clients
    - Custom pluggable command handlers

    Usage:
        router = MessageRouter()
        router.set_osc_sender(osc_send_fn)
        router.set_mixer_state(mixer_state)
        # On WS connect:
        router.register_ws_client(client_id, websocket)
        # On WS message:
        result = await router.route_ws_message(client_id, raw_json)
        # On OSC message:
        router.route_osc_message(address, *args)
    """

    def __init__(self) -> None:
        # WS clients
        self._ws_clients: Dict[str, WSClient] = {}
        self._clients_lock = threading.Lock()

        # OSC send function: address, args -> bool
        self._osc_send: Optional[Callable[[str, Tuple[Any, ...]], bool]] = None

        # Mixer state reference (for state queries and snapshot operations)
        self._mixer_state: Optional[Any] = None

        # Registered handlers: OSC address patterns -> callbacks
        self._osc_handlers: Dict[str, Callable[..., Any]] = {}

        # WS command handlers: action -> async/sync handler
        self._ws_command_handlers: Dict[str, Callable[..., Any]] = {}

        # Per-command validators
        self._command_validators: Dict[str, Callable[..., Optional[str]]] = {}

        # Statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            "ws_messages_in": 0,
            "ws_messages_out": 0,
            "osc_messages_routed": 0,
            "validation_errors": 0,
            "broadcast_errors": 0,
        }

        logger.info("MessageRouter initialized")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_osc_sender(
        self, sender: Callable[[str, Tuple[Any, ...]], bool]
    ) -> None:
        """Set the function used to send OSC messages to the mixer."""
        self._osc_send = sender

    def set_mixer_state(self, mixer_state: Any) -> None:
        """Set the MixerState instance for state queries and snapshots."""
        self._mixer_state = mixer_state

    # ------------------------------------------------------------------
    # WS client management
    # ------------------------------------------------------------------

    def register_ws_client(self, client_id: str, websocket: Any) -> WSClient:
        """Register a new WebSocket client."""
        client = WSClient(client_id=client_id, websocket=websocket)
        with self._clients_lock:
            self._ws_clients[client_id] = client
        logger.info("WS client registered: %s (total: %d)",
                     client_id, self.client_count)
        return client

    def unregister_ws_client(self, client_id: str) -> None:
        """Remove a WebSocket client."""
        with self._clients_lock:
            self._ws_clients.pop(client_id, None)
        logger.info("WS client unregistered: %s (total: %d)",
                     client_id, self.client_count)

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._ws_clients)

    def get_client_ids(self) -> List[str]:
        with self._clients_lock:
            return list(self._ws_clients.keys())

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe_client(self, client_id: str, topic: str) -> None:
        """Subscribe a client to a topic for filtered updates."""
        with self._clients_lock:
            client = self._ws_clients.get(client_id)
        if client:
            client.subscriptions.add(topic)
            logger.debug("Client %s subscribed to '%s'", client_id, topic)

    def unsubscribe_client(self, client_id: str, topic: str) -> None:
        """Unsubscribe a client from a topic."""
        with self._clients_lock:
            client = self._ws_clients.get(client_id)
        if client:
            client.subscriptions.discard(topic)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_command_handler(
        self,
        command: str,
        handler: Callable[..., Any],
        validator: Optional[Callable[..., Optional[str]]] = None,
    ) -> None:
        """
        Register a handler for a WS command type.

        The handler receives (client_id: str, data: dict) and can be
        sync or async.  It should return a dict response or None.
        """
        self._ws_command_handlers[command] = handler
        if validator:
            self._command_validators[command] = validator
        logger.debug("Command handler registered: %s", command)

    def register_osc_handler(
        self, address_pattern: str, handler: Callable[..., Any]
    ) -> None:
        """Register a handler for incoming OSC address patterns."""
        self._osc_handlers[address_pattern] = handler
        logger.debug("OSC handler registered: %s", address_pattern)

    # ------------------------------------------------------------------
    # WS -> OSC routing
    # ------------------------------------------------------------------

    async def route_ws_message(
        self, client_id: str, message: str
    ) -> Optional[Dict[str, Any]]:
        """
        Route an incoming WS message: parse, validate, translate to OSC,
        send to mixer, and return a response dict.
        """
        with self._stats_lock:
            self._stats["ws_messages_in"] += 1

        # Parse
        try:
            cmd = parse_ws_message(message, client_id=client_id)
        except ValueError as exc:
            logger.warning("Parse error from %s: %s", client_id, exc)
            return {"error": str(exc)}

        # Check for custom WS command handler first (legacy path)
        if cmd.action in self._ws_command_handlers:
            handler = self._ws_command_handlers[cmd.action]

            # Run per-command validator if registered
            if cmd.action in self._command_validators:
                val_error = self._command_validators[cmd.action](cmd.to_dict())
                if val_error:
                    with self._stats_lock:
                        self._stats["validation_errors"] += 1
                    return {"error": val_error}

            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(client_id, cmd.to_dict())
                else:
                    result = handler(client_id, cmd.to_dict())
                return result
            except Exception as exc:
                logger.error("Handler error for '%s': %s", cmd.action, exc)
                return {"error": str(exc)}

        # Schema-based validation
        error = validate_command(cmd)
        if error:
            with self._stats_lock:
                self._stats["validation_errors"] += 1
            return {"error": error, "request_id": cmd.request_id}

        # Handle internal commands (no OSC needed)
        internal_result = await self._handle_internal_command(cmd, client_id)
        if internal_result is not None:
            return internal_result

        # Translate to OSC and send
        if not self._osc_send:
            return {"error": "No OSC sender configured"}

        osc_actions = command_to_osc(cmd)
        sent = 0
        for osc_action in osc_actions:
            if self._osc_send(osc_action.address, osc_action.args):
                sent += 1

        if sent > 0:
            return {
                "event": "ack",
                "action": cmd.action,
                "channel": cmd.channel,
                "osc_messages_sent": sent,
                "request_id": cmd.request_id,
            }
        else:
            return {
                "error": f"Failed to send OSC for {cmd.action}",
                "request_id": cmd.request_id,
            }

    async def _handle_internal_command(
        self, cmd: WSCommand, client_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Handle commands that don't require OSC (ping, state queries,
        snapshot operations).  Returns None if the command is not internal.
        """
        if cmd.action == "ping":
            return {
                "event": "pong",
                "server_time": time.time(),
                "request_id": cmd.request_id,
            }

        if cmd.action == "get_state" and self._mixer_state:
            data = json.loads(self._mixer_state.export_json())
            return {
                "event": "full_state",
                "data": data,
                "request_id": cmd.request_id,
            }

        if cmd.action == "get_channel_state" and self._mixer_state:
            try:
                ch = self._mixer_state.get_channel(cmd.channel)
                return {
                    "event": "channel_state",
                    "data": ch.to_dict(),
                    "request_id": cmd.request_id,
                }
            except KeyError:
                return {
                    "error": f"Channel {cmd.channel} not found",
                    "request_id": cmd.request_id,
                }

        if cmd.action == "snapshot_save" and self._mixer_state:
            try:
                self._mixer_state.snapshot_save(cmd.params["name"])
                return {
                    "event": "ack",
                    "action": "snapshot_save",
                    "name": cmd.params["name"],
                    "request_id": cmd.request_id,
                }
            except Exception as exc:
                return {"error": str(exc), "request_id": cmd.request_id}

        if cmd.action == "snapshot_recall" and self._mixer_state:
            ok = self._mixer_state.snapshot_recall(cmd.params["name"])
            if ok:
                # Broadcast recall event to all clients
                await self.broadcast({
                    "event": "snapshot_recalled",
                    "name": cmd.params["name"],
                })
                return {
                    "event": "ack",
                    "action": "snapshot_recall",
                    "name": cmd.params["name"],
                    "request_id": cmd.request_id,
                }
            return {
                "error": f"Snapshot '{cmd.params['name']}' not found",
                "request_id": cmd.request_id,
            }

        if cmd.action == "snapshot_list" and self._mixer_state:
            snapshots = self._mixer_state.snapshot_list()
            return {
                "event": "snapshot_list",
                "snapshots": snapshots,
                "request_id": cmd.request_id,
            }

        # Not an internal command
        return None

    # ------------------------------------------------------------------
    # OSC -> WS routing
    # ------------------------------------------------------------------

    def route_osc_message(self, address: str, *args: Any) -> None:
        """
        Route an incoming OSC message from the mixer.

        1. Dispatch to registered OSC handlers.
        2. Update the MixerState if available.
        3. Schedule a broadcast to all WS clients.
        """
        with self._stats_lock:
            self._stats["osc_messages_routed"] += 1

        # Dispatch to pattern-matched handlers
        for pattern, handler in self._osc_handlers.items():
            if self._match_osc_pattern(pattern, address):
                try:
                    handler(address, *args)
                except Exception as exc:
                    logger.error("OSC handler error for %s: %s", address, exc)

        # Update internal state
        if self._mixer_state:
            value = args[-1] if args else None
            self._mixer_state.update_from_osc(address, value)

        # Build WS notification and schedule broadcast
        notification = {
            "event": "osc_update",
            "address": address,
            "args": list(args),
            "timestamp": time.time(),
        }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.broadcast(notification), loop
                )
        except RuntimeError:
            logger.debug("No event loop for OSC broadcast: %s", address)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[str] = None,
    ) -> None:
        """Broadcast a message to all connected WS clients."""
        text = json.dumps(message, default=str)

        with self._stats_lock:
            self._stats["ws_messages_out"] += 1

        disconnected: List[str] = []
        with self._clients_lock:
            clients = dict(self._ws_clients)

        for client_id, client in clients.items():
            if client_id == exclude:
                continue
            try:
                await client.websocket.send(text)
            except Exception:
                disconnected.append(client_id)
                with self._stats_lock:
                    self._stats["broadcast_errors"] += 1

        for cid in disconnected:
            self.unregister_ws_client(cid)

    async def send_to_client(
        self, client_id: str, message: Dict[str, Any]
    ) -> bool:
        """Send a message to a specific WS client. Returns False on failure."""
        with self._clients_lock:
            client = self._ws_clients.get(client_id)
        if not client:
            return False
        try:
            await client.websocket.send(json.dumps(message, default=str))
            with self._stats_lock:
                self._stats["ws_messages_out"] += 1
            return True
        except Exception:
            self.unregister_ws_client(client_id)
            return False

    async def broadcast_to_subscribers(
        self, topic: str, message: Dict[str, Any]
    ) -> None:
        """Send a message only to clients subscribed to a given topic."""
        text = json.dumps(message, default=str)
        disconnected: List[str] = []

        with self._clients_lock:
            clients = dict(self._ws_clients)

        for client_id, client in clients.items():
            if topic in client.subscriptions:
                try:
                    await client.websocket.send(text)
                except Exception:
                    disconnected.append(client_id)

        for cid in disconnected:
            self.unregister_ws_client(cid)

    # ------------------------------------------------------------------
    # OSC pattern matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_osc_pattern(pattern: str, address: str) -> bool:
        """
        Match an OSC address against a pattern.

        Supports:
        - Exact match: '/ch/1/fdr' matches '/ch/1/fdr'
        - Wildcard '*': '/ch/*/fdr' matches '/ch/5/fdr'
        - Global wildcard: '*' matches everything
        """
        if pattern == address:
            return True
        if pattern == "*":
            return True
        if "*" not in pattern:
            return False

        # Convert glob-style pattern to regex
        regex_parts = []
        for part in pattern.split("*"):
            regex_parts.append(re.escape(part))
        regex_str = ".*".join(regex_parts)
        return bool(re.match(f"^{regex_str}$", address))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["connected_clients"] = self.client_count
        stats["registered_commands"] = list(self._ws_command_handlers.keys())
        stats["registered_osc_patterns"] = list(self._osc_handlers.keys())
        return stats

    def __repr__(self) -> str:
        return (
            f"<MessageRouter clients={self.client_count} "
            f"handlers={len(self._ws_command_handlers)}>"
        )
