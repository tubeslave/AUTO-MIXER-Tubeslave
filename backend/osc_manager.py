"""
OSC send/receive management extracted from server.py.

Centralizes all OSC communication with the Wing mixer, including
connection management, message routing, and state synchronization.
"""

import asyncio
import logging
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import fnmatch

logger = logging.getLogger(__name__)

try:
    from pythonosc import udp_client, dispatcher, osc_server
    from pythonosc.osc_message_builder import OscMessageBuilder
    HAS_OSC = True
except ImportError:
    HAS_OSC = False
    logger.warning("python-osc not installed, OSC features disabled")


@dataclass
class OSCMessage:
    """Represents an OSC message."""
    address: str
    args: Tuple
    timestamp: float = 0.0
    source: str = ""


@dataclass
class OSCSubscription:
    """Pattern subscription wrapper kept for compatibility with older tests/code."""
    pattern: str
    callback: Callable

    def matches(self, address: str) -> bool:
        return fnmatch.fnmatchcase(address, self.pattern)


class OSCManager:
    """
    Manages all OSC communication with the Wing mixer.

    Handles:
    - Connection lifecycle (connect, disconnect, reconnect)
    - Sending OSC commands with throttling
    - Receiving and dispatching OSC messages
    - State cache synchronization
    - Subscription management (/xremote)
    """

    def __init__(
        self,
        ip: str = "192.168.1.102",
        send_port: int = 2223,
        recv_port: int = 2223,
        throttle_hz: float = 10.0,
        port: Optional[int] = None,
        rate_limit_hz: Optional[float] = None,
        health_timeout_sec: float = 10.0,
    ):
        if port is not None:
            send_port = port
            recv_port = port
        if rate_limit_hz is not None:
            throttle_hz = rate_limit_hz

        self.ip = ip
        self.port = send_port
        self.send_port = send_port
        self.recv_port = recv_port
        self.throttle_hz = throttle_hz
        self.rate_limit_hz = throttle_hz
        self.health_timeout_sec = health_timeout_sec

        self._state: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._subscriptions: List[OSCSubscription] = []
        self._global_listeners: List[Callable] = []
        self._throttle_times: Dict[str, float] = {}
        self._connected = False
        self._lock = threading.Lock()
        self._send_count = 0
        self._recv_count = 0
        self._wing_client = None
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    def connect(self, wing_client) -> bool:
        """
        Initialize with an existing WingClient instance.

        Args:
            wing_client: Connected WingClient
        Returns:
            True if setup successful
        """
        self._wing_client = wing_client
        self._connected = wing_client.is_connected
        if self._connected:
            self._state = wing_client.state
            logger.info(f"OSCManager connected via WingClient to {self.ip}")
            if self._on_connected:
                self._on_connected()
        return self._connected

    def disconnect(self):
        """Disconnect from the mixer."""
        self._connected = False
        self._wing_client = None
        if self._on_disconnected:
            self._on_disconnected()
        logger.info("OSCManager disconnected")

    def send(self, address: str, *values, force: bool = False) -> bool:
        """
        Send an OSC message with optional throttling.

        Args:
            address: OSC address
            *values: Message values
            force: Bypass throttle

        Returns:
            True if sent
        """
        if not self._connected or not self._wing_client:
            return False

        # Throttle check
        if not force and values and self.throttle_hz > 0:
            now = time.time()
            min_interval = 1.0 / self.throttle_hz
            last = self._throttle_times.get(address, 0)
            if now - last < min_interval:
                return False
            self._throttle_times[address] = now

        result = self._wing_client.send(address, *values)
        if result:
            self._send_count += 1
            if values:
                with self._lock:
                    self._state[address] = values[0] if len(values) == 1 else values
        return result

    def query(self, address: str) -> Any:
        """Send a query (no value) and return cached state."""
        if not self._connected or not self._wing_client:
            return False
        self._wing_client.send(address)
        return self._state.get(address, False)

    def subscribe(self, address_pattern: str, callback: Callable):
        """Subscribe to state changes matching a pattern."""
        if address_pattern not in self._callbacks:
            self._callbacks[address_pattern] = []
        self._callbacks[address_pattern].append(callback)
        self._subscriptions.append(OSCSubscription(address_pattern, callback))

    def unsubscribe(self, address_pattern: str, callback: Callable):
        """Remove a callback subscription."""
        removed = False
        if address_pattern in self._callbacks:
            callbacks = self._callbacks[address_pattern]
            new_callbacks = [cb for cb in callbacks if cb != callback]
            removed = len(new_callbacks) != len(callbacks)
            if new_callbacks:
                self._callbacks[address_pattern] = new_callbacks
            else:
                self._callbacks.pop(address_pattern, None)
        if removed:
            self._subscriptions = [
                sub for sub in self._subscriptions
                if not (sub.pattern == address_pattern and sub.callback == callback)
            ]
        return removed

    def get_state(self, address: str, default: Any = None) -> Any:
        """Get cached state for an address."""
        return self._state.get(address, default)

    def set_channel_fader(self, channel: int, value_db: float):
        """Set channel fader in dB."""
        value_db = max(-144.0, min(10.0, float(value_db)))
        self.send(f"/ch/{channel}/fdr", value_db)

    def set_channel_mute(self, channel: int, muted: int):
        """Set channel mute."""
        self.send(f"/ch/{channel}/mute", muted)

    def set_channel_pan(self, channel: int, pan: float):
        """Set channel pan (-100 to 100)."""
        self.send(f"/ch/{channel}/pan", pan)

    def set_eq_band(self, channel: int, band: int, freq: float = None,
                    gain: float = None, q: float = None):
        """Set EQ band parameters."""
        if freq is not None:
            self.send(f"/ch/{channel}/eq/{band}f", freq)
        if gain is not None:
            self.send(f"/ch/{channel}/eq/{band}g", gain)
        if q is not None:
            self.send(f"/ch/{channel}/eq/{band}q", q)

    def get_stats(self) -> Dict:
        """Get communication statistics."""
        return {
            "connected": self._connected,
            "ip": self.ip,
            "port": self.port,
            "messages_sent": self._send_count,
            "messages_received": self._recv_count,
            "cached_params": len(self._state),
            "subscriptions": len(self._subscriptions),
            "queue_size": 0,
        }

    def send_batch(self, messages: List[Tuple[str, Tuple]]) -> int:
        """Send multiple OSC messages and report how many were accepted."""
        sent = 0
        for address, args in messages:
            if self.send(address, *(args or ())):
                sent += 1
        return sent

    def add_global_listener(self, callback: Callable):
        self._global_listeners.append(callback)

    def remove_global_listener(self, callback: Callable) -> bool:
        before = len(self._global_listeners)
        self._global_listeners = [cb for cb in self._global_listeners if cb != callback]
        return len(self._global_listeners) != before

    def on_connected(self, callback: Callable):
        self._on_connected = callback

    def on_disconnected(self, callback: Callable):
        self._on_disconnected = callback

    def _dispatch(self, address: str, *args):
        """Dispatch an inbound OSC update to subscribers and global listeners."""
        self._recv_count += 1
        self._state[address] = args[0] if len(args) == 1 else args

        for sub in list(self._subscriptions):
            if sub.matches(address):
                try:
                    sub.callback(address, *args)
                except Exception as exc:
                    logger.debug("OSC subscriber callback failed: %s", exc)

        for callback in list(self._global_listeners):
            try:
                callback(address, *args)
            except Exception as exc:
                logger.debug("OSC global listener failed: %s", exc)

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"OSCManager(ip={self.ip!r}, port={self.port}, status={status})"
