"""
Enhanced OSC Client for Behringer Wing with auto-reconnect and state machine.

Wraps WingClient and adds:
- Connection state machine (DISCONNECTED → CONNECTING → CONNECTED → RECONNECTING)
- Automatic reconnection on connection loss
- Message send/receive statistics
- Structured logging of connection events
"""

import enum
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ConnectionState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class EnhancedOSCClient:
    """
    Enhanced OSC client wrapping WingClient with auto-reconnect.

    Drop-in replacement: exposes the same public API as WingClient
    so server.py can switch imports without changing call sites.
    """

    def __init__(self, ip: str = "192.168.1.102", port: int = 2223,
                 reconnect_interval: float = 5.0, max_reconnect_attempts: int = 0):
        """
        Args:
            ip: Wing mixer IP address.
            port: Wing OSC port (default 2223).
            reconnect_interval: Seconds between reconnect attempts.
            max_reconnect_attempts: 0 = unlimited retries.
        """
        # Lazy import to avoid circular deps
        from wing_client import WingClient

        self._wing = WingClient(ip=ip, port=port)
        self.ip = ip
        self.port = port

        # State machine
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        # Monotonic, process-private identity for the underlying transport.
        # Every successful initial connection or automatic reconnect changes
        # this value so authorities bound to an older socket cannot survive a
        # transparent reconnect performed inside this wrapper.
        self._transport_generation = 0
        self._manual_disconnect = False
        self._manual_disconnect_epoch = 0

        # Reconnect settings
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_reconnect = False

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "reconnect_count": 0,
            "last_connected": None,
            "last_disconnected": None,
            "connection_uptime_sec": 0.0,
        }
        self._connected_since: Optional[float] = None

        logger.info(f"EnhancedOSCClient created for {ip}:{port}")

    # ---- Properties delegated to WingClient ----

    @property
    def is_connected(self) -> bool:
        return self._wing.is_connected

    @is_connected.setter
    def is_connected(self, value: bool):
        self._wing.is_connected = value

    @property
    def state(self) -> dict:
        return self._wing.state

    @state.setter
    def state(self, value: dict):
        self._wing.state = value

    @property
    def callbacks(self) -> dict:
        return self._wing.callbacks

    @property
    def connection_state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    @property
    def transport_identity(self) -> tuple[ConnectionState, int]:
        """Atomically snapshot wrapper state and underlying transport generation."""
        with self._state_lock:
            return self._state, self._transport_generation

    @property
    def transport_generation(self) -> int:
        """Return the generation of the currently connected Wing transport."""
        with self._state_lock:
            return self._transport_generation

    # ---- Connection lifecycle ----

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to Wing with auto-reconnect enabled."""
        with self._state_lock:
            self._manual_disconnect = False
            self._state = ConnectionState.CONNECTING
            disconnect_epoch = self._manual_disconnect_epoch
        logger.info(f"Connecting to Wing at {self.ip}:{self.port}...")

        ok = self._wing.connect(timeout=timeout)
        if ok:
            if not self._on_connected(expected_disconnect_epoch=disconnect_epoch):
                self._wing.disconnect()
                return False
        else:
            with self._state_lock:
                self._state = ConnectionState.DISCONNECTED
            self._start_reconnect_loop()
        return ok

    def disconnect(self):
        """Disconnect and stop reconnect loop."""
        with self._state_lock:
            self._manual_disconnect = True
            self._manual_disconnect_epoch += 1
            self._stop_reconnect = True
            if self._state != ConnectionState.DISCONNECTED:
                self._transport_generation += 1
            self._state = ConnectionState.DISCONNECTED
        self._wing.disconnect()
        reconnect_thread = self._reconnect_thread
        if (
            reconnect_thread is not None
            and reconnect_thread is not threading.current_thread()
            and reconnect_thread.is_alive()
        ):
            reconnect_thread.join(timeout=5.5)
            if reconnect_thread.is_alive():
                logger.error("EnhancedOSCClient reconnect thread did not stop")
        self._stats["last_disconnected"] = time.time()
        if self._connected_since:
            self._stats["connection_uptime_sec"] += time.time() - self._connected_since
            self._connected_since = None
        logger.info("EnhancedOSCClient disconnected")

    # ---- Send / subscribe (delegated) ----

    def send(self, address: str, *values) -> bool:
        result = self._wing.send(address, *values)
        if result:
            self._stats["messages_sent"] += 1
        elif not self._wing.is_connected:
            self._handle_disconnect()
        return result

    def subscribe(self, address_pattern: str, callback: Callable):
        self._wing.subscribe(address_pattern, callback)

    def unsubscribe(self, address_pattern: str, callback: Callable) -> bool:
        return self._wing.unsubscribe(address_pattern, callback)

    def install_mutation_boundary(
        self,
        boundary: Callable[[str, tuple[Any, ...]], bool],
    ) -> object:
        return self._wing.install_mutation_boundary(boundary)

    def remove_mutation_boundary(self, token: object) -> bool:
        return self._wing.remove_mutation_boundary(token)

    def mutation_boundary_matches(self, token: object) -> bool:
        return self._wing.mutation_boundary_matches(token)

    def query_parameter(self, address: str, timeout: float = 0.5) -> Dict[str, Any]:
        """Query through the wrapper so send failures trigger reconnect state."""
        return self._wing.query_parameter(
            address,
            timeout,
            send_callable=self.send,
        )

    def set_osc_throttle(self, enabled: bool = True, hz: float = 10.0):
        self._wing.set_osc_throttle(enabled, hz)

    # ---- Delegate all WingClient public methods ----

    def __getattr__(self, name: str):
        """Forward attribute access to the underlying WingClient."""
        return getattr(self._wing, name)

    # ---- Auto-reconnect machinery ----

    def _on_connected(self, *, expected_disconnect_epoch: Optional[int] = None) -> bool:
        with self._state_lock:
            if self._manual_disconnect or (
                expected_disconnect_epoch is not None
                and expected_disconnect_epoch != self._manual_disconnect_epoch
            ):
                return False
            self._state = ConnectionState.CONNECTED
            self._transport_generation += 1
        self._connected_since = time.time()
        self._stats["last_connected"] = self._connected_since
        self._stop_reconnect = True
        logger.info("EnhancedOSCClient connected")
        return True

    def _handle_disconnect(self):
        with self._state_lock:
            if self._manual_disconnect or self._state in {
                ConnectionState.DISCONNECTED,
                ConnectionState.RECONNECTING,
            }:
                return
            self._state = ConnectionState.RECONNECTING
            # Invalidate authorities immediately, before WingClient's next
            # connect() can expose is_connected=True during its setup window.
            self._transport_generation += 1
        self._stats["last_disconnected"] = time.time()
        if self._connected_since:
            self._stats["connection_uptime_sec"] += time.time() - self._connected_since
            self._connected_since = None
        logger.warning("Connection lost — starting reconnect loop")
        self._start_reconnect_loop()

    def _start_reconnect_loop(self):
        with self._state_lock:
            if self._manual_disconnect:
                return
            if self._reconnect_thread and self._reconnect_thread.is_alive():
                return
            self._stop_reconnect = False
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        attempt = 0
        while not self._stop_reconnect:
            with self._state_lock:
                if self._manual_disconnect:
                    break
                disconnect_epoch = self._manual_disconnect_epoch
            attempt += 1
            if self._max_reconnect_attempts and attempt > self._max_reconnect_attempts:
                logger.error(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached")
                break
            logger.info(f"Reconnect attempt {attempt}...")
            time.sleep(self._reconnect_interval)
            if self._stop_reconnect:
                break
            try:
                self._wing.disconnect()
                ok = self._wing.connect(timeout=5.0)
                if ok:
                    if not self._on_connected(
                        expected_disconnect_epoch=disconnect_epoch,
                    ):
                        self._wing.disconnect()
                        return
                    self._stats["reconnect_count"] += 1
                    return
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")

    # ---- Statistics ----

    def get_stats(self) -> Dict[str, Any]:
        uptime = self._stats["connection_uptime_sec"]
        if self._connected_since:
            uptime += time.time() - self._connected_since
        return {
            **self._stats,
            "connection_state": self.connection_state.value,
            "current_uptime_sec": uptime,
        }
