"""
OSC Send/Receive Manager for Behringer Wing Rack

Provides a high-level OSC transport layer with:
- Thread-safe outbound message queue with configurable rate limiting
- Subscription-based inbound message routing with pattern matching
- Connection health monitoring via periodic keepalive
- Batched send support for efficient parameter updates

Uses python-osc (pythonosc) for message serialization.
"""

import fnmatch
import logging
import queue
import re
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from pythonosc.osc_message_builder import OscMessageBuilder
    from pythonosc.osc_message import OscMessage
except ImportError:
    raise ImportError(
        "python-osc is required: pip install python-osc"
    )

logger = logging.getLogger(__name__)


class OSCSubscription:
    """A single subscription binding an address pattern to a callback."""

    __slots__ = ("pattern", "callback", "regex")

    def __init__(self, pattern: str, callback: Callable[..., Any]):
        self.pattern = pattern
        self.callback = callback
        # Pre-compile a regex from the glob-style pattern for fast matching.
        # Supports '*' (any segment) and '?' (single char), standard OSC style.
        escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
        self.regex = re.compile(f"^{escaped}$")

    def matches(self, address: str) -> bool:
        return self.regex.match(address) is not None


class OSCManager:
    """
    Manages bidirectional OSC communication with a Behringer Wing mixer.

    Wraps low-level UDP send/receive into a clean interface with:
    - Queued, rate-limited sending from any thread
    - Pattern-based subscription for incoming messages
    - Background keepalive to maintain /xremote subscription
    - Connection health tracking with configurable timeout
    """

    # Wing protocol constants
    WING_DISCOVERY_PORT = 2222
    WING_OSC_PORT = 2223
    XREMOTE_INTERVAL_SEC = 8.0
    HEALTH_CHECK_INTERVAL_SEC = 5.0
    DEFAULT_RATE_HZ = 50.0  # Max outbound messages per second

    def __init__(
        self,
        ip: str = "192.168.1.102",
        port: int = WING_OSC_PORT,
        rate_limit_hz: float = DEFAULT_RATE_HZ,
        health_timeout_sec: float = 15.0,
    ):
        """
        Args:
            ip: Wing mixer IP address.
            port: Wing OSC port (default 2223).
            rate_limit_hz: Maximum outbound message rate in Hz. 0 = unlimited.
            health_timeout_sec: Seconds without a received message before
                                the connection is considered unhealthy.
        """
        self.ip = ip
        self.port = port
        self.rate_limit_hz = rate_limit_hz
        self.health_timeout_sec = health_timeout_sec

        # UDP socket
        self._sock: Optional[socket.socket] = None
        self._local_port: int = 0

        # Connection state
        self._connected = False
        self._last_rx_time: float = 0.0
        self._last_tx_time: float = 0.0

        # Outbound queue (thread-safe)
        self._send_queue: queue.Queue[Tuple[str, Tuple[Any, ...]]] = queue.Queue(
            maxsize=4096
        )

        # Per-address throttle tracking
        self._per_address_last_send: Dict[str, float] = {}
        self._per_address_lock = threading.Lock()

        # Subscriptions
        self._subscriptions: List[OSCSubscription] = []
        self._subscriptions_lock = threading.Lock()

        # Global listeners (receive every message regardless of address)
        self._global_listeners: List[Callable[..., Any]] = []
        self._global_lock = threading.Lock()

        # Background threads
        self._stop_event = threading.Event()
        self._sender_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "queue_overflows": 0,
            "health_timeouts": 0,
        }

        # Health callbacks
        self._on_connected: Optional[Callable[[], None]] = None
        self._on_disconnected: Optional[Callable[[], None]] = None

        logger.info(
            "OSCManager initialized for %s:%d (rate_limit=%.0f Hz, "
            "health_timeout=%.1f s)",
            ip, port, rate_limit_hz, health_timeout_sec,
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to Wing mixer: perform discovery handshake, then start
        background send/receive/keepalive threads.

        Args:
            timeout: Seconds to wait for the Wing discovery response.

        Returns:
            True on success, False on failure.
        """
        if self._connected:
            logger.warning("Already connected")
            return True

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(timeout)
            self._sock.bind(("0.0.0.0", 0))
            self._local_port = self._sock.getsockname()[1]
            logger.info("Bound to local UDP port %d", self._local_port)

            # Wing discovery: send 'WING?' to port 2222
            logger.info("Sending WING? discovery to %s:%d", self.ip, self.WING_DISCOVERY_PORT)
            self._sock.sendto(b"WING?", (self.ip, self.WING_DISCOVERY_PORT))

            try:
                data, addr = self._sock.recvfrom(4096)
                info = data.decode("utf-8", errors="ignore")
                logger.info("Wing discovery response: %s", info)
            except socket.timeout:
                logger.error(
                    "Wing at %s did not respond to discovery within %.1f s",
                    self.ip, timeout,
                )
                self._sock.close()
                self._sock = None
                return False

            # Verify OSC works by querying ch/1/fdr
            builder = OscMessageBuilder(address="/ch/1/fdr")
            msg = builder.build()
            self._sock.sendto(msg.dgram, (self.ip, self.port))

            try:
                data, _ = self._sock.recvfrom(4096)
                osc_msg = OscMessage(data)
                logger.info("OSC handshake OK — /ch/1/fdr = %s", osc_msg.params)
            except socket.timeout:
                logger.error("OSC handshake failed on port %d", self.port)
                self._sock.close()
                self._sock = None
                return False

            # Mark as connected
            self._connected = True
            self._last_rx_time = time.monotonic()
            self._last_tx_time = time.monotonic()
            self._stop_event.clear()

            # Start background threads
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop, name="osc-rx", daemon=True
            )
            self._sender_thread = threading.Thread(
                target=self._sender_loop, name="osc-tx", daemon=True
            )
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, name="osc-keepalive", daemon=True
            )
            self._health_thread = threading.Thread(
                target=self._health_loop, name="osc-health", daemon=True
            )

            self._receiver_thread.start()
            self._sender_thread.start()
            self._keepalive_thread.start()
            self._health_thread.start()

            # Subscribe to Wing updates
            self._send_immediate("/xremote")
            logger.info("Connected to Wing at %s:%d", self.ip, self.port)

            if self._on_connected:
                try:
                    self._on_connected()
                except Exception as exc:
                    logger.error("on_connected callback error: %s", exc)

            return True

        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            self._connected = False
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            return False

    def disconnect(self) -> None:
        """Gracefully shut down all threads and close the socket."""
        if not self._connected:
            return

        logger.info("Disconnecting from Wing at %s:%d", self.ip, self.port)
        self._connected = False
        self._stop_event.set()

        # Drain the send queue so the sender thread can exit
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for threads (short timeout to avoid blocking)
        for t in (self._sender_thread, self._receiver_thread,
                  self._keepalive_thread, self._health_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)

        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        if self._on_disconnected:
            try:
                self._on_disconnected()
            except Exception as exc:
                logger.error("on_disconnected callback error: %s", exc)

        logger.info("Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send(self, address: str, *args: Any) -> bool:
        """
        Enqueue an OSC message for sending.  Thread-safe.

        Args:
            address: OSC address string (e.g. '/ch/1/fdr').
            *args: OSC arguments (float, int, str, bytes).

        Returns:
            True if queued, False if the queue is full or not connected.
        """
        if not self._connected:
            logger.debug("send() called while disconnected: %s", address)
            return False

        try:
            self._send_queue.put_nowait((address, args))
            return True
        except queue.Full:
            with self._stats_lock:
                self._stats["queue_overflows"] += 1
            logger.warning("Send queue full — dropping %s", address)
            return False

    def send_batch(self, messages: List[Tuple[str, Tuple[Any, ...]]]) -> int:
        """
        Enqueue multiple OSC messages at once.

        Args:
            messages: List of (address, args_tuple) pairs.

        Returns:
            Number of messages successfully queued.
        """
        queued = 0
        for address, args in messages:
            if self.send(address, *args):
                queued += 1
        return queued

    def query(self, address: str) -> bool:
        """
        Send an OSC query (message with no arguments).

        Args:
            address: OSC address to query.

        Returns:
            True if queued successfully.
        """
        return self.send(address)

    # ------------------------------------------------------------------
    # Receiving / subscriptions
    # ------------------------------------------------------------------

    def subscribe(self, pattern: str, callback: Callable[..., Any]) -> None:
        """
        Register a callback for incoming messages matching *pattern*.

        The pattern uses glob-style matching:
        - '/ch/*/fdr' matches any channel fader
        - '/ch/1/*' matches all parameters of channel 1
        - '/ch/1/fdr' matches exactly that address

        The callback signature is: callback(address: str, *args)

        Args:
            pattern: Glob-style OSC address pattern.
            callback: Function to call when a matching message arrives.
        """
        sub = OSCSubscription(pattern, callback)
        with self._subscriptions_lock:
            self._subscriptions.append(sub)
        logger.debug("Subscribed to pattern '%s'", pattern)

    def unsubscribe(self, pattern: str, callback: Callable[..., Any]) -> bool:
        """
        Remove a previously registered subscription.

        Args:
            pattern: The exact pattern string used in subscribe().
            callback: The exact callback reference used in subscribe().

        Returns:
            True if a matching subscription was removed.
        """
        with self._subscriptions_lock:
            before = len(self._subscriptions)
            self._subscriptions = [
                s for s in self._subscriptions
                if not (s.pattern == pattern and s.callback is callback)
            ]
            removed = before - len(self._subscriptions)
        if removed:
            logger.debug("Unsubscribed from '%s'", pattern)
        return removed > 0

    def add_global_listener(self, callback: Callable[..., Any]) -> None:
        """Register a callback that receives every incoming OSC message."""
        with self._global_lock:
            self._global_listeners.append(callback)

    def remove_global_listener(self, callback: Callable[..., Any]) -> bool:
        """Remove a global listener. Returns True if found and removed."""
        with self._global_lock:
            before = len(self._global_listeners)
            self._global_listeners = [
                c for c in self._global_listeners if c is not callback
            ]
            return len(self._global_listeners) < before

    # ------------------------------------------------------------------
    # Health callbacks
    # ------------------------------------------------------------------

    def on_connected(self, callback: Callable[[], None]) -> None:
        """Set a callback invoked when the connection is established."""
        self._on_connected = callback

    def on_disconnected(self, callback: Callable[[], None]) -> None:
        """Set a callback invoked when the connection is lost."""
        self._on_disconnected = callback

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of send/receive statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
        stats["queue_size"] = self._send_queue.qsize()
        stats["connected"] = self._connected
        stats["subscriptions"] = len(self._subscriptions)
        elapsed = time.monotonic() - self._last_rx_time if self._last_rx_time else 0
        stats["seconds_since_last_rx"] = round(elapsed, 2)
        return stats

    # ------------------------------------------------------------------
    # Internal: sender loop (runs in background thread)
    # ------------------------------------------------------------------

    def _sender_loop(self) -> None:
        """Drain the outbound queue, respecting the global rate limit."""
        min_interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0.0

        while not self._stop_event.is_set():
            try:
                address, args = self._send_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Per-address throttle: skip if too soon for this address (only
            # for commands with values, not queries)
            if args and min_interval > 0:
                now = time.monotonic()
                with self._per_address_lock:
                    last = self._per_address_last_send.get(address, 0.0)
                    if (now - last) < min_interval:
                        with self._stats_lock:
                            self._stats["messages_dropped"] += 1
                        continue
                    self._per_address_last_send[address] = now

            self._send_immediate(address, *args)

    def _send_immediate(self, address: str, *args: Any) -> bool:
        """
        Build and send an OSC message on the wire immediately.
        Only call from the sender thread or during connect handshake.
        """
        if not self._sock:
            return False

        try:
            builder = OscMessageBuilder(address=address)
            for v in args:
                if isinstance(v, float):
                    builder.add_arg(v)
                elif isinstance(v, int):
                    builder.add_arg(v)
                elif isinstance(v, str):
                    builder.add_arg(v)
                elif isinstance(v, bytes):
                    builder.add_arg(v)
                else:
                    builder.add_arg(v)
            msg = builder.build()
            self._sock.sendto(msg.dgram, (self.ip, self.port))
            self._last_tx_time = time.monotonic()
            with self._stats_lock:
                self._stats["messages_sent"] += 1
            logger.debug("TX: %s %s", address, args)
            return True
        except OSError as exc:
            if exc.errno in (9, 32, 57):  # Bad fd / Broken pipe / Not connected
                logger.warning("Socket error during send: %s", exc)
                self._connected = False
            else:
                logger.error("Send OSError: %s", exc)
            return False
        except Exception as exc:
            logger.error("Send failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal: receiver loop
    # ------------------------------------------------------------------

    def _receiver_loop(self) -> None:
        """Continuously read incoming UDP datagrams and dispatch them."""
        if not self._sock:
            return

        try:
            self._sock.settimeout(0.5)
        except OSError:
            return

        while not self._stop_event.is_set() and self._connected:
            if not self._sock:
                break
            try:
                data, addr = self._sock.recvfrom(4096)
                self._last_rx_time = time.monotonic()

                try:
                    osc_msg = OscMessage(data)
                except Exception:
                    logger.debug("Failed to parse incoming OSC datagram")
                    continue

                with self._stats_lock:
                    self._stats["messages_received"] += 1

                self._dispatch(osc_msg.address, *osc_msg.params)

            except socket.timeout:
                continue
            except OSError as exc:
                if exc.errno in (9, 57):
                    break
                if self._connected:
                    logger.debug("Receiver OS error: %s", exc)
            except Exception as exc:
                if self._connected:
                    logger.debug("Receiver error: %s", exc)

    def _dispatch(self, address: str, *args: Any) -> None:
        """Route an incoming message to matching subscriptions and globals."""
        # Pattern-matched subscriptions
        with self._subscriptions_lock:
            subs = list(self._subscriptions)
        for sub in subs:
            if sub.matches(address):
                try:
                    sub.callback(address, *args)
                except Exception as exc:
                    logger.error(
                        "Subscription callback error for '%s' on %s: %s",
                        sub.pattern, address, exc,
                    )

        # Global listeners
        with self._global_lock:
            globals_copy = list(self._global_listeners)
        for listener in globals_copy:
            try:
                listener(address, *args)
            except Exception as exc:
                logger.error("Global listener error on %s: %s", address, exc)

    # ------------------------------------------------------------------
    # Internal: keepalive (renew /xremote subscription)
    # ------------------------------------------------------------------

    def _keepalive_loop(self) -> None:
        """Periodically send /xremote to keep the Wing subscription alive."""
        while not self._stop_event.is_set() and self._connected:
            self._stop_event.wait(timeout=self.XREMOTE_INTERVAL_SEC)
            if self._connected and not self._stop_event.is_set():
                self._send_immediate("/xremote")
                logger.debug("Renewed /xremote subscription")

    # ------------------------------------------------------------------
    # Internal: health monitoring
    # ------------------------------------------------------------------

    def _health_loop(self) -> None:
        """Check connection health based on time since last received message."""
        was_healthy = True

        while not self._stop_event.is_set() and self._connected:
            self._stop_event.wait(timeout=self.HEALTH_CHECK_INTERVAL_SEC)
            if self._stop_event.is_set() or not self._connected:
                break

            elapsed = time.monotonic() - self._last_rx_time
            healthy = elapsed < self.health_timeout_sec

            if not healthy and was_healthy:
                with self._stats_lock:
                    self._stats["health_timeouts"] += 1
                logger.warning(
                    "Connection unhealthy — no messages for %.1f s "
                    "(threshold %.1f s)",
                    elapsed, self.health_timeout_sec,
                )
                # Attempt to prod the mixer
                self._send_immediate("/xremote")
                self._send_immediate("/ch/1/fdr")

            elif healthy and not was_healthy:
                logger.info("Connection health restored")

            was_healthy = healthy

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "OSCManager":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        state = "connected" if self._connected else "disconnected"
        return f"<OSCManager {self.ip}:{self.port} [{state}]>"
