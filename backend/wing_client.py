from pythonosc import udp_client, dispatcher, osc_server
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
import socket
import threading
import logging
import os
import time
from typing import Callable, Dict, Any, Optional

from mixer_client_base import MixerClientBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WING compressor ratio — дискретный enum. Невалидные токены ("10.0", "4.3")
# пульт отвергает МОЛЧА (задокументированный паттерн отказа), поэтому
# конверсия к ближайшему валидному токену обязана жить на транспортной границе.
WING_RATIO_STRINGS = ["1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", "3.0",
                      "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"]
WING_RATIO_VALUES = [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0,
                     3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0]


class _WingPhysicalReadbackReceipt(dict):
    """Process-local proof that the canonical query path observed two replies.

    A plain mapping is intentionally insufficient: server-side write proof must
    remain bound to the exact ``WingClient`` instance that performed the query.
    """

    __slots__ = ("_client_token", "_issued_monotonic", "_consumed")

    def __init__(self, payload: Dict[str, Any], *, client_token: object):
        super().__init__(payload)
        self._client_token = client_token
        self._issued_monotonic = time.monotonic()
        self._consumed = False


def verify_wing_physical_readback_receipt(
    client: object,
    receipt: object,
    address: str,
    *,
    maximum_age_sec: float = 2.0,
) -> bool:
    """Reject OSC observations as authority-bearing physical proof.

    WING OSC has no authenticated peer identity or request-bound response
    nonce. A UDP source tuple, even with two fresh observations and a private
    in-process wrapper, is forgeable by another LAN peer and therefore cannot
    authorize or verify a physical mutation.
    """

    return False


def _verify_legacy_wing_observation_receipt(
    client: object,
    receipt: object,
    address: str,
    *,
    maximum_age_sec: float = 2.0,
) -> bool:
    """Validate a process-local observation receipt without granting authority."""

    transport = None
    if type(client) is WingClient:
        query = getattr(client, "query_parameter", None)
        if getattr(query, "__func__", None) is not WingClient.query_parameter:
            return False
        transport = client
    else:
        # Imported lazily to avoid the module cycle used by the reconnecting
        # wrapper. Exact types prevent duck-typed in-process clients from
        # minting authority-bearing physical proof.
        try:
            from osc.enhanced_osc_client import EnhancedOSCClient
        except ImportError:
            return False
        if type(client) is not EnhancedOSCClient:
            return False
        query = getattr(client, "query_parameter", None)
        if getattr(query, "__func__", None) is not EnhancedOSCClient.query_parameter:
            return False
        transport = getattr(client, "_wing", None)
        if type(transport) is not WingClient:
            return False

    if (
        not getattr(transport, "is_connected", False)
        or getattr(transport, "sock", None) is None
        or type(receipt) is not _WingPhysicalReadbackReceipt
        or receipt._client_token is not getattr(
            transport,
            "_physical_readback_receipt_token",
            None,
        )
        or receipt._consumed
        or receipt.get("status") != "received"
        or receipt.get("address") != address
        or receipt.get("confirmed_round_trips") != 2
        or receipt.get("peer_ip")
        != getattr(transport, "_configured_peer_ip", None)
        or receipt.get("peer_port") != getattr(transport, "port", None)
    ):
        return False
    sequences = receipt.get("receive_sequences")
    if (
        not isinstance(sequences, tuple)
        or len(sequences) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in sequences)
        or sequences[0] >= sequences[1]
    ):
        return False
    age = time.monotonic() - receipt._issued_monotonic
    if age < 0.0 or age > max(0.0, float(maximum_age_sec)):
        return False
    receipt._consumed = True
    return True


def ratio_to_wing_token(ratio) -> str:
    """Nearest valid WING ratio token for a float/str ratio."""
    if isinstance(ratio, str) and ratio in WING_RATIO_STRINGS:
        return ratio
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return "1.1"
    if value <= 0:
        return "1.1"
    best_idx = min(range(len(WING_RATIO_VALUES)),
                   key=lambda i: abs(WING_RATIO_VALUES[i] - value))
    return WING_RATIO_STRINGS[best_idx]


class WingClient(MixerClientBase):
    """OSC client for Behringer Wing mixer - uses correct Wing OSC address format"""
    
    def __init__(self, ip: str = "192.168.1.102", port: int = 2223, *, fader_ceiling_db: Optional[float] = None):
        """
        Initialize Wing client

        Args:
            ip: Wing mixer IP address
            port: Wing OSC port (default 2223)
            fader_ceiling_db: software fader ceiling in dB (default 0.0).
                Resolution order: this argument, then the
                AUTOMIXER_FADER_CEILING_DB env var, then 0.0. Writes above
                the ceiling are clamped with a warning; the hardware range
                (-144..+10 dB) still applies after the ceiling.
        """
        self.ip = ip
        self.port = port
        self._configured_peer_ip = str(ip)

        if fader_ceiling_db is None:
            try:
                fader_ceiling_db = float(os.getenv("AUTOMIXER_FADER_CEILING_DB", "") or 0.0)
            except ValueError:
                fader_ceiling_db = 0.0
        # Defence-in-depth: fader > 0 dBFS requires an explicit operator-level
        # opt-in, even if an upper layer fails to clamp (CLAUDE.md rule).
        self._fader_ceiling_db = max(-144.0, min(10.0, float(fader_ceiling_db)))
        
        self.sock = None
        self.receiver_thread = None
        self._stop_receiver = False
        
        self.state = {}
        self.state_timestamps = {}
        self._state_observation_lock = threading.Lock()
        self._state_change_generation = 0
        self._state_value_generations: Dict[str, int] = {}
        self.commanded_state = {}
        self.command_timestamps = {}
        self.callbacks = {}
        self._callbacks_lock = threading.RLock()
        # A process-local mutation boundary can temporarily narrow every
        # value-bearing OSC send, including sends attempted by unrelated
        # server tasks.  Zero-argument reads remain available.  M110 uses this
        # only while its finite maintenance transaction (and any rollback) is
        # active; it is not standing live authority.
        self._mutation_boundary_lock = threading.RLock()
        self._mutation_boundary: Optional[Callable[[str, tuple[Any, ...]], bool]] = None
        self._mutation_boundary_token: object | None = None
        # Only this exact transport instance can mint receipts accepted by the
        # supervised write proof boundary. The token never enters JSON output.
        self._physical_readback_receipt_token = object()
        self._physical_readback_context = threading.local()
        self._physical_receive_lock = threading.Lock()
        self._physical_receive_sequence = 0
        
        self.is_connected = False
        self.handshake_info: Optional[str] = None
        
        # OSC throttle: track last send time per address to limit rate
        self._osc_throttle_enabled = True
        self._osc_throttle_hz = 10.0  # Default: 10 Hz = 100ms minimum interval
        self._osc_last_send_time: Dict[str, float] = {}  # address -> last send time
        self._osc_send_lock = threading.Lock()
        # Trailing-edge буфер троттла: address -> (values, lane, due_time)
        self._osc_pending: Dict[str, tuple] = {}
        self._osc_flush_wakeup = threading.Event()
        self._osc_flusher_thread: Optional[threading.Thread] = None
        self._osc_flusher_stop = False
        self._teardown_failed = False
        
        logger.info(f"WingClient initialized for {ip}:{port}")

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "on", "yes"}
        return bool(value)

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_dca_assignments(tags: Any) -> list:
        """Parse WING DCA assignment tags (#D1..#D16) from a tags field."""
        assignments = set()

        def visit(item: Any):
            if item is None:
                return
            if isinstance(item, (list, tuple, set)):
                for nested in item:
                    visit(nested)
                return
            text = str(item).replace(",", " ").replace(";", " ")
            for token in text.split():
                token = token.strip().upper()
                if not token.startswith("#D"):
                    continue
                suffix = token[2:]
                if suffix.isdigit():
                    dca = int(suffix)
                    if 1 <= dca <= 16:
                        assignments.add(dca)

        visit(tags)
        return sorted(assignments)

    @classmethod
    def _is_active_send(cls, send: Dict[str, Any]) -> bool:
        if not cls._coerce_bool(send.get("on")):
            return False
        level = cls._coerce_float(send.get("level_db"), -144.0)
        return level > -143.0

    def _normalize_fx_slot(self, fx_slot: str) -> tuple[str, str]:
        """Return canonical FX slot label and numeric suffix."""
        slot = str(fx_slot).upper()
        if slot.startswith("FX"):
            fx_num = slot[2:]
            return slot, fx_num
        return f"FX{slot}", slot

    def _insert_base(self, target_type: str, target: int, position: str) -> str:
        """Build OSC base path for pre/post inserts on supported target types."""
        type_map = {
            "channel": "ch",
            "ch": "ch",
            "aux": "aux",
            "bus": "bus",
            "main": "main",
            "matrix": "mtx",
            "mtx": "mtx",
        }
        pos_map = {
            "pre": "preins",
            "post": "postins",
        }
        target_key = str(target_type).lower()
        position_key = str(position).lower()
        if target_key not in type_map:
            raise ValueError(f"Unsupported insert target type: {target_type}")
        if position_key not in pos_map:
            raise ValueError(f"Unsupported insert position: {position}")
        return f"/{type_map[target_key]}/{int(target)}/{pos_map[position_key]}"

    def _set_state_value(self, address: str, value: Any) -> None:
        """Record only state received from the physical console."""
        with self._state_observation_lock:
            previous_present = address in self.state
            previous = self.state.get(address)
            self.state[address] = value
            self.state_timestamps[address] = time.time()
            if not previous_present or previous != value:
                self._state_change_generation += 1
                self._state_value_generations[address] = (
                    self._state_change_generation
                )

    def get_observed_state_revision(
        self,
        addresses: Any = None,
    ) -> Dict[str, Any]:
        """Atomically expose value-change generations for readiness fences."""

        requested = [
            str(address)
            for address in (addresses or self.state.keys())
            if str(address).strip()
        ]
        with self._state_observation_lock:
            return {
                "generation": int(self._state_change_generation),
                "addresses": {
                    address: {
                        "generation": int(
                            self._state_value_generations.get(address, 0)
                        ),
                        "value": self.state.get(address),
                        "observed_at_epoch": self.state_timestamps.get(address),
                    }
                    for address in requested
                },
                "captured_at_epoch": time.time(),
            }

    def _set_commanded_value(self, address: str, value: Any) -> None:
        """Record transport intent separately from observed console state."""
        self.commanded_state[address] = value
        self.command_timestamps[address] = time.time()

    def _active_readback(self, address: str, ttl: float = 0.4,
                         timeout: float = 0.15) -> Any:
        """Return a surface parameter, refreshing it from the mixer when stale.

        Some parameters (pan, eq/*, in/set/trim) are NOT pushed by the
        ``/xremote`` subscription, so a passive ``self.state`` read returns
        ``None``/stale — which silently breaks readback verification of applied
        corrections. When connected and the cached value is older than ``ttl``
        seconds, send a bare GET and wait up to ``timeout`` for the physical
        readback (early-exiting the moment it lands) before returning the cached
        value. Repeated reads within ``ttl`` hit the cache, so hot callers are
        not slowed. GET messages carry no value and bypass the write throttle.
        Mirrors the active-query pattern already used by
        :meth:`get_gate_threshold` / :meth:`get_compressor_threshold`.
        """
        if getattr(self, "is_connected", False):
            prev_ts = self.state_timestamps.get(address, 0.0)
            if (time.time() - prev_ts) > ttl:
                self.send(address)
                deadline = time.time() + max(0.0, float(timeout))
                while time.time() < deadline:
                    if self.state_timestamps.get(address, 0.0) > prev_ts:
                        break
                    time.sleep(0.005)
        return self.state.get(address)

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to Wing mixer and validate connection
        
        Per Wing Remote Protocols documentation:
        1. Send 'WING?' to port 2222 to initiate connection
        2. Then use OSC on port 2223
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully, False otherwise
        """
        if self._teardown_failed:
            logger.error("Wing connection refused after unverified transport teardown")
            return False
        try:
            self._configured_peer_ip = socket.gethostbyname(self.ip)
            # Create UDP socket for bidirectional communication
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(timeout)
            self.sock.bind(('0.0.0.0', 0))  # Bind to any available port
            
            local_port = self.sock.getsockname()[1]
            logger.info(f"Bound to local port {local_port}")
            
            # Step 1: Initiate connection by sending 'WING?' to port 2222
            # Per documentation: "Initiating a communication with WING starts with 
            # sending the 5 bytes [UDP] datagram 'WING?' to port 2222"
            logger.info(f"Initiating connection to Wing at {self.ip}...")
            self.sock.sendto(b'WING?', (self.ip, 2222))
            
            try:
                data, addr = self.sock.recvfrom(4096)
                if addr != (self._configured_peer_ip, 2222):
                    raise ConnectionError("unexpected_wing_handshake_peer")
                wing_info = data.decode('utf-8', errors='ignore')
                self.handshake_info = wing_info
                logger.info(f"Wing found: {wing_info}")
            except socket.timeout:
                logger.error(f"Connection timeout: Wing at {self.ip} did not respond to handshake")
                self._close_socket_verified()
                return False
            
            # Step 2: Verify OSC is working by querying a parameter
            builder = OscMessageBuilder(address='/ch/1/fdr')
            msg = builder.build()
            self.sock.sendto(msg.dgram, (self.ip, self.port))
            
            try:
                data, addr = self.sock.recvfrom(4096)
                if addr != (self._configured_peer_ip, self.port):
                    raise ConnectionError("unexpected_wing_osc_peer")
                osc_msg = OscMessage(data)
                logger.info(f"OSC verified, ch1 fader: {osc_msg.params}")
            except socket.timeout:
                logger.error(f"OSC timeout: Wing not responding on port {self.port}")
                self._close_socket_verified()
                return False
            
            self.is_connected = True
            
            # Start receiver thread
            self._stop_receiver = False
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()
            
            # Subscribe to updates
            self._subscribe_to_updates()
            
            # Scan initial state
            self._scan_console_state()
            
            logger.info(f"Connected to Wing at {self.ip}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            if self.sock is not None:
                self._close_socket_verified()
            return False

    def _close_socket_verified(self) -> bool:
        sock = self.sock
        self.sock = None
        if sock is None:
            return not self._teardown_failed
        try:
            sock.close()
        except Exception as exc:
            self._teardown_failed = True
            logger.error("Error closing Wing socket: %s", exc)
            return False
        return not self._teardown_failed

    def disconnect(self) -> bool:
        """Disconnect from Wing and return verified teardown status."""
        self._stop_receiver = True
        self._osc_flusher_stop = True
        self._osc_flush_wakeup.set()
        with self._osc_send_lock:
            if self._osc_pending:
                logger.warning(
                    f"Disconnect drops {len(self._osc_pending)} pending "
                    f"trailing-edge writes: {sorted(self._osc_pending)}"
                )
                self._osc_pending.clear()
        self.is_connected = False
        teardown_ok = self._close_socket_verified()
        for attr_name in ("receiver_thread", "_osc_flusher_thread"):
            thread = getattr(self, attr_name, None)
            if (
                thread is not None
                and thread is not threading.current_thread()
                and thread.is_alive()
            ):
                thread.join(timeout=2.0)
                if thread.is_alive():
                    teardown_ok = False
                    self._teardown_failed = True
                    logger.error("Wing transport thread did not stop: %s", attr_name)
                else:
                    setattr(self, attr_name, None)
        if teardown_ok:
            logger.info("Disconnected from Wing")
        return teardown_ok

    def _receiver_loop(self):
        """Background thread to receive OSC messages from Wing"""
        if not self.sock:
            return
        try:
            self.sock.settimeout(0.5)
        except Exception:
            return
            
        while not self._stop_receiver and self.is_connected:
            if not self.sock:
                break
            try:
                data, addr = self.sock.recvfrom(4096)
                try:
                    osc_msg = OscMessage(data)
                    self._handle_transport_message(
                        osc_msg.address,
                        tuple(osc_msg.params),
                        addr,
                    )
                except Exception as e:
                    logger.debug(f"Error parsing OSC message: {e}")
            except socket.timeout:
                pass
            except OSError as e:
                # Socket was closed
                if e.errno in (9, 57):  # Bad file descriptor or Socket not connected
                    break
                if self.is_connected:
                    logger.debug(f"Receiver OS error: {e}")
            except Exception as e:
                if self.is_connected:
                    logger.debug(f"Receiver error: {e}")

    def _handle_transport_message(
        self,
        address: str,
        args: tuple[Any, ...],
        peer: tuple[Any, ...],
    ) -> bool:
        """Accept one OSC message only from the configured WING UDP peer."""

        if (
            not isinstance(peer, tuple)
            or len(peer) < 2
            or str(peer[0]) != self._configured_peer_ip
            or isinstance(peer[1], bool)
            or not isinstance(peer[1], int)
            or peer[1] != self.port
        ):
            logger.debug("Ignored OSC datagram from unexpected peer: %r", peer)
            return False
        with self._physical_receive_lock:
            self._physical_receive_sequence += 1
            sequence = self._physical_receive_sequence
        self._physical_readback_context.current = {
            "peer_ip": str(peer[0]),
            "peer_port": int(peer[1]),
            "receive_sequence": sequence,
        }
        try:
            self._handle_message(address, *args)
        finally:
            self._physical_readback_context.current = None
        return True

    def _handle_message(self, address: str, *args):
        """Handle incoming OSC message from Wing"""
        # Log responses for name-related addresses (including $name)
        if '/name' in address or '/$name' in address:
            logger.info(f"Name response: {address} = {args}")
        else:
            logger.debug(f"Received: {address} {args}")
        
        # Wing returns [display_string, normalized_value, actual_value]
        # For string values like names, use the first argument
        if ('/name' in address or '/$name' in address) and args:
            # $name returns the actual name string
            self._set_state_value(address, args[0] if args else '')
        elif len(args) >= 3:
            self._set_state_value(address, args[2])  # Actual value
        elif len(args) == 1:
            self._set_state_value(address, args[0])
        else:
            self._set_state_value(address, args)
        
        # Call registered callbacks
        with self._callbacks_lock:
            address_callbacks = tuple(self.callbacks.get(address, ()))
            wildcard_callbacks = tuple(self.callbacks.get("*", ()))
        for callback in address_callbacks:
            try:
                callback(address, *args)
            except Exception as e:
                logger.error(f"Callback error for {address}: {e}")
        for callback in wildcard_callbacks:
            try:
                callback(address, *args)
            except Exception as e:
                logger.error(f"Global callback error: {e}")

    def set_osc_throttle(self, enabled: bool = True, hz: float = 10.0):
        """
        Configure OSC throttle to limit command rate.
        
        Args:
            enabled: Enable/disable throttle
            hz: Maximum frequency in Hz (default 10 Hz = 100ms minimum interval)
        """
        self._osc_throttle_enabled = enabled
        self._osc_throttle_hz = hz
        logger.info(f"OSC throttle: {'enabled' if enabled else 'disabled'}, {hz} Hz")
    
    # Приоритетные полосы троттла: аварийная запись не ждёт косметику
    _OSC_PRIORITY_LANES = {"emergency": 0, "operator": 1, "correction": 2}

    def send(self, address: str, *values, priority: str = "correction"):
        """
        Send OSC message to Wing.

        Trailing-edge throttle (совет 2026-07-29, DEC-0003): запись в «горячий»
        адрес (чаще ~10 Гц) НЕ блокирует вызывающий поток и НЕ теряется —
        она коалесцируется в per-address буфере (последнее значение побеждает)
        и отправляется фоновым flush-таймером, когда адрес остынет. Записи в
        разные адреса независимы: head-of-line блокировки больше нет.

        Args:
            address: OSC address (e.g., '/ch/1/fdr')
            values: Optional values to send. If empty, this is a query.
            priority: "emergency" | "operator" | "correction" — порядок
                flush при одновременной готовности нескольких адресов.

        Returns True когда запись отправлена ЛИБО принята trailing-edge
        буфером с гарантией доставки последнего значения (проверяется
        readback-слоем executor'а как и раньше).
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        if not self.sock:
            logger.warning("Socket not available")
            self.is_connected = False
            return False

        lane = self._OSC_PRIORITY_LANES.get(priority, 2)

        try:
            with self._osc_send_lock:
                if self._osc_throttle_enabled and values and self._osc_throttle_hz > 0:
                    min_interval = 1.0 / self._osc_throttle_hz
                    last_send = self._osc_last_send_time.get(address)
                    now = time.monotonic()
                    if last_send is not None and (now - last_send) < min_interval:
                        # Горячий адрес: ранний отказ границы — честный False
                        # сейчас; принятое значение доставит flush-таймер.
                        if not self._check_mutation_boundary(address, values):
                            return False
                        due = last_send + min_interval
                        self._osc_pending[address] = (tuple(values), lane, due)
                        self._ensure_flusher_running()
                        self._osc_flush_wakeup.set()
                        logger.debug(
                            f"Queued (trailing-edge): {address} {values} lane={priority}"
                        )
                        return True
                return self._send_now_locked(address, values)
        except OSError as e:
            # Socket error - mark as disconnected
            if e.errno in (9, 57, 32):  # Bad file descriptor, Socket not connected, Broken pipe
                logger.warning(f"Socket error, connection lost: {e}")
                self.is_connected = False
            else:
                logger.error(f"Send OSError: {e}")
            return False
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def _check_mutation_boundary(self, address: str, values: tuple) -> bool:
        """Authority-bearing preflight; runs at accept-time AND at flush-time
        (linearization point для отложенных записей)."""
        if not values:
            return True
        with self._mutation_boundary_lock:
            boundary = self._mutation_boundary
            if boundary is None:
                logger.warning(
                    "Blocked WING OSC mutation without an installed authority boundary: %s",
                    address,
                )
                return False
            try:
                allowed = boundary(address, tuple(values))
            except Exception as exc:
                logger.error("Mutation boundary rejected %s: %s", address, exc)
                return False
            if allowed is not True:
                logger.warning("Mutation boundary blocked OSC write: %s", address)
                return False
            return True

    def _send_now_locked(self, address: str, values: tuple) -> bool:
        """Собрать и отправить датаграмму. Вызывать под _osc_send_lock."""
        if not self._check_mutation_boundary(address, values):
            return False
        builder = OscMessageBuilder(address=address)
        for v in values:
            builder.add_arg(v)
        msg = builder.build()
        self.sock.sendto(msg.dgram, (self.ip, self.port))
        if values:
            self._osc_last_send_time[address] = time.monotonic()
            commanded = values[0] if len(values) == 1 else values
            self._set_commanded_value(address, commanded)
        logger.debug(f"Sent: {address} {values}")
        return True

    def _ensure_flusher_running(self):
        if self._osc_flusher_thread is not None and self._osc_flusher_thread.is_alive():
            return
        self._osc_flusher_stop = False
        self._osc_flusher_thread = threading.Thread(
            target=self._osc_flush_loop, name="wing-osc-flush", daemon=True
        )
        self._osc_flusher_thread.start()

    def _osc_flush_loop(self):
        """Фоновый flush trailing-edge буфера: готовые адреса отправляются
        в порядке приоритетной полосы, затем по времени готовности."""
        while not self._osc_flusher_stop:
            with self._osc_send_lock:
                if not self._osc_pending:
                    self._osc_flush_wakeup.clear()
                    wait_for = None
                else:
                    now = time.monotonic()
                    ready = sorted(
                        (
                            (lane, due, addr, vals)
                            for addr, (vals, lane, due) in self._osc_pending.items()
                            if due <= now
                        ),
                    )
                    for lane, due, addr, vals in ready:
                        self._osc_pending.pop(addr, None)
                        if not self.is_connected or not self.sock:
                            logger.warning(
                                f"Trailing-edge flush dropped (disconnected): {addr}"
                            )
                            continue
                        try:
                            self._send_now_locked(addr, vals)
                        except OSError as e:
                            if e.errno in (9, 57, 32):
                                self.is_connected = False
                                logger.warning(
                                    f"Flush socket error, connection lost: {e}"
                                )
                            else:
                                logger.error(f"Flush OSError for {addr}: {e}")
                        except Exception as e:
                            logger.error(f"Flush failed for {addr}: {e}")
                    if self._osc_pending:
                        wait_for = max(
                            0.0,
                            min(d for (_v, _l, d) in self._osc_pending.values())
                            - time.monotonic(),
                        )
                    else:
                        self._osc_flush_wakeup.clear()
                        wait_for = None
            if wait_for is None:
                self._osc_flush_wakeup.wait(timeout=1.0)
            else:
                time.sleep(min(wait_for, 0.05) if wait_for > 0 else 0.0)

    def flush_pending_writes(self, timeout: float = 2.0) -> bool:
        """Дождаться опустошения trailing-edge буфера (для тестов/шатдауна)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._osc_send_lock:
                if not self._osc_pending:
                    return True
            time.sleep(0.005)
        return False

    def install_mutation_boundary(
        self,
        boundary: Callable[[str, tuple[Any, ...]], bool],
    ) -> object:
        """Install one exclusive process-local guard for value-bearing sends."""

        if not callable(boundary):
            raise TypeError("mutation_boundary_callable_required")
        with self._osc_send_lock:
            with self._mutation_boundary_lock:
                if self._mutation_boundary is not None:
                    raise RuntimeError("mutation_boundary_already_active")
                token = object()
                self._mutation_boundary = boundary
                self._mutation_boundary_token = token
                return token

    def remove_mutation_boundary(self, token: object) -> bool:
        """Remove exactly the boundary installed with ``token``."""

        with self._osc_send_lock:
            with self._mutation_boundary_lock:
                if (
                    self._mutation_boundary is None
                    or self._mutation_boundary_token is not token
                ):
                    return False
                self._mutation_boundary = None
                self._mutation_boundary_token = None
                return True

    def mutation_boundary_matches(self, token: object) -> bool:
        """Return whether ``token`` still owns the installed send boundary."""

        with self._osc_send_lock:
            with self._mutation_boundary_lock:
                return bool(
                    self._mutation_boundary is not None
                    and self._mutation_boundary_token is token
                )

    def query_parameter(
        self,
        address: str,
        timeout: float = 0.5,
        *,
        send_callable: Optional[Callable[..., bool]] = None,
    ) -> Dict[str, Any]:
        """Query one OSC address and wait briefly for physical readback.

        The caller decides whether a timeout can fall back to the state cache.
        OSC replies have no request identifier, so one queued update cannot prove
        that the console answered the current query. The first request drains any
        delayed matching packet; a second request must receive another reply before
        the readback is reported as fresh.
        """

        event = threading.Event()
        captured: Dict[str, Any] = {
            "value": None,
            "args": None,
            "transport": None,
        }
        confirmations: list[Dict[str, Any]] = []

        def callback(_address: str, *args: Any) -> None:
            captured["args"] = args
            if len(args) >= 3:
                captured["value"] = args[2]
            elif len(args) == 1:
                captured["value"] = args[0]
            else:
                captured["value"] = self.state.get(address)
            context = getattr(self._physical_readback_context, "current", None)
            captured["transport"] = dict(context) if isinstance(context, dict) else None
            event.set()

        self.subscribe(address, callback)
        try:
            for phase in ("drain", "confirm"):
                event.clear()
                captured["value"] = None
                captured["args"] = None
                captured["transport"] = None
                send = send_callable or self.send
                if not send(address):
                    return {
                        "status": "send_failed",
                        "address": address,
                        "value": None,
                        "phase": phase,
                    }
                if not event.wait(max(0.0, float(timeout))):
                    return {
                        "status": "timeout",
                        "address": address,
                        "value": None,
                        "phase": phase,
                    }
                transport = captured.get("transport")
                if not isinstance(transport, dict):
                    return {
                        "status": "unverified_transport",
                        "address": address,
                        "value": None,
                        "phase": phase,
                    }
                confirmations.append(dict(transport))
            return _WingPhysicalReadbackReceipt(
                {
                    "status": "received",
                    "authoritative": False,
                    "authentication": "none",
                    "provenance": "unauthenticated_udp_observation",
                    "address": address,
                    "value": captured["value"],
                    "args": captured["args"],
                    "confirmed_round_trips": 2,
                    "peer_ip": confirmations[-1]["peer_ip"],
                    "peer_port": confirmations[-1]["peer_port"],
                    "receive_sequences": tuple(
                        item["receive_sequence"] for item in confirmations
                    ),
                },
                client_token=self._physical_readback_receipt_token,
            )
        finally:
            self.unsubscribe(address, callback)

    def subscribe(self, address_pattern: str, callback: Callable):
        """Subscribe to OSC address changes"""
        if not isinstance(address_pattern, str) or not address_pattern:
            raise ValueError("address_pattern_required")
        if not callable(callback):
            raise TypeError("callback_must_be_callable")
        with self._callbacks_lock:
            callbacks = self.callbacks.setdefault(address_pattern, [])
            if callback not in callbacks:
                callbacks.append(callback)

    def unsubscribe(self, address_pattern: str, callback: Callable) -> bool:
        """Atomically remove one callback without disturbing other observers."""

        with self._callbacks_lock:
            callbacks = self.callbacks.get(address_pattern)
            if not callbacks or callback not in callbacks:
                return False
            callbacks.remove(callback)
            if not callbacks:
                self.callbacks.pop(address_pattern, None)
            return True

    def _subscribe_to_updates(self):
        """Subscribe to Wing updates"""
        self.send("/xremote")
        logger.info("Subscribed to Wing updates via /xremote")
        
        # Start periodic renewal
        self._start_xremote_renewal()
    
    def _start_xremote_renewal(self):
        """Periodically send /xremote to maintain subscription"""
        def renewal_loop():
            while self.is_connected and not self._stop_receiver:
                time.sleep(8)
                if self.is_connected:
                    self.send("/xremote")
                    logger.debug("Renewed /xremote subscription")
        
        thread = threading.Thread(target=renewal_loop, daemon=True)
        thread.start()

    def _scan_console_state(self):
        """Scan initial Wing state"""
        logger.info("Scanning Wing console state...")
        
        # Scan first 8 channels (basic scan)
        for ch in range(1, 9):
            self.send(f"/ch/{ch}/fdr")      # Fader
            self.send(f"/ch/{ch}/mute")     # Mute
            self.send(f"/ch/{ch}/$name")    # Source/linked name
            self.send(f"/ch/{ch}/name")     # Strip name fallback
            time.sleep(0.02)
        
        # Scan mains
        for m in range(1, 5):
            self.send(f"/main/{m}/fdr")
        
        time.sleep(0.3)
        
        logger.info(f"Initial state scan complete, received {len(self.state)} parameters")
        
        # Now scan all channels with inserts in background
        threading.Thread(target=self._scan_all_channels_with_inserts, daemon=True).start()
    
    def _scan_all_channels_with_inserts(self):
        """Scan all channels (1-40) with their inserts, FX modules, and names"""
        logger.info("Starting full channel scan with inserts...")
        
        # First, scan all channel names (important for voice control)
        logger.info("Scanning channel names...")
        for ch in range(1, 41):
            self.send(f"/ch/{ch}/$name")
            self.send(f"/ch/{ch}/name")
            self.send(f"/ch/{ch}/fdr")
            self.send(f"/ch/{ch}/mute")
            time.sleep(0.02)
        
        time.sleep(0.3)  # Wait for name responses
        
        # Log found channel names
        found_names = []
        for ch in range(1, 41):
            name = self.state.get(f"/ch/{ch}/$name") or self.state.get(f"/ch/{ch}/name")
            if name:
                found_names.append(f"Ch{ch}: {name}")
        if found_names:
            logger.info(f"Found channel names: {', '.join(found_names)}")
        
        # Scan all channels for inserts
        for ch in range(1, 41):
            self._scan_channel_inserts(ch)
            time.sleep(0.05)  # Small delay between channels
        
        # Log summary
        channels_with_inserts = self.get_all_channels_with_inserts()
        logger.info(f"Full channel scan complete. Total parameters: {len(self.state)}")
        logger.info(f"Found {len(channels_with_inserts)} channels with inserts:")
        for ch_num, ch_info in channels_with_inserts.items():
            inserts_info = []
            if ch_info['inserts']['pre_insert']:
                fx = ch_info['inserts']['pre_insert']['fx_module']
                inserts_info.append(f"Pre: {fx['slot']} ({fx.get('model', 'N/A')})")
            if ch_info['inserts']['post_insert']:
                fx = ch_info['inserts']['post_insert']['fx_module']
                inserts_info.append(f"Post: {fx['slot']} ({fx.get('model', 'N/A')})")
            logger.info(f"  Channel {ch_num}: {', '.join(inserts_info)}")
    
    def _scan_channel_inserts(self, channel: int):
        """Scan a single channel's insert information and FX modules"""
        ch = channel
        
        # Query insert information
        self.send(f"/ch/{ch}/preins/on")
        self.send(f"/ch/{ch}/preins/ins")
        self.send(f"/ch/{ch}/preins/$stat")
        self.send(f"/ch/{ch}/postins/on")
        self.send(f"/ch/{ch}/postins/mode")
        self.send(f"/ch/{ch}/postins/ins")
        self.send(f"/ch/{ch}/postins/$stat")
        time.sleep(0.1)  # Wait for responses
        
        # Check if there are inserts and scan FX modules
        preins_slot = self.state.get(f'/ch/{ch}/preins/ins')
        postins_slot = self.state.get(f'/ch/{ch}/postins/ins')
        
        fx_slots = []
        if preins_slot and preins_slot != 'NONE':
            fx_slots.append(preins_slot)
        if postins_slot and postins_slot != 'NONE':
            fx_slots.append(postins_slot)
        
        # Scan FX modules if present
        for fx_slot in fx_slots:
            self._scan_fx_module(fx_slot)
    
    def _scan_fx_module(self, fx_slot: str):
        """Scan an FX module to get its model and parameters"""
        _, fx_num = self._normalize_fx_slot(fx_slot)

        # Official FW 3.1 FX slot metadata has model and fxmix, not /fx/N/on.
        self.send(f"/fx/{fx_num}/mdl")
        self.send(f"/fx/{fx_num}/fxmix")
        self.send(f"/fx/{fx_num}/")  # Root node
        time.sleep(0.05)
        
        # Query numbered parameters (WING uses numbered parameters for FX)
        # Query first 32 parameters (most FX modules have parameters 1-32)
        for i in range(1, 33):
            self.send(f"/fx/{fx_num}/{i}")
            if i % 8 == 0:  # Small delay every 8 parameters
                time.sleep(0.02)
        
        time.sleep(0.1)  # Final wait for responses
        
        fx_model = self.state.get(f'/fx/{fx_num}/mdl')
        if fx_model:
            logger.debug(f"Scanned {fx_slot}: {fx_model}")

    # ========== Channel Methods ==========
    
    def get_channel_fader(self, channel: int) -> Optional[float]:
        """Get channel fader value"""
        address = f"/ch/{channel}/fdr"
        return self.state.get(address)

    def set_channel_fader(self, channel: int, value: float):
        """
        Set channel fader value.

        According to Wing Remote Protocols v3.0.5 documentation:
        /ch/{ch}/fdr accepts dB values in range -144..10

        This method accepts dB values directly and sends them to the mixer.
        Values above the software fader ceiling (self._fader_ceiling_db,
        default 0.0 dBFS) are clamped with a warning — defence-in-depth so
        no programmatic caller can raise a live channel above unity without
        an explicit operator-level opt-in (AUTOMIXER_FADER_CEILING_DB).
        """
        address = f"/ch/{channel}/fdr"
        db_value = self._clamp_fader_db(address, value)
        logger.info(f"Setting fader /ch/{channel}/fdr = {db_value:.2f} dB")
        return self.send(address, db_value)

    def _clamp_fader_db(self, address: str, value: float) -> float:
        """Clamp any level-type write (fader/send lvl) to hardware range and
        the software fader ceiling. Ceiling applies to EVERY level surface
        (ch/bus/dca/main/fx-return/sends): a DCA or bus above unity defeats
        the per-channel ceiling (defence-in-depth, council 2026-07-29)."""
        db_value = max(-144.0, min(10.0, float(value)))
        if db_value > self._fader_ceiling_db:
            logger.warning(
                f"Fader ceiling enforced: {address} requested "
                f"{db_value:.2f} dB > ceiling {self._fader_ceiling_db:.2f} dB; "
                f"clamping (operator opt-in: AUTOMIXER_FADER_CEILING_DB)"
            )
            db_value = self._fader_ceiling_db
        return db_value

    def get_channel_mute(self, channel: int) -> Optional[int]:
        """Get channel mute state"""
        address = f"/ch/{channel}/mute"
        return self.state.get(address)

    def set_channel_mute(self, channel: int, value: int):
        """Set channel mute (0=unmuted, 1=muted)"""
        address = f"/ch/{channel}/mute"
        return self.send(address, value)

    def get_channel_gain(self, channel: int) -> Optional[float]:
        """Get channel input trim/gain (active readback; not streamed)."""
        return self._active_readback(f"/ch/{channel}/in/set/trim")

    def set_channel_gain(self, channel: int, value: float):
        """Set channel input trim/gain (dB, range: -18..18)"""
        address = f"/ch/{channel}/in/set/trim"
        # Ensure value is float and within valid range
        trim_value = float(value)
        trim_value = max(-18.0, min(18.0, trim_value))  # Clamp to valid range
        logger.info(f"Setting TRIM for channel {channel} to {trim_value} dB (address: {address})")
        result = self.send(address, trim_value)
        if result:
            logger.info(f"TRIM command sent successfully for channel {channel}")
        else:
            logger.error(f"Failed to send TRIM command for channel {channel}")
        return result
    
    def set_channel_balance(self, channel: int, value: float):
        """Set channel input balance (dB, range: -9..9)"""
        return self.send(f"/ch/{channel}/in/set/bal", value)
    
    def set_channel_phase_invert(self, channel: int, value: int):
        """Set channel phase invert (0=normal, 1=inverted)"""
        return self.send(f"/ch/{channel}/in/set/inv", value)
    
    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        """
        Set channel input delay
        
        Args:
            channel: Channel number (1-40)
            value: Delay value (depends on mode)
            mode: Delay mode - "M" (meters 0..150), "FT" (feet 0.5..500), 
                  "MS" (milliseconds 0.5..500), "SMP" (samples 16..500)
        """
        results = [
            self.send(f"/ch/{channel}/in/set/dlymode", mode),
            self.send(f"/ch/{channel}/in/set/dly", value),
            self.send(f"/ch/{channel}/in/set/dlyon", 1),
        ]
        return all(results)

    def get_channel_pan(self, channel: int) -> Optional[float]:
        """Get channel pan (active readback; /xremote does not stream pan)."""
        return self._active_readback(f"/ch/{channel}/pan")

    def set_channel_pan(self, channel: int, value: float):
        """Set channel pan (-100=left, 0=center, 100=right)"""
        address = f"/ch/{channel}/pan"
        return self.send(address, value)
    
    # ========== EQ Methods ==========
    
    def set_eq_on(self, channel: int, on: int):
        """Set EQ on/off (0=off, 1=on)"""
        return self.send(f"/ch/{channel}/eq/on", on)
    
    def get_eq_on(self, channel: int) -> Optional[int]:
        """Get EQ on/off state (active readback; not streamed)."""
        return self._active_readback(f"/ch/{channel}/eq/on")
    
    def set_eq_band_gain(self, channel: int, band: str, gain: float):
        """
        Set EQ band gain (dB, range: -15..15)
        
        Args:
            channel: Channel number (1-40)
            band: Band name - "lg" (low shelf), "hg" (high shelf), "1g", "2g", "3g", "4g" (parametric bands)
            gain: Gain in dB (-15 to 15)
        """
        return self.send(f"/ch/{channel}/eq/{band}", gain)
    
    def get_eq_band_gain(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band gain (active readback; not streamed)."""
        return self._active_readback(f"/ch/{channel}/eq/{band}")
    
    def set_eq_band_frequency(self, channel: int, band: str, frequency: float):
        """
        Set EQ band frequency (Hz)
        
        Args:
            channel: Channel number (1-40)
            band: Band name - "lf" (low shelf), "hf" (high shelf), "1f", "2f", "3f", "4f"
            frequency: Frequency in Hz (20-20000)
        """
        return self.send(f"/ch/{channel}/eq/{band}", frequency)
    
    def get_eq_band_frequency(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band frequency (active readback; not streamed)."""
        return self._active_readback(f"/ch/{channel}/eq/{band}")
    
    def set_eq_band_q(self, channel: int, band: str, q: float):
        """
        Set EQ band Q (0.44-10)
        
        Args:
            channel: Channel number (1-40)
            band: Band name - "lq" (low shelf), "hq" (high shelf), "1q", "2q", "3q", "4q"
            q: Q value (0.44-10)
        """
        return self.send(f"/ch/{channel}/eq/{band}", q)
    
    # ========== Compressor (Dynamics) Methods ==========
    
    def set_compressor_on(self, channel: int, on: int):
        """Set compressor on/off (0=off, 1=on)"""
        return self.send(f"/ch/{channel}/dyn/on", on)
    
    def get_compressor_on(self, channel: int) -> Optional[int]:
        """Get compressor on/off state"""
        return self.state.get(f"/ch/{channel}/dyn/on")
    
    def set_compressor_threshold(self, channel: int, threshold: float):
        """Set compressor threshold (dB, range: -60..0).

        Адрес порога зависит от загруженной модели динамики: стандартный
        COMP использует /dyn/thr, ряд эмуляций — /dyn/thr26. Пишем в оба
        адреса: несуществующий пульт молча игнорирует (подтверждено на
        реальном WING: запись только в thr26 при модели COMP теряется).
        """
        results = [
            self.send(f"/ch/{channel}/dyn/thr", threshold),
            self.send(f"/ch/{channel}/dyn/thr26", threshold),
        ]
        return all(results)
    
    def get_compressor_threshold(self, channel: int) -> Optional[float]:
        """Get compressor threshold (свежий запрос + чтение state).

        Порядок предпочтения — thr, затем thr26: при модели COMP thr26
        может держать устаревшее значение (запись в него молча теряется).
        """
        if self.is_connected:
            self.send(f"/ch/{channel}/dyn/thr")
            self.send(f"/ch/{channel}/dyn/thr26")
            time.sleep(0.05)
        value = self.state.get(f"/ch/{channel}/dyn/thr")
        if value is not None:
            return value
        return self.state.get(f"/ch/{channel}/dyn/thr26")

    def get_gate_threshold(self, channel: int) -> Optional[float]:
        """Get gate threshold (свежий запрос + чтение state)."""
        if self.is_connected:
            self.send(f"/ch/{channel}/gate/thr")
            time.sleep(0.05)
        return self.state.get(f"/ch/{channel}/gate/thr")

    def get_hpf(self, channel: int) -> Optional[dict]:
        """Get low-cut (HPF) state: {'freq_hz': ..., 'enabled': ...}."""
        if self.is_connected:
            self.send(f"/ch/{channel}/flt/lcf")
            self.send(f"/ch/{channel}/flt/lc")
            time.sleep(0.05)
        freq = self.state.get(f"/ch/{channel}/flt/lcf")
        if freq is None:
            return None
        return {"freq_hz": float(freq),
                "enabled": bool(self.state.get(f"/ch/{channel}/flt/lc") or 0)}
    
    def set_compressor_ratio(self, channel: int, ratio: str):
        """
        Set compressor ratio
        
        Args:
            channel: Channel number (1-40)
            ratio: Ratio string - "1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", "3.0",
                   "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"
        """
        return self.send(f"/ch/{channel}/dyn/ratio", ratio_to_wing_token(ratio))
    
    def get_compressor_ratio(self, channel: int) -> Optional[str]:
        """Get compressor ratio"""
        return self.state.get(f"/ch/{channel}/dyn/ratio")
    
    def set_compressor_gain(self, channel: int, gain: float):
        """Set compressor make-up gain (dB, range: -6..12)"""
        return self.send(f"/ch/{channel}/dyn/gain", gain)
    
    def get_compressor_gain(self, channel: int) -> Optional[float]:
        """Get compressor make-up gain (свежий запрос + чтение state)"""
        if self.is_connected:
            self.send(f"/ch/{channel}/dyn/gain")
            time.sleep(0.05)
        return self.state.get(f"/ch/{channel}/dyn/gain")

    def get_compressor_gr(self, channel: int) -> Optional[float]:
        """
        Get current gain reduction (GR) from compressor meter if supported by mixer.
        Wing may expose this as read-only. Returns None if not available.
        """
        # Try common read-only meter addresses (WING protocol may use $gr or gr)
        v = self.state.get(f"/ch/{channel}/dyn/gr")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        v = self.state.get(f"/ch/{channel}/dyn/$gr")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return None

    def set_compressor_attack(self, channel: int, attack: float):
        """Set compressor attack (ms, range: 0..120)"""
        return self.send(f"/ch/{channel}/dyn/att", attack)
    
    def set_compressor_release(self, channel: int, release: float):
        """Set compressor release (ms, range: 4..4000)"""
        return self.send(f"/ch/{channel}/dyn/rel", release)
    
    def set_compressor_knee(self, channel: int, knee: float):
        """Set compressor knee (0-5)"""
        return self.send(f"/ch/{channel}/dyn/knee", knee)
    
    def set_compressor_mix(self, channel: int, mix: float):
        """Set compressor mix (%, range: 0..100)"""
        return self.send(f"/ch/{channel}/dyn/mix", mix)
    
    def set_channel_width(self, channel: int, value: float):
        """Set channel width (%) -150..150"""
        return self.send(f"/ch/{channel}/wid", value)
    
    def set_channel_name(self, channel: int, name: str):
        """Set channel name (max 16 chars)"""
        return self.send(f"/ch/{channel}/name", name)
    
    def get_channel_name(self, channel: int, retries: int = 3) -> Optional[str]:
        """Get channel name with retries
        Uses /ch/X/$name (read-only) which reflects linked source or current strip value
        """
        # Try $name first (read-only, reflects linked source)
        address_dollar = f"/ch/{channel}/$name"
        address_regular = f"/ch/{channel}/name"
        
        for attempt in range(retries):
            # Query $name (read-only address)
            self.send(address_dollar)
            time.sleep(0.15)
            
            name = self.state.get(address_dollar)
            if name and name != '':
                logger.debug(f"Channel {channel} $name: '{name}' (attempt {attempt + 1})")
                return name
        
        # Fallback to regular name
        self.send(address_regular)
        time.sleep(0.15)
        name = self.state.get(address_regular)
        if name:
            return name
        
        logger.debug(f"Channel {channel} name not found after {retries} attempts")
        return None
    
    def get_all_channel_names(self, num_channels: int = 40) -> Dict[int, str]:
        """
        Get names of all channels (batch query for speed)
        Uses /ch/X/$name which reflects linked source or current strip value
        
        Returns:
            Dict mapping channel number to name, e.g. {1: "Kick", 2: "Snare", ...}
        """
        # First, send all $name queries at once (read-only, reflects linked source)
        for ch in range(1, num_channels + 1):
            self.send(f"/ch/{ch}/$name")
            time.sleep(0.02)  # Small delay between sends
        
        # Wait for all responses
        time.sleep(0.5)
        
        # Now collect all names from state
        names = {}
        for ch in range(1, num_channels + 1):
            # Try $name first
            name = self.state.get(f"/ch/{ch}/$name")
            if not name or name == '':
                # Fallback to regular name
                name = self.state.get(f"/ch/{ch}/name")
            
            if name and name != '':
                names[ch] = name
                logger.debug(f"Channel {ch} name: '{name}'")
        
        logger.info(f"Retrieved {len(names)} channel names")
        return names
    
    def set_channel_color(self, channel: int, color: int):
        """Set channel color (1-12)"""
        return self.send(f"/ch/{channel}/col", color)
    
    # ========== Filter Methods ==========
    
    def set_low_cut(self, channel: int, enabled: int, frequency: float = None, slope: str = None):
        """
        Set low cut filter
        
        Args:
            channel: Channel number (1-40)
            enabled: 0=off, 1=on
            frequency: Frequency in Hz (20..2000)
            slope: Slope "6", "12", "18", or "24"
        """
        results = [self.send(f"/ch/{channel}/flt/lc", enabled)]
        if frequency is not None:
            results.append(self.send(f"/ch/{channel}/flt/lcf", frequency))
        if slope is not None:
            results.append(self.send(f"/ch/{channel}/flt/lcs", slope))
        return all(results)
    
    def set_high_cut(self, channel: int, enabled: int, frequency: float = None, slope: str = None):
        """
        Set high cut filter
        
        Args:
            channel: Channel number (1-40)
            enabled: 0=off, 1=on
            frequency: Frequency in Hz (50..20000)
            slope: Slope "6" or "12"
        """
        results = [self.send(f"/ch/{channel}/flt/hc", enabled)]
        if frequency is not None:
            results.append(self.send(f"/ch/{channel}/flt/hcf", frequency))
        if slope is not None:
            results.append(self.send(f"/ch/{channel}/flt/hcs", slope))
        return all(results)

    # ========== EQ Methods ==========
    
    def set_eq_on(self, channel: int, on: int):
        """Enable/disable EQ (0=off, 1=on)"""
        return self.send(f"/ch/{channel}/eq/on", on)
    
    def set_eq_model(self, channel: int, model: str):
        """Set EQ model: STD, SOUL, E88, E84, F110, PULSAR, MACH4"""
        return self.send(f"/ch/{channel}/eq/mdl", model)
    
    def set_eq_mix(self, channel: int, mix: float):
        """Set EQ mix (%) 0..125"""
        return self.send(f"/ch/{channel}/eq/mix", mix)
    
    def set_eq_low_shelf(self, channel: int, gain: float = None, freq: float = None, q: float = None, eq_type: str = None):
        """Set EQ low shelf parameters"""
        results = []
        if gain is not None:
            results.append(self.send(f"/ch/{channel}/eq/lg", gain))  # -15..15 dB
        if freq is not None:
            results.append(self.send(f"/ch/{channel}/eq/lf", freq))  # 20..2000 Hz
        if q is not None:
            results.append(self.send(f"/ch/{channel}/eq/lq", q))  # 0.44..10
        if eq_type is not None:
            results.append(self.send(f"/ch/{channel}/eq/leq", eq_type))  # PEQ, SHV
        return all(results) if results else True

    def set_eq_band(self, channel: int, band: int, freq: float = None, gain: float = None, q: float = None):
        """
        Set EQ band parameters (bands 1-4)
        
        Args:
            channel: Channel number (1-40)
            band: EQ band (1-4)
            freq: Frequency in Hz (20..20000)
            gain: Gain in dB (-15..15)
            q: Q factor (0.44..10)
        """
        if band < 1 or band > 4:
            raise ValueError("Band must be 1-4")
        results = []
        if freq is not None:
            results.append(self.send(f"/ch/{channel}/eq/{band}f", freq))
        if gain is not None:
            results.append(self.send(f"/ch/{channel}/eq/{band}g", gain))
        if q is not None:
            results.append(self.send(f"/ch/{channel}/eq/{band}q", q))
        return all(results) if results else True
    
    def set_eq_high_shelf(self, channel: int, gain: float = None, freq: float = None, q: float = None, eq_type: str = None):
        """Set EQ high shelf parameters"""
        results = []
        if gain is not None:
            results.append(self.send(f"/ch/{channel}/eq/hg", gain))  # -15..15 dB
        if freq is not None:
            results.append(self.send(f"/ch/{channel}/eq/hf", freq))  # 50..20000 Hz
        if q is not None:
            results.append(self.send(f"/ch/{channel}/eq/hq", q))  # 0.44..10
        if eq_type is not None:
            results.append(self.send(f"/ch/{channel}/eq/heq", eq_type))  # SHV, PEQ
        return all(results) if results else True

    # ========== Dynamics Methods ==========
    
    def set_compressor_on(self, channel: int, on: int):
        """Enable/disable compressor"""
        return self.send(f"/ch/{channel}/dyn/on", on)

    def set_compressor(self, channel: int, threshold: float = None, ratio: str = None,
                      knee: int = None, attack: float = None, hold: float = None,
                      release: float = None, gain: float = None, mix: float = None,
                      det: str = None, env: str = None, auto: int = None,
                      threshold_db: float = None, attack_ms: float = None,
                      release_ms: float = None, makeup_db: float = None,
                      enabled: bool = None):
        """
        Set compressor/dynamics parameters
        
        Args:
            channel: Channel number (1-40)
            threshold: Threshold in dB (-60..0)
            ratio: Ratio string: "1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", 
                   "3.0", "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"
            knee: Knee (0..5)
            attack: Attack time in ms (0..120)
            hold: Hold time in ms (1..200)
            release: Release time in ms (4..4000)
            gain: Make-up gain in dB (-6..12)
            mix: Mix percentage (0..100)
            det: Detection mode: "PEAK" or "RMS"
            env: Envelope: "LIN" or "LOG"
            auto: Auto switch (0/1)
        """
        base = f"/ch/{channel}/dyn"
        results = []
        if enabled is not None:
            results.append(self.send(f"{base}/on", 1 if enabled else 0))
            time.sleep(0.01)
        if threshold is None:
            threshold = threshold_db
        if attack is None:
            attack = attack_ms
        if release is None:
            release = release_ms
        if gain is None:
            gain = makeup_db
        if ratio is not None:
            ratio = ratio_to_wing_token(ratio)
        # Small delay between OSC commands to ensure mixer processes them
        if threshold is not None:
            # Адрес порога зависит от модели динамики (COMP -> thr,
            # эмуляции -> thr26) — пишем в оба, лишний молча игнорируется
            results.append(self.send(f"{base}/thr", threshold))
            time.sleep(0.01)
            results.append(self.send(f"{base}/thr26", threshold))
            time.sleep(0.01)
        if ratio is not None:
            results.append(self.send(f"{base}/ratio", ratio))
            time.sleep(0.01)
        if knee is not None:
            results.append(self.send(f"{base}/knee", knee))
            time.sleep(0.01)
        if attack is not None:
            results.append(self.send(f"{base}/att", attack))
            time.sleep(0.01)
        if hold is not None:
            results.append(self.send(f"{base}/hld", hold))
            time.sleep(0.01)
        if release is not None:
            results.append(self.send(f"{base}/rel", release))
            time.sleep(0.01)
        if gain is not None:
            results.append(self.send(f"{base}/gain", gain))
            time.sleep(0.01)
        if mix is not None:
            results.append(self.send(f"{base}/mix", mix))
            time.sleep(0.01)
        if det is not None:
            results.append(self.send(f"{base}/det", det))
            time.sleep(0.01)
        if env is not None:
            results.append(self.send(f"{base}/env", env))
            time.sleep(0.01)
        if auto is not None:
            results.append(self.send(f"{base}/auto", auto))
            time.sleep(0.01)
        return all(results) if results else True
    
    def set_compressor_model(self, channel: int, model: str):
        """Set compressor model: COMP, EXP, B160, B560, D241, ECL33, 9000C, SBUS, 
        RED3, 76LA, LA, F670, BLISS, NSTR, WAVE, RIDE, 2250, L100, CMB"""
        return self.send(f"/ch/{channel}/dyn/mdl", model)

    def set_gate_on(self, channel: int, on: int):
        """Enable/disable gate"""
        return self.send(f"/ch/{channel}/gate/on", on)

    def set_gate(self, channel: int, threshold: float = None, range_db: float = None, 
                 attack: float = None, hold: float = None, release: float = None,
                 accent: float = None, ratio: str = None):
        """
        Set gate parameters
        
        Args:
            channel: Channel number (1-40)
            threshold: Threshold in dB (-80..0)
            range_db: Range in dB (3..60)
            attack: Attack time in ms (0..120)
            hold: Hold time in ms (0..200)
            release: Release time in ms (4..4000)
            accent: Accent (0..100)
            ratio: Ratio string: "1:1.5", "1:2", "1:3", "1:4", "GATE"
        """
        base = f"/ch/{channel}/gate"
        results = []
        if threshold is not None:
            results.append(self.send(f"{base}/thr", threshold))
        if range_db is not None:
            results.append(self.send(f"{base}/range", range_db))
        if attack is not None:
            results.append(self.send(f"{base}/att", attack))
        if hold is not None:
            results.append(self.send(f"{base}/hld", hold))
        if release is not None:
            results.append(self.send(f"{base}/rel", release))
        if accent is not None:
            results.append(self.send(f"{base}/acc", accent))
        if ratio is not None:
            results.append(self.send(f"{base}/ratio", ratio))
        return all(results) if results else True
    
    def set_gate_model(self, channel: int, model: str):
        """Set gate model: GATE, DUCK, E88, 9000G, D241, DS902, WAVE, DEQ, WARM, 76LA, LA, RIDE, PSE, CMB"""
        return self.send(f"/ch/{channel}/gate/mdl", model)

    # ========== Main/DCA Methods ==========
    
    def get_main_fader(self, main: int = 1) -> Optional[float]:
        """Get main fader value"""
        address = f"/main/{main}/fdr"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        return self.state.get(address)

    def set_main_fader(self, main: int, value: float):
        """Set main fader value (dB, -144..10; ceiling-clamped)"""
        address = f"/main/{main}/fdr"
        return self.send(address, self._clamp_fader_db(address, value))

    def get_dca_fader(self, dca: int) -> Optional[float]:
        """Get DCA fader value"""
        address = f"/dca/{dca}/fdr"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        value = self.state.get(address)
        return float(value) if value is not None else None

    def set_dca_fader(self, dca: int, value: float):
        """Set DCA fader value (ceiling-clamped)"""
        address = f"/dca/{dca}/fdr"
        return self.send(address, self._clamp_fader_db(address, value))

    def set_dca_level(self, dca: int, value: float):
        """Compatibility wrapper for safety actions controlling DCA level."""
        return self.set_dca_fader(dca, value)
    
    def set_dca_mute(self, dca: int, value: int):
        """Set DCA mute (0=unmuted, 1=muted)"""
        return self.send(f"/dca/{dca}/mute", value)
    
    # ========== Bus Methods ==========
    
    def set_bus_fader(self, bus: int, value: float):
        """Set bus fader value (dB, -144..10; ceiling-clamped)"""
        address = f"/bus/{bus}/fdr"
        return self.send(address, self._clamp_fader_db(address, value))

    def get_bus_fader(self, bus: int) -> Optional[float]:
        """Get bus fader value."""
        address = f"/bus/{bus}/fdr"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        value = self.state.get(address)
        return float(value) if value is not None else None
    
    def set_bus_mute(self, bus: int, value: int):
        """Set bus mute (0=unmuted, 1=muted)"""
        return self.send(f"/bus/{bus}/mute", value)
    
    def set_bus_pan(self, bus: int, value: float):
        """Set bus pan (-100..100)"""
        return self.send(f"/bus/{bus}/pan", value)
    
    def set_bus_eq_on(self, bus: int, on: int):
        """Enable/disable bus EQ (0=off, 1=on)"""
        return self.send(f"/bus/{bus}/eq/on", on)

    def get_bus_eq_band_gain(self, bus: int, band: int) -> Optional[float]:
        """Get bus EQ band gain."""
        address = f"/bus/{bus}/eq/{band}g"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        value = self.state.get(address)
        return float(value) if value is not None else None

    def get_bus_eq_band_frequency(self, bus: int, band: int) -> Optional[float]:
        """Get bus EQ band frequency."""
        address = f"/bus/{bus}/eq/{band}f"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        value = self.state.get(address)
        return float(value) if value is not None else None
    
    def set_bus_eq_band(self, bus: int, band: int, freq: float = None, gain: float = None, q: float = None):
        """Set bus EQ band (1-8) parameters"""
        if band < 1 or band > 8:
            raise ValueError("Bus EQ band must be 1-8")
        results = []
        if freq is not None:
            results.append(self.send(f"/bus/{bus}/eq/{band}f", freq))
        if gain is not None:
            results.append(self.send(f"/bus/{bus}/eq/{band}g", gain))
        if q is not None:
            results.append(self.send(f"/bus/{bus}/eq/{band}q", q))
        return all(results) if results else True

    def set_bus_compressor(
        self,
        bus: int,
        threshold_db: float = None,
        ratio: float = None,
        attack_ms: float = None,
        release_ms: float = None,
        makeup_db: float = None,
        enabled: bool = None,
    ):
        """Set bus dynamics parameters with WING bus threshold address."""
        base = f"/bus/{bus}/dyn"
        results = []
        if enabled is not None:
            results.append(self.send(f"{base}/on", 1 if enabled else 0))
            time.sleep(0.01)
        if threshold_db is not None:
            # WING Rack fw3.0.5 standard bus compressor exposes /dyn/thr.
            # /dyn/thr37 is model-specific and times out on the verified COMP
            # model, so it cannot serve as the canonical write/readback path.
            results.append(self.send(f"{base}/thr", threshold_db))
            time.sleep(0.01)
        if ratio is not None:
            ratio_value = f"{float(ratio):.1f}" if isinstance(ratio, (int, float)) else str(ratio)
            results.append(self.send(f"{base}/ratio", ratio_value))
            time.sleep(0.01)
        if attack_ms is not None:
            results.append(self.send(f"{base}/att", attack_ms))
            time.sleep(0.01)
        if release_ms is not None:
            results.append(self.send(f"{base}/rel", release_ms))
            time.sleep(0.01)
        if makeup_db is not None:
            results.append(self.send(f"{base}/gain", makeup_db))
            time.sleep(0.01)
        return all(results) if results else True

    def set_bus_compressor_gain(self, bus: int, gain: float):
        """Set bus compressor make-up gain."""
        return self.send(f"/bus/{bus}/dyn/gain", gain)
    
    # ========== Main Methods ==========
    
    def set_main_mute(self, main: int, value: int):
        """Set main mute (0=unmuted, 1=muted)"""
        return self.send(f"/main/{main}/mute", value)
    
    def set_main_pan(self, main: int, value: float):
        """Set main pan (-100..100)"""
        return self.send(f"/main/{main}/pan", value)
    
    def set_main_eq_on(self, main: int, on: int):
        """Enable/disable main EQ (0=off, 1=on)"""
        return self.send(f"/main/{main}/eq/on", on)

    def set_main_eq_band(self, main: int, band: int, freq: float = None, gain: float = None, q: float = None):
        """Set main EQ band (1-8) parameters."""
        if band < 1 or band > 8:
            raise ValueError("Main EQ band must be 1-8")
        results = []
        if freq is not None:
            results.append(self.send(f"/main/{main}/eq/{band}f", freq))
        if gain is not None:
            results.append(self.send(f"/main/{main}/eq/{band}g", gain))
        if q is not None:
            results.append(self.send(f"/main/{main}/eq/{band}q", q))
        return all(results) if results else True

    def set_matrix_eq_on(self, matrix: int, on: int):
        """Enable/disable matrix EQ (0=off, 1=on)."""
        return self.send(f"/mtx/{matrix}/eq/on", on)

    def set_matrix_eq_band(self, matrix: int, band: int, freq: float = None, gain: float = None, q: float = None):
        """Set matrix EQ band (1-8) parameters."""
        if band < 1 or band > 8:
            raise ValueError("Matrix EQ band must be 1-8")
        results = []
        if freq is not None:
            results.append(self.send(f"/mtx/{matrix}/eq/{band}f", freq))
        if gain is not None:
            results.append(self.send(f"/mtx/{matrix}/eq/{band}g", gain))
        if q is not None:
            results.append(self.send(f"/mtx/{matrix}/eq/{band}q", q))
        return all(results) if results else True
    
    # ========== Send Methods ==========
    
    # ========== FX Module Methods ==========
    
    def set_fx_parameter(self, fx_slot: str, parameter: int, value):
        """
        Set an FX module parameter
        
        Args:
            fx_slot: FX slot (e.g., 'FX1', 'FX13')
            parameter: Parameter number (1-32)
            value: Parameter value
        """
        _, fx_num = self._normalize_fx_slot(fx_slot)
        return self.send(f"/fx/{fx_num}/{parameter}", value)
    
    def set_fx_model(self, fx_slot: str, model: str):
        """
        Set FX module model
        
        Args:
            fx_slot: FX slot (e.g., 'FX1', 'FX13')
            model: FX model name (e.g., 'P-BASS', 'REVERB', 'PCORR')
        """
        _, fx_num = self._normalize_fx_slot(fx_slot)
        return self.send(f"/fx/{fx_num}/mdl", model)
    
    def set_fx_on(self, fx_slot: str, on: int):
        """Fail closed for the unsupported /fx/N/on path.

        WING FX slots are active when a model is loaded. The official FW 3.1
        slot paths include /mdl and /fxmix, but not /on, so reporting success
        here would turn an ignored OSC write into a false applied state.
        """
        return False

    def set_fx_mix(self, fx_slot: str, mix: float):
        """Set FX mix percentage (0-100)"""
        _, fx_num = self._normalize_fx_slot(fx_slot)
        return self.send(f"/fx/{fx_num}/fxmix", mix)

    def query_fx_slot(self, fx_slot: str):
        """Query primary FX slot metadata from the console."""
        _, fx_num = self._normalize_fx_slot(fx_slot)
        for suffix in ("mdl", "fxmix", "$esrc", "$emode", "$a_chn", "$a_pos"):
            self.send(f"/fx/{fx_num}/{suffix}")

    def get_fx_slot_info(self, fx_slot: str) -> Dict[str, Any]:
        """Return cached FX slot information using official FW 3.1 paths."""
        slot, fx_num = self._normalize_fx_slot(fx_slot)
        model = self.state.get(f"/fx/{fx_num}/mdl")
        active = bool(model and str(model).strip().upper() != "NONE")
        return {
            "slot": slot,
            "model": model,
            "on": 1 if active else 0,
            "active": active,
            "mix": self.state.get(f"/fx/{fx_num}/fxmix"),
            "source": self.state.get(f"/fx/{fx_num}/$esrc"),
            "mode": self.state.get(f"/fx/{fx_num}/$emode"),
            "assigned_channel": self.state.get(f"/fx/{fx_num}/$a_chn"),
            "assigned_position": self.state.get(f"/fx/{fx_num}/$a_pos"),
        }

    def set_insert(self, target_type: str, target: int, position: str, slot: str = "NONE", on: Optional[int] = None, mode: Optional[str] = None):
        """Assign or clear an insert slot on channel/aux/bus/main/matrix."""
        base = self._insert_base(target_type, target, position)
        results = []
        if on is not None:
            results.append(self.send(f"{base}/on", on))
        if mode is not None and position.lower() == "post":
            results.append(self.send(f"{base}/mode", mode))
        results.append(self.send(f"{base}/ins", str(slot).upper()))
        return all(results)

    def query_insert(self, target_type: str, target: int, position: str):
        """Query insert state for supported channel types."""
        base = self._insert_base(target_type, target, position)
        self.send(f"{base}/on")
        if position.lower() == "post":
            try:
                self.send(f"{base}/mode")
            except Exception:
                pass
            try:
                self.send(f"{base}/w")
            except Exception:
                pass
        self.send(f"{base}/ins")
        self.send(f"{base}/$stat")

    def get_insert(self, target_type: str, target: int, position: str) -> Dict[str, Any]:
        """Return cached insert state and linked FX metadata."""
        base = self._insert_base(target_type, target, position)
        result: Dict[str, Any] = {
            "target_type": str(target_type).lower(),
            "target": int(target),
            "position": str(position).lower(),
            "on": self.state.get(f"{base}/on"),
            "slot": self.state.get(f"{base}/ins"),
            "status": self.state.get(f"{base}/$stat"),
        }
        if position.lower() == "post":
            result["mode"] = self.state.get(f"{base}/mode")
            weight = self.state.get(f"{base}/w")
            if weight is not None:
                result["weight"] = weight
        slot = result.get("slot")
        if slot and slot != "NONE":
            result["fx_module"] = self._get_fx_module_info(slot)
        else:
            result["fx_module"] = None
        return result

    def _active_insert_or_none(self, target_type: str, target: int, position: str):
        """Preserve legacy semantics: return None when no insert is assigned."""
        insert = self.get_insert(target_type, target, position)
        slot = insert.get("slot")
        if insert.get("on") == 1 and slot and slot != "NONE":
            return insert
        return None

    def get_bus_inserts(self, bus: int) -> Dict[str, Any]:
        return {
            "pre_insert": self._active_insert_or_none("bus", bus, "pre"),
            "post_insert": self._active_insert_or_none("bus", bus, "post"),
        }

    def get_main_inserts(self, main: int) -> Dict[str, Any]:
        return {
            "pre_insert": self._active_insert_or_none("main", main, "pre"),
            "post_insert": self._active_insert_or_none("main", main, "post"),
        }

    def get_matrix_inserts(self, matrix: int) -> Dict[str, Any]:
        return {
            "pre_insert": self._active_insert_or_none("matrix", matrix, "pre"),
            "post_insert": self._active_insert_or_none("matrix", matrix, "post"),
        }

    def set_fx_return_level(self, bus: int, level_db: float):
        """Set an FX return bus level in dB when returns are mapped to buses."""
        address = f"/bus/{bus}/fdr"
        return self.send(address, self._clamp_fader_db(address, level_db))

    def set_reverb_send(self, channel: int, level_db: float, send_bus: int = 13):
        """Compatibility wrapper used by AutoReverb/AutoFX planners."""
        return self.set_channel_send(channel, send_bus, level=level_db, on=1, mode="POST")

    def set_reverb_params(
        self,
        channel: int,
        reverb_type: str,
        decay_time: float,
        predelay: float,
        brightness: float,
        density: float,
        hpf: float,
        lpf: float,
        fx_slot: str = "FX1",
    ):
        """Compatibility wrapper for shared reverb FX parameter setup."""
        model = str(reverb_type).upper()
        results = [
            self.set_fx_model(fx_slot, model),
            self.set_fx_mix(fx_slot, 100.0),
            self.set_fx_parameter(fx_slot, 1, decay_time),
            self.set_fx_parameter(fx_slot, 2, predelay),
            self.set_fx_parameter(fx_slot, 3, density),
            self.set_fx_parameter(fx_slot, 4, brightness),
            self.set_fx_parameter(fx_slot, 5, hpf),
            self.set_fx_parameter(fx_slot, 6, lpf),
        ]
        return all(results)
    
    def get_fx_parameter(self, fx_slot: str, parameter: int):
        """Get an FX module parameter value"""
        _, fx_num = self._normalize_fx_slot(fx_slot)
        return self.state.get(f'/fx/{fx_num}/{parameter}')
    
    def set_channel_send(self, channel: int, send: int, level: float = None, on: int = None,
                         mode: str = None, pan: float = None):
        """
        Set channel send parameters
        
        Args:
            channel: Channel number (1-40)
            send: Send number (1-16)
            level: Send level in dB (-144..10)
            on: Send on/off (0/1)
            mode: Send mode: "PRE", "POST", "GRP"
            pan: Send pan (-100..100)
        """
        base = f"/ch/{channel}/send/{send}"
        results = []
        if level is not None:
            results.append(self.send(f"{base}/lvl", self._clamp_fader_db(f"{base}/lvl", level)))
        if on is not None:
            results.append(self.send(f"{base}/on", on))
        if mode is not None:
            results.append(self.send(f"{base}/mode", mode))
        if pan is not None:
            results.append(self.send(f"{base}/pan", pan))
        return all(results) if results else True
    
    def set_channel_main_send(self, channel: int, main: int, level: float = None, on: int = None, pre: int = None):
        """
        Set channel send to main
        
        Args:
            channel: Channel number (1-40)
            main: Main number (1-4)
            level: Level in dB (-144..10)
            on: On/off (0/1)
            pre: Pre fader (0/1)
        """
        results = []
        if level is not None:
            addr = f"/ch/{channel}/main/{main}/lvl"
            results.append(self.send(addr, self._clamp_fader_db(addr, level)))
        if on is not None:
            results.append(self.send(f"/ch/{channel}/main/{main}/on", on))
        if pre is not None:
            results.append(self.send(f"/ch/{channel}/main/pre", pre))
        return all(results) if results else True

    # ========== Utility Methods ==========
    
    def get_state(self) -> Dict[str, Any]:
        """Get copy of current mixer state"""
        return self.state.copy()
    
    def get_channel_inserts(self, channel: int) -> Dict[str, Any]:
        """
        Get insert information for a channel
        
        Returns:
            Dict with 'pre_insert' and 'post_insert' info including FX module details
        """
        return {
            'pre_insert': self._active_insert_or_none("channel", channel, "pre"),
            'post_insert': self._active_insert_or_none("channel", channel, "post"),
        }
    
    def _get_fx_module_info(self, fx_slot: str) -> Dict[str, Any]:
        """Get FX module information including model and parameters"""
        slot, fx_num = self._normalize_fx_slot(fx_slot)

        fx_model = self.state.get(f'/fx/{fx_num}/mdl')
        fx_mix = self.state.get(f'/fx/{fx_num}/fxmix')
        fx_node = self.state.get(f'/fx/{fx_num}/')
        fx_active = bool(fx_model and str(fx_model).strip().upper() != "NONE")
        
        # Get all FX parameters
        fx_params = {}
        for key, value in self.state.items():
            if key.startswith(f'/fx/{fx_num}/') and key not in [
                f'/fx/{fx_num}/mdl', f'/fx/{fx_num}/fxmix',
                f'/fx/{fx_num}/', f'/fx/{fx_num}/name', f'/fx/{fx_num}/type',
                f'/fx/{fx_num}/node'
            ]:
                param_name = key.replace(f'/fx/{fx_num}/', '')
                if param_name and param_name.isdigit():
                    fx_params[int(param_name)] = value
        
        return {
            'slot': slot,
            'model': fx_model,
            'on': 1 if fx_active else 0,
            'active': fx_active,
            'mix': fx_mix,
            'node_type': fx_node,
            'parameters': fx_params
        }
    
    def get_all_channels_with_inserts(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all channels (1-40) with their inserts
        
        Returns:
            Dict mapping channel number to channel info including inserts
        """
        channels = {}
        for ch in range(1, 41):
            inserts = self.get_channel_inserts(ch)
            if inserts['pre_insert'] or inserts['post_insert']:
                channels[ch] = {
                    'channel': ch,
                    'inserts': inserts
                }
        return channels

    def query(self, address: str) -> Optional[Any]:
        """Query a parameter and wait for response"""
        self.send(address)
        time.sleep(0.1)
        return self.state.get(address)
    
    # ========== Routing Methods ==========
    
    def route_output(self, output_group: str, output_number: int, source_group: str, source_channel: int):
        """
        Маршрутизация выхода на источник
        
        Args:
            output_group: OUTPUT GROUP (куда посылать) - например, "MOD" для DANTE модуля, "CRD" для карт
            output_number: Номер выхода в OUTPUT GROUP (1..64)
            source_group: SOURCE GROUP (откуда брать сигнал) - например, "CRD" для Card, "PLAY" для USB Player
            source_channel: Номер канала источника из SOURCE GROUP (1..64)
        
        Примеры:
            # Маршрутизация DANTE выхода 1 на WLIVE PLAY канал 1:
            route_output("MOD", 1, "CRD", 1)
            
            # Маршрутизация CRD выхода 5 на канал пульта 10:
            route_output("CRD", 5, "CH", 10)
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False
        
        try:
            # Устанавливаем SOURCE GROUP
            result1 = self.send(f"/io/out/{output_group}/{output_number}/grp", source_group)
            time.sleep(0.01)
            
            # Устанавливаем номер канала источника
            result2 = self.send(f"/io/out/{output_group}/{output_number}/in", source_channel)
            time.sleep(0.01)
            
            if result1 and result2:
                logger.debug(f"Routed {output_group}/{output_number} <- {source_group}/{source_channel}")
                return True
            else:
                logger.warning(f"Failed to route {output_group}/{output_number}")
                return False
        except Exception as e:
            logger.error(f"Error routing {output_group}/{output_number}: {e}")
            return False
    
    def route_multiple_outputs(self, output_group: str, start_output: int, num_outputs: int, 
                               source_group: str, start_source_channel: int):
        """
        Маршрутизация нескольких выходов последовательно
        
        Args:
            output_group: OUTPUT GROUP (куда посылать) - "MOD", "CRD", и т.д.
            start_output: Начальный номер выхода
            num_outputs: Количество выходов для маршрутизации
            source_group: SOURCE GROUP (откуда брать сигнал) - "CRD", "PLAY", "CH", и т.д.
            start_source_channel: Начальный номер канала источника
        
        Пример:
            # Маршрутизация 24 DANTE выходов (MOD 1-24) на WLIVE PLAY каналы (CRD 1-24):
            route_multiple_outputs("MOD", 1, 24, "CRD", 1)
        """
        success_count = 0
        for i in range(num_outputs):
            output_num = start_output + i
            source_ch = start_source_channel + i
            if self.route_output(output_group, output_num, source_group, source_ch):
                success_count += 1
            time.sleep(0.01)
        return success_count
    
    def get_output_routing(self, output_group: str, output_number: int) -> Optional[Dict[str, Any]]:
        """
        Получить текущую маршрутизацию выхода
        
        Args:
            output_group: OUTPUT GROUP - "MOD", "CRD", и т.д.
            output_number: Номер выхода
        
        Returns:
            Dict с 'source_group' и 'source_channel', или None если не найдено
        """
        if not self.is_connected:
            return None
        
        try:
            self.send(f"/io/out/{output_group}/{output_number}/grp")
            self.send(f"/io/out/{output_group}/{output_number}/in")
            time.sleep(0.1)
            
            source_group = self.state.get(f"/io/out/{output_group}/{output_number}/grp")
            source_channel = self.state.get(f"/io/out/{output_group}/{output_number}/in")
            
            if source_group is not None:
                return {
                    "output_group": output_group,
                    "output_number": output_number,
                    "source_group": source_group,
                    "source_channel": source_channel
                }
        except Exception as e:
            logger.error(f"Error getting routing for {output_group}/{output_number}: {e}")
        
        return None
    
    # ========== Snapshot/Scene Methods ==========
    
    def get_current_show(self) -> Optional[str]:
        """
        Получить имя текущего активного Show
        
        Returns:
            Имя Show или None
        """
        if not self.is_connected:
            return None
        
        self.send("/$ctl/lib/$actshow")
        time.sleep(0.1)
        return self.state.get("/$ctl/lib/$actshow")
    
    def get_snap_list(self) -> Dict[int, str]:
        """
        Получить список snapshots с их индексами
        
        Возвращает словарь {индекс: имя_snapshot}
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return {}
        
        snapshots = {}
        
        # Получаем текущий активный snapshot
        self.send("/$ctl/lib/$active")
        self.send("/$ctl/lib/$actidx")
        time.sleep(0.2)
        
        current_active = self.state.get("/$ctl/lib/$active")
        current_idx = self.state.get("/$ctl/lib/$actidx")
        
        if current_active and current_idx is not None:
            snapshots[current_idx] = current_active
        
        return snapshots
    
    def find_snap_by_name(self, snap_name: str, max_index: int = 200) -> Optional[int]:
        """
        Найти индекс snapshot по имени БЕЗ загрузки (неразрушающий поиск).

        C-04 FIX: The original implementation sent a "GO" (load) command for
        each index to read back the name, which destructively changed the
        mixer state on every iteration during a live performance.  This
        rewrite queries snapshot names via the read-only OSC address
        ``/$ctl/lib/$name`` (set index first, then query name without GO)
        so the active scene is never altered during the search.

        Args:
            snap_name: Имя snapshot для поиска
            max_index: Максимальный индекс для проверки

        Returns:
            Индекс snapshot или None если не найден
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return None

        search_name_upper = snap_name.upper().strip()

        # Query names without loading — set index then read $name (not $active)
        for idx in range(1, max_index + 1):
            try:
                # Point the library cursor at this index (read-only operation)
                self.send("/$ctl/lib/$actionidx", idx)
                time.sleep(0.05)

                # Request the snapshot name at the current index.
                # $name is a read-only query; it does NOT load the snapshot.
                self.send("/$ctl/lib/$name")
                time.sleep(0.05)
                snap_name_at_idx = self.state.get("/$ctl/lib/$name")

                if snap_name_at_idx:
                    name_upper = str(snap_name_at_idx).upper()
                    name_clean = name_upper.replace("I:/", "").replace(".SNAP", "").strip()

                    if (search_name_upper == name_upper
                            or search_name_upper == name_clean
                            or search_name_upper in name_upper):
                        logger.debug(f"Found snapshot '{snap_name_at_idx}' at index {idx}")
                        return idx

            except Exception as e:
                logger.debug(f"Error checking snapshot index {idx}: {e}")

        logger.warning(f"Snapshot '{snap_name}' not found in first {max_index} slots")
        return None
    
    def load_snap(self, snap_name: str, max_index: int = 200) -> bool:
        """
        Загрузить snapshot/scene по имени
        
        Процесс:
        1. Находит индекс snapshot по имени через перебор
        2. Устанавливает индекс через /$ctl/lib/$actionidx
        3. Загружает через /$ctl/lib/$action = "GO"
        
        Args:
            snap_name: Имя snapshot для загрузки
            max_index: Максимальный индекс для поиска (по умолчанию 200)
        
        Примеры:
            # Загрузить snapshot "HULI REPA AC"
            load_snap("HULI REPA AC")
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False
        
        try:
            # Находим индекс snapshot по имени
            snap_index = self.find_snap_by_name(snap_name, max_index)
            
            if snap_index is None:
                logger.warning(f"Snapshot '{snap_name}' not found")
                return False
            
            # Загружаем snapshot по найденному индексу
            return self.load_snap_by_index(snap_index)
            
        except Exception as e:
            logger.error(f"Error loading snapshot {snap_name}: {e}")
            return False
    
    def load_snap_by_index(self, snap_index: int) -> bool:
        """
        Загрузить snapshot по индексу
        
        Args:
            snap_index: Индекс snapshot для загрузки
        
        Returns:
            True если команды отправлены успешно
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False
        
        try:
            # Шаг 1: Устанавливаем индекс snapshot
            result1 = self.send("/$ctl/lib/$actionidx", snap_index)
            time.sleep(0.1)
            
            if not result1:
                logger.warning(f"Failed to set snapshot index: {snap_index}")
                return False
            
            # Шаг 2: Отправляем команду GO для загрузки
            result2 = self.send("/$ctl/lib/$action", "GO")
            time.sleep(0.2)
            
            if result1 and result2:
                logger.debug(f"Loaded snapshot at index: {snap_index}")
                return True
            else:
                logger.warning(f"Failed to load snapshot at index: {snap_index}")
                return False
        except Exception as e:
            logger.error(f"Error loading snapshot at index {snap_index}: {e}")
            return False
    
    def save_snap(self, snap_name: str) -> bool:
        """
        Сохранить текущее состояние как snapshot/scene
        
        Args:
            snap_name: Имя snapshot для сохранения
        
        Примеры:
            # Сохранить текущее состояние как "HULI REPA AC"
            save_snap("HULI REPA AC")
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False
        
        try:
            # Возможные адреса для сохранения snapshot
            # Пробуем /snap/store с именем
            result = self.send("/snap/store", snap_name)
            time.sleep(0.1)
            
            if result:
                logger.debug(f"Saved snapshot: {snap_name}")
                return True
            else:
                logger.warning(f"Failed to save snapshot: {snap_name}")
                return False
        except Exception as e:
            logger.error(f"Error saving snapshot {snap_name}: {e}")
            return False
    
    # ========== Channel Input Routing Methods ==========
    
    def set_channel_input(self, channel: int, source_group: str, source_channel: int):
        """
        Назначить входной источник для канала (Channel Main)
        
        Args:
            channel: Номер канала пульта (1-40)
            source_group: SOURCE GROUP (откуда брать сигнал) - "MOD", "CRD", "CH", "AUX", "BUS", "MAIN", "MTX", "SEND", "MON", "USR", "OSC"
            source_channel: Номер канала источника из SOURCE GROUP (1-64)
        
        Примеры:
            # Канал 1 получает сигнал с MOD канала 1:
            set_channel_input(1, "MOD", 1)
            
            # Канал 10 получает сигнал с CRD канала 5:
            set_channel_input(10, "CRD", 5)
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False
        
        try:
            # Устанавливаем SOURCE GROUP для Main input
            result1 = self.send(f"/ch/{channel}/in/conn/grp", source_group)
            time.sleep(0.01)
            
            # Устанавливаем номер канала источника
            result2 = self.send(f"/ch/{channel}/in/conn/in", source_channel)
            time.sleep(0.01)
            
            if result1 and result2:
                logger.debug(f"Channel {channel} Main input: {source_group}/{source_channel}")
                return True
            else:
                logger.warning(f"Failed to set channel {channel} Main input")
                return False
        except Exception as e:
            logger.error(f"Error setting channel {channel} Main input: {e}")
            return False
    
    def set_channel_alt_input(self, channel: int, source_group: str, source_channel: int):
        """
        Назначить альтернативный входной источник для канала (Channel ALT)
        
        Args:
            channel: Номер канала пульта (1-40)
            source_group: SOURCE GROUP (откуда брать сигнал) - "MOD", "CRD", "CH", "AUX", "BUS", "MAIN", "MTX", "SEND", "MON", "USR", "OSC"
            source_channel: Номер канала источника из SOURCE GROUP (1-64)
        
        Примеры:
            # Канал 1 ALT получает сигнал с CRD канала 1:
            set_channel_alt_input(1, "CRD", 1)
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False
        
        try:
            # Устанавливаем SOURCE GROUP для ALT input
            result1 = self.send(f"/ch/{channel}/in/conn/altgrp", source_group)
            time.sleep(0.01)
            
            # Устанавливаем номер канала источника
            result2 = self.send(f"/ch/{channel}/in/conn/altin", source_channel)
            time.sleep(0.01)
            
            if result1 and result2:
                logger.debug(f"Channel {channel} ALT input: {source_group}/{source_channel}")
                return True
            else:
                logger.warning(f"Failed to set channel {channel} ALT input")
                return False
        except Exception as e:
            logger.error(f"Error setting channel {channel} ALT input: {e}")
            return False
    
    def get_channel_input_routing(self, channel: int) -> Optional[Dict[str, Any]]:
        """
        Получить текущую маршрутизацию входов канала
        
        Args:
            channel: Номер канала пульта (1-40)
        
        Returns:
            Dict с 'main_group', 'main_channel', 'alt_group', 'alt_channel', или None если не найдено
        """
        if not self.is_connected:
            return None
        
        try:
            self.send(f"/ch/{channel}/in/conn/grp")
            self.send(f"/ch/{channel}/in/conn/in")
            self.send(f"/ch/{channel}/in/conn/altgrp")
            self.send(f"/ch/{channel}/in/conn/altin")
            time.sleep(0.1)
            
            main_group = self.state.get(f"/ch/{channel}/in/conn/grp")
            main_channel = self.state.get(f"/ch/{channel}/in/conn/in")
            alt_group = self.state.get(f"/ch/{channel}/in/conn/altgrp")
            alt_channel = self.state.get(f"/ch/{channel}/in/conn/altin")
            
            if main_group is not None:
                return {
                    "channel": channel,
                    "main_group": main_group,
                    "main_channel": main_channel,
                    "alt_group": alt_group,
                    "alt_channel": alt_channel
                }
        except Exception as e:
            logger.error(f"Error getting channel {channel} input routing: {e}")

        return None

    # ── MixerClientBase ABC bridge methods ──────────────────────
    def set_fader(self, channel: int, value_db: float):
        return self.set_channel_fader(channel, value_db)

    def get_fader(self, channel: int) -> float:
        address = f"/ch/{channel}/fdr"
        val = self.state.get(address)
        return float(val) if val is not None else -144.0

    def set_mute(self, channel: int, muted: bool):
        return self.set_channel_mute(channel, 1 if muted else 0)

    def get_mute(self, channel: int) -> bool:
        val = self.get_channel_mute(channel)
        return bool(val) if val is not None else False

    def set_gain(self, channel: int, value_db: float):
        return self.set_channel_gain(channel, value_db)

    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        return self.set_low_cut(channel, enabled=1 if enabled else 0, frequency=freq)

    def set_pan(self, channel: int, pan: float):
        return self.set_channel_pan(channel, pan)

    def set_polarity(self, channel: int, inverted: bool):
        return self.set_channel_phase_invert(channel, 1 if inverted else 0)

    def get_polarity(self, channel: int) -> bool:
        address = f"/ch/{channel}/in/set/inv"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        return bool(self.state.get(address) or 0)

    def set_delay(self, channel: int, delay_ms: float, enabled: bool = True):
        if enabled:
            return self.set_channel_delay(channel, delay_ms, mode="MS")
        return self.send(f"/ch/{channel}/in/set/dlyon", 0)

    def get_delay(self, channel: int) -> float:
        address = f"/ch/{channel}/in/set/dly"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        value = self.state.get(address)
        return float(value) if value is not None else 0.0

    def set_send_level(self, channel: int, send_bus: int, level_db: float, channel_type: str = "input"):
        return self.set_channel_send(channel, send_bus, level=level_db)

    def get_send_level(self, channel: int, send_bus: int) -> Optional[float]:
        """Get channel send level (свежий запрос + чтение state)."""
        address = f"/ch/{channel}/send/{send_bus}/lvl"
        if self.is_connected:
            self.send(address)
            time.sleep(0.05)
        value = self.state.get(address)
        return float(value) if value is not None else None

    def reset_channel_eq(self, channel: int):
        self.set_eq_on(channel, 0)
        self.send(f"/ch/{channel}/eq/lg", 0.0)
        for band in range(1, 5):
            self.send(f"/ch/{channel}/eq/{band}g", 0.0)
        self.send(f"/ch/{channel}/eq/hg", 0.0)

    def reset_channel_hpf(self, channel: int):
        self.set_low_cut(channel, enabled=0, frequency=20.0)

    def reset_channel_dynamics(self, channel: int):
        self.set_compressor_on(channel, 0)
        self.set_gate_on(channel, 0)

    def reset_channel_processing(self, channel: int):
        self.reset_channel_eq(channel)
        self.reset_channel_hpf(channel)
        self.reset_channel_dynamics(channel)
        self.set_polarity(channel, False)
        self.set_delay(channel, 0.0, enabled=False)

    def get_bus_settings(self, bus: int) -> Dict[str, Any]:
        """Read processing and assignment state for a WING bus."""
        query_addresses = [
            f"/bus/{bus}/name",
            f"/bus/{bus}/$name",
            f"/bus/{bus}/mute",
            f"/bus/{bus}/fdr",
            f"/bus/{bus}/pan",
            f"/bus/{bus}/wid",
            f"/bus/{bus}/busmono",
            f"/bus/{bus}/eq/on",
            f"/bus/{bus}/dyn/on",
            f"/bus/{bus}/dyn/mdl",
            f"/bus/{bus}/dyn/thr",
            f"/bus/{bus}/dyn/ratio",
            f"/bus/{bus}/dyn/knee",
            f"/bus/{bus}/dyn/det",
            f"/bus/{bus}/dyn/att",
            f"/bus/{bus}/dyn/rel",
            f"/bus/{bus}/dyn/gain",
            f"/bus/{bus}/dyn/mix",
            f"/bus/{bus}/main/1/on",
            f"/bus/{bus}/main/1/lvl",
            f"/bus/{bus}/main/1/pre",
            f"/bus/{bus}/tags",
        ]
        for band in range(1, 9):
            query_addresses.extend(
                [
                    f"/bus/{bus}/eq/{band}f",
                    f"/bus/{bus}/eq/{band}g",
                    f"/bus/{bus}/eq/{band}q",
                ]
            )

        for address in query_addresses:
            self.send(address)
        time.sleep(0.12)

        eq_bands = []
        for band in range(1, 9):
            eq_bands.append(
                (
                    self.state.get(f"/bus/{bus}/eq/{band}f"),
                    self.state.get(f"/bus/{bus}/eq/{band}g"),
                    self.state.get(f"/bus/{bus}/eq/{band}q"),
                )
            )

        tags = self.state.get(f"/bus/{bus}/tags")
        return {
            "bus": bus,
            "name": self.state.get(f"/bus/{bus}/$name") or self.state.get(f"/bus/{bus}/name") or "",
            "fader_db": self.state.get(f"/bus/{bus}/fdr"),
            "muted": self.state.get(f"/bus/{bus}/mute"),
            "pan": self.state.get(f"/bus/{bus}/pan"),
            "width": self.state.get(f"/bus/{bus}/wid"),
            "busmono": self.state.get(f"/bus/{bus}/busmono"),
            "eq_on": self.state.get(f"/bus/{bus}/eq/on"),
            "eq_bands": eq_bands,
            "compressor_enabled": self.state.get(f"/bus/{bus}/dyn/on"),
            "compressor_model": self.state.get(f"/bus/{bus}/dyn/mdl"),
            "compressor_threshold_db": (
                self.state.get(f"/bus/{bus}/dyn/thr")
                if self.state.get(f"/bus/{bus}/dyn/thr") is not None
                else self.state.get(f"/bus/{bus}/dyn/thr37")
            ),
            "compressor_ratio": self.state.get(f"/bus/{bus}/dyn/ratio"),
            "compressor_knee": self.state.get(f"/bus/{bus}/dyn/knee"),
            "compressor_detector": self.state.get(f"/bus/{bus}/dyn/det"),
            "compressor_attack_ms": self.state.get(f"/bus/{bus}/dyn/att"),
            "compressor_release_ms": self.state.get(f"/bus/{bus}/dyn/rel"),
            "compressor_makeup_db": self.state.get(f"/bus/{bus}/dyn/gain"),
            "compressor_mix_pct": self.state.get(f"/bus/{bus}/dyn/mix"),
            "main_send": {
                "on": self.state.get(f"/bus/{bus}/main/1/on"),
                "level_db": self.state.get(f"/bus/{bus}/main/1/lvl"),
                "pre": self.state.get(f"/bus/{bus}/main/1/pre"),
            },
            "tags": tags,
            "dca_assignments": self._parse_dca_assignments(tags),
        }

    def get_dca_settings(self, dca: int) -> Dict[str, Any]:
        """Read state for a WING DCA group."""
        query_addresses = [
            f"/dca/{dca}/name",
            f"/dca/{dca}/mute",
            f"/dca/{dca}/fdr",
            f"/dca/{dca}/col",
            f"/dca/{dca}/icon",
            f"/dca/{dca}/mon",
        ]
        for address in query_addresses:
            self.send(address)
        time.sleep(0.1)

        return {
            "dca": dca,
            "name": self.state.get(f"/dca/{dca}/name") or "",
            "muted": self.state.get(f"/dca/{dca}/mute"),
            "fader_db": self.state.get(f"/dca/{dca}/fdr"),
            "color": self.state.get(f"/dca/{dca}/col"),
            "icon": self.state.get(f"/dca/{dca}/icon"),
            "monitor": self.state.get(f"/dca/{dca}/mon"),
        }

    def get_channel_settings(self, channel: int) -> Dict[str, Any]:
        query_addresses = [
            f"/ch/{channel}/fdr",
            f"/ch/{channel}/mute",
            f"/ch/{channel}/in/set/trim",
            f"/ch/{channel}/in/set/inv",
            f"/ch/{channel}/in/set/dlyon",
            f"/ch/{channel}/in/set/dly",
            f"/ch/{channel}/in/conn/grp",
            f"/ch/{channel}/in/conn/in",
            f"/ch/{channel}/in/conn/altgrp",
            f"/ch/{channel}/in/conn/altin",
            f"/ch/{channel}/pan",
            f"/ch/{channel}/flt/lc",
            f"/ch/{channel}/flt/lcf",
            f"/ch/{channel}/eq/on",
            f"/ch/{channel}/dyn/on",
            f"/ch/{channel}/dyn/mdl",
            f"/ch/{channel}/dyn/thr26",
            f"/ch/{channel}/dyn/thr",
            f"/ch/{channel}/dyn/ratio",
            f"/ch/{channel}/dyn/att",
            f"/ch/{channel}/dyn/rel",
            f"/ch/{channel}/dyn/gain",
            f"/ch/{channel}/dyn/mix",
            f"/ch/{channel}/dyn/det",
            f"/ch/{channel}/dyn/gr",
            f"/ch/{channel}/dyn/$gr",
            f"/ch/{channel}/gate/on",
            f"/ch/{channel}/gate/mdl",
            f"/ch/{channel}/gate/thr",
            f"/ch/{channel}/gate/range",
            f"/ch/{channel}/gate/att",
            f"/ch/{channel}/gate/hld",
            f"/ch/{channel}/gate/rel",
            f"/ch/{channel}/gate/acc",
            f"/ch/{channel}/gate/ratio",
            f"/ch/{channel}/main/1/on",
            f"/ch/{channel}/main/1/lvl",
            f"/ch/{channel}/main/1/pre",
            f"/ch/{channel}/tags",
        ]
        for band in range(1, 5):
            query_addresses.extend(
                [
                    f"/ch/{channel}/eq/{band}f",
                    f"/ch/{channel}/eq/{band}g",
                    f"/ch/{channel}/eq/{band}q",
                ]
            )
        for send in range(1, 17):
            query_addresses.extend(
                [
                    f"/ch/{channel}/send/{send}/on",
                    f"/ch/{channel}/send/{send}/lvl",
                    f"/ch/{channel}/send/{send}/pon",
                    f"/ch/{channel}/send/{send}/mode",
                    f"/ch/{channel}/send/{send}/plink",
                    f"/ch/{channel}/send/{send}/pan",
                ]
            )

        for address in query_addresses:
            self.send(address)

        # Give the receiver thread a short window to populate state from WING replies.
        time.sleep(0.12)

        eq_bands = []
        for band in range(1, 5):
            eq_bands.append(
                (
                    self.state.get(f"/ch/{channel}/eq/{band}f"),
                    self.state.get(f"/ch/{channel}/eq/{band}g"),
                    self.state.get(f"/ch/{channel}/eq/{band}q"),
                )
            )

        sends = []
        active_bus_sends = []
        for send in range(1, 17):
            base = f"/ch/{channel}/send/{send}"
            send_state = {
                "bus": send,
                "on": self.state.get(f"{base}/on"),
                "level_db": self.state.get(f"{base}/lvl"),
                "pre_on": self.state.get(f"{base}/pon"),
                "mode": self.state.get(f"{base}/mode"),
                "plink": self.state.get(f"{base}/plink"),
                "pan": self.state.get(f"{base}/pan"),
            }
            send_state["active"] = self._is_active_send(send_state)
            sends.append(send_state)
            if send_state["active"]:
                active_bus_sends.append(send_state)

        tags = self.state.get(f"/ch/{channel}/tags")
        readback_presence = {
            "fader": self.state.get(f"/ch/{channel}/fdr") is not None,
            "mute": self.state.get(f"/ch/{channel}/mute") is not None,
            "main_send_on": self.state.get(f"/ch/{channel}/main/1/on") is not None,
            "main_send_level": self.state.get(f"/ch/{channel}/main/1/lvl") is not None,
            "main_send_pre": self.state.get(f"/ch/{channel}/main/1/pre") is not None,
            "input_main_group": self.state.get(f"/ch/{channel}/in/conn/grp") is not None,
            "input_main_channel": self.state.get(f"/ch/{channel}/in/conn/in") is not None,
            "input_alt_group": self.state.get(f"/ch/{channel}/in/conn/altgrp") is not None,
            "input_alt_channel": self.state.get(f"/ch/{channel}/in/conn/altin") is not None,
            "tags": self.state.get(f"/ch/{channel}/tags") is not None,
        }

        return {
            "channel": channel,
            "name": self.get_channel_name(channel),
            "fader_db": self.get_fader(channel),
            "muted": self.state.get(f"/ch/{channel}/mute"),
            "gain_db": self.get_channel_gain(channel) or 0.0,
            "pan": self.get_channel_pan(channel),
            "polarity_inverted": self.state.get(f"/ch/{channel}/in/set/inv"),
            "delay_enabled": self.state.get(f"/ch/{channel}/in/set/dlyon"),
            "delay_ms": self.state.get(f"/ch/{channel}/in/set/dly"),
            "input_routing": {
                "main_group": self.state.get(f"/ch/{channel}/in/conn/grp"),
                "main_channel": self.state.get(f"/ch/{channel}/in/conn/in"),
                "alt_group": self.state.get(f"/ch/{channel}/in/conn/altgrp"),
                "alt_channel": self.state.get(f"/ch/{channel}/in/conn/altin"),
            },
            "main_send": {
                "on": self.state.get(f"/ch/{channel}/main/1/on"),
                "level_db": self.state.get(f"/ch/{channel}/main/1/lvl"),
                "pre": self.state.get(f"/ch/{channel}/main/1/pre"),
            },
            "sends": sends,
            "active_bus_sends": active_bus_sends,
            "tags": tags,
            "dca_assignments": self._parse_dca_assignments(tags),
            "readback_presence": readback_presence,
            "hpf_enabled": self.state.get(f"/ch/{channel}/flt/lc"),
            "hpf_freq": self.state.get(f"/ch/{channel}/flt/lcf"),
            "eq_on": self.get_eq_on(channel),
            "eq_bands": eq_bands,
            "compressor_enabled": self.get_compressor_on(channel),
            "compressor_model": self.state.get(f"/ch/{channel}/dyn/mdl"),
            "compressor_threshold_db": self.get_compressor_threshold(channel),
            "compressor_ratio": self.get_compressor_ratio(channel),
            "compressor_attack_ms": self.state.get(f"/ch/{channel}/dyn/att"),
            "compressor_release_ms": self.state.get(f"/ch/{channel}/dyn/rel"),
            "compressor_makeup_db": self.get_compressor_gain(channel),
            "compressor_mix_pct": self.state.get(f"/ch/{channel}/dyn/mix"),
            "compressor_detector": self.state.get(f"/ch/{channel}/dyn/det"),
            "compressor_gr_db": self.get_compressor_gr(channel),
            "gate_enabled": self.state.get(f"/ch/{channel}/gate/on"),
            "gate_model": self.state.get(f"/ch/{channel}/gate/mdl"),
            "gate_threshold_db": self.state.get(f"/ch/{channel}/gate/thr"),
            "gate_range_db": self.state.get(f"/ch/{channel}/gate/range"),
            "gate_attack_ms": self.state.get(f"/ch/{channel}/gate/att"),
            "gate_hold_ms": self.state.get(f"/ch/{channel}/gate/hld"),
            "gate_release_ms": self.state.get(f"/ch/{channel}/gate/rel"),
            "gate_accent": self.state.get(f"/ch/{channel}/gate/acc"),
            "gate_ratio": self.state.get(f"/ch/{channel}/gate/ratio"),
        }

    def recall_scene(self, scene_number: int):
        self.load_snap_by_index(scene_number)
