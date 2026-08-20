"""
Ableton Live 12 Client - connects via AbletonOSC Remote Script

AbletonOSC provides a UDP/OSC interface to Ableton Live:
- Send commands to port 11000
- Receive replies on port 11001

Requires AbletonOSC Remote Script installed and activated in Ableton Live.
Audio routing from Ableton to the app is via BlackHole 64ch.
"""

from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
import socket
import threading
import logging
import math
import time
from typing import Callable, Dict, Any, Optional, List, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Volume Conversion ==========

def _db_to_ableton_vol(db: float) -> float:
    """Convert dB to Ableton 0.0-1.0 volume.
    Mapping: -inf->0.0, -70->0.0, 0dB->0.85, +6dB->1.0
    Uses Ableton's log curve approximation.
    """
    if db <= -70:
        return 0.0
    if db >= 6.0:
        return 1.0
    if db >= 0:
        return 0.85 + (db / 6.0) * 0.15
    return 0.85 * (10 ** (db / 38.0))


def _ableton_vol_to_db(vol: float) -> float:
    """Inverse of _db_to_ableton_vol."""
    if vol <= 0.0:
        return -144.0
    if vol >= 1.0:
        return 6.0
    if vol >= 0.85:
        return (vol - 0.85) / 0.15 * 6.0
    return 38.0 * math.log10(vol / 0.85)


class AbletonClient:
    """OSC client for Ableton Live 12 via AbletonOSC Remote Script"""

    _MAX_TRACK_ID = 255
    _MAX_PENDING_REPLIES = 128

    def __init__(self, ip: str = "127.0.0.1", send_port: int = 11000, recv_port: int = 11001):
        """
        Initialize Ableton client

        Args:
            ip: Ableton host IP (usually localhost)
            send_port: AbletonOSC send port (default 11000)
            recv_port: AbletonOSC receive/reply port (default 11001)
        """
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port

        self.sock = None            # UDP socket for sending
        self.recv_sock = None       # UDP socket for receiving replies
        self.receiver_thread = None
        self._stop_receiver = False

        self.state = {}
        self.callbacks = {}

        self.is_connected = False

        self._track_count = 0
        self._track_names: Dict[int, str] = {}  # track_id (0-based) -> name

        # Only correlated replies from the configured AbletonOSC peer may
        # become observed state. Keys are (reply address, optional track id).
        self._pending_replies: Dict[Tuple[str, Optional[int]], float] = {}
        self._reply_timeout_seconds = 1.0
        self._allowed_reply_ips: Set[str] = set()

        # OSC throttle: track last send time per address to limit rate
        self._osc_throttle_enabled = True
        self._osc_throttle_hz = 10.0
        self._osc_last_send_time: Dict[str, float] = {}

        self._lock = threading.Lock()

        logger.info(f"AbletonClient initialized for {ip}:{send_port} (recv: {recv_port})")

    # ========== Connection ==========

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to Ableton Live via AbletonOSC.
        Sends /live/test and waits for a reply on recv_port.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Create send socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Create receive socket bound to recv_port
            self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.recv_sock.settimeout(timeout)
            self._allowed_reply_ips = self._resolve_allowed_reply_ips()
            bind_ip = '127.0.0.1' if self._allowed_reply_ips and all(
                ip.startswith('127.') for ip in self._allowed_reply_ips
            ) else '0.0.0.0'
            self.recv_sock.bind((bind_ip, self.recv_port))
            logger.info(f"Bound receive socket to port {self.recv_port}")

            # Send test message
            logger.info(f"Sending /live/test to {self.ip}:{self.send_port}...")
            builder = OscMessageBuilder(address='/live/test')
            msg = builder.build()
            self.sock.sendto(msg.dgram, (self.ip, self.send_port))

            try:
                osc_msg = self._receive_valid_handshake(timeout)
                logger.info(f"Ableton responded: {osc_msg.address} {osc_msg.params}")
            except socket.timeout:
                logger.error(f"Connection timeout: Ableton at {self.ip}:{self.send_port} did not respond")
                logger.error("Please ensure:")
                logger.error("  1. Ableton Live is running")
                logger.error("  2. AbletonOSC Remote Script is activated in MIDI preferences")
                self.sock.close()
                self.recv_sock.close()
                return False

            self.is_connected = True

            # Start receiver thread
            self._stop_receiver = False
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()

            # Scan initial state
            self._scan_initial_state()

            logger.info(f"Connected to Ableton Live at {self.ip}:{self.send_port}")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from Ableton Live"""
        self._stop_receiver = True
        self.is_connected = False
        with self._lock:
            self._pending_replies.clear()
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.debug(f"Error closing send socket: {e}")
            self.sock = None
        if self.recv_sock:
            try:
                self.recv_sock.close()
            except Exception as e:
                logger.debug(f"Error closing recv socket: {e}")
            self.recv_sock = None
        logger.info("Disconnected from Ableton Live")

    # ========== Receiver ==========

    def _receiver_loop(self):
        """Background thread to receive OSC replies from AbletonOSC"""
        if not self.recv_sock:
            return
        try:
            self.recv_sock.settimeout(0.5)
        except Exception:
            return

        while not self._stop_receiver and self.is_connected:
            if not self.recv_sock:
                break
            try:
                data, addr = self.recv_sock.recvfrom(8192)
                if not self._is_expected_peer(addr):
                    logger.debug(f"Rejected OSC datagram from unexpected peer {addr}")
                    continue
                try:
                    osc_msg = OscMessage(data)
                    self._handle_message(osc_msg.address, *osc_msg.params)
                except Exception as e:
                    logger.debug(f"Error parsing OSC message: {e}")
            except socket.timeout:
                pass
            except OSError as e:
                if e.errno in (9, 57):  # Bad file descriptor or Socket not connected
                    break
                if self.is_connected:
                    logger.debug(f"Receiver OS error: {e}")
            except Exception as e:
                if self.is_connected:
                    logger.debug(f"Receiver error: {e}")

    def _resolve_allowed_reply_ips(self) -> Set[str]:
        """Resolve the configured Ableton host to the exact accepted IPv4 peers."""
        resolved: Set[str] = set()
        for item in socket.getaddrinfo(
            self.ip, self.send_port, socket.AF_INET, socket.SOCK_DGRAM
        ):
            resolved.add(str(item[4][0]))
        if not resolved:
            raise OSError(f"Unable to resolve Ableton host {self.ip!r}")
        return resolved

    def _receive_valid_handshake(self, timeout: float) -> OscMessage:
        """Ignore unrelated UDP traffic until a valid peer handshake or timeout."""
        if not self.recv_sock:
            raise socket.timeout
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout
            self.recv_sock.settimeout(remaining)
            data, addr = self.recv_sock.recvfrom(4096)
            if not self._is_expected_peer(addr):
                logger.debug(f"Ignored Ableton handshake from unexpected peer {addr}")
                continue
            try:
                osc_msg = OscMessage(data)
            except Exception:
                logger.debug("Ignored malformed Ableton handshake datagram")
                continue
            if osc_msg.address != '/live/test':
                logger.debug(f"Ignored unexpected Ableton handshake address {osc_msg.address}")
                continue
            return osc_msg

    def _is_expected_peer(self, addr: Tuple[str, int]) -> bool:
        """Accept replies only from the configured host and AbletonOSC port."""
        try:
            peer_ip, peer_port = addr[0], int(addr[1])
        except (IndexError, TypeError, ValueError):
            return False
        return peer_ip in self._allowed_reply_ips and peer_port == int(self.send_port)

    @staticmethod
    def _reply_key(
        address: str, args: Tuple[Any, ...]
    ) -> Optional[Tuple[str, Optional[int]]]:
        track_replies = {
            '/live/track/get/volume',
            '/live/track/get/mute',
            '/live/track/get/name',
            '/live/track/get/panning',
        }
        if address in track_replies:
            if not args:
                return None
            try:
                track_id = int(args[0])
            except (TypeError, ValueError):
                return None
            if track_id < 0 or track_id > AbletonClient._MAX_TRACK_ID:
                return None
            return address, track_id
        if address in {
            '/live/master/get/volume',
            '/live/song/get/track_names',
            '/live/song/get/num_tracks',
        }:
            return address, None
        return None

    def _register_pending_reply(
        self, address: str, values: Tuple[Any, ...]
    ) -> Optional[Tuple[str, Optional[int]]]:
        key = self._reply_key(address, values)
        if key is None:
            return None
        now = time.monotonic()
        with self._lock:
            self._pending_replies = {
                pending_key: deadline
                for pending_key, deadline in self._pending_replies.items()
                if deadline >= now
            }
            if (
                key not in self._pending_replies
                and len(self._pending_replies) >= self._MAX_PENDING_REPLIES
            ):
                logger.warning("Rejected Ableton query: pending reply limit reached")
                return None
            self._pending_replies[key] = now + self._reply_timeout_seconds
        return key

    def _consume_pending_reply(self, address: str, args: Tuple[Any, ...]) -> bool:
        key = self._reply_key(address, args)
        if key is None:
            return False
        now = time.monotonic()
        with self._lock:
            deadline = self._pending_replies.pop(key, None)
            expired = [
                pending_key
                for pending_key, value in self._pending_replies.items()
                if value < now
            ]
            for pending_key in expired:
                self._pending_replies.pop(pending_key, None)
        return deadline is not None and deadline >= now

    def _send_query(self, address: str, *values: Any) -> bool:
        """Send a fixed-schema query and correlate its next fresh reply."""
        key = self._register_pending_reply(address, values)
        if key is None:
            logger.warning(f"Rejected unsupported Ableton query address: {address}")
            return False
        if self._send_osc(address, *values) is True:
            return True
        with self._lock:
            self._pending_replies.pop(key, None)
        return False

    def _handle_message(self, address: str, *args):
        """Promote a correlated AbletonOSC reply to observed internal state."""
        logger.debug(f"Received: {address} {args}")
        if not self._consume_pending_reply(address, args):
            logger.debug(f"Ignored unsolicited, stale, or unsupported OSC reply: {address}")
            return

        # Translate AbletonOSC replies to internal /ch/{N}/... format for callbacks
        # Volume replies: /live/track/get/volume <track_id> <value>
        if address == '/live/track/get/volume' and len(args) >= 2:
            track_id = int(args[0])
            vol = float(args[1])
            channel = track_id + 1  # 0-based -> 1-based
            db_val = _ableton_vol_to_db(vol)
            internal_addr = f"/ch/{channel}/fdr"
            self.state[internal_addr] = db_val
            self._emit_update(internal_addr, db_val)
            return

        # Mute replies: /live/track/get/mute <track_id> <value>
        if address == '/live/track/get/mute' and len(args) >= 2:
            track_id = int(args[0])
            mute_val = int(args[1])
            channel = track_id + 1
            internal_addr = f"/ch/{channel}/mute"
            self.state[internal_addr] = mute_val
            self._emit_update(internal_addr, mute_val)
            return

        # Name replies: /live/track/get/name <track_id> <name>
        if address == '/live/track/get/name' and len(args) >= 2:
            track_id = int(args[0])
            name = str(args[1])
            channel = track_id + 1
            self._track_names[track_id] = name
            internal_addr = f"/ch/{channel}/name"
            self.state[internal_addr] = name
            self._emit_update(internal_addr, name)
            return

        # Panning replies: /live/track/get/panning <track_id> <value>
        if address == '/live/track/get/panning' and len(args) >= 2:
            track_id = int(args[0])
            pan_norm = float(args[1])  # -1..1
            channel = track_id + 1
            pan_100 = pan_norm * 100.0  # -> -100..100
            internal_addr = f"/ch/{channel}/pan"
            self.state[internal_addr] = pan_100
            self._emit_update(internal_addr, pan_100)
            return

        # Master volume: /live/master/get/volume <value>
        if address == '/live/master/get/volume' and len(args) >= 1:
            vol = float(args[0])
            db_val = _ableton_vol_to_db(vol)
            internal_addr = "/main/1/fdr"
            self.state[internal_addr] = db_val
            self._emit_update(internal_addr, db_val)
            return

        # Track names list: /live/song/get/track_names <name1> <name2> ...
        if address == '/live/song/get/track_names' and 0 < len(args) <= self._MAX_TRACK_ID + 1:
            for i, name in enumerate(args):
                self._track_names[i] = str(name)
                channel = i + 1
                internal_addr = f"/ch/{channel}/name"
                self.state[internal_addr] = str(name)
            self._track_count = len(args)
            self.state[address] = tuple(str(name) for name in args)
            logger.info(f"Received {len(args)} track names from Ableton")
            self._emit_update(address, *args)
            return

        # Track count: /live/song/get/num_tracks <count>
        if address == '/live/song/get/num_tracks' and len(args) >= 1:
            track_count = int(args[0])
            if track_count < 0 or track_count > self._MAX_TRACK_ID + 1:
                logger.debug(f"Ignored out-of-range Ableton track count: {track_count}")
                return
            self._track_count = track_count
            self.state[address] = self._track_count
            logger.info(f"Ableton track count: {self._track_count}")
            self._emit_update(address, self._track_count)
            return

        logger.debug(f"Ignored malformed OSC reply: {address} {args}")

    def _emit_update(self, address: str, *args):
        """Dispatch update to registered callbacks"""
        if address in self.callbacks:
            for callback in self.callbacks[address]:
                try:
                    callback(address, *args)
                except Exception as e:
                    logger.error(f"Callback error for {address}: {e}")
        if "*" in self.callbacks:
            for callback in self.callbacks["*"]:
                try:
                    callback(address, *args)
                except Exception as e:
                    logger.error(f"Global callback error: {e}")

    # ========== Subscribe / State ==========

    def subscribe(self, address_pattern: str, callback: Callable):
        """Subscribe to OSC address changes"""
        if address_pattern not in self.callbacks:
            self.callbacks[address_pattern] = []
        self.callbacks[address_pattern].append(callback)

    def get_state(self) -> Dict[str, Any]:
        """Get copy of current mixer state"""
        return self.state.copy()

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

    def query(self, address: str) -> Optional[Any]:
        """Query a parameter and wait for response"""
        if self._send_query(address) is not True:
            return None
        time.sleep(0.15)
        return self.state.get(address)

    # ========== OSC Send ==========

    def _send_osc(self, address: str, *values):
        """Send raw OSC message to AbletonOSC"""
        if not self.is_connected or not self.sock:
            logger.warning("Not connected to Ableton")
            return False

        try:
            builder = OscMessageBuilder(address=address)
            for v in values:
                builder.add_arg(v)
            msg = builder.build()
            self.sock.sendto(msg.dgram, (self.ip, self.send_port))
            logger.debug(f"Sent: {address} {values}")
            return True
        except OSError as e:
            if e.errno in (9, 57, 32):
                logger.warning(f"Socket error, connection lost: {e}")
                self.is_connected = False
            else:
                logger.error(f"Send OSError: {e}")
            return False
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def send(self, address: str, *values):
        """
        Send OSC message — compatibility shim.
        Translates Wing-style addresses to AbletonOSC if needed,
        otherwise sends directly.
        """
        if not self.is_connected:
            logger.warning("Not connected to Ableton")
            return False

        if not self.sock:
            logger.warning("Socket not available")
            self.is_connected = False
            return False

        # Throttle check: only for commands with values (not queries)
        if self._osc_throttle_enabled and values:
            current_time = time.time()
            min_interval = 1.0 / self._osc_throttle_hz

            if address in self._osc_last_send_time:
                time_since_last = current_time - self._osc_last_send_time[address]
                if time_since_last < min_interval:
                    logger.debug(f"OSC throttle: skipping {address}")
                    return False

            self._osc_last_send_time[address] = current_time

        # Translate Wing-style /ch/N/fdr addresses to AbletonOSC
        import re
        ch_fdr_match = re.match(r'^/ch/(\d+)/fdr$', address)
        if ch_fdr_match:
            ch = int(ch_fdr_match.group(1))
            if values:
                return self.set_channel_fader(ch, float(values[0])) is True
            return self._send_query('/live/track/get/volume', ch - 1) is True

        ch_mute_match = re.match(r'^/ch/(\d+)/mute$', address)
        if ch_mute_match:
            ch = int(ch_mute_match.group(1))
            if values:
                return self.set_channel_mute(ch, int(values[0])) is True
            return self._send_query('/live/track/get/mute', ch - 1) is True

        ch_name_match = re.match(r'^/ch/(\d+)/(\$?name)$', address)
        if ch_name_match:
            ch = int(ch_name_match.group(1))
            if values:
                return self.set_channel_name(ch, str(values[0])) is True
            return self._send_query('/live/track/get/name', ch - 1) is True

        ch_pan_match = re.match(r'^/ch/(\d+)/pan$', address)
        if ch_pan_match:
            ch = int(ch_pan_match.group(1))
            if values:
                return self.set_channel_pan(ch, float(values[0])) is True
            return self._send_query('/live/track/get/panning', ch - 1) is True

        main_fdr_match = re.match(r'^/main/(\d+)/fdr$', address)
        if main_fdr_match:
            if values:
                return self.set_main_fader(1, float(values[0])) is True
            return self._send_query('/live/master/get/volume') is True

        # Pass through as raw AbletonOSC
        return self._send_osc(address, *values)

    # ========== Initial State Scan ==========

    def _scan_initial_state(self):
        """Scan initial Ableton Live state"""
        logger.info("Scanning Ableton Live state...")

        # Get track count
        self._send_query('/live/song/get/num_tracks')
        time.sleep(0.1)

        # Get all track names
        self._send_query('/live/song/get/track_names')
        time.sleep(0.3)

        # Scan first 16 tracks (fader, mute, name, panning)
        for track_id in range(16):
            self._send_query('/live/track/get/volume', track_id)
            self._send_query('/live/track/get/mute', track_id)
            self._send_query('/live/track/get/name', track_id)
            self._send_query('/live/track/get/panning', track_id)
            time.sleep(0.02)

        # Get master volume
        self._send_query('/live/master/get/volume')

        time.sleep(0.5)
        logger.info(f"Initial state scan complete, received {len(self.state)} parameters")

        # Log found track names
        if self._track_names:
            names_str = ', '.join(f"T{tid}: {name}" for tid, name in sorted(self._track_names.items()))
            logger.info(f"Found track names: {names_str}")

    # ========== Channel Methods ==========

    def get_channel_fader(self, channel: int) -> Optional[float]:
        """Get channel fader value in dB"""
        internal_addr = f"/ch/{channel}/fdr"
        val = self.state.get(internal_addr)
        if val is not None:
            return float(val)
        # Try querying from Ableton
        track_id = channel - 1
        self._send_query('/live/track/get/volume', track_id)
        time.sleep(0.1)
        return self.state.get(internal_addr)

    def set_channel_fader(self, channel: int, value: float):
        """
        Set channel fader value.
        Accepts dB values (same interface as WingClient) and converts to Ableton 0-1.
        """
        track_id = channel - 1
        db_value = float(value)
        if db_value < -144.0:
            db_value = -144.0
        elif db_value > 6.0:
            db_value = 6.0

        vol = _db_to_ableton_vol(db_value)
        logger.info(f"Setting fader ch {channel} (track {track_id}) = {db_value:.2f} dB -> vol {vol:.4f}")
        return self._send_osc('/live/track/set/volume', track_id, vol) is True

    def get_channel_mute(self, channel: int) -> Optional[int]:
        """Get channel mute state (0=unmuted, 1=muted)"""
        internal_addr = f"/ch/{channel}/mute"
        return self.state.get(internal_addr)

    def set_channel_mute(self, channel: int, value: int):
        """Set channel mute (0=unmuted, 1=muted)"""
        track_id = channel - 1
        return self._send_osc('/live/track/set/mute', track_id, int(value)) is True

    def get_channel_gain(self, channel: int) -> Optional[float]:
        """Get channel input trim/gain — not available in Ableton, returns 0"""
        return 0.0

    def set_channel_gain(self, channel: int, value: float):
        """Set channel input trim/gain — no pre-fader gain in Ableton, no-op"""
        logger.warning(f"set_channel_gain: No pre-fader gain in Ableton (ch {channel}, value {value})")
        return False

    def get_channel_pan(self, channel: int) -> Optional[float]:
        """Get channel pan (-100..100)"""
        internal_addr = f"/ch/{channel}/pan"
        return self.state.get(internal_addr)

    def set_channel_pan(self, channel: int, value: float):
        """Set channel pan (-100=left, 0=center, 100=right)"""
        track_id = channel - 1
        # Convert -100..100 to -1..1
        pan_norm = max(-1.0, min(1.0, float(value) / 100.0))
        return self._send_osc('/live/track/set/panning', track_id, pan_norm) is True

    def get_channel_name(self, channel: int, retries: int = 3) -> Optional[str]:
        """Get channel name"""
        track_id = channel - 1
        internal_addr = f"/ch/{channel}/name"

        # Check cached first
        name = self.state.get(internal_addr)
        if name:
            return name

        # Query from Ableton
        for attempt in range(retries):
            self._send_query('/live/track/get/name', track_id)
            time.sleep(0.15)
            name = self.state.get(internal_addr)
            if name:
                return name

        return None

    def get_all_channel_names(self, num_channels: int = 64) -> Dict[int, str]:
        """
        Get names of all channels (batch query).

        Returns:
            Dict mapping channel number (1-based) to name
        """
        # Request all track names at once
        self._send_query('/live/song/get/track_names')
        time.sleep(0.5)

        names = {}
        for ch in range(1, num_channels + 1):
            name = self.state.get(f"/ch/{ch}/name")
            if name and name != '':
                names[ch] = name

        logger.info(f"Retrieved {len(names)} channel names")
        return names

    def set_channel_name(self, channel: int, name: str):
        """Set channel name"""
        track_id = channel - 1
        return self._send_osc('/live/track/set/name', track_id, name) is True

    def set_channel_color(self, channel: int, color: int):
        """Set channel color — Ableton uses different color model, no-op"""
        logger.warning(f"set_channel_color: Not mapped for Ableton (ch {channel})")
        return False

    # ========== Main / Master ==========

    def get_main_fader(self, main: int = 1) -> Optional[float]:
        """Get master fader value in dB"""
        internal_addr = "/main/1/fdr"
        val = self.state.get(internal_addr)
        if val is not None:
            return float(val)
        self._send_query('/live/master/get/volume')
        time.sleep(0.1)
        return self.state.get(internal_addr)

    def set_main_fader(self, main: int, value: float):
        """Set master fader value (dB)"""
        db_value = max(-144.0, min(6.0, float(value)))
        vol = _db_to_ableton_vol(db_value)
        logger.info(f"Setting master fader = {db_value:.2f} dB -> vol {vol:.4f}")
        return self._send_osc('/live/master/set/volume', vol) is True

    def set_main_mute(self, main: int, value: int):
        """Set main mute — not directly available in Ableton"""
        logger.warning("set_main_mute: Not available in Ableton")
        return False

    def set_main_pan(self, main: int, value: float):
        """Set main pan — limited support in Ableton"""
        pan_norm = max(-1.0, min(1.0, float(value) / 100.0))
        return self._send_osc('/live/master/set/panning', pan_norm) is True

    def set_main_eq_on(self, main: int, on: int):
        """Set main EQ on/off — not directly available in Ableton"""
        logger.warning("set_main_eq_on: Not available in Ableton")
        return False

    # ========== EQ Methods (via EQ Eight device) ==========

    def set_eq_on(self, channel: int, on: int):
        """Enable/disable EQ — toggles first device on track (assumes EQ Eight)"""
        track_id = channel - 1
        return self._send_osc('/live/device/set/enabled', track_id, 0, int(on)) is True

    def get_eq_on(self, channel: int) -> Optional[int]:
        """Get EQ on/off state"""
        return self.state.get(f"/ch/{channel}/eq/on")

    def set_eq_band(self, channel: int, band: int, freq: float = None,
                    gain: float = None, q: float = None):
        """
        Set EQ band parameters via EQ Eight device (device index 0 assumed).
        EQ Eight parameter indices (per band, 0-based):
          Band N: Gain = 1 + (N-1)*8 + 2, Freq = 1 + (N-1)*8 + 3, Q = 1 + (N-1)*8 + 4
          (Approximate — actual indices depend on EQ Eight version)

        For simplicity, we use:
          Band 1: Gain=1, Freq=2, Q=3
          Band 2: Gain=9, Freq=10, Q=11
          Band 3: Gain=17, Freq=18, Q=19
          Band 4: Gain=25, Freq=26, Q=27
        """
        track_id = channel - 1
        device_id = 0  # Assumes EQ Eight is the first device

        # EQ Eight parameter offset per band (approximate mapping)
        band_offsets = {1: 1, 2: 9, 3: 17, 4: 25}
        if band not in band_offsets:
            logger.warning(f"set_eq_band: Invalid band {band} (expected 1-4)")
            return False

        base_param = band_offsets[band]
        results: List[bool] = []

        if gain is not None:
            # Normalize gain from dB (-15..15) to 0..1 (center = 0.5)
            gain_norm = max(0.0, min(1.0, (gain + 15.0) / 30.0))
            results.append(self._send_osc(
                '/live/device/set/parameter/value',
                track_id, device_id, base_param, gain_norm,
            ) is True)

        if freq is not None:
            # Normalize freq from Hz (20..20000) to 0..1 (log scale)
            if freq <= 20:
                freq_norm = 0.0
            elif freq >= 20000:
                freq_norm = 1.0
            else:
                freq_norm = math.log10(freq / 20.0) / math.log10(20000.0 / 20.0)
            results.append(self._send_osc(
                '/live/device/set/parameter/value',
                track_id, device_id, base_param + 1, freq_norm,
            ) is True)

        if q is not None:
            # Normalize Q from 0.1..18 to 0..1
            q_norm = max(0.0, min(1.0, (q - 0.1) / 17.9))
            results.append(self._send_osc(
                '/live/device/set/parameter/value',
                track_id, device_id, base_param + 2, q_norm,
            ) is True)

        return bool(results) and all(results)

    def set_eq_band_gain(self, channel: int, band: str, gain: float):
        """Set EQ band gain — translates band string to numeric band"""
        band_map = {"1g": 1, "2g": 2, "3g": 3, "4g": 4, "lg": 1, "hg": 4}
        band_num = band_map.get(band)
        if band_num:
            return self.set_eq_band(channel, band_num, gain=gain) is True
        return False

    def get_eq_band_gain(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band gain — returns from state if available"""
        return self.state.get(f"/ch/{channel}/eq/{band}")

    def set_eq_band_frequency(self, channel: int, band: str, frequency: float):
        """Set EQ band frequency"""
        band_map = {"1f": 1, "2f": 2, "3f": 3, "4f": 4, "lf": 1, "hf": 4}
        band_num = band_map.get(band)
        if band_num:
            return self.set_eq_band(channel, band_num, freq=frequency) is True
        return False

    def get_eq_band_frequency(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band frequency"""
        return self.state.get(f"/ch/{channel}/eq/{band}")

    def set_eq_band_q(self, channel: int, band: str, q: float):
        """Set EQ band Q"""
        band_map = {"1q": 1, "2q": 2, "3q": 3, "4q": 4, "lq": 1, "hq": 4}
        band_num = band_map.get(band)
        if band_num:
            return self.set_eq_band(channel, band_num, q=q) is True
        return False

    def set_eq_high_shelf(self, channel: int, gain: float = None, freq: float = None,
                          q: float = None, eq_type: str = None):
        """Set EQ high shelf — maps to band 4"""
        return self.set_eq_band(channel, 4, freq=freq, gain=gain, q=q) is True

    def set_eq_low_shelf(self, channel: int, gain: float = None, freq: float = None,
                         q: float = None, eq_type: str = None):
        """Set EQ low shelf — maps to band 1"""
        return self.set_eq_band(channel, 1, freq=freq, gain=gain, q=q) is True

    def set_eq_mix(self, channel: int, mix: float):
        """Set EQ mix — not available in EQ Eight"""
        logger.warning("set_eq_mix: Not available in Ableton EQ Eight")
        return False

    def set_eq_model(self, channel: int, model: str):
        """Set EQ model — Ableton only has EQ Eight"""
        logger.warning("set_eq_model: Not available in Ableton")
        return False

    # ========== Compressor / Dynamics (unsupported) ==========

    def set_compressor_on(self, channel: int, on: int):
        """Not supported — Ableton compressor requires device discovery"""
        logger.warning(f"set_compressor_on: Not supported for Ableton (ch {channel})")
        return False

    def get_compressor_on(self, channel: int) -> Optional[int]:
        """Not supported"""
        return None

    def set_compressor(self, channel: int, threshold: float = None, ratio: str = None,
                       knee: int = None, attack: float = None, hold: float = None,
                       release: float = None, gain: float = None, mix: float = None,
                       det: str = None, env: str = None, auto: int = None):
        """Not supported — Ableton compressor requires device discovery"""
        logger.warning(f"set_compressor: Not supported for Ableton (ch {channel})")
        return False

    def set_compressor_threshold(self, channel: int, threshold: float):
        """Not supported"""
        logger.warning(f"set_compressor_threshold: Not supported for Ableton (ch {channel})")
        return False

    def set_compressor_ratio(self, channel: int, ratio: str):
        """Not supported"""
        logger.warning(f"set_compressor_ratio: Not supported for Ableton (ch {channel})")
        return False

    def set_compressor_gain(self, channel: int, gain: float):
        """Not supported"""
        logger.warning(f"set_compressor_gain: Not supported for Ableton (ch {channel})")
        return False

    def get_compressor_gain(self, channel: int) -> Optional[float]:
        """Not supported"""
        return None

    def set_compressor_attack(self, channel: int, attack: float):
        """Not supported"""
        logger.warning(f"set_compressor_attack: Not supported for Ableton (ch {channel})")
        return False

    def set_compressor_release(self, channel: int, release: float):
        """Not supported"""
        logger.warning(f"set_compressor_release: Not supported for Ableton (ch {channel})")
        return False

    def set_compressor_knee(self, channel: int, knee: float):
        """Not supported"""
        return False

    def set_compressor_mix(self, channel: int, mix: float):
        """Not supported"""
        return False

    def set_compressor_model(self, channel: int, model: str):
        """Not supported"""
        return False

    def get_compressor_gr(self, channel: int) -> Optional[float]:
        """Not supported"""
        return None

    # ========== Gate (unsupported) ==========

    def set_gate_on(self, channel: int, on: int):
        """Not supported"""
        logger.warning(f"set_gate_on: Not supported for Ableton (ch {channel})")
        return False

    def set_gate(self, channel: int, threshold: float = None, range_db: float = None,
                 attack: float = None, hold: float = None, release: float = None,
                 accent: float = None, ratio: str = None):
        """Not supported"""
        return False

    def set_gate_model(self, channel: int, model: str):
        """Not supported"""
        return False

    # ========== Filter Methods (unsupported) ==========

    def set_low_cut(self, channel: int, enabled: int, frequency: float = None, slope: str = None):
        """Not directly supported — requires Auto Filter device"""
        logger.warning(f"set_low_cut: Not supported for Ableton (ch {channel})")
        return False

    def set_high_cut(self, channel: int, enabled: int, frequency: float = None, slope: str = None):
        """Not directly supported"""
        logger.warning(f"set_high_cut: Not supported for Ableton (ch {channel})")
        return False

    # ========== DCA (unsupported — no DCAs in Ableton) ==========

    def get_dca_fader(self, dca: int) -> Optional[float]:
        """No DCAs in Ableton"""
        return None

    def set_dca_fader(self, dca: int, value: float):
        """No DCAs in Ableton"""
        logger.warning("set_dca_fader: No DCAs in Ableton")
        return False

    def set_dca_mute(self, dca: int, value: int):
        """No DCAs in Ableton"""
        return False

    # ========== Bus Methods (unsupported — Ableton uses return tracks) ==========

    def set_bus_fader(self, bus: int, value: float):
        """Not directly supported — Ableton uses return tracks"""
        logger.warning(f"set_bus_fader: Not mapped for Ableton (bus {bus})")
        return False

    def set_bus_mute(self, bus: int, value: int):
        """Not supported"""
        return False

    def set_bus_pan(self, bus: int, value: float):
        """Not supported"""
        return False

    def set_bus_eq_on(self, bus: int, on: int):
        """Not supported"""
        return False

    def set_bus_eq_band(self, bus: int, band: int, freq: float = None,
                        gain: float = None, q: float = None):
        """Not supported"""
        return False

    # ========== Send Methods ==========

    def set_channel_send(self, channel: int, send: int, level: float = None,
                         on: int = None, mode: str = None, pan: float = None):
        """
        Set channel send level.
        In Ableton, sends go to return tracks.
        """
        track_id = channel - 1
        send_id = send - 1  # 0-based
        results: List[bool] = []
        if level is not None:
            vol = _db_to_ableton_vol(level)
            results.append(
                self._send_osc('/live/track/set/send', track_id, send_id, vol) is True
            )
        if on is not None:
            # Ableton doesn't have send on/off — set to 0 volume if off
            if not on:
                results.append(
                    self._send_osc('/live/track/set/send', track_id, send_id, 0.0) is True
                )
        return bool(results) and all(results)

    def set_channel_main_send(self, channel: int, main: int, level: float = None,
                              on: int = None, pre: int = None):
        """Not directly mapped in Ableton"""
        logger.warning("set_channel_main_send: Not mapped for Ableton")
        return False

    # ========== Unsupported Methods (hardware / Wing-specific) ==========

    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        """Not supported — no channel delay in Ableton"""
        logger.warning(f"set_channel_delay: Not supported for Ableton (ch {channel})")
        return False

    def set_channel_phase_invert(self, channel: int, value: int):
        """Not supported — requires Utility device"""
        logger.warning(f"set_channel_phase_invert: Not supported for Ableton (ch {channel})")
        return False

    def set_channel_width(self, channel: int, value: float):
        """Not supported"""
        return False

    def set_channel_balance(self, channel: int, value: float):
        """Not supported"""
        return False

    def set_preamp_pad(self, channel: int, value: int):
        """Not supported — hardware only"""
        return False

    def set_preamp_48v(self, channel: int, value: int):
        """Not supported — hardware only"""
        return False

    # ========== Routing (unsupported) ==========

    def route_output(self, output_group: str, output_number: int,
                     source_group: str, source_channel: int):
        """Not supported"""
        return False

    def route_multiple_outputs(self, output_group: str, start_output: int,
                               num_outputs: int, source_group: str,
                               start_source_channel: int):
        """Not supported"""
        return 0

    def get_output_routing(self, output_group: str, output_number: int):
        """Not supported"""
        return None

    def set_channel_input(self, channel: int, source_group: str, source_channel: int):
        """Not supported"""
        return False

    def set_channel_alt_input(self, channel: int, source_group: str, source_channel: int):
        """Not supported"""
        return False

    def get_channel_input_routing(self, channel: int):
        """Not supported"""
        return None

    # ========== Snapshot / Scene (unsupported) ==========

    def load_snap(self, snap_name: str, max_index: int = 200):
        """Not supported"""
        logger.warning("load_snap: Not supported for Ableton")
        return False

    def find_snap_by_name(self, snap_name: str, max_index: int = 200):
        """Not supported"""
        return None

    def load_snap_by_index(self, snap_index: int):
        """Not supported"""
        return False

    def save_snap(self, snap_name: str):
        """Not supported"""
        return False

    def get_snap_list(self):
        """Not supported"""
        return {}

    def get_current_show(self):
        """Not supported"""
        return None

    # ========== FX Module Methods (unsupported) ==========

    def set_fx_parameter(self, fx_slot: str, parameter: int, value):
        """Not supported"""
        return False

    def set_fx_model(self, fx_slot: str, model: str):
        """Not supported"""
        return False

    def set_fx_on(self, fx_slot: str, on: int):
        """Not supported"""
        return False

    def set_fx_mix(self, fx_slot: str, mix: float):
        """Not supported"""
        return False

    def get_fx_parameter(self, fx_slot: str, parameter: int):
        """Not supported"""
        return None

    def get_channel_inserts(self, channel: int):
        """Not supported"""
        return {'pre_insert': None, 'post_insert': None}

    def get_all_channels_with_inserts(self):
        """Not supported"""
        return {}

    # ========== Mute Group (unsupported) ==========

    def set_mute_group_assign(self, channel: int, group: int, value: int):
        """Not supported"""
        return False

    def set_dca_assign(self, channel: int, dca: int, value: int):
        """Not supported"""
        return False
