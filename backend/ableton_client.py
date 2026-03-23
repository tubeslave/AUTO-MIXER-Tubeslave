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
import re
import socket
import threading
import logging
import math
import subprocess
import time
from typing import Callable, Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Volume Conversion ==========
# Ableton Live track volume curve (marcobn, AbletonOSC #44).
# Linear 0-1 maps to dB -inf..+6. 0 dB ≈ 0.85, +6 dB = 1.0.

_ABLETON_ALPHA = 799.503788
_ABLETON_BETA = 12630.61132
_ABLETON_GAMMA = 201.871345
_ABLETON_DELTA = 399.751894
_ABLETON_A = 70.0
_ABLETON_B = 118.426374
_ABLETON_G = 7504.0 / 5567.0


def _db_to_ableton_vol(db: float) -> float:
    """Convert dB to Ableton 0.0-1.0 volume (marcobn curve)."""
    if db >= 6.0:
        return 1.0
    if db >= -18.0:
        return (db + 34.0) / 40.0
    if db >= -41.0:
        return -(math.sqrt(-_ABLETON_ALPHA * db - _ABLETON_BETA) - _ABLETON_GAMMA) / _ABLETON_DELTA
    if db >= -70.0:
        return math.pow((db + _ABLETON_A) / _ABLETON_B, _ABLETON_G)
    return 0.0


def _ableton_vol_to_db(vol: float) -> float:
    """Convert Ableton 0.0-1.0 volume to dB (marcobn curve)."""
    if vol <= 0.0:
        return -144.0
    if vol >= 1.0:
        return 6.0
    if vol >= 0.4:
        return 40.0 * vol - 34.0
    if vol >= 0.15:
        return -((_ABLETON_DELTA * vol - _ABLETON_GAMMA) ** 2 + _ABLETON_BETA) / _ABLETON_ALPHA
    return _ABLETON_B * math.pow(vol, 1.0 / _ABLETON_G) - _ABLETON_A


def _eq_norm_to_gain_db(norm: float) -> float:
    """Inverse of EQ Eight gain mapping in ``_set_eq_eight_physical_band`` (0..1 → −15..+15 dB)."""
    n = max(0.0, min(1.0, float(norm)))
    return n * 30.0 - 15.0


def _eq_norm_to_freq_hz(norm: float) -> float:
    """Inverse log fader 20 Hz … 20 kHz (same as set side)."""
    n = max(0.0, min(1.0, float(norm)))
    if n <= 0.0:
        return 20.0
    if n >= 1.0:
        return 20000.0
    return 20.0 * (1000.0**n)


def _eq_norm_to_q(norm: float) -> float:
    """Inverse Q mapping (0..1 → 0.1 … 18.0)."""
    n = max(0.0, min(1.0, float(norm)))
    return n * 17.9 + 0.1


class AbletonClient:
    """OSC client for Ableton Live 12 via AbletonOSC Remote Script"""

    # Live device.parameters: index 0 = Device On/Off (bypass), never touch it!
    # Utility: 0=Device On, 1=Left Inv, 2=Right Inv, 3=Channel Mode, 4=Stereo Width, 5=Mono, ...
    UTILITY_PHASE_INVERT_L = 1
    UTILITY_PHASE_INVERT_R = 2
    # Delay: mixer control / track option — встроенный Track Delay (значение в ms)
    MAX_DELAY_MS = 500.0
    DEFAULT_LIVE_WINDOW_TITLE = "AUTOMIX ABLETON"
    DEFAULT_TRACK_DELAY_X_RATIO = 0.875
    DEFAULT_TRACK_DELAY_FIRST_ROW_CENTER_Y_RATIO = 0.200
    DEFAULT_TRACK_DELAY_ROW_PITCH_Y_RATIO = 0.098
    DEFAULT_TRACK_DELAY_BASE_CHANNEL = 1

    def __init__(self, ip: str = "127.0.0.1", send_port: int = 11000, recv_port: int = 11001,
                 channel_offset: int = 0, utility_device_index: int = 0,
                 eq_eight_device_index: int = 1, delay_device_index: int = 1,
                 live_window_title: str = DEFAULT_LIVE_WINDOW_TITLE,
                 track_delay_x_ratio: float = DEFAULT_TRACK_DELAY_X_RATIO,
                 track_delay_first_row_center_y_ratio: float = DEFAULT_TRACK_DELAY_FIRST_ROW_CENTER_Y_RATIO,
                 track_delay_row_pitch_y_ratio: float = DEFAULT_TRACK_DELAY_ROW_PITCH_Y_RATIO,
                 track_delay_base_channel: int = DEFAULT_TRACK_DELAY_BASE_CHANNEL):
        """
        Initialize Ableton client

        Args:
            ip: Ableton host IP (usually localhost)
            send_port: AbletonOSC send port (default 11000)
            recv_port: AbletonOSC receive/reply port (default 11001)
            channel_offset: Track offset (channel 1 -> Ableton track channel_offset)
            utility_device_index: Device index for Utility (phase invert), default 0
            eq_eight_device_index: Device index for EQ Eight (e.g. 1 if Utility is 0), default 1
            delay_device_index: (legacy, не используется — задержка через Track Option)
            live_window_title: Visible Ableton window title for UI automation fallback
            track_delay_x_ratio: Relative X of Track Delay input within Live window
            track_delay_first_row_center_y_ratio: Relative Y center of the base channel row
            track_delay_row_pitch_y_ratio: Relative vertical distance between channel rows
            track_delay_base_channel: Channel number used for the first calibrated row
        """
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.channel_offset = channel_offset
        self.utility_device_index = utility_device_index
        self.eq_eight_device_index = int(eq_eight_device_index)
        self.delay_device_index = delay_device_index
        self.live_window_title = live_window_title
        self.track_delay_x_ratio = track_delay_x_ratio
        self.track_delay_first_row_center_y_ratio = track_delay_first_row_center_y_ratio
        self.track_delay_row_pitch_y_ratio = track_delay_row_pitch_y_ratio
        self.track_delay_base_channel = track_delay_base_channel

        self.sock = None            # UDP socket for sending
        self.recv_sock = None       # UDP socket for receiving replies
        self.receiver_thread = None
        self._stop_receiver = False

        self.state = {}
        self.callbacks = {}

        self.is_connected = False

        self._track_count = 0
        self._track_names: Dict[int, str] = {}  # track_id (0-based) -> name

        # OSC throttle: track last send time per address to limit rate
        self._osc_throttle_enabled = True
        self._osc_throttle_hz = 10.0
        self._osc_last_send_time: Dict[str, float] = {}

        self._lock = threading.Lock()

        # Latency param discovery: mixer device (index 0) parameter for Track Delay
        self._latency_param_index: Optional[int] = None
        self._latency_param_min: Optional[float] = None
        self._latency_param_max: Optional[float] = None
        self._latency_discovery_event = threading.Event()
        self._latency_discovery_done = False

        # Utility param discovery: Left Inv, Right Inv (по имени, не хардкод)
        self._utility_left_inv_index: Optional[int] = None
        self._utility_right_inv_index: Optional[int] = None
        self._utility_device_index_found: Optional[int] = None
        self._utility_discovery_event = threading.Event()
        self._utility_discovery_done = False
        self._track_delay_ui_calibrated = False

        # EQ Eight: Gain/Freq/Q param indices per physical band 1..8
        self._eq_band_param_indices: Optional[Dict[int, Tuple[int, int, int]]] = None
        # Last raw parameter name list for eq_eight_device (same order as param indices)
        self._eq_eight_last_param_names: Optional[List[str]] = None
        self._eq_discovery_event = threading.Event()
        self._eq_discovery_done = False

        # Last values from /live/device/get/parameter/value (track_id, device_id, param_index)
        self._device_param_values: Dict[Tuple[int, int, int], float] = {}

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
            self.recv_sock.bind(('0.0.0.0', self.recv_port))
            logger.info(f"Bound receive socket to port {self.recv_port}")

            # Send test message
            logger.info(f"Sending /live/test to {self.ip}:{self.send_port}...")
            builder = OscMessageBuilder(address='/live/test')
            msg = builder.build()
            self.sock.sendto(msg.dgram, (self.ip, self.send_port))

            try:
                data, addr = self.recv_sock.recvfrom(4096)
                osc_msg = OscMessage(data)
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

    def _handle_message(self, address: str, *args):
        """Handle incoming OSC reply from AbletonOSC and translate to internal format"""
        logger.debug(f"Received: {address} {args}")

        # Store raw Ableton state
        if len(args) == 1:
            self.state[address] = args[0]
        elif len(args) > 1:
            self.state[address] = args
        else:
            self.state[address] = None

        # Translate AbletonOSC replies to internal /ch/{N}/... format for callbacks
        # Volume replies: /live/track/get/volume <track_id> <value>
        if address == '/live/track/get/volume' and len(args) >= 2:
            track_id = int(args[0])
            vol = float(args[1])
            channel = track_id - self.channel_offset + 1  # track_id -> 1-based channel
            if channel < 1:
                return  # Track below our offset, ignore
            db_val = _ableton_vol_to_db(vol)
            internal_addr = f"/ch/{channel}/fdr"
            self.state[internal_addr] = db_val
            self._emit_update(internal_addr, db_val)
            return

        # Mute replies: /live/track/get/mute <track_id> <value>
        if address == '/live/track/get/mute' and len(args) >= 2:
            track_id = int(args[0])
            mute_val = int(args[1])
            channel = track_id - self.channel_offset + 1
            if channel < 1:
                return
            internal_addr = f"/ch/{channel}/mute"
            self.state[internal_addr] = mute_val
            self._emit_update(internal_addr, mute_val)
            return

        # Name replies: /live/track/get/name <track_id> <name>
        if address == '/live/track/get/name' and len(args) >= 2:
            track_id = int(args[0])
            name = str(args[1])
            channel = track_id - self.channel_offset + 1
            if channel < 1:
                return
            self._track_names[track_id] = name
            internal_addr = f"/ch/{channel}/name"
            self.state[internal_addr] = name
            self._emit_update(internal_addr, name)
            return

        # Panning replies: /live/track/get/panning <track_id> <value>
        if address == '/live/track/get/panning' and len(args) >= 2:
            track_id = int(args[0])
            pan_norm = float(args[1])  # -1..1
            channel = track_id - self.channel_offset + 1
            if channel < 1:
                return
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
        if address == '/live/song/get/track_names' and args:
            for i, name in enumerate(args):
                self._track_names[i] = str(name)
                channel = i - self.channel_offset + 1
                if channel >= 1:
                    internal_addr = f"/ch/{channel}/name"
                    self.state[internal_addr] = str(name)
            self._track_count = len(args)
            logger.info(f"Received {len(args)} track names from Ableton")
            self._emit_update(address, *args)
            return

        # Track count: /live/song/get/num_tracks <count>
        if address == '/live/song/get/num_tracks' and len(args) >= 1:
            self._track_count = int(args[0])
            logger.info(f"Ableton track count: {self._track_count}")

        # Single parameter readback: /live/device/get/parameter/value <track> <device> <param> <0..1>
        if address == "/live/device/get/parameter/value" and len(args) >= 4:
            try:
                track_id = int(args[0])
                device_id = int(args[1])
                param_idx = int(args[2])
                value = float(args[3])
            except (TypeError, ValueError):
                logger.debug("get/parameter/value: bad args %s", args)
            else:
                key = (track_id, device_id, param_idx)
                with self._lock:
                    self._device_param_values[key] = value
                logger.debug(
                    "Ableton readback: track=%s device=%s param=%s value=%.6f",
                    track_id,
                    device_id,
                    param_idx,
                    value,
                )

        # Device parameters: /live/device/get/parameters/name track_id device_id name1 name2 ...
        if address == '/live/device/get/parameters/name' and len(args) >= 3:
            track_id = int(args[0])
            device_id = int(args[1])
            names = [str(a).strip() for a in args[2:]]

            if device_id == 0:  # Mixer device
                for i, n in enumerate(names):
                    nl = n.lower().strip()
                    if nl in ('track delay', 'track_delay', 'latency', 'delay') or 'track delay' in nl:
                        with self._lock:
                            self._latency_param_index = i
                            self._latency_discovery_done = True
                        logger.info(f"Found Track Delay param at mixer device index {i} (names: {names})")
                        self._latency_discovery_event.set()
                        break

            # Utility: ищем Left Inv, Right Inv (любой device, не только 0)
            left_idx = right_idx = None
            for i, n in enumerate(names):
                nl = n.lower().strip()
                if 'left' in nl and ('inv' in nl or 'phase' in nl):
                    left_idx = i
                elif 'right' in nl and ('inv' in nl or 'phase' in nl):
                    right_idx = i
            if left_idx is not None and right_idx is not None:
                with self._lock:
                    self._utility_left_inv_index = left_idx
                    self._utility_right_inv_index = right_idx
                    self._utility_device_index_found = device_id
                    self._utility_discovery_done = True
                logger.info(f"Found Utility phase: device={device_id} Left Inv={left_idx}, Right Inv={right_idx}")
                self._utility_discovery_event.set()

            # EQ Eight on device eq_eight_device_index (e.g. 1 when Utility is 0)
            if device_id == self.eq_eight_device_index:
                looks_like_eq8 = len(names) >= 8 or any(
                    re.match(r"^\d+\s+Gain\b", str(x).strip(), re.I) for x in names
                )
                if looks_like_eq8:
                    parsed = self._parse_eq_eight_parameter_names(names)
                    with self._lock:
                        self._eq_band_param_indices = parsed
                        self._eq_eight_last_param_names = list(names)
                        self._eq_discovery_done = True
                    logger.info(
                        "EQ Eight parameter map (device %s, phys bands 1–8 gain/freq/q): %s",
                        device_id,
                        parsed,
                    )
                    self._eq_discovery_event.set()

        # For all other messages, emit with original address
        self._emit_update(address, *args)

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
        self._send_osc(address)
        time.sleep(0.15)
        return self.state.get(address)

    # ========== OSC Send ==========

    def _send_osc(self, address: str, *values):
        """Send raw OSC message to AbletonOSC"""
        if not self.is_connected or not self.sock:
            logger.warning("Not connected to Ableton")
            return False

        try:
            vals = list(values)
            # AbletonOSC expects int indices for device param address (strict typing in Live API)
            if address == "/live/device/set/parameter/value" and len(vals) >= 4:
                vals[0] = int(vals[0])
                vals[1] = int(vals[1])
                vals[2] = int(vals[2])
                vals[3] = float(vals[3])
            if address == "/live/device/get/parameter/value" and len(vals) >= 3:
                vals[0] = int(vals[0])
                vals[1] = int(vals[1])
                vals[2] = int(vals[2])
            builder = OscMessageBuilder(address=address)
            for v in vals:
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
                self.set_channel_fader(ch, float(values[0]))
            else:
                self._send_osc('/live/track/get/volume', (ch - 1) + self.channel_offset)
            return True

        ch_mute_match = re.match(r'^/ch/(\d+)/mute$', address)
        if ch_mute_match:
            ch = int(ch_mute_match.group(1))
            if values:
                self.set_channel_mute(ch, int(values[0]))
            else:
                self._send_osc('/live/track/get/mute', (ch - 1) + self.channel_offset)
            return True

        ch_name_match = re.match(r'^/ch/(\d+)/(\$?name)$', address)
        if ch_name_match:
            ch = int(ch_name_match.group(1))
            if values:
                self.set_channel_name(ch, str(values[0]))
            else:
                self._send_osc('/live/track/get/name', (ch - 1) + self.channel_offset)
            return True

        ch_pan_match = re.match(r'^/ch/(\d+)/pan$', address)
        if ch_pan_match:
            ch = int(ch_pan_match.group(1))
            if values:
                self.set_channel_pan(ch, float(values[0]))
            else:
                self._send_osc('/live/track/get/panning', (ch - 1) + self.channel_offset)
            return True

        main_fdr_match = re.match(r'^/main/(\d+)/fdr$', address)
        if main_fdr_match:
            if values:
                self.set_main_fader(1, float(values[0]))
            else:
                self._send_osc('/live/master/get/volume')
            return True

        # Wing-style delay: dlyon/dly/dlymode — translate to Ableton set_channel_delay
        ch_dly_match = re.match(r'^/ch/(\d+)/in/set/(dlyon|dly|dlymode)$', address)
        if ch_dly_match:
            ch = int(ch_dly_match.group(1))
            if address.endswith('dlyon') and values and values[0] == 0:
                self.set_channel_delay(ch, 0.0, mode="MS")
            elif address.endswith('dly') and values:
                try:
                    val = float(values[0])
                    self.set_channel_delay(ch, val, mode="MS")
                except (TypeError, ValueError):
                    pass
            # dlymode alone: no-op; dlyon=1 without dly: ignore
            return True

        # Wing-style EQ gain only (reset_eq / reset_all_eq use these paths)
        ch_eq_lg = re.match(r"^/ch/(\d+)/eq/lg$", address)
        if ch_eq_lg and values:
            ch = int(ch_eq_lg.group(1))
            self.set_eq_band_gain(ch, "lg", float(values[0]))
            return True

        ch_eq_hg = re.match(r"^/ch/(\d+)/eq/hg$", address)
        if ch_eq_hg and values:
            ch = int(ch_eq_hg.group(1))
            self.set_eq_band_gain(ch, "hg", float(values[0]))
            return True

        ch_eq_ng = re.match(r"^/ch/(\d+)/eq/([1-4])g$", address)
        if ch_eq_ng and values:
            ch = int(ch_eq_ng.group(1))
            n = ch_eq_ng.group(2)
            self.set_eq_band_gain(ch, f"{n}g", float(values[0]))
            return True

        # Pass through as raw AbletonOSC
        return self._send_osc(address, *values)

    # ========== Initial State Scan ==========

    def _scan_initial_state(self):
        """Scan initial Ableton Live state"""
        logger.info("Scanning Ableton Live state...")

        # Get track count
        self._send_osc('/live/song/get/num_tracks')
        time.sleep(0.1)

        # Get all track names
        self._send_osc('/live/song/get/track_names')
        time.sleep(0.3)

        # Scan first 16 tracks (fader, mute, name, panning) from offset
        for i in range(16):
            track_id = self.channel_offset + i
            self._send_osc('/live/track/get/volume', track_id)
            self._send_osc('/live/track/get/mute', track_id)
            self._send_osc('/live/track/get/name', track_id)
            self._send_osc('/live/track/get/panning', track_id)
            time.sleep(0.02)

        # Get master volume
        self._send_osc('/live/master/get/volume')

        # Discover params: mixer (Track Delay), Utility, EQ Eight (unique device indices)
        first_track = self.channel_offset
        devices_to_scan = sorted(
            {
                self.MIXER_DEVICE_INDEX,
                self.utility_device_index,
                self.eq_eight_device_index,
            }
        )
        for dev_id in devices_to_scan:
            self._send_osc(
                '/live/device/get/parameters/name', first_track, dev_id
            )
            time.sleep(0.04)

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
        track_id = (channel - 1) + self.channel_offset
        self._send_osc('/live/track/get/volume', track_id)
        time.sleep(0.1)
        return self.state.get(internal_addr)

    def set_channel_fader(self, channel: int, value: float):
        """
        Set channel fader value.
        Accepts dB values (same interface as WingClient) and converts to Ableton 0-1.
        """
        track_id = (channel - 1) + self.channel_offset
        db_value = float(value)
        if db_value < -144.0:
            db_value = -144.0
        elif db_value > 6.0:
            db_value = 6.0

        vol = _db_to_ableton_vol(db_value)
        logger.info(f"Setting fader ch {channel} (track {track_id}) = {db_value:.2f} dB -> vol {vol:.4f}")
        self._send_osc('/live/track/set/volume', track_id, vol)

        # Update local state
        internal_addr = f"/ch/{channel}/fdr"
        self.state[internal_addr] = db_value

    def get_channel_mute(self, channel: int) -> Optional[int]:
        """Get channel mute state (0=unmuted, 1=muted)"""
        internal_addr = f"/ch/{channel}/mute"
        return self.state.get(internal_addr)

    def set_channel_mute(self, channel: int, value: int):
        """Set channel mute (0=unmuted, 1=muted)"""
        track_id = (channel - 1) + self.channel_offset
        self._send_osc('/live/track/set/mute', track_id, int(value))
        self.state[f"/ch/{channel}/mute"] = int(value)

    def get_channel_gain(self, channel: int) -> Optional[float]:
        """Get channel gain — Ableton has no pre-fader trim, returns track volume (fader) instead."""
        val = self.get_channel_fader(channel)
        return float(val) if val is not None else 0.0

    def set_channel_gain(self, channel: int, value: float):
        """Set channel gain — Ableton has no pre-fader trim; applies to track volume (fader) instead."""
        self.set_channel_fader(channel, value)
        return True

    # ── MixerClientBase bridge methods ─────────────────────────────
    def set_fader(self, channel: int, value_db: float):
        self.set_channel_fader(channel, value_db)

    def get_fader(self, channel: int) -> float:
        val = self.get_channel_fader(channel)
        return float(val) if val is not None else -144.0

    def set_mute(self, channel: int, muted: bool):
        self.set_channel_mute(channel, 1 if muted else 0)

    def get_mute(self, channel: int) -> bool:
        val = self.get_channel_mute(channel)
        return bool(val) if val is not None else False

    def set_gain(self, channel: int, value_db: float):
        self.set_channel_gain(channel, value_db)

    def recall_scene(self, scene_number: int):
        """Recall scene — not supported in Ableton, no-op"""
        logger.warning("recall_scene: Not supported for Ableton")
        return

    def get_channel_pan(self, channel: int) -> Optional[float]:
        """Get channel pan (-100..100)"""
        internal_addr = f"/ch/{channel}/pan"
        return self.state.get(internal_addr)

    def set_channel_pan(self, channel: int, value: float):
        """Set channel pan (-100=left, 0=center, 100=right)"""
        track_id = (channel - 1) + self.channel_offset
        # Convert -100..100 to -1..1
        pan_norm = max(-1.0, min(1.0, float(value) / 100.0))
        self._send_osc('/live/track/set/panning', track_id, pan_norm)
        self.state[f"/ch/{channel}/pan"] = float(value)

    def get_channel_name(self, channel: int, retries: int = 3) -> Optional[str]:
        """Get channel name"""
        track_id = (channel - 1) + self.channel_offset
        internal_addr = f"/ch/{channel}/name"

        # Check cached first
        name = self.state.get(internal_addr)
        if name:
            return name

        # Query from Ableton
        for attempt in range(retries):
            self._send_osc('/live/track/get/name', track_id)
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
        self._send_osc('/live/song/get/track_names')
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
        track_id = (channel - 1) + self.channel_offset
        self._send_osc('/live/track/set/name', track_id, name)
        self.state[f"/ch/{channel}/name"] = name

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
        self._send_osc('/live/master/get/volume')
        time.sleep(0.1)
        return self.state.get(internal_addr)

    def set_main_fader(self, main: int, value: float):
        """Set master fader value (dB)"""
        db_value = float(value)
        vol = _db_to_ableton_vol(db_value)
        logger.info(f"Setting master fader = {db_value:.2f} dB -> vol {vol:.4f}")
        self._send_osc('/live/master/set/volume', vol)
        self.state["/main/1/fdr"] = db_value

    def set_main_mute(self, main: int, value: int):
        """Set main mute — not directly available in Ableton"""
        logger.warning("set_main_mute: Not available in Ableton")
        return False

    def set_main_pan(self, main: int, value: float):
        """Set main pan — limited support in Ableton"""
        pan_norm = max(-1.0, min(1.0, float(value) / 100.0))
        self._send_osc('/live/master/set/panning', pan_norm)

    def set_main_eq_on(self, main: int, on: int):
        """Set main EQ on/off — not directly available in Ableton"""
        logger.warning("set_main_eq_on: Not available in Ableton")
        return False

    # EQ Eight physical bands 1..8: Gain/Freq/Q param indices (from OSC names or fallback).
    # Live 11/12 spacing is +10 per band once Filter rows are skipped (see discovery logs).
    _EQ8_FALLBACK_STEP = 10

    @classmethod
    def _eq_eight_band_fallback(cls) -> Dict[int, Tuple[int, int, int]]:
        """
        Per-physical-band (gain, freq, q) param indices if OSC name discovery fails.

        **Must** match stock Ableton EQ Eight (Live 11/12, EN): ``Device On`` = 0, then
        per-band Filter rows + Frequency / Gain / Resonance. The old fallback ``(1,2,3)``
        was wrong and sent resets to unrelated parameters — EQ in UI did not move.

        Pattern from real OSC logs: band 1 → (7,6,8), then +10 per band for bands 2–8.
        """
        out: Dict[int, Tuple[int, int, int]] = {}
        for b in range(1, 9):
            off = (b - 1) * cls._EQ8_FALLBACK_STEP
            out[b] = (7 + off, 6 + off, 8 + off)
        return out

    @staticmethod
    def _parse_eq_eight_parameter_names(names: List[str]) -> Dict[int, Tuple[int, int, int]]:
        """
        Parse Ableton Live stock EQ Eight parameter names (English UI).

        Skips ``N Filter …`` rows; uses ``N Frequency…``, ``N Gain…``, ``N Resonance…``
        (or ``Res``) for physical bands 1–8.
        """
        fallback = AbletonClient._eq_eight_band_fallback()
        found: Dict[int, Dict[str, int]] = {b: {} for b in range(1, 9)}

        for i, raw in enumerate(names):
            line = raw.strip()
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if not m:
                continue
            bnum = int(m.group(1))
            if bnum < 1 or bnum > 8:
                continue
            tail = m.group(2).strip().lower()
            if tail.startswith("filter "):
                continue
            bmap = found[bnum]
            if "gain" not in bmap and re.match(r"^gain\b", tail):
                bmap["gain"] = i
            if "freq" not in bmap and re.match(r"^frequency\b|^freq\b", tail):
                bmap["freq"] = i
            if "q" not in bmap and re.match(r"^resonance\b|^res\b", tail):
                bmap["q"] = i

        result: Dict[int, Tuple[int, int, int]] = {}
        for b in range(1, 9):
            g = found[b].get("gain")
            f = found[b].get("freq")
            q = found[b].get("q")
            if g is not None and f is not None and q is not None:
                result[b] = (g, f, q)
            else:
                result[b] = fallback[b]
                if found[b]:
                    logger.warning(
                        "EQ Eight phys band %s: incomplete name parse (%s), using fallback %s",
                        b,
                        found[b],
                        fallback[b],
                    )
        return result

    def _ensure_eq_eight_indices(self, track_id: int, timeout: float = 2.0) -> bool:
        """Ensure ``_eq_band_param_indices`` is set (OSC query or fallback)."""
        if self._eq_band_param_indices is not None:
            return True
        self._eq_discovery_event.clear()
        self._send_osc(
            "/live/device/get/parameters/name",
            track_id,
            self.eq_eight_device_index,
        )
        if not self._eq_discovery_event.wait(timeout):
            logger.warning(
                "EQ Eight discovery timeout (device %s, track %s) — using fallback indices",
                self.eq_eight_device_index,
                track_id,
            )
        with self._lock:
            if self._eq_band_param_indices is None:
                self._eq_band_param_indices = self._eq_eight_band_fallback()
                logger.warning(
                    "EQ Eight: using numeric fallback map (discovery missing); "
                    "verify eq_eight_device_index matches EQ Eight on track"
                )
        return True

    def invalidate_eq_eight_indices(self) -> None:
        """Clear cached param map so next ensure re-queries Live (e.g. before EQ reset)."""
        with self._lock:
            self._eq_band_param_indices = None
            self._eq_eight_last_param_names = None

    # ========== EQ Methods (via EQ Eight device) ==========

    def set_eq_on(self, channel: int, on: int):
        """Enable/disable EQ Eight at ``eq_eight_device_index`` (not device 0)."""
        track_id = (channel - 1) + self.channel_offset
        self._send_osc(
            "/live/device/set/enabled",
            track_id,
            self.eq_eight_device_index,
            int(on),
        )

    def get_eq_on(self, channel: int) -> Optional[int]:
        """Get EQ on/off state"""
        return self.state.get(f"/ch/{channel}/eq/on")

    # Wing-style EQ has low shelf + 4 PEQ + high shelf (6 slots). EQ Eight shares one
    # Gain/Freq/Q triple per band — without remapping, low+PEQ1 and PEQ4+high collide.
    # Mapping: low_shelf -> phys 1, Wing PEQ 1..4 -> phys 2..5, high_shelf -> phys 6.
    _EQ8_PHYS_LOW = 1
    _EQ8_PHYS_HIGH = 6

    def _set_eq_eight_physical_band(
        self,
        channel: int,
        phys_band: int,
        freq: float = None,
        gain: float = None,
        q: float = None,
        log_label: str = "",
    ) -> None:
        """Write Gain/Freq/Q to EQ Eight **physical** band index (1..8)."""
        if phys_band < 1 or phys_band > 8:
            logger.warning("EQ Eight phys band out of range: %s", phys_band)
            return

        track_id = (channel - 1) + self.channel_offset
        device_id = self.eq_eight_device_index

        self._ensure_eq_eight_indices(track_id)
        if phys_band not in self._eq_band_param_indices:
            logger.error("Missing EQ Eight index map for phys band %s", phys_band)
            return

        g_idx, f_idx, q_idx = self._eq_band_param_indices[phys_band]

        gain_norm = freq_norm = q_norm = None
        if gain is not None:
            gain_norm = max(0.0, min(1.0, (gain + 15.0) / 30.0))
            self._send_osc(
                "/live/device/set/parameter/value",
                track_id,
                device_id,
                g_idx,
                gain_norm,
            )

        if freq is not None:
            if freq <= 20:
                freq_norm = 0.0
            elif freq >= 20000:
                freq_norm = 1.0
            else:
                freq_norm = math.log10(freq / 20.0) / math.log10(20000.0 / 20.0)
            self._send_osc(
                "/live/device/set/parameter/value",
                track_id,
                device_id,
                f_idx,
                freq_norm,
            )

        if q is not None:
            q_norm = max(0.0, min(1.0, (q - 0.1) / 17.9))
            self._send_osc(
                "/live/device/set/parameter/value",
                track_id,
                device_id,
                q_idx,
                q_norm,
            )

        label = log_label or f"phys{phys_band}"
        logger.info(
            "Ableton EQ OSC: ch=%s track=%s dev=%s %s eq8_phys=%s "
            "param_idx_gfq=(%s,%s,%s) dB=%s Hz=%s Q=%s norm_gfq=(%s,%s,%s)",
            channel,
            track_id,
            device_id,
            label,
            phys_band,
            g_idx,
            f_idx,
            q_idx,
            gain,
            freq,
            q,
            gain_norm,
            freq_norm,
            q_norm,
        )

    def set_eq_band(self, channel: int, band: int, freq: float = None,
                    gain: float = None, q: float = None):
        """
        Set Wing **parametric** band 1..4 on EQ Eight physical bands 2..5 (no collision
        with low shelf on phys 1 or high shelf on phys 6).
        """
        if band not in (1, 2, 3, 4):
            logger.warning("set_eq_band: Invalid Wing PEQ band %s (expected 1-4)", band)
            return
        phys = band + 1
        self._set_eq_eight_physical_band(
            channel, phys, freq=freq, gain=gain, q=q,
            log_label=f"wing_peq{band}->phys{phys}",
        )

    def set_eq_band_gain(self, channel: int, band: str, gain: float):
        """Set EQ band gain — translates band string to numeric band"""
        if band == "lg":
            self._set_eq_eight_physical_band(
                channel, self._EQ8_PHYS_LOW, gain=gain, log_label="lg->phys1",
            )
            return
        if band == "hg":
            self._set_eq_eight_physical_band(
                channel, self._EQ8_PHYS_HIGH, gain=gain, log_label="hg->phys6",
            )
            return
        band_map = {"1g": 1, "2g": 2, "3g": 3, "4g": 4}
        band_num = band_map.get(band)
        if band_num:
            self.set_eq_band(channel, band_num, gain=gain)

    def get_eq_band_gain(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band gain — returns from state if available"""
        return self.state.get(f"/ch/{channel}/eq/{band}")

    def set_eq_band_frequency(self, channel: int, band: str, frequency: float):
        """Set EQ band frequency"""
        if band == "lf":
            self._set_eq_eight_physical_band(
                channel, self._EQ8_PHYS_LOW, freq=frequency, log_label="lf->phys1",
            )
            return
        if band == "hf":
            self._set_eq_eight_physical_band(
                channel, self._EQ8_PHYS_HIGH, freq=frequency, log_label="hf->phys6",
            )
            return
        band_map = {"1f": 1, "2f": 2, "3f": 3, "4f": 4}
        band_num = band_map.get(band)
        if band_num:
            self.set_eq_band(channel, band_num, freq=frequency)

    def get_eq_band_frequency(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band frequency"""
        return self.state.get(f"/ch/{channel}/eq/{band}")

    def set_eq_band_q(self, channel: int, band: str, q: float):
        """Set EQ band Q"""
        if band == "lq":
            self._set_eq_eight_physical_band(
                channel, self._EQ8_PHYS_LOW, q=q, log_label="lq->phys1",
            )
            return
        if band == "hq":
            self._set_eq_eight_physical_band(
                channel, self._EQ8_PHYS_HIGH, q=q, log_label="hq->phys6",
            )
            return
        band_map = {"1q": 1, "2q": 2, "3q": 3, "4q": 4}
        band_num = band_map.get(band)
        if band_num:
            self.set_eq_band(channel, band_num, q=q)

    def set_eq_high_shelf(self, channel: int, gain: float = None, freq: float = None,
                          q: float = None, eq_type: str = None):
        """High shelf -> EQ Eight **physical band 6** (not 4 — avoids PEQ4 clash)."""
        if eq_type is not None:
            logger.debug("set_eq_high_shelf: eq_type ignored on Ableton EQ Eight")
        self._set_eq_eight_physical_band(
            channel,
            self._EQ8_PHYS_HIGH,
            freq=freq,
            gain=gain,
            q=q,
            log_label="high_shelf->phys6",
        )

    def set_eq_low_shelf(self, channel: int, gain: float = None, freq: float = None,
                         q: float = None, eq_type: str = None):
        """Low shelf -> EQ Eight **physical band 1** (Wing PEQ1 uses phys 2)."""
        if eq_type is not None:
            logger.debug("set_eq_low_shelf: eq_type ignored on Ableton EQ Eight")
        self._set_eq_eight_physical_band(
            channel,
            self._EQ8_PHYS_LOW,
            freq=freq,
            gain=gain,
            q=q,
            log_label="low_shelf->phys1",
        )

    def set_eq_mix(self, channel: int, mix: float):
        """Set EQ mix — not available in EQ Eight"""
        logger.warning("set_eq_mix: Not available in Ableton EQ Eight")
        return False

    def set_eq_model(self, channel: int, model: str):
        """Set EQ model — Ableton only has EQ Eight"""
        logger.warning("set_eq_model: Not available in Ableton")
        return False

    def _zero_eq_eight_gains_by_param_names(self, track_id: int) -> int:
        """
        Set every parameter whose English name looks like ``N Gain …`` to 0 dB.

        Uses the same index order as ``/live/device/get/parameters/name`` — most
        reliable reset path when Live UI is English (avoids wrong triple mapping).
        """
        with self._lock:
            names = self._eq_eight_last_param_names
        if not names:
            return 0
        dev = self.eq_eight_device_index
        gain_norm = (0.0 + 15.0) / 30.0  # 0 dB
        count = 0
        for i, raw in enumerate(names):
            line = str(raw).strip()
            if not re.match(r"^\d+\s+Gain\b", line, re.I):
                continue
            if self._send_osc(
                "/live/device/set/parameter/value",
                int(track_id),
                int(dev),
                int(i),
                float(gain_norm),
            ):
                count += 1
            time.sleep(0.02)
        return count

    def reset_channel_eq_gains_zero(self, channel: int) -> bool:
        """
        Reset EQ Eight gains to 0 dB for one logical channel.

        Prefer **name-based** reset (``1 Gain``, ``2 Gain``, … from OSC names).
        If discovery did not return names, fall back to phys map + Wing routing.
        """
        try:
            ch = int(channel)
        except (TypeError, ValueError):
            logger.warning("reset_channel_eq_gains_zero: invalid channel %r", channel)
            return False
        if not self.is_connected or not self.sock:
            logger.warning("reset_channel_eq_gains_zero: not connected")
            return False

        track_id = (ch - 1) + self.channel_offset
        self.invalidate_eq_eight_indices()
        self._ensure_eq_eight_indices(track_id, timeout=3.0)
        with self._lock:
            pmap = self._eq_band_param_indices
            names = self._eq_eight_last_param_names

        self.set_eq_on(ch, 1)
        time.sleep(0.12)

        n_name = self._zero_eq_eight_gains_by_param_names(track_id)
        if n_name > 0:
            logger.info(
                "Ableton: EQ reset by Gain names — %s params -> 0 dB "
                "(ch=%s track=%s dev=%s)",
                n_name,
                ch,
                track_id,
                self.eq_eight_device_index,
            )
            return True

        required = (1, 2, 3, 4, 5, 6)
        if not pmap or not all(p in pmap for p in required):
            logger.error(
                "reset_channel_eq_gains_zero: no OSC names and no phys map 1–6 "
                "(track=%s dev=%s). Check: AbletonOSC recv port, eq_eight_device_index, "
                "EQ Eight on this track.",
                track_id,
                self.eq_eight_device_index,
            )
            return False

        self.set_eq_band_gain(ch, "lg", 0.0)
        for n in (1, 2, 3, 4):
            self.set_eq_band_gain(ch, f"{n}g", 0.0)
            time.sleep(0.03)
        self.set_eq_band_gain(ch, "hg", 0.0)
        for phys in (7, 8):
            if phys in pmap:
                self._set_eq_eight_physical_band(
                    ch, phys, gain=0.0, log_label="reset_flat"
                )

        logger.info(
            "Ableton: EQ Eight gains 0 dB (phys fallback ch=%s track=%s dev=%s)",
            ch,
            track_id,
            self.eq_eight_device_index,
        )
        return True

    # ========== EQ Eight readback (Live → log) ==========

    def request_device_parameter_value(
        self, track_id: int, device_id: int, param_index: int
    ) -> bool:
        """Ask AbletonOSC for one normalized parameter value (reply on recv port)."""
        return self._send_osc(
            "/live/device/get/parameter/value",
            int(track_id),
            int(device_id),
            int(param_index),
        )

    def fetch_eq_eight_physical_bands(
        self,
        channel: int,
        phys_bands: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        settle_s: float = 0.55,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Query Live for EQ Eight Gain/Freq/Q on physical bands, decode to dB / Hz / Q.

        Uses the same normalization as ``_set_eq_eight_physical_band`` (inverse).
        """
        track_id = (channel - 1) + self.channel_offset
        dev = self.eq_eight_device_index
        self._ensure_eq_eight_indices(track_id)
        with self._lock:
            indices = self._eq_band_param_indices
        if not indices:
            logger.warning("fetch_eq_eight_physical_bands: no EQ Eight index map")
            return {}

        keys_to_query: List[Tuple[int, int, int]] = []
        for phys in phys_bands:
            if phys not in indices:
                continue
            g_idx, f_idx, q_idx = indices[phys]
            keys_to_query.extend(
                [
                    (track_id, dev, g_idx),
                    (track_id, dev, f_idx),
                    (track_id, dev, q_idx),
                ]
            )
        # Dedupe while preserving order
        seen = set()
        unique_keys: List[Tuple[int, int, int]] = []
        for k in keys_to_query:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)

        with self._lock:
            for k in unique_keys:
                self._device_param_values.pop(k, None)

        for _, _, pidx in unique_keys:
            self._send_osc(
                "/live/device/get/parameter/value",
                track_id,
                dev,
                int(pidx),
            )
            time.sleep(0.012)

        time.sleep(settle_s)

        out: Dict[int, Dict[str, Any]] = {}
        for phys in phys_bands:
            if phys not in indices:
                continue
            g_idx, f_idx, q_idx = indices[phys]
            with self._lock:
                rg = self._device_param_values.get((track_id, dev, g_idx))
                rf = self._device_param_values.get((track_id, dev, f_idx))
                rq = self._device_param_values.get((track_id, dev, q_idx))
            if rg is None or rf is None or rq is None:
                out[phys] = {
                    "gain_db": None,
                    "freq_hz": None,
                    "q": None,
                    "raw_norm": (rg, rf, rq),
                }
            else:
                out[phys] = {
                    "gain_db": _eq_norm_to_gain_db(rg),
                    "freq_hz": _eq_norm_to_freq_hz(rf),
                    "q": _eq_norm_to_q(rq),
                    "raw_norm": (rg, rf, rq),
                }
        return out

    def log_eq_eight_readback(
        self,
        channel: int,
        *,
        tag: str = "readback",
        phys_bands: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    ) -> None:
        """Poll Live once and log decoded EQ Eight state at INFO."""
        if not self.is_connected:
            logger.warning("log_eq_eight_readback: not connected (ch=%s)", channel)
            return

        _phys_labels = {
            1: "low(Wing shelf)",
            2: "peq1",
            3: "peq2",
            4: "peq3",
            5: "peq4",
            6: "high(Wing shelf)",
        }

        data = self.fetch_eq_eight_physical_bands(channel, phys_bands=phys_bands)
        if not data:
            logger.info(
                "Ableton EQ Live readback [%s]: ch=%s track=%s — no bands (device %s?)",
                tag,
                channel,
                (channel - 1) + self.channel_offset,
                self.eq_eight_device_index,
            )
            return

        parts: List[str] = []
        for phys in sorted(data.keys()):
            row = data[phys]
            label = _phys_labels.get(phys, f"band{phys}")
            if row["gain_db"] is None:
                parts.append(f"phys{phys}({label})=<missing>")
            else:
                parts.append(
                    f"phys{phys}({label}): {row['gain_db']:+.2f}dB "
                    f"{row['freq_hz']:.1f}Hz Q={row['q']:.2f} "
                    f"norm={tuple(round(x, 4) for x in row['raw_norm'])}"
                )

        logger.info(
            "Ableton EQ Live readback [%s]: ch=%s track=%s eq8_dev=%s → %s",
            tag,
            channel,
            (channel - 1) + self.channel_offset,
            self.eq_eight_device_index,
            " | ".join(parts),
        )

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
        track_id = (channel - 1) + self.channel_offset
        send_id = send - 1  # 0-based
        if level is not None:
            vol = _db_to_ableton_vol(level)
            self._send_osc('/live/track/set/send', track_id, send_id, vol)
        if on is not None:
            # Ableton doesn't have send on/off — set to 0 volume if off
            if not on:
                self._send_osc('/live/track/set/send', track_id, send_id, 0.0)

    def set_channel_main_send(self, channel: int, main: int, level: float = None,
                              on: int = None, pre: int = None):
        """Not directly mapped in Ableton"""
        logger.warning("set_channel_main_send: Not mapped for Ableton")
        return False

    # ========== Delay & Phase (via Utility + Track Delay) ==========

    MIXER_DEVICE_INDEX = 0  # Mixer is always device 0 on each track

    def _discover_utility_params(self, track_id: int, timeout: float = 2.0) -> bool:
        """Query device params, find Left Inv / Right Inv. Blocks until response or timeout."""
        if self._utility_discovery_done:
            return True
        self._utility_discovery_event.clear()
        # Only query devices 0,1 — avoid IndexError for tracks with fewer devices
        for dev_id in range(0, 2):
            self._send_osc('/live/device/get/parameters/name', track_id, dev_id)
            time.sleep(0.1)
        if self._utility_discovery_event.wait(timeout):
            return self._utility_left_inv_index is not None
        return False

    def _discover_latency_param(self, track_id: int, timeout: float = 2.0) -> bool:
        """Query mixer device params, find Latency index. Blocks until response or timeout."""
        if self._latency_discovery_done and self._latency_param_index is not None:
            return True
        self._latency_discovery_event.clear()
        self._send_osc('/live/device/get/parameters/name', track_id, self.MIXER_DEVICE_INDEX)
        if self._latency_discovery_event.wait(timeout):
            return self._latency_param_index is not None
        logger.warning("Latency param discovery timeout — mixer params not received")
        return False

    def _run_osascript(self, script: str) -> subprocess.CompletedProcess:
        """Run osascript and return the completed process."""
        return subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )

    def _activate_live_window(self) -> bool:
        """Bring Ableton Live to the front."""
        result = self._run_osascript('tell application "Live" to activate')
        if result.returncode != 0:
            logger.error("Failed to activate Live: %s", result.stderr.strip())
            return False
        time.sleep(0.15)
        return True

    def _get_live_window_bounds(self) -> Optional[tuple[int, int, int, int]]:
        """Get position and size of the main Ableton window."""
        position_script = (
            'tell application "System Events" to tell process "Live" '
            f'to get position of window "{self.live_window_title}"'
        )
        size_script = (
            'tell application "System Events" to tell process "Live" '
            f'to get size of window "{self.live_window_title}"'
        )
        pos_result = self._run_osascript(position_script)
        size_result = self._run_osascript(size_script)
        if pos_result.returncode != 0 or size_result.returncode != 0:
            logger.error(
                "Failed to read Live window bounds: pos=%s size=%s",
                pos_result.stderr.strip(),
                size_result.stderr.strip(),
            )
            return None

        try:
            x_str, y_str = [part.strip() for part in pos_result.stdout.strip().split(",")]
            w_str, h_str = [part.strip() for part in size_result.stdout.strip().split(",")]
            return int(x_str), int(y_str), int(w_str), int(h_str)
        except Exception as exc:
            logger.error("Failed to parse Live window bounds: %s", exc)
            return None

    def _is_track_options_enabled(self) -> bool:
        """Check whether View -> Arrangement Track Controls -> Track Options is enabled."""
        script = (
            'tell application "System Events" to tell process "Live" '
            'to get value of attribute "AXMenuItemMarkChar" '
            'of menu item "Track Options" of menu 1 of menu item "Arrangement Track Controls" '
            'of menu 1 of menu bar item "View" of menu bar 1'
        )
        result = self._run_osascript(script)
        return result.returncode == 0 and result.stdout.strip() == "✓"

    def _ensure_track_options_enabled(self) -> bool:
        """Ensure Track Options are visible in Arrangement view."""
        if self._is_track_options_enabled():
            return True

        click_script = (
            'tell application "System Events" to tell process "Live" '
            'to click menu item "Track Options" of menu 1 of menu item "Arrangement Track Controls" '
            'of menu 1 of menu bar item "View" of menu bar 1'
        )
        result = self._run_osascript(click_script)
        if result.returncode != 0:
            logger.error("Failed to enable Track Options: %s", result.stderr.strip())
            return False
        time.sleep(0.2)
        return self._is_track_options_enabled()

    def calibrate_track_delay_ui_from_mouse(self, channel: int = 1) -> bool:
        """
        Capture Track Delay field coordinates from the current mouse position.
        Use after manually clicking the Track Delay field for the given channel.
        """
        try:
            from Quartz import NSEvent
        except Exception as exc:
            logger.error("Quartz unavailable for calibration: %s", exc)
            return False

        bounds = self._get_live_window_bounds()
        if bounds is None:
            return False

        x0, y0, width, height = bounds
        mouse = NSEvent.mouseLocation()
        mouse_x = float(mouse.x)
        # Quartz origin is bottom-left; UI scripting uses top-left.
        # Convert using the current display/window geometry.
        mouse_y_from_top = float((y0 + height) - mouse.y)
        rel_x = (mouse_x - x0) / width
        rel_y = mouse_y_from_top / height

        self.track_delay_x_ratio = rel_x
        self.track_delay_first_row_center_y_ratio = (
            rel_y - ((channel - self.track_delay_base_channel) * self.track_delay_row_pitch_y_ratio)
        )
        self._track_delay_ui_calibrated = True
        logger.info(
            "Calibrated Track Delay UI from mouse: channel=%s x_ratio=%.4f first_row_y_ratio=%.4f",
            channel,
            self.track_delay_x_ratio,
            self.track_delay_first_row_center_y_ratio,
        )
        return True

    def _click_track_delay_field(self, channel: int) -> bool:
        """Click the Track Delay field for the given channel using calibrated ratios."""
        bounds = self._get_live_window_bounds()
        if bounds is None:
            return False

        try:
            from Quartz import (
                CGEventCreateMouseEvent,
                CGEventPost,
                kCGEventLeftMouseDown,
                kCGEventLeftMouseUp,
                kCGHIDEventTap,
            )
        except Exception as exc:
            logger.error("Quartz unavailable for UI click: %s", exc)
            return False

        x0, y0, width, height = bounds
        target_x = int(round(x0 + (width * self.track_delay_x_ratio)))
        target_y = int(
            round(
                y0
                + (
                    height
                    * (
                        self.track_delay_first_row_center_y_ratio
                        + ((channel - self.track_delay_base_channel) * self.track_delay_row_pitch_y_ratio)
                    )
                )
            )
        )

        logger.info(
            "Clicking Track Delay field: ch=%s x=%s y=%s (ratios x=%.4f first_y=%.4f pitch=%.4f)",
            channel,
            target_x,
            target_y,
            self.track_delay_x_ratio,
            self.track_delay_first_row_center_y_ratio,
            self.track_delay_row_pitch_y_ratio,
        )

        mouse_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (target_x, target_y), 0)
        mouse_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (target_x, target_y), 0)
        CGEventPost(kCGHIDEventTap, mouse_down)
        CGEventPost(kCGHIDEventTap, mouse_up)
        time.sleep(0.12)
        return True

    def _input_text_into_focused_field(self, text: str) -> bool:
        """Replace the value in the currently focused text field."""
        script = (
            'tell application "Live" to activate\n'
            'tell application "System Events"\n'
            '    keystroke "a" using command down\n'
            '    delay 0.08\n'
            f'    keystroke "{text}"\n'
            '    delay 0.08\n'
            '    key code 36\n'
            'end tell'
        )
        result = self._run_osascript(script)
        if result.returncode != 0:
            logger.error("Failed to input Track Delay value: %s", result.stderr.strip())
            return False
        time.sleep(0.1)
        return True

    def set_channel_delay_ui(self, channel: int, value: float) -> bool:
        """Set Track Delay via Ableton UI automation."""
        delay_ms = max(0.0, min(self.MAX_DELAY_MS, float(value)))
        delay_text = f"{delay_ms:.2f}".rstrip("0").rstrip(".")

        if not self._activate_live_window():
            return False
        if not self._ensure_track_options_enabled():
            return False
        if not self._click_track_delay_field(channel):
            return False
        if not self._input_text_into_focused_field(delay_text):
            return False

        logger.info("Channel %s: Delay %.2f ms (UI Track Delay)", channel, delay_ms)
        return True

    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        """
        Set channel delay — UI automation отключён (не работает должным образом).
        value: delay in ms. AbletonOSC не предоставляет Track Delay по OSC.
        """
        logger.warning(
            "set_channel_delay: UI automation отключён для Ableton (ch %s, %.2f ms). "
            "Track Delay настраивайте вручную в Ableton.",
            channel, value
        )
        return False

    def set_channel_phase_invert(self, channel: int, value: int):
        """
        Set channel phase invert via Utility device.
        Индексы Left Inv / Right Inv определяются автоматически по имени параметра.
        value: 0=normal, 1=inverted.
        """
        track_id = (channel - 1) + self.channel_offset
        phase_val = 1.0 if value else 0.0

        if self._utility_left_inv_index is None:
            self._discover_utility_params(track_id)

        dev_id = self._utility_device_index_found if self._utility_device_index_found is not None else self.utility_device_index
        left_idx = self._utility_left_inv_index if self._utility_left_inv_index is not None else self.UTILITY_PHASE_INVERT_L
        right_idx = self._utility_right_inv_index if self._utility_right_inv_index is not None else self.UTILITY_PHASE_INVERT_R

        ok_l = self._send_osc('/live/device/set/parameter/value',
                              track_id, dev_id, left_idx, phase_val)
        ok_r = self._send_osc('/live/device/set/parameter/value',
                              track_id, dev_id, right_idx, phase_val)
        ok = ok_l and ok_r
        if ok:
            logger.info(f"Channel {channel}: Phase invert {value} (Utility dev={dev_id} L={left_idx} R={right_idx})")
        return ok

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
