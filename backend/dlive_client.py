"""
Allen & Heath dLive mixer client — MIDI over TCP/IP.

Protocol reference:
  https://www.allen-heath.com/content/uploads/2024/06/dLive-MIDI-Over-TCP-Protocol-V2.0.pdf

Default port: 51328 (plain TCP) or 51329 (TLS).
"""

import logging
import math
import socket
import ssl
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from mixer_client_base import MixerClientBase

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
DLIVE_TCP_PORT = 51328
DLIVE_TLS_PORT = 51329
SYSEX_HEADER = bytes([0xF0, 0x00, 0x00, 0x1A, 0x50, 0x10, 0x01, 0x00])
SYSEX_END = 0xF7

# Fader value mapping:
#   0x0000 = -inf,  0x2AAA = 0 dB,  0x3FFF = +10 dB
FADER_MAX = 0x3FFF          # 16383
FADER_0DB = 0x2AAA          # 10922
FADER_NEG_INF = 0x0000

# NRPN MSB constants for PEQ bands (Input channels)
PEQ_BANDS = {
    1: {"freq": 0x40, "gain": 0x41, "q": 0x42},
    2: {"freq": 0x44, "gain": 0x45, "q": 0x46},
    3: {"freq": 0x48, "gain": 0x49, "q": 0x4A},
    4: {"freq": 0x4C, "gain": 0x4D, "q": 0x4E},
}

# HPF
HPF_FREQ_MSB = 0x50
HPF_ON_MSB = 0x51

# Channel type → MIDI channel offset from base
CHANNEL_TYPE_OFFSET = {
    "input": 0,
    "mono_group": 1,
    "stereo_group": 1,
    "mono_aux": 2,
    "mono_fx_send": 2,
    "stereo_aux": 2,
    "stereo_fx_send": 2,
    "mono_matrix": 3,
    "stereo_matrix": 3,
    "dca": 4,
}

# dLive colour codes
COLOURS = {
    "off": 0x00, "red": 0x01, "green": 0x02, "yellow": 0x03,
    "blue": 0x04, "purple": 0x05, "lt_blue": 0x06, "white": 0x07,
}


# ══════════════════════════════════════════════════════════════
#  Value conversion helpers
# ══════════════════════════════════════════════════════════════

def db_to_fader_value(db: float) -> int:
    """Convert dB to dLive 14-bit fader value (0x0000–0x3FFF).

    Mapping:  -inf → 0x0000,  0 dB → 0x2AAA,  +10 dB → 0x3FFF.
    Uses linear interpolation between -inf/0dB and 0dB/+10dB regions.
    """
    if db <= -100.0:
        return FADER_NEG_INF
    if db >= 10.0:
        return FADER_MAX
    if db <= 0.0:
        # Map -100 .. 0 dB → 0 .. 0x2AAA  (log-linear approximation)
        ratio = (db + 100.0) / 100.0
        return int(ratio * FADER_0DB)
    # 0 .. +10 dB → 0x2AAA .. 0x3FFF
    ratio = db / 10.0
    return FADER_0DB + int(ratio * (FADER_MAX - FADER_0DB))


def fader_value_to_db(value: int) -> float:
    """Convert 14-bit fader value to dB."""
    value = max(0, min(FADER_MAX, value))
    if value == 0:
        return -100.0
    if value <= FADER_0DB:
        ratio = value / FADER_0DB
        return ratio * 100.0 - 100.0
    ratio = (value - FADER_0DB) / (FADER_MAX - FADER_0DB)
    return ratio * 10.0


def db_to_gain_value(db: float) -> int:
    """Convert preamp gain dB to 14-bit pitchbend value.

    Range roughly -inf .. +60 dB mapped to 0x0000 .. 0x3FFF.
    """
    if db <= -100.0:
        return 0
    if db >= 60.0:
        return FADER_MAX
    ratio = (db + 100.0) / 160.0
    return max(0, min(FADER_MAX, int(ratio * FADER_MAX)))


def freq_to_nrpn(freq_hz: float) -> int:
    """Convert frequency in Hz to 14-bit NRPN value (log scale 20–20000 Hz)."""
    freq_hz = max(20.0, min(20000.0, freq_hz))
    log_ratio = math.log10(freq_hz / 20.0) / math.log10(20000.0 / 20.0)
    return max(0, min(FADER_MAX, int(log_ratio * FADER_MAX)))


def db_to_eq_gain(db: float) -> int:
    """Convert EQ gain dB (±15 dB) to 14-bit NRPN value."""
    db = max(-15.0, min(15.0, db))
    ratio = (db + 15.0) / 30.0
    return max(0, min(FADER_MAX, int(ratio * FADER_MAX)))


def q_to_nrpn(q: float) -> int:
    """Convert Q value (0.3 – 35) to 14-bit NRPN value (log scale)."""
    q = max(0.3, min(35.0, q))
    log_ratio = math.log10(q / 0.3) / math.log10(35.0 / 0.3)
    return max(0, min(FADER_MAX, int(log_ratio * FADER_MAX)))


# ══════════════════════════════════════════════════════════════
#  MIDI message builders (pure functions → easy to unit-test)
# ══════════════════════════════════════════════════════════════

def build_nrpn(channel: int, msb: int, lsb: int, value: int) -> bytes:
    """Build a 4-message NRPN sequence (8 bytes).

    channel:  MIDI channel 0-15
    msb/lsb:  NRPN parameter address
    value:    14-bit value 0-16383
    """
    ch = channel & 0x0F
    val_msb = (value >> 7) & 0x7F
    val_lsb = value & 0x7F
    return bytes([
        0xB0 | ch, 0x63, msb & 0x7F,   # NRPN MSB
        0xB0 | ch, 0x62, lsb & 0x7F,   # NRPN LSB
        0xB0 | ch, 0x06, val_msb,       # Data Entry MSB
        0xB0 | ch, 0x26, val_lsb,       # Data Entry LSB
    ])


def build_note_on(channel: int, note: int, velocity: int) -> bytes:
    """Note On message (3 bytes)."""
    return bytes([0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F])


def build_pitchbend(channel: int, value: int) -> bytes:
    """Pitch Bend message (3 bytes).  value is 14-bit (0-16383)."""
    lsb = value & 0x7F
    msb = (value >> 7) & 0x7F
    return bytes([0xE0 | (channel & 0x0F), lsb, msb])


def build_sysex(payload: bytes) -> bytes:
    """Wrap *payload* in the dLive SysEx header + F7 trailer."""
    return SYSEX_HEADER + payload + bytes([SYSEX_END])


def build_program_change(channel: int, bank_msb: int, bank_lsb: int, program: int) -> bytes:
    """Bank Select + Program Change (5 bytes)."""
    ch = channel & 0x0F
    return bytes([
        0xB0 | ch, 0x00, bank_msb & 0x7F,
        0xB0 | ch, 0x20, bank_lsb & 0x7F,
        0xC0 | ch, program & 0x7F,
    ])


# ══════════════════════════════════════════════════════════════
#  DLiveClient
# ══════════════════════════════════════════════════════════════

class DLiveClient(MixerClientBase):
    """Allen & Heath dLive mixer client using MIDI over TCP."""

    def __init__(
        self,
        ip: str = "192.168.1.70",
        port: int = DLIVE_TCP_PORT,
        tls: bool = False,
        midi_base_channel: int = 0,
        user_profile: int = 0,
        password: str = "",
    ):
        self.ip = ip
        self.port = port
        self.tls = tls
        self.midi_base_channel = midi_base_channel & 0x0F
        self.user_profile = user_profile
        self.password = password

        self._sock: Optional[socket.socket] = None
        self._ssl_sock: Optional[ssl.SSLSocket] = None
        self.is_connected = False

        self._lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reconnect_interval = 3.0

        self.state: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self._channel_names: Dict[int, str] = {}

        logger.info(f"DLiveClient initialized for {ip}:{port} (TLS={tls}, base_ch={midi_base_channel})")

    # ── Connection ─────────────────────────────────────────────

    def connect(self, timeout: float = 5.0) -> bool:
        if self.is_connected:
            return True
        try:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw.settimeout(timeout)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw.connect((self.ip, self.port))

            if self.tls:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                self._ssl_sock = ctx.wrap_socket(raw, server_hostname=self.ip)
                self._sock = self._ssl_sock
                # Authenticate
                auth_msg = f"{chr(self.user_profile)},{self.password}"
                self._sock.sendall(auth_msg.encode("utf-8"))
                resp = self._sock.recv(256).decode("utf-8", errors="ignore")
                if "AuthOK" not in resp:
                    logger.error(f"dLive auth failed: {resp}")
                    self._sock.close()
                    return False
                logger.info("dLive TLS auth successful")
            else:
                self._sock = raw

            self.is_connected = True
            self._stop_event.clear()

            self._recv_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self._recv_thread.start()

            logger.info(f"Connected to dLive at {self.ip}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"dLive connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        self._stop_event.set()
        self.is_connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._ssl_sock = None
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        logger.info("dLive disconnected")

    # ── Send raw bytes (thread-safe) ───────────────────────────

    def _send_raw(self, data: bytes):
        with self._lock:
            if not self.is_connected or not self._sock:
                return
            try:
                self._sock.sendall(data)
            except Exception as e:
                logger.error(f"dLive send error: {e}")
                self.is_connected = False

    # ── Receiver loop ──────────────────────────────────────────

    def _receiver_loop(self):
        """Read incoming MIDI bytes from dLive and dispatch callbacks."""
        while not self._stop_event.is_set() and self.is_connected:
            try:
                if self._sock is None:
                    break
                self._sock.settimeout(1.0)
                data = self._sock.recv(4096)
                if not data:
                    logger.warning("dLive connection closed by remote")
                    self.is_connected = False
                    break
                self._parse_incoming(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_connected:
                    logger.error(f"dLive receiver error: {e}")
                    self.is_connected = False
                break

        # Auto-reconnect
        if not self._stop_event.is_set():
            logger.info(f"dLive reconnecting in {self._reconnect_interval}s …")
            time.sleep(self._reconnect_interval)
            self.connect()

    def _parse_incoming(self, data: bytes):
        """Best-effort parse of incoming MIDI messages for state tracking."""
        i = 0
        while i < len(data):
            status = data[i]
            if status == 0xF0:
                # SysEx — find F7
                end = data.find(b'\xF7', i)
                if end == -1:
                    break
                i = end + 1
            elif status & 0xF0 == 0xB0:
                # CC / NRPN — 3 bytes
                if i + 2 < len(data):
                    self._handle_cc(status & 0x0F, data[i + 1], data[i + 2])
                i += 3
            elif status & 0xF0 == 0x90:
                if i + 2 < len(data):
                    self._handle_note(status & 0x0F, data[i + 1], data[i + 2])
                i += 3
            elif status & 0xF0 == 0xE0:
                if i + 2 < len(data):
                    self._handle_pitchbend(status & 0x0F, data[i + 1], data[i + 2])
                i += 3
            else:
                i += 1

    def _handle_cc(self, channel: int, cc: int, value: int):
        key = f"cc/{channel}/{cc}"
        self.state[key] = value
        self._notify(key, value)

    def _handle_note(self, channel: int, note: int, velocity: int):
        key = f"note/{channel}/{note}"
        self.state[key] = velocity
        self._notify(key, velocity)

    def _handle_pitchbend(self, channel: int, lsb: int, msb: int):
        value = (msb << 7) | lsb
        key = f"pitchbend/{channel}"
        self.state[key] = value
        self._notify(key, value)

    def _notify(self, address: str, *args):
        for pattern, cbs in list(self.callbacks.items()):
            if pattern == "*" or pattern == address:
                for cb in cbs:
                    try:
                        cb(address, *args)
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")

    # ── MixerClientBase interface ──────────────────────────────

    def send(self, address: str, *args):
        """Generic send — for dLive we accept raw bytes in args[0]."""
        if args and isinstance(args[0], (bytes, bytearray)):
            self._send_raw(args[0])

    def subscribe(self, address: str, callback: Callable):
        self.callbacks.setdefault(address, []).append(callback)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    # ── Faders (NRPN) ─────────────────────────────────────────

    def _midi_channel(self, channel_type: str = "input") -> int:
        return (self.midi_base_channel + CHANNEL_TYPE_OFFSET.get(channel_type, 0)) & 0x0F

    def set_fader(self, channel: int, value_db: float, channel_type: str = "input"):
        midi_ch = self._midi_channel(channel_type)
        fader_val = db_to_fader_value(value_db)
        msg = build_nrpn(midi_ch, 0x00, channel - 1, fader_val)
        self._send_raw(msg)

    def get_fader(self, channel: int) -> float:
        key = f"fader/{channel}"
        val = self.state.get(key)
        if val is not None:
            return fader_value_to_db(int(val))
        return -100.0

    # ── Mutes (Note On) ───────────────────────────────────────

    def set_mute(self, channel: int, muted: bool, channel_type: str = "input"):
        midi_ch = self._midi_channel(channel_type)
        note = channel - 1
        velocity = 0x7F if muted else 0x3F
        msg = build_note_on(midi_ch, note, velocity) + build_note_on(midi_ch, note, 0x00)
        self._send_raw(msg)

    def get_mute(self, channel: int) -> bool:
        key = f"note/{self._midi_channel()}/{channel - 1}"
        val = self.state.get(key, 0)
        return val >= 0x40

    # ── Gain / Preamp (Pitchbend) ──────────────────────────────

    def set_gain(self, channel: int, value_db: float):
        midi_ch = self._midi_channel("input")
        gain_val = db_to_gain_value(value_db)
        msg = build_pitchbend(midi_ch, gain_val)
        self._send_raw(msg)

    # ── Phantom 48V (SysEx) ────────────────────────────────────

    def set_phantom(self, channel: int, enabled: bool):
        midi_ch = self._midi_channel("input")
        val = 0x7F if enabled else 0x00
        payload = bytes([midi_ch, 0x0B, channel - 1, val])
        self._send_raw(build_sysex(payload))

    # ── Pad (SysEx) ────────────────────────────────────────────

    def set_pad(self, channel: int, enabled: bool):
        midi_ch = self._midi_channel("input")
        val = 0x7F if enabled else 0x00
        payload = bytes([midi_ch, 0x0A, channel - 1, val])
        self._send_raw(build_sysex(payload))

    # ── PEQ (NRPN) ────────────────────────────────────────────

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        """Set PEQ band parameters (band 1-4)."""
        if band not in PEQ_BANDS:
            logger.warning(f"Invalid PEQ band {band}, must be 1-4")
            return
        midi_ch = self._midi_channel("input")
        ch_idx = channel - 1
        params = PEQ_BANDS[band]

        msgs = b""
        msgs += build_nrpn(midi_ch, params["freq"], ch_idx, freq_to_nrpn(freq))
        msgs += build_nrpn(midi_ch, params["gain"], ch_idx, db_to_eq_gain(gain))
        msgs += build_nrpn(midi_ch, params["q"], ch_idx, q_to_nrpn(q))
        self._send_raw(msgs)

    # ── HPF (NRPN) ─────────────────────────────────────────────

    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        midi_ch = self._midi_channel("input")
        ch_idx = channel - 1
        msgs = build_nrpn(midi_ch, HPF_FREQ_MSB, ch_idx, freq_to_nrpn(freq))
        on_val = FADER_MAX if enabled else 0
        msgs += build_nrpn(midi_ch, HPF_ON_MSB, ch_idx, on_val)
        self._send_raw(msgs)

    # ── Scene Recall (Program Change) ──────────────────────────

    def recall_scene(self, scene_number: int):
        """Recall scene by number (0-499).  Scenes span 4 banks of 125."""
        midi_ch = self._midi_channel("input")
        bank = scene_number // 128
        program = scene_number % 128
        msg = build_program_change(midi_ch, 0, bank, program)
        self._send_raw(msg)

    # ── Channel Name (SysEx) ───────────────────────────────────

    def set_channel_name(self, channel: int, name: str, channel_type: str = "input"):
        midi_ch = self._midi_channel(channel_type)
        name_bytes = name.encode("ascii", errors="replace")[:8]
        payload = bytes([midi_ch, 0x03, channel - 1]) + name_bytes
        self._send_raw(build_sysex(payload))

    def get_channel_name(self, channel: int) -> str:
        return self._channel_names.get(channel, f"Ch {channel}")

    # ── Channel Colour (SysEx) ─────────────────────────────────

    def set_channel_colour(self, channel: int, colour: str, channel_type: str = "input"):
        midi_ch = self._midi_channel(channel_type)
        col_val = COLOURS.get(colour, 0x00)
        payload = bytes([midi_ch, 0x05, channel - 1, col_val])
        self._send_raw(build_sysex(payload))

    # ── Find snap by name (not directly available via MIDI) ────

    def find_snap_by_name(self, name: str) -> Optional[int]:
        """dLive does not support scene query by name over MIDI — returns None."""
        logger.info(f"find_snap_by_name not available over dLive MIDI protocol")
        return None
